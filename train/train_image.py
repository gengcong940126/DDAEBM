import argparse
import torch
import numpy as np
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
import datetime
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from data_process.lsun import LSUN
from data_process.lmdb_datasets import LMDBDataset
import json
import math
import itertools
from utils.utils import overwrite_opt,CropCelebA64
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil



def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def is_debugging():
    import sys
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        assert 0, ('No sys.gettrace')
    elif gettrace():
        return True
    else:
        return False


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def diag_normal_NLL(z, z_mu, z_log_sigma):
    # define the Negative Log Probability of Normal which has diagonal cov
    # input: [batch nz, 1, 1] squeeze it to batch nz
    # return: shape is [batch]
    nll = 0.5 * torch.sum(z_log_sigma, dim=1) + \
          0.5 * torch.sum((torch.mul(z - z_mu, z - z_mu) / (1e-6 + torch.exp(z_log_sigma))).squeeze(), dim=1)
    return nll.squeeze()

def get_time_schedule(args,device):
    if args.dataset=='lsun':
        temp = np.concatenate([np.arange(args.num_timesteps) * (1000 // ((args.num_timesteps - 1) * 2)), [999]])
        t=torch.from_numpy(temp)/999
        return t.to(device)
    else:
        n_timestep = args.num_timesteps
        eps_small = 1e-3
        t = np.arange(0, n_timestep + 1, dtype=np.float64)
        t = t / n_timestep
        t = torch.from_numpy(t) * (1. - eps_small) + eps_small
        return t.to(device)
    
def get_sigma_schedule(args, device):
    beta_min = args.beta_min
    beta_max = args.beta_max
    t=get_time_schedule(args,device)

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
        
    alpha_bars = 1.0 - var
    betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(max=0.999)

    first = var[0]
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)

def p_condition(coeff, x_t, x_tp1, t):
    logp = torch.distributions.Normal(extract(coeff.a_s, t + 1, x_t.shape) * x_t, \
                                      extract(coeff.sigmas, t + 1, x_t.shape)).log_prob(x_tp1).sum([1, 2, 3])
    return logp[:, None]

def q_condition(coefficients, x_t, x_tp1, x_0, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    mean, _, log_var = q_posterior(x_0, x_tp1, t)
    log_var = log_var.clamp(min=coefficients.posterior_log_variance_clipped[1])
    #log_var = log_var.clamp(min=math.log(1e-2))
    logq = torch.distributions.Normal(mean, torch.exp(0.5 * log_var)).log_prob(x_t).sum([1, 2, 3])
    return logq[:, None]

def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_pos = self.alphas_cumprod[1:]
        self.alphas_cumprod_prev = self.alphas_cumprod[:-1]
        self.posterior_variance = self.betas[1:] * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod_pos)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas[1:] * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod_pos))
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas[1:]) / (1 - self.alphas_cumprod_pos))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-4))


def sample_posterior(coefficients, x_0, x_t, t,gen=True):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t,gen):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)
        if gen:
            nonzero_mask = (1 - (t == 0).type(torch.float32))
            return mean + nonzero_mask[:,None,None,None]* torch.exp(0.5 * log_var) * noise
        else:

            return mean + torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t,gen)

    return sample_x_pos

def sample_posterior_mean(coefficients, x_0, x_t, t):

    mean = extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0 \
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
    return mean

def sample_from_model(coefficients, generator, n_time, x_init, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time,latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            
    return x


def train(rank, gpu, args):
    from network.image.encoder import Encoder_small, Encoder_64, Encoder_large
    from network.image.ncsnpp_generator_adagn import NCSNpp
    from network.image.ncsnpp import NCSNpp as NCSN_D
   
    from EMA import EMA
    if args.seed == -1:
        args.seed = random.randint(1, 4096)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    else:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    random.seed(args.seed+ rank)
    np.random.seed(args.seed+ rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
   
    device = torch.device('cuda:{}'.format(gpu))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    if args.dataset == 'cifar10':
        dataset = CIFAR10('./data/cifar10', train=True, transform=transforms.Compose([
        transforms.Resize(32), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    elif args.dataset == 'celeba64':
        train_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = LMDBDataset(root='./data/celeba64_lmdb/', name='celeba64', \
                            train=True, transform=train_transform,is_encoded=True)

    elif args.dataset == 'lsun128':
        train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = LSUN(root='data/lsun_church', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)


    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)


    netG = NCSNpp(args).to(device)

    netD = NCSN_D(args).to(device)

    if args.dataset=='cifar10':
        netE = Encoder_small(nc=2*args.num_channels, ngf=args.ngf, nz=args.nz,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    elif args.dataset == 'celeba64':
        netE = Encoder_64(nc=2*args.num_channels, ngf=args.ngf, nz=args.nz,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    elif args.dataset == 'lsun128':
        netE = Encoder_large(nc=2*args.num_channels, ngf=args.ngf, nz=args.nz,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)


    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())
    broadcast_params(netE.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(itertools.chain(netG.parameters(),netE.parameters()), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
        optimizerD = EMA(optimizerD, ema_decay=args.ema_decay)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])
    netE = nn.parallel.DistributedDataParallel(netE, device_ids=[gpu])

    exp = args.exp
    parent_dir = "./saved_info/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('network/image', os.path.join(exp_path, 'network/image'))
    
        with open("{}/args.txt".format(exp_path), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
            print('\n', netD, '\n', netG, '\n', netE, '\n', file=f)

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    weight = 0.6 * (T ** math.log(0.2 / 0.6, 0.5))

    if args.resume:
        checkpoint_file = os.path.join(args.resume_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G,E
        netG.load_state_dict(checkpoint['netG_dict'])
        netE.load_state_dict(checkpoint['netE_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:

        global_step, epoch, init_epoch = 0, 0, 0

    mse_loss = nn.MSELoss(reduction='sum').to(device)
    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            real_data = x.to(device, non_blocking=True)
           
            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            D_real_r = netD(x_t, t)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1, t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            D_fake_r = netD(x_pos_sample.detach(), t)
            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                logpxtp1_condition_t = p_condition(coeff, x_t, x_tp1, t)
                D_real = D_real_r + logpxtp1_condition_t
             
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                errD = (-D_real_r + D_fake_r).mean()+grad_penalty
            else:
                errD = (-D_real_r + D_fake_r).mean()
            errD.backward()
            #torch.nn.utils.clip_grad_norm_(netD.parameters(), 1)
            optimizerD.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            netE.zero_grad()

            logpxtp1_condition_t_pos = p_condition(coeff, x_pos_sample, x_tp1, t)
            E_F = (netD(x_pos_sample, t) + logpxtp1_condition_t_pos).mean()
            latent_z_gen, mean_gen, _, logvar_gen = netE(x_pos_sample, t, x_tp1)
            errLatent = 0.1 * torch.mean(diag_normal_NLL(latent_z, mean_gen, logvar_gen))
            latent_z_pos, mean, std, logvar = netE(x_t, t, x_tp1)
            x_0_predict_E = netG(x_tp1, t, latent_z_pos)
            if args.dataset=='lsun128':
                logqxt_condition_tp1 = q_condition(pos_coeff, x_t, x_tp1, x_0_predict_E, t)
                err_Recon = -(logqxt_condition_tp1.squeeze()*weight[t+1].to(device)).mean()
            else:
                x_pos_mean = sample_posterior_mean(pos_coeff, x_0_predict, x_tp1, t)
                err_Recon = mse_loss(x_t, x_pos_mean) / batch_size

            errKld = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
            errG = -E_F + errLatent + errKld + err_Recon

            errG.backward()
            optimizerG.step()

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, Recon:{}, kld:{}, errlatent:{},E_F:{}, D Loss: {}'.format(epoch, iteration, errG.item(),
                                             err_Recon.item(), errKld.item(),errLatent.item(),E_F.item(), errD.item()))
                   
                    with open("{}/args.txt".format(exp_path), 'a') as f:
                        print(
                            'epoch {} iteration{}, G Loss: {}, Recon:{}, kld:{}, errlatent:{},E_F:{}, D Loss: {}'.format(
                                epoch, iteration, errG.item(),
                                err_Recon.item(), errKld.item(), errLatent.item(), E_F.item(), errD.item()),file=f)
                   
        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            netD.eval()
            netG.eval()
            
            if epoch % args.visualize_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizerD.swap_parameters_with_ema(store_params_in_ema=True)
                x_t_1 = torch.randn_like(real_data)
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)
                fake_sample = to_range_0_1(fake_sample)
                torchvision.utils.save_image(fake_sample,
                                            os.path.join(exp_path, 'sample_epoch_{}.png'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizerD.swap_parameters_with_ema(store_params_in_ema=True)
            netG.train()
            netD.train()
           

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'netE_dict': netE.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizerD.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                torch.save(netD.state_dict(), os.path.join(exp_path, 'netD_{}.pth'.format(epoch)))
                torch.save(netE.state_dict(), os.path.join(exp_path, 'netE_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizerD.swap_parameters_with_ema(store_params_in_ema=True)
        if args.num_process_per_node > 1:
            dist.barrier()

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6022'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
"""
    Usage:

        export PORT=6006
        export TIME_STR=1
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        export PYTHONPATH=./
        python ./train/train_image.py

    :return:
    """
if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser('DDAEBM parameters')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')
    parser.add_argument('--network_type', type=str, default='ncsnpp', choices=['ncsnpp', 'ddpmpp'],
                        help='network type')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=4,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 2, 2],
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--num_res_blocks_E', type=int, default=2,
                        help='number of resnet blocks per scale for energy')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='experiment_image', help='name of experiment')
    parser.add_argument('--resume_path', default='', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'celeba64', 'lsun128'], help='name of dataset')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--r1_gamma', type=float, default=0.02, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')
    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=50, help='save ckpt every x epochs')
    parser.add_argument('--visualize_every', type=int, default=10, help='save ckpt every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.2',
                        help='address for master')

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        from config.cifar10 import cifar10_config
        args = overwrite_opt(args, cifar10_config)
    elif args.dataset == 'celeba64':
        from config.celeba64 import celeba64_config
        args = overwrite_opt(args, celeba64_config)
    elif args.dataset == 'lsun128':
        from config.lsun128 import lsun128_config
        args = overwrite_opt(args, lsun128_config)
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    if is_debugging() == False:
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d_%H-%M-%S")
        args.exp = os.path.join(args.exp, time)
    print('start')
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)


        for p in processes:
            p.join()
    else:
        
        init_processes(0, size, train, args)

