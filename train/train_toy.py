import argparse
import torch
import numpy as np
import json
import os
import math
import datetime
import random
import torch.nn as nn
import itertools
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from data_process import toy_data
import torchvision.transforms as transforms
import torchplot as plt
import shutil


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def is_debugging():
  import sys
  gettrace = getattr(sys, 'gettrace', None)

  if gettrace is None:
    assert 0, ('No sys.gettrace')
  elif gettrace():
    return True
  else:
    return False

def visualize_results(real_data,fake_samples,n_time,epoch,path,netD):

    real_data = real_data.detach().cpu().numpy()
    fake_samples = fake_samples.detach().cpu().numpy()
    plt.clf()
    ax1 = plt.subplot(1, 3, 1, aspect="equal", title='real data')
    ax1.scatter(real_data[:, 0], real_data[:, 1], s=1)
    ax2 = plt.subplot(1, 3, 2, aspect="equal", title='fake data')
    ax2.scatter(fake_samples[:, 0], fake_samples[:, 1], s=1)
    ax3 = plt.subplot(1, 3, 3, aspect="equal")
    plt_toy_density(lambda x: netD(x,torch.full((x.size(0),),0).to(x.device)), ax3,
                             low=-4, high=4,
                             title="p(x)")

    plt.savefig(os.path.join(path, 'visualization_epoch_{}.png'.format(epoch)))
def plt_toy_density(logdensity, ax, npts=100,
                    title="$q(x)$", device="cuda:0", low=-4, high=4, exp=True):
    """
    Plot density of toy data.
    """
    side = np.linspace(low, high, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    logpx = logdensity(x)
    logpx = logpx.squeeze()

    if exp:
        logpx = logpx
        logpx = logpx - logpx.logsumexp(0)
        px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
        px = px / px.sum()

    else:
        logpx = logpx - logpx.logsumexp(0)
        px = logpx.cpu().detach().numpy().reshape(npts, npts)

    im = ax.imshow(px, origin='lower', extent=[low, high, low, high])
    # plt.colorbar(im)
    ax.set_title(title)

# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def diag_normal_NLL(z, z_mu, z_log_sigma):
   
    nll = 0.5 * torch.sum(z_log_sigma, dim=1) + \
          0.5 * torch.sum((torch.mul(z - z_mu, z - z_mu) / (1e-6 + torch.exp(z_log_sigma))).squeeze(), dim=1)
    return nll.squeeze()

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
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
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = var[0]
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


def get_sigmas(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)

    return var.to(device)

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.sigmas=self.sigmas.to(dtype=torch.float32)
        self.a_s=self.a_s.to(dtype=torch.float32)
        self.a_s_cum = self.a_s_cum.to(device,dtype=torch.float32)
        self.sigmas_cum = self.sigmas_cum.to(device,dtype=torch.float32)
        self.a_s_prev = self.a_s_prev.to(device,dtype=torch.float32)

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

def p_condition(coeff,x_t,x_tp1,t):
    logp = torch.distributions.Normal(extract(coeff.a_s, t + 1, x_t.shape) * x_t, \
                                                     extract(coeff.sigmas, t + 1, x_t.shape)).log_prob(x_tp1).sum(1,keepdims=True)
    return logp
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
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)

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
            p_samples=mean + nonzero_mask[:,None]* torch.exp(0.5 * log_var) * noise
        else:
            p_samples = mean +  torch.exp(0.5 * log_var) * noise

        return p_samples
    sample_x_pos= p_sample(x_0, x_t, t,gen)

    return sample_x_pos

def sample_posterior_mean(coefficients, x_0, x_t, t):

    mean = extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0 \
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t


    return mean

def q_condition(coefficients, x_t, x_tp1, x_0,t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    mean, _, log_var = q_posterior(x_0, x_tp1, t)
    #log_var = log_var.clamp(min=coefficients.posterior_log_variance_clipped[1])
    log_var = log_var.clamp(min=math.log(1e-2))
    logq = torch.distributions.Normal(mean, torch.exp(0.5 * log_var)).log_prob(x_t).sum(1,keepdims=True)
    return logq

def sample_from_model(coefficients, generator, n_time, x_init, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat([x,latent_z],1), t_time)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
    return x

# %%

def train(args):
    from network.toy.mlp import Energy_small, Generator_small,Encoder_small
    if args.seed == -1:
        args.seed = random.randint(1,4096)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
    else:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:0')

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    netG = Generator_small(x_dim=4).to(device)

    netD = Energy_small().to(device)
   
    netE = Encoder_small(x_dim=4).to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(itertools.chain(netG.parameters(),netE.parameters()), lr=args.lr_g, betas=(args.beta1, args.beta2))

    exp = args.exp
    parent_dir = "./saved_info/{}".format(args.dataset)
    exp_path = os.path.join(parent_dir, exp)
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)
        shutil.copytree('network/toy', os.path.join(exp_path, 'network/toy'))
    with open("{}/args.txt".format(exp_path), 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)
        print('\n', netD, '\n', netG, '\n', netE, file=f)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    mse_loss = nn.MSELoss(reduction='sum').to(device)

    for epoch in range(init_epoch, args.num_epoch + 1):
       
        data = torch.from_numpy(toy_data.data_process(args.dataset, batch_size=args.batch_size)).float()
        
        for p in netD.parameters():
            p.requires_grad = True

        netD.zero_grad()

        # sample from p(x_0)
        real_data =data.to(device, non_blocking=True)

        t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
       
        x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
        x_t.requires_grad_()
        latent_z = torch.randn(batch_size, nz, device=device)
        x_0_predict = netG(torch.cat([x_tp1.detach(), latent_z], 1), t)
        x_pos_sample= sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

        D_real_r=netD(x_t, t)

        D_fake_r = netD(x_pos_sample.detach(), t)

        if args.lazy_reg is None or global_step % args.lazy_reg == 0:
            logpxtp1_condition_t = p_condition(coeff, x_t, x_tp1, t)
            D_real = D_real_r + logpxtp1_condition_t

            grad_real = torch.autograd.grad(
                outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
      
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                         )
            grad_penalty = ( args.r1_gamma/ 2 * grad_penalty).mean()
            
            errD = (-D_real_r + D_fake_r).mean() + grad_penalty
        else:
           
            errD =  (-D_real_r + D_fake_r).mean()

        errD.backward()
        #torch.nn.utils.clip_grad_norm_(netD.parameters(), args.max_normE)
        # Update D
        optimizerD.step()

    # update G
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        netE.zero_grad()
       
        latent_z_pos, mean, std,logvar = netE(torch.cat([x_t, x_tp1], 1), t)
        x_0_predict_E = netG(torch.cat([x_tp1,latent_z_pos],1), t)
        logqxt_condition_tp1 = q_condition(pos_coeff, x_t, x_tp1, x_0_predict_E, t)
        err_Recon = -logqxt_condition_tp1.mean()
        errKld = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
        logpxtp1_condition_t_pos = p_condition(coeff, x_pos_sample, x_tp1, t)
        E_F = (netD(x_pos_sample, t) + logpxtp1_condition_t_pos).mean()
        latent_z_gen, mean_gen, std_gen,logvar_gen = netE(torch.cat([x_pos_sample, x_tp1], 1),t)
        errLatent = 0.1*torch.mean(diag_normal_NLL(latent_z, mean_gen, logvar_gen))
        errG = -E_F+errLatent+err_Recon+errKld
        errG.backward()
        optimizerG.step()

        global_step += 1
        if epoch % args.visualize_every == 0:
           
            print('epoch {} , G Loss: {}, D Loss: {}'.format(epoch,  errG.item(), errD.item()))
            test_data = torch.from_numpy(toy_data.data_process(args.dataset, batch_size=1000)).float().to(device)
            x_t_1 = torch.randn_like(test_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1,args)

            with torch.no_grad():
                visualize_results(test_data, fake_sample,args.num_timesteps, epoch,exp_path,netD)
        if args.save_content:
            if epoch % args.save_content_every == 0:
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                            'netD_dict': netD.state_dict(), 'netE_dict': netE.state_dict(),
                            'optimizerD': optimizerD.state_dict()}

                torch.save(content, os.path.join(exp_path, 'content.pth'))


"""
    Usage:

        export PORT=6007
        export TIME_STR=1
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=./
        python ./train/train_toy.py
    :return:
    """

# %%
if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser('DDAEBM parameters')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=True)

    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.01,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=0.1,
                        help='beta_max for diffusion')

    # geenrator and training
    parser.add_argument('--exp', default='experiment_toy_default', help='name of experiment')
    parser.add_argument('--dataset', default='25gaussians', choices=['25gaussians','pinwheel','swissroll'],help='name of dataset')
    parser.add_argument('--nz', type=int, default=2)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_dim', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=100000)

    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')

    parser.add_argument('--r1_gamma', type=float, default=0.02, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')
    parser.add_argument('--max_normE', type=float, default=0.1, help='max norm allowed for E')

    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--visualize_every', type=int, default=5000, help='save content for resuming every x epochs')
    parser.add_argument('--save_content_every', type=int, default=20000, help='save ckpt every x epochs')


    args = parser.parse_args()
    
    if is_debugging() == False:
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d_%H-%M-%S")
        args.exp=os.path.join(args.exp, time)

   
    print('start')

    train(args)
    
