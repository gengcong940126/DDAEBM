# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from . import up_or_down_sampling
from . import dense_layer
from . import layers

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb
#%%
class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1),
        spec_norm=False, use_scale=False

    ):
        super().__init__()
     
        
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(
                    conv2d(in_channel, out_channel, kernel_size, padding=padding,spec_norm=spec_norm),
                    )

        
        self.conv2 = nn.Sequential(
                    conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.,spec_norm=spec_norm,use_scale=use_scale)
                    )
        self.dense_t1= dense(t_emb_dim, out_channel,spec_norm=spec_norm)


        self.act = act
        
            
        self.skip = nn.Sequential(
                    conv2d(in_channel, out_channel, 1, padding=0, bias=False),
                    )
        
            

    def forward(self, input, t_emb):
        
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
       
        out = self.act(out)
       
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        
        
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)


        return out
    
class Encoder_small(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=128, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        self.conv1 = nn.Conv2d(nc, ngf, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.dense_t1 = dense(t_emb_dim, ngf)
        self.dense_t2 = dense(t_emb_dim, ngf*2)
        self.dense_t3 = dense(t_emb_dim, ngf*4)
        self.dense_t4 = dense(t_emb_dim, ngf*8)
        self.conv51 = nn.Conv2d(ngf * 8, nz, 4, 1, 0)  # for mu
        self.conv52 = nn.Conv2d(ngf * 8, nz, 4, 1, 0)
        self.act = act

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))
        input_x = torch.cat((x, x_t), dim=1)
        oE_l1 = self.act(self.conv1(input_x))
        oE_l1 += self.dense_t1(t_embed)[..., None, None]
        oE_l2 = self.act(self.conv2(oE_l1))
        oE_l2 += self.dense_t2(t_embed)[..., None, None]
        oE_l3 = self.act(self.conv3(oE_l2))
        oE_l3 += self.dense_t3(t_embed)[..., None, None]
        oE_l4 = self.act(self.conv4(oE_l3))
        oE_l4 += self.dense_t4(t_embed)[..., None, None]
        mean = self.conv51(oE_l4).squeeze()
        logvar = self.conv52(oE_l4).squeeze()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps * std + mean
        return sample,mean,std,logvar
    def logprob(self, z,mean,std):
        logprob=torch.distributions.Normal(mean,std).log_prob(z).sum(1, keepdims=True)
        return logprob
class Encoder_64(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=128, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        self.conv1 = nn.Conv2d(nc, ngf, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
        self.dense_t1 = dense(t_emb_dim, ngf)
        self.dense_t2 = dense(t_emb_dim, ngf*2)
        self.dense_t3 = dense(t_emb_dim, ngf*4)
        self.dense_t4 = dense(t_emb_dim, ngf*8)
        self.dense_t5 = dense(t_emb_dim, ngf * 16)
        self.conv61 = nn.Conv2d(ngf * 16, nz, 4, 1, 0)  # for mu
        self.conv62 = nn.Conv2d(ngf * 16, nz, 4, 1, 0)
        self.act = act

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))
        input_x = torch.cat((x, x_t), dim=1)
        oE_l1 = self.act(self.conv1(input_x))
        oE_l1 += self.dense_t1(t_embed)[..., None, None]
        oE_l2 = self.act(self.conv2(oE_l1))
        oE_l2 += self.dense_t2(t_embed)[..., None, None]
        oE_l3 = self.act(self.conv3(oE_l2))
        oE_l3 += self.dense_t3(t_embed)[..., None, None]
        oE_l4 = self.act(self.conv4(oE_l3))
        oE_l4 += self.dense_t4(t_embed)[..., None, None]
        oE_l5 = self.act(self.conv5(oE_l4))
        oE_l5 += self.dense_t5(t_embed)[..., None, None]
        mean = self.conv61(oE_l5).squeeze()
        logvar = self.conv62(oE_l5).squeeze()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps * std + mean
        return sample,mean,std,logvar
    def logprob(self, z,mean,std):
        logprob=torch.distributions.Normal(mean,std).log_prob(z).sum(1, keepdims=True)
        return logprob

class Encoder_large(nn.Module):
    
    def __init__(self, nc=1, ngf=32, t_emb_dim=128,nz=100, act=nn.LeakyReLU(0.2)):
        super().__init__()
       
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv2 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv5 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv6 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear_mean = dense(ngf * 8, nz)
        self.end_linear_logvar = dense(ngf * 8, nz)
        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h = self.start_conv(input_x)
        h = self.conv1(h, t_embed)

        h = self.conv2(h, t_embed)

        h = self.conv3(h, t_embed)
        h = self.conv4(h, t_embed)
        h = self.conv5(h, t_embed)

        out = self.conv6(h, t_embed)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        mean = self.end_linear_mean(out)
        logvar = self.end_linear_logvar(out)
        logvar = -F.softplus(logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps * std + mean
        return sample, mean, std, logvar


