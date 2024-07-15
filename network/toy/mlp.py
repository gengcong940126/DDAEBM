import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1., spec_norm=False):
    lin = nn.Linear(in_channels, out_channels)
    if spec_norm:
        lin=nn.utils.spectral_norm(lin)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

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
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb
#%%


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final=False, activation_fn=nn.ReLU,batch_norm=False):
        super(MLP, self).__init__()
        layers = []
        layers_bn=[]
        activation_fn_list=[]
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            if batch_norm:
                layers_bn.append(nn.BatchNorm1d(layer_width))
            activation_fn_list.append(activation_fn())
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.layers_bn = torch.nn.ModuleList(layers_bn)
        self.activate_final = activate_final
        self.activation_fn = torch.nn.ModuleList(activation_fn_list)
        self.batch_norm = batch_norm

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if self.batch_norm:
                x=self.layers_bn[i](layer(x))
                x = self.activation_fn[i](x)
            else:
                x = self.activation_fn[i](layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn[-1](x)
        return x


class Energy_small(torch.nn.Module):

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[300, 300], x_dim=2, act_fn=nn.PReLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(2*t_enc_dim,
                       layer_widths=decoder_layers + [1],
                       activate_final=False,
                       activation_fn=act_fn)

        self.t_encoder = MLP(self.temb_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn)

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn)

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out = self.net(h)
        return out

class Generator_small(nn.Module):
   
    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[300, 300], x_dim=2,x_out_dim=2, act_fn=nn.PReLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(2*t_enc_dim,
                       layer_widths=decoder_layers + [x_out_dim],
                       activate_final=False,
                       activation_fn=act_fn,batch_norm=True)

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn)

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn,batch_norm=True)

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out = self.net(h)
        return out

class Encoder_small(nn.Module):

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[300, 300], x_dim=2,x_out_dim=2, act_fn=nn.PReLU):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(2*t_enc_dim,
                       layer_widths=decoder_layers + encoder_layers,
                       activate_final=False,
                       activation_fn=act_fn,batch_norm=True)

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn)

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn,batch_norm=True)

        self.fc_mu = nn.Linear(encoder_layers[-1] , x_out_dim)
        self.fc_var = nn.Linear(encoder_layers[-1], x_out_dim)
    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out= self.net(h)
        mean=self.fc_mu(out)
        logvar=self.fc_var(out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps * std + mean
        return sample,mean,std,logvar
    def logprob(self, z,mean,std):
        logprob=torch.distributions.Normal(mean,std).log_prob(z).sum(1, keepdims=True)
        return logprob

