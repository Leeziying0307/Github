import torch
import torch.nn as nn
import numpy as np
from DAMSM import CA_NET
from miscc.config import cfg
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d
from GlobalAttention import MaskCrossAtten as MaskAtten
from GlobalAttention import MultiHeadAttention as MaskAttention
BatchNorm = SynchronizedBatchNorm2d

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.BatchNorm2d(in_planes),
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()

        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM

        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()
        self.ca_net = CA_NET()
        self.IN = InstanceNorm()
        self.conv_img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )
        

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        nz = self.in_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, self.gf_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(self.gf_dim * 8 * 4 * 4),
            GLU())
        
        self.att1 = MaskAtten(self.gf_dim, self.ef_dim)
        self.att2 = MaskAttention(self.gf_dim, self.ef_dim, n_head=3)
        self.residual = self._make_layer(ResBlock, self.gf_dim * 3)
        self.upsample1 = upBlock(self.gf_dim * 4, self.gf_dim * 8)
        self.upsample2 = upBlock(self.gf_dim * 8, self.gf_dim * 4)
        self.upsample3 = upBlock(self.gf_dim * 4, self.gf_dim * 2)
        self.upsample4 = upBlock(self.gf_dim * 2, self.gf_dim * 1)
        self.upsample = upBlock(self.gf_dim * 3, self.gf_dim * 1)
        self.block = nn.Sequential(
        # nn.BatchNorm2d(in_planes),
                      conv3x3(self.gf_dim * 4, self.gf_dim * 2),
                      InstanceNorm(self.gf_dim * 2),
                      GLU())
    

    def forward(self, z_code, c_code, word_embs):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x 3 x 256 x 256
        """
        c_code, mu, logvar = self.ca_net(c_code)
        c_z_code = torch.cat((c_code, z_code), 1)
        out_code = self.fc(c_z_code)
        h_code = out_code.view(-1, self.gf_dim*4, 4, 4)
        #out_code = self.upsample1(out_code)
        #out_code = self.upsample2(out_code)
        #out_code = self.upsample3(out_code)
        h_code = self.block(h_code)
        c_code, att = self.att1(h_code, word_embs)
        c_code_channel, att_channel = self.att2(c_code, word_embs)
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_code = self.residual(h_c_c_code)
        h_code = self.upsample(h_code)
        # 8x8
        c_code, att = self.att1(h_code, word_embs)
        c_code_channel, att_channel = self.att2(c_code, word_embs)
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_code = self.residual(h_c_c_code)
        h_code = self.upsample(h_code)
        #16x16
        c_code, att = self.att1(h_code, word_embs)
        c_code_channel, att_channel = self.att2(c_code, word_embs)
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_code = self.residual(h_c_c_code)
        h_code = self.upsample(h_code)
        #32x32
        c_code, att = self.att1(h_code, word_embs)
        c_code_channel, att_channel = self.att2(c_code, word_embs)
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_code = self.residual(h_c_c_code)
        h_code = self.upsample(h_code)
        #64x64

        c_code, att = self.att1(h_code, word_embs)
        c_code_channel, att_channel = self.att2(c_code, word_embs)
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_code = self.residual(h_c_c_code)
        h_code = self.upsample(h_code)
        #128x128
        c_code, att = self.att1(h_code, word_embs)
        c_code_channel, att_channel = self.att2(c_code, word_embs)
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        out_code = self.residual(h_c_c_code)
        out_code = self.upsample(out_code)
        #256x256
        out_img = self.conv_img(out_code)

        return out_img, att, mu, logvar



class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):

        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64
        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 16)  # 8
        self.block4 = resD(ndf * 16, ndf * 16)  # 4
        self.block5 = resD(ndf * 16, ndf * 16)  # 4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self, x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True, spectral_norm=False):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


def linear(in_feat, out_feat, bias=True, spectral_norm=False):
    lin = nn.Linear(in_feat, out_feat, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(lin)
    else:
        return lin