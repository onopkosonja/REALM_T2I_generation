from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class affine(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.c1(h)

        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return self.c2(h)


class Generator(nn.Module):
    def __init__(self, resnet_path, ngf=64, nz=100, nn_ch=15):
        super().__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.load_state_dict(torch.load(resnet_path))
        self.ngf = ngf

        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)  # 100 x 1 x 1 -> (ngf * 8) x 4 x 4
        self.block0 = G_Block(ngf * 8, ngf * 8)  # 4x4
        self.block1 = G_Block(ngf * 8 + 512 * 5, ngf * 8)  # 4x4
        self.block2 = G_Block(ngf * 8 + 256 * 5, ngf * 8)  # 8x8
        self.block3 = G_Block(ngf * 8 + 128 * 5, ngf * 8)  # 16x16
        self.block4 = G_Block(ngf * 8 + 64 * 5, ngf * 4)  # 32x32
        self.block5 = G_Block(ngf * 4, ngf * 2)  # 64x64
        self.block6 = G_Block(ngf * 2, ngf * 1)  # 128x128 -> 256x256

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, c, nn_imgs):
        bs = len(x)
        nn_imgs = nn_imgs.reshape(bs * 5, 3, 256, 256)
        out_nn = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(nn_imgs))))  # 64 x 64 x 64
        nn4 = self.resnet.layer1(out_nn)  # 64 x 64 x 64
        nn3 = self.resnet.layer2(nn4)     # 128 x 32 x 32
        nn2 = self.resnet.layer3(nn3)     # 256 x 16 x 16
        nn1 = self.resnet.layer4(nn2)     # 512 x 8 x 8

        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)
        out = self.block0(out, c)  # 4 x 4

        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, nn1.reshape(bs, -1, 8, 8)], 1)
        out = self.block1(out, c)  # 8 x 8

        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, nn2.reshape(bs, -1, 16, 16)], 1)
        out = self.block2(out, c)  # 16 x 16

        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, nn3.reshape(bs, -1, 32, 32)], 1)
        out = self.block3(out, c)  # 32 x 32

        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, nn4.reshape(bs, -1, 64, 64)], 1)
        out = self.block4(out, c)  # 64 x 64

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out, c)  # 128 x 128

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out, c)  # 256 x 256

        out = self.conv_img(out)
        return out


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super().__init__()
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


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()

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
