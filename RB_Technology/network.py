import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm


# -----------------------------------------------------------------------
#                          Base Network
# -----------------------------------------------------------------------
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[{self.__class__.__name__}] Total parameters: {num_params / 1e6:.2f}M")

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if 'BatchNorm2d' in classname:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif ('Conv' in classname or 'Linear' in classname) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f"Init type '{init_type}' not implemented")
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)

    def forward(self, *inputs):
        pass


# -----------------------------------------------------------------------
#                       Segmentation Generator
# -----------------------------------------------------------------------
class SegGenerator(BaseNetwork):
    """
    U-Net style segmentation network that predicts the semantic parse map
    of the person wearing the target clothing item.
    """
    def __init__(self, opt, input_nc, output_nc=13, norm_layer=nn.InstanceNorm2d):
        super(SegGenerator, self).__init__()

        self.enc1 = self._block(input_nc, 64, norm_layer)
        self.enc2 = self._block(64, 128, norm_layer)
        self.enc3 = self._block(128, 256, norm_layer)
        self.enc4 = self._block(256, 512, norm_layer)
        self.enc5 = self._block(512, 1024, norm_layer)

        self.up6  = self._up(1024, 512, norm_layer)
        self.dec6 = self._block(1024, 512, norm_layer)
        self.up7  = self._up(512, 256, norm_layer)
        self.dec7 = self._block(512, 256, norm_layer)
        self.up8  = self._up(256, 128, norm_layer)
        self.dec8 = self._block(256, 128, norm_layer)
        self.up9  = self._up(128, 64, norm_layer)
        self.dec9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), norm_layer(64), nn.ReLU(),
            nn.Conv2d(64,  64, 3, padding=1), norm_layer(64), nn.ReLU(),
            nn.Conv2d(64, output_nc, 3, padding=1)
        )

        self.pool    = nn.MaxPool2d(2)
        self.drop    = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)

    def _block(self, in_c, out_c, norm):
        return nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1), norm(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), norm(out_c), nn.ReLU()
        )

    def _up(self, in_c, out_c, norm):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_c, out_c, 3, padding=1), norm(out_c), nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.drop(self.enc4(self.pool(e3)))
        e5 = self.drop(self.enc5(self.pool(e4)))

        d6 = self.dec6(torch.cat((e4, self.up6(e5)), 1))
        d7 = self.dec7(torch.cat((e3, self.up7(d6)), 1))
        d8 = self.dec8(torch.cat((e2, self.up8(d7)), 1))
        d9 = self.dec9(torch.cat((e1, self.up9(d8)), 1))
        return self.sigmoid(d9)


# -----------------------------------------------------------------------
#                   Geometric Matching Module (GMM)
# -----------------------------------------------------------------------
class FeatureExtraction(BaseNetwork):
    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        super(FeatureExtraction, self).__init__()
        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, 4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]
        for _ in range(1, num_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]
        layers += [
            nn.Conv2d(nf, 512, 3, padding=1), nn.ReLU(), norm_layer(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        ]
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.model(x)


class FeatureCorrelation(nn.Module):
    def forward(self, fA, fB):
        b, c, h, w = fA.size()
        fA = fA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        fB = fB.reshape(b, c, h * w)
        return torch.bmm(fA, fB).reshape(b, w * h, h, w)


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, 4, stride=2, padding=1), norm_layer(512), nn.ReLU(),
            nn.Conv2d(512, 256, 4, stride=2, padding=1), norm_layer(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), norm_layer(128), nn.ReLU(),
            nn.Conv2d(128,  64, 3, padding=1), norm_layer(64),  nn.ReLU()
        )
        self.linear = nn.Linear(64 * (input_nc // 16), output_size)
        self.tanh   = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        return self.tanh(self.linear(x.reshape(x.size(0), -1)))


class TpsGridGen(nn.Module):
    """Thin Plate Spline grid generator for clothing deformation."""
    def __init__(self, opt, dtype=torch.float):
        super(TpsGridGen, self).__init__()
        gX, gY = np.meshgrid(
            np.linspace(-0.9, 0.9, opt.load_width),
            np.linspace(-0.9, 0.9, opt.load_height)
        )
        gX = torch.tensor(gX, dtype=dtype).unsqueeze(0).unsqueeze(3)
        gY = torch.tensor(gY, dtype=dtype).unsqueeze(0).unsqueeze(3)

        self.N = opt.grid_size ** 2
        coords = np.linspace(-0.9, 0.9, opt.grid_size)
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype).reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype).reshape(self.N, 1)
        P_X_base, P_Y_base = P_X.clone(), P_Y.clone()
        Li = self._compute_L_inverse(P_X, P_Y).unsqueeze(0)
        P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
        P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)

        for name, buf in [('grid_X', gX), ('grid_Y', gY), ('P_X_base', P_X_base),
                          ('P_Y_base', P_Y_base), ('Li', Li), ('P_X', P_X), ('P_Y', P_Y)]:
            self.register_buffer(name, buf, False)

    def _compute_L_inverse(self, X, Y):
        N = X.size(0)
        Xmat, Ymat = X.expand(N, N), Y.expand(N, N)
        D2 = (Xmat - Xmat.t()).pow(2) + (Ymat - Ymat.t()).pow(2)
        D2[D2 == 0] = 1
        K = D2 * torch.log(D2)
        O = torch.ones(N, 1)
        Z = torch.zeros(3, 3)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.t(), Z), 1)), 0)
        return torch.inverse(L)

    def _apply_tps(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        bs = theta.size(0)
        Q_X = theta[:, :self.N].squeeze(3) + self.P_X_base.expand(bs, -1, -1).squeeze(2)
        Q_Y = theta[:, self.N:].squeeze(3) + self.P_Y_base.expand(bs, -1, -1).squeeze(2)

        ph, pw = points.size(1), points.size(2)
        P_X = self.P_X.expand(1, ph, pw, 1, self.N)
        P_Y = self.P_Y.expand(1, ph, pw, 1, self.N)
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand(bs, -1, -1), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand(bs, -1, -1), Q_Y)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand(bs, -1, -1), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand(bs, -1, -1), Q_Y)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)

        pX = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, 1, self.N)
        pY = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, 1, self.N)
        if bs != 1:
            P_X = P_X.expand_as(pX)
            P_Y = P_Y.expand_as(pY)
        dX, dY = pX - P_X, pY - P_Y
        D2 = dX.pow(2) + dY.pow(2)
        D2[D2 == 0] = 1
        U = D2 * torch.log(D2)

        pbX = points[:, :, :, 0].unsqueeze(3)
        pbY = points[:, :, :, 1].unsqueeze(3)
        if bs == 1:
            pbX = pbX.expand(bs, ph, pw, 1)
            pbY = pbY.expand(bs, ph, pw, 1)

        X_prime = (A_X[:, :, :, :, 0] + A_X[:, :, :, :, 1] * pbX
                   + A_X[:, :, :, :, 2] * pbY + (W_X * U.expand_as(W_X)).sum(4))
        Y_prime = (A_Y[:, :, :, :, 0] + A_Y[:, :, :, :, 1] * pbX
                   + A_Y[:, :, :, :, 2] * pbY + (W_Y * U.expand_as(W_Y)).sum(4))
        return torch.cat((X_prime, Y_prime), 3)

    def forward(self, theta):
        return self._apply_tps(theta, torch.cat((self.grid_X, self.grid_Y), 3))


class GMM(nn.Module):
    """Geometric Matching Module — warps cloth to fit person body."""
    def __init__(self, opt, inputA_nc, inputB_nc):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(inputA_nc)
        self.extractionB = FeatureExtraction(inputB_nc)
        self.correlation = FeatureCorrelation()
        self.regression  = FeatureRegression(
            input_nc=(opt.load_width // 64) * (opt.load_height // 64),
            output_size=2 * opt.grid_size ** 2
        )
        self.gridGen = TpsGridGen(opt)

    def forward(self, inputA, inputB):
        fA = F.normalize(self.extractionA(inputA), dim=1)
        fB = F.normalize(self.extractionB(inputB), dim=1)
        theta = self.regression(self.correlation(fA, fB))
        return theta, self.gridGen(theta)


# -----------------------------------------------------------------------
#                        ALIAS Generator
# -----------------------------------------------------------------------
class MaskNorm(nn.Module):
    def __init__(self, norm_nc):
        super(MaskNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

    def _norm_region(self, region, mask):
        b, c, h, w = region.size()
        n = mask.sum((2, 3), keepdim=True).clamp(min=1)
        mu = region.sum((2, 3), keepdim=True) / n
        return self.norm(region + (1 - mask) * mu) * torch.sqrt(n / (h * w))

    def forward(self, x, mask):
        mask = mask.detach()
        return self._norm_region(x * mask, mask) + self._norm_region(x * (1 - mask), 1 - mask)


class ALIASNorm(nn.Module):
    def __init__(self, norm_type, norm_nc, label_nc):
        super(ALIASNorm, self).__init__()
        self.noise_scale = nn.Parameter(torch.zeros(norm_nc))
        pf_type = norm_type[len('alias'):]
        if pf_type == 'batch':    self.pf_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif pf_type == 'instance': self.pf_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif pf_type == 'mask':   self.pf_norm = MaskNorm(norm_nc)
        else: raise ValueError(f"Unknown norm type: {pf_type}")
        self.conv_shared = nn.Sequential(nn.Conv2d(label_nc, 128, 3, padding=1), nn.ReLU())
        self.conv_gamma  = nn.Conv2d(128, norm_nc, 3, padding=1)
        self.conv_beta   = nn.Conv2d(128, norm_nc, 3, padding=1)

    def forward(self, x, seg, misalign_mask=None):
        b, c, h, w = x.size()
        noise = (torch.randn(b, w, h, 1).cuda() * self.noise_scale).transpose(1, 3)
        normalized = self.pf_norm(x + noise) if misalign_mask is None else self.pf_norm(x + noise, misalign_mask)
        actv  = self.conv_shared(seg)
        return normalized * (1 + self.conv_gamma(actv)) + self.conv_beta(actv)


class ALIASResBlock(nn.Module):
    def __init__(self, opt, in_nc, out_nc, use_mask_norm=True):
        super(ALIASResBlock, self).__init__()
        self.learned_shortcut = (in_nc != out_nc)
        mid_nc = min(in_nc, out_nc)
        self.conv0 = nn.Conv2d(in_nc,  mid_nc, 3, padding=1)
        self.conv1 = nn.Conv2d(mid_nc, out_nc, 3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_nc, out_nc, 1, bias=False)
        stype = opt.norm_G
        if stype.startswith('spectral'):
            stype = stype[len('spectral'):]
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        snc = opt.semantic_nc + (1 if use_mask_norm else 0)
        ntype = 'aliasmask' if use_mask_norm else stype
        self.norm0 = ALIASNorm(ntype, in_nc,  snc)
        self.norm1 = ALIASNorm(ntype, mid_nc, snc)
        if self.learned_shortcut:
            self.norm_s = ALIASNorm(ntype, in_nc, snc)
        self.relu = nn.LeakyReLU(0.2)

    def _shortcut(self, x, seg, mm):
        return self.conv_s(self.norm_s(x, seg, mm)) if self.learned_shortcut else x

    def forward(self, x, seg, mm=None):
        seg = F.interpolate(seg, size=x.shape[2:], mode='nearest')
        if mm is not None:
            mm = F.interpolate(mm, size=x.shape[2:], mode='nearest')
        xs = self._shortcut(x, seg, mm)
        dx = self.conv0(self.relu(self.norm0(x, seg, mm)))
        dx = self.conv1(self.relu(self.norm1(dx, seg, mm)))
        return xs + dx


class ALIASGenerator(BaseNetwork):
    """Final synthesis network that generates the try-on image."""
    def __init__(self, opt, input_nc):
        super(ALIASGenerator, self).__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers
        self.sh, self.sw = self._latent_size(opt)
        nf = opt.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, 3, padding=1)
        for i in range(1, 8):
            self.add_module(f'conv_{i}', nn.Conv2d(input_nc, 16, 3, padding=1))
        self.head   = ALIASResBlock(opt, nf*16,      nf*16)
        self.mid0   = ALIASResBlock(opt, nf*16 + 16, nf*16)
        self.mid1   = ALIASResBlock(opt, nf*16 + 16, nf*16)
        self.up0    = ALIASResBlock(opt, nf*16 + 16, nf*8)
        self.up1    = ALIASResBlock(opt, nf*8  + 16, nf*4)
        self.up2    = ALIASResBlock(opt, nf*4  + 16, nf*2, use_mask_norm=False)
        self.up3    = ALIASResBlock(opt, nf*2  + 16, nf,   use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up4 = ALIASResBlock(opt, nf + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)

    def _latent_size(self, opt):
        n = {'normal': 5, 'more': 6, 'most': 7}[self.num_upsampling_layers]
        return opt.load_height // 2**n, opt.load_width // 2**n

    def forward(self, x, seg, seg_div, mm):
        samples  = [F.interpolate(x, (self.sh * 2**i, self.sw * 2**i), mode='nearest') for i in range(8)]
        features = [self._modules[f'conv_{i}'](samples[i]) for i in range(8)]
        x = self.head(features[0], seg_div, mm)
        x = self.up(x)
        x = self.mid0(torch.cat((x, features[1]), 1), seg_div, mm)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.mid1(torch.cat((x, features[2]), 1), seg_div, mm)
        x = self.up(x)
        x = self.up0(torch.cat((x, features[3]), 1), seg_div, mm)
        x = self.up(x)
        x = self.up1(torch.cat((x, features[4]), 1), seg_div, mm)
        x = self.up(x)
        x = self.up2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up4(torch.cat((x, features[7]), 1), seg)
        return self.tanh(self.conv_img(self.relu(x)))
