import math
import torch
import torch.nn as nn
from networks.drn import drn_c_26


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, classes, pretrained_drn=False,
            pretrained_model=None, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = drn_c_26(pretrained=pretrained_drn)
        self.base = nn.Sequential(*list(model.children())[:-2])
        if pretrained_model:
            self.load_pretrained(pretrained_model)

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

    def load_pretrained(self, pretrained_model):
        print("loading the pretrained drn model from %s" % pretrained_model)
        state_dict = torch.load(pretrained_model, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # filter out unnecessary keys
        pretrained_dict = state_dict['model']
        pretrained_dict = {k[5:]: v for k, v in pretrained_dict.items() if k.split('.')[0] == 'base'}

        # load the pretrained state dict
        self.base.load_state_dict(pretrained_dict)


class DRNSub(nn.Module):
    def __init__(self, num_classes, pretrained_model=None, fix_base=False):
        super(DRNSub, self).__init__()

        drnseg = DRNSeg(2)
        if pretrained_model:
            print("loading the pretrained drn model from %s" % pretrained_model)
            state_dict = torch.load(pretrained_model, map_location='cpu')
            drnseg.load_state_dict(state_dict['model'])

        self.base = drnseg.base
        if fix_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
