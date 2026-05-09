import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from . import ViCBlock, act_layer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vic_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vic_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class Stem(nn.Module):
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class LKStem(nn.Module):
    def __init__(self, in_dim=3, out_dim=768, act='relu',
                 patch_size=7, patch_stride=4, down_stride=2):
        super().__init__()

        mid_dim = out_dim // 2

        self.proj = nn.Conv2d(
            in_dim, mid_dim,
            kernel_size=patch_size, stride=patch_stride,
            padding=patch_size // 2
        )
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.act1 = act_layer(act)

        self.down1 = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, 3, stride=down_stride, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
        )
        self.down2 = nn.Conv2d(out_dim, out_dim, 3, stride=down_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.bn2(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DeepViC(torch.nn.Module):
    def __init__(self, opt):
        super(DeepViC, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = getattr(opt, 'norm', 'batch')
        bias = getattr(opt, 'bias', True)
        epsilon = getattr(opt, 'epsilon', 0.1)
        stochastic = getattr(opt, 'use_stochastic', True)
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]

        relative_pos = getattr(opt, 'relative_pos', False)
        use_dilation = getattr(opt, 'use_dilation', True)
        num_hyperedges = getattr(opt, 'num_hyperedges', 16)
        laplacian_alpha = getattr(opt, 'laplacian_alpha', 0.1)
        use_structure_induction = getattr(opt, 'use_structure_induction', True)
        e2v_ratio = getattr(opt, 'e2v_ratio', 2.0)
        hyperedge_dropout = getattr(opt, 'hyperedge_dropout', False)
        e_drop_rate = getattr(opt, 'e_drop_rate', 0.2)

        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]

        num_hyperedges_list = [int(x.item()) for x in torch.linspace(num_hyperedges, 2*num_hyperedges, self.n_blocks)]
        print('num_knn', num_knn)
        print('num_hyperedges_list', num_hyperedges_list)
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = (224 // 4) * (224 // 4)

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                if use_dilation:
                    self.backbone += [
                        Seq(ViCBlock(channels[i],
                                              kernel_size=num_knn[idx],
                                              dilation=min(idx // 4 + 1, max_dilation),
                                              act=act, norm=norm, bias=bias,
                                              stochastic=stochastic, epsilon=epsilon, r=reduce_ratios[i], n=HW,
                                              drop_path=dpr[idx], relative_pos=relative_pos,
                                              num_hyperedges=num_hyperedges_list[idx],
                                              laplacian_alpha=laplacian_alpha,
                                              hypergraph_k=num_knn[idx],
                                              use_structure_induction=use_structure_induction,
                                              e2v_ratio=e2v_ratio,
                                              hyperedge_dropout=hyperedge_dropout,
                                              e_drop_rate=e_drop_rate),
                              FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                             )]
                else:
                    self.backbone += [
                        Seq(ViCBlock(channels[i],
                                              kernel_size=num_knn[idx],
                                              dilation=1,
                                              act=act, norm=norm, bias=bias,
                                              stochastic=stochastic, epsilon=epsilon, r=reduce_ratios[i], n=HW,
                                              drop_path=dpr[idx], relative_pos=relative_pos,
                                              num_hyperedges=num_hyperedges_list[idx],
                                              laplacian_alpha=laplacian_alpha,
                                              hypergraph_k=num_knn[idx],
                                              use_structure_induction=use_structure_induction,
                                              e2v_ratio=e2v_ratio,
                                              hyperedge_dropout=hyperedge_dropout,
                                              e_drop_rate=e_drop_rate),
                              FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                             )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

@register_model
def pvic_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0,
                     k=9, num_knn=None, relative_pos=False, use_dilation=True, **kwargs):
            if num_knn is not None:
                k = num_knn
            self.act = 'gelu'
            self.k = k
            self.relative_pos = relative_pos
            self.use_dilation = use_dilation
            self.num_hyperedges = kwargs.get('num_hyperedges', 16)
            self.laplacian_alpha = kwargs.get('laplacian_alpha', 0.1)
            self.e2v_ratio = kwargs.get('e2v_ratio', 2.0)
            self.hyperedge_dropout = kwargs.get('hyperedge_dropout', False)
            self.e_drop_rate = kwargs.get('e_drop_rate', 0.2)

            self.blocks = [2,2,6,2]
            self.channels = [48, 96, 240, 384]
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.drop_path = drop_path_rate
            self.emb_dims = 1024

            self.norm = kwargs.get('norm', 'batch')
            self.bias = kwargs.get('bias', True)
            self.epsilon = kwargs.get('epsilon', 0.1)
            self.use_stochastic = kwargs.get('use_stochastic', True)

    opt = OptInit(**kwargs)
    model = DeepViC(opt)
    model.default_cfg = default_cfgs['vic_224_gelu']
    return model

@register_model
def pvic_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0,
                     k=9, num_knn=None, relative_pos=False, use_dilation=True, **kwargs):
            if num_knn is not None:
                k = num_knn
            self.act = 'gelu'
            self.k = k
            self.relative_pos = relative_pos
            self.use_dilation = use_dilation
            self.num_hyperedges = kwargs.get('num_hyperedges', 16)
            self.laplacian_alpha = kwargs.get('laplacian_alpha', 0.1)
            self.e2v_ratio = kwargs.get('e2v_ratio', 2.0)
            self.hyperedge_dropout = kwargs.get('hyperedge_dropout', False)
            self.e_drop_rate = kwargs.get('e_drop_rate', 0.2)

            self.blocks = [2,2,6,2]
            self.channels = [80, 160, 400, 640]
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.drop_path = drop_path_rate
            self.emb_dims = 1024

            self.norm = kwargs.get('norm', 'batch')
            self.bias = kwargs.get('bias', True)
            self.epsilon = kwargs.get('epsilon', 0.1)
            self.use_stochastic = kwargs.get('use_stochastic', True)

    opt = OptInit(**kwargs)
    model = DeepViC(opt)
    model.default_cfg = default_cfgs['vic_224_gelu']
    return model

@register_model
def pvic_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0,
                     k=9, num_knn=None, relative_pos=False, use_dilation=True, **kwargs):
            if num_knn is not None:
                k = num_knn
            self.act = 'gelu'
            self.k = k
            self.relative_pos = relative_pos
            self.use_dilation = use_dilation
            self.num_hyperedges = kwargs.get('num_hyperedges', 16)
            self.laplacian_alpha = kwargs.get('laplacian_alpha', 0.1)
            self.e2v_ratio = kwargs.get('e2v_ratio', 2.0)
            self.hyperedge_dropout = kwargs.get('hyperedge_dropout', False)
            self.e_drop_rate = kwargs.get('e_drop_rate', 0.2)

            self.blocks = [2,2,18,2]
            self.channels = [128, 256, 512, 1024]
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.drop_path = drop_path_rate
            self.emb_dims = 1024

            self.norm = kwargs.get('norm', 'batch')
            self.bias = kwargs.get('bias', True)
            self.epsilon = kwargs.get('epsilon', 0.1)
            self.use_stochastic = kwargs.get('use_stochastic', True)

    opt = OptInit(**kwargs)
    model = DeepViC(opt)
    model.default_cfg = default_cfgs['vic_b_224_gelu']
    return model
