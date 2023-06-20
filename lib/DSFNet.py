import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import os.path as osp
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from lib.UNet_ResNet34 import DecoderBlock
from lib.UNet_ResNet34 import ResNet34Unet
from lib.modules import *
import timm
import torch.nn.init as init
from typing import List

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.contiguous().view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.contiguous().view(b, c, -1)
        node_v = node_v.contiguous().view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.contiguous().view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out


class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)


        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.contiguous().view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.contiguous().view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out


class DualGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(DualGCNHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output1 = self.conva(x)
        output2 = self.dualgcn(output1)
        output3 = self.convb(output2)
        output4 = self.bottleneck(torch.cat([x, output3], 1))

        return output4


class AttentionConv_self(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv_self, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out_x = self.query_conv(x)
        k_out_x = self.key_conv(padded_x)
        v_out_x = self.value_conv(padded_x)

        q_out = q_out_x
        k_out = k_out_x
        v_out = v_out_x

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out


    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x, y):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        padded_y = F.pad(y, [self.padding, self.padding, self.padding, self.padding])
        q_out_x = self.query_conv(x)
        k_out_x = self.key_conv(padded_x)

        v_out_y = self.value_conv(padded_y)

        q_out = q_out_x
        k_out = k_out_x
        v_out = v_out_y

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class FpnCombine(nn.Module):
    def __init__(self,
            fpn_channels,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            apply_resample_bn=False,
            redundant_bias=False,
            weight_method='attn',
    ):
        super(FpnCombine, self).__init__()
        self.weight_method = weight_method
        self.resample = nn.ModuleDict()

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(3), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for i in x:
            nodes.append(i)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights

        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)

        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)

        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))

        out = torch.sum(out, dim=-1)

        return out


class DSFNet(ResNet34Unet):
    def __init__(self,
                 bank_size =16,
                 decay_rate=0.99,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=768
                 ):
        super().__init__(num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True)
        
        self.feat_channels = feat_channels     
        self.decay_rate = decay_rate
        self.bank_size = bank_size
        self.register_buffer("pixelmemory", torch.zeros(self.bank_size, self.feat_channels))  # pixel semantic memory

        self.l = nn.Conv2d(self.feat_channels, num_classes, 1)
        self.conv_v_x = conv1d(self.feat_channels, self.feat_channels)
        self.conv_memory = conv1d(self.feat_channels, self.feat_channels)
        self.logit_softmax_x = nn.Softmax(dim=1)
        self.logit_softmax_m = nn.Softmax(dim=0)
        self.gamma = Parameter(torch.zeros(1))


        self.conv_l3 =  nn.Sequential(conv2d(self.feat_channels, 384, kernel_size=1),
                                        nn.Upsample(scale_factor=2))
        self.conv_l2 =  nn.Sequential(conv2d(self.feat_channels, 192, kernel_size=1),
                                        nn.Upsample(scale_factor=4))
        self.conv_l1 =  nn.Sequential(conv2d(self.feat_channels, 96, kernel_size=1),
                                        nn.Upsample(scale_factor=8))
        # # # DualGCN
        self.head = DualGCNHead(768, 384, num_classes)
        self.conv_Dual = nn.Sequential(conv2d(1, 768, kernel_size=3),
                                       BatchNorm2d(768)
        )

        # # # ConvNext:
        self.convnext = timm.create_model('convnext_tiny', features_only=True, out_indices=(0,1,2,3), pretrained=True)


        self.SaAtt1 = AttentionConv_self(96, 96, kernel_size=3, padding=1)
        self.SaAtt2 = AttentionConv_self(192, 192, kernel_size=3, padding=1)
        self.SaAtt3 = AttentionConv_self(384, 384, kernel_size=3, padding=1)

        self.combine3 = FpnCombine(
            384,
            weight_method='fastattn'
        )
        self.combine2 = FpnCombine(
            192,
            weight_method='fastattn'
        )
        self.combine1 = FpnCombine(
            96,
            weight_method='fastattn'
        )

        self.conv_c24 = conv2d(768, 384, kernel_size=1)
        self.conv_423 = conv2d(384, 192, kernel_size=1)
        self.conv_322 = conv2d(192, 96, kernel_size=1)
        self.conv_221 = conv2d(96, 96, kernel_size=1)

        self.SaAtt = AttentionConv(1, 32, kernel_size=3, padding=1)

        self.conv_out = nn.Sequential(
            conv2d(192, 96, kernel_size=3),
            BatchNorm2d(96),
            nn.ReLU(96),
            conv2d(96, 48, kernel_size=3),
            BatchNorm2d(48),
            nn.ReLU(48),
            conv2d(48, 24, kernel_size=3),
            BatchNorm2d(24),
            nn.ReLU(24),
            conv2d(24, 12, kernel_size=3),
            BatchNorm2d(12),
            nn.ReLU(12),
            conv2d(12, 1, kernel_size=3),
            BatchNorm2d(1)
        )



    def GCN_att(self, x, flag):
        aux_out = self.head(x)
        out = self.conv_Dual(aux_out)
        # memory update
        if (flag == 'train'):
            x_obj = x * aux_out.sigmoid()
            self.memory_update(x_obj)
        return aux_out, out

    @torch.no_grad()
    def memory_update(self, x_obj):
        # value_x = B * C * HW
        batch_size, _, height, width = x_obj.shape
        value_x = self.conv_v_x(x_obj.contiguous().view(batch_size, self.feat_channels, -1))
        logit = torch.matmul(self.pixelmemory, value_x)
        attn_map = self.logit_softmax_m(logit) # B * S * HW
        memory_update = attn_map.transpose(0,1).contiguous().view(self.bank_size, -1) @ value_x.transpose(1,2).contiguous().view(-1, self.feat_channels)  #S * BHW @ BHW * C = S * C
        self.pixelmemory = self.decay_rate * self.pixelmemory + (1 - self.decay_rate) * memory_update


    def up(self, feat, e3, e2, e1, x):
        feat_l3 = self.conv_l3(feat)
        feat_l2 = self.conv_l2(feat)
        feat_l1 = self.conv_l1(feat)
        center = self.center(feat)

        ddd4 = self.conv_c24(center)
        ddd4_Sa = self.SaAtt3(ddd4)
        dd4 = self.combine3([e3, feat_l3, ddd4_Sa])
        d4 = self.decoder4(dd4) #[2, 384, 28, 28]

        ddd3 = self.conv_423(d4)
        ddd3_Sa = self.SaAtt2(ddd3)
        dd3 = self.combine2([e2, feat_l2, ddd3_Sa])
        d3 = self.decoder3(dd3)  # [2, 192, 56, 56]

        ddd2 = self.conv_322(d3)
        ddd2_Sa = self.SaAtt1(ddd2)
        dd2 = self.combine2([e1, feat_l1, ddd2_Sa])
        d2 = self.decoder2(dd2)  # [2, 192, 56, 56]

        d1 = self.decoder1(torch.cat([d2, x], 1))

        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)


        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        return f4, f3, f2, f1

    def forward(self, x, flag):
        batch_size = x.shape[0]  # x is input image

        e = self.convnext(x)
        e1 = e[0]
        e2 = e[1]
        e3 = e[2]
        e4 = e[3]


        x = F.interpolate(e1, size=(112,112), mode='bilinear')

        aux_out, feats = self.GCN_att(e4, flag)

        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(feats, e3, e2, e1, x)

        #=== SASA attention ===#
        s3 = self.SaAtt(f3, f4)
        s2 = self.SaAtt(f2, f3)
        s1 = self.SaAtt(f1, f2)

        X = F.interpolate(x, size=(224,224), mode='bilinear')

        S = torch.cat((s1, s2, s3, X), 1)

        output = self.conv_out(S)

        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)


        return aux_out, f4, f3, f2, f1, output
