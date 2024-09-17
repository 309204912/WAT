# Modified from https://github.com/JingyunLiang/SwinIR
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
import os
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import to_2tuple, trunc_normal_

from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F


class WTAttention(nn.Module):
    """
    使用小波变换的下采样模块，结合卷积层来调整特征图的维度。
    参数:
    - in_ch: 输入通道数
    - out_ch: 输出通道数
    """

    def __init__(self, in_ch, out_ch):
        super(WTAttention, self).__init__()
        # 初始化小波变换模块，使用haar小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 1x1卷积用于合并特征并进行通道数的转换
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch * 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.conv0 = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, stride=1)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.gelu(x)
        x = self.conv1(x)
        identity = x
        # 应用小波变换，分离低频和高频部分
        yL, yH = self.wt(x)  # yL是低频部分，yH是高频部分（包含3个方向的细节）
        # 提取不同方向的高频分量
        y_HL = yH[0][:, :, 0, ::]  # 水平高频分量
        y_LH = yH[0][:, :, 1, ::]  # 垂直高频分量
        y_HH = yH[0][:, :, 2, ::]  # 对角高频分量
        # 将低频和高频分量在通道维度上进行拼接
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.pool(x)
        # 通过卷积层进行通道数的调整
        x = self.conv_bn_relu(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x * identity


# class StarReLU(nn.Module):
#     """
#     StarReLU 实现了一个自定义的激活函数，形式为：s * relu(x) ** 2 + b。
#     其中，s 和 b 分别是可学习的缩放系数和偏置项。
#     """
#     def __init__(self, scale_value=1.0, bias_value=0.0,
#                  scale_learnable=True, bias_learnable=True,
#                  mode=None, inplace=False):
#         super().__init__()
#         # inplace 参数决定了 ReLU 激活是否采用原地操作，减少内存消耗
#         self.relu = nn.ReLU(inplace=inplace)
#         # 缩放系数，初始化为 scale_value，并根据 scale_learnable 决定是否可学习
#         self.scale = nn.Parameter(scale_value * torch.ones(1),
#                                   requires_grad=scale_learnable)
#         # 偏置项，初始化为 bias_value，并根据 bias_learnable 决定是否可学习
#         self.bias = nn.Parameter(bias_value * torch.ones(1),
#                                  requires_grad=bias_learnable)
#
#     def forward(self, x):
#         # 在前向传播中应用 StarReLU 激活函数：s * relu(x) ** 2 + b
#         return self.scale * self.relu(x) ** 2 + self.bias
#
#
#
#
# class ddformerMlp(nn.Module):
#
#     """
#     实现了 MetaFormer 模型中使用的 MLP 结构，常见于 Transformer、MLP-Mixer 等模型中。
#     它包括两个全连接层和一个激活函数，中间可能包含 Dropout。
#     """
#     def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
#                  bias=False, **kwargs):
#         from timm.layers import to_2tuple
#         super().__init__()
#         # 计算中间层的特征维数，mlp_ratio 决定了隐藏层的放大比例
#         hidden_features = int(mlp_ratio * dim)
#         # 处理 dropout 参数，确保其为二元组形式，对应两个 Dropout 层
#         drop_probs = to_2tuple(drop)
#
#         # 第一个全连接层，将输入特征映射到隐藏层
#         self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
#         # 激活函数层，默认为 StarReLU
#         self.act = act_layer()
#         # 第一个 Dropout 层
#         self.drop1 = nn.Dropout(drop_probs[0])
#         # 第二个全连接层，将隐藏层特征映射回输出特征维度
#         out_features = out_features or dim  # 如果未指定输出特征维度，使用输入特征维度
#         self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
#         # 第二个 Dropout 层
#         self.drop2 = nn.Dropout(drop_probs[1])
#
#     def forward(self, x):
#         # MLP 的前向传播流程
#         x = self.fc1(x)  # 应用第一个全连接层
#         x = self.act(x)  # 应用激活函数
#         x = self.drop1(x)  # 应用第一个 Dropout
#         x = self.fc2(x)  # 应用第二个全连接层
#         x = self.drop2(x)  # 应用第二个 Dropout
#         return x
#
#
# def resize_complex_weight(origin_weight, new_h, new_w):
#     """
#     调整复数权重的大小。
#
#     参数:
#     - origin_weight (Tensor): 原始的复数权重张量，维度为 (高度, 宽度, 头数, 2)。
#                               其中最后一个维度的 2 表示复数的实部和虚部。
#     - new_h (int): 新的高度。
#     - new_w (int): 新的宽度。
#
#     返回:
#     - new_weight (Tensor): 调整大小后的复数权重张量，维度为 (新高度, 新宽度, 头数, 2)。
#     """
#
#     # 获取原始权重的维度信息
#     h, w, num_heads = origin_weight.shape[0:3]
#     # 将原始权重张量重新塑形并调整维度顺序，以适应 interpolate 函数的要求
#     origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
#
#     # 使用双三次插值方法（bicubic）调整权重大小
#     new_weight = torch.nn.functional.interpolate(
#         origin_weight,
#         size=(new_h, new_w),  # 目标大小
#         mode='bicubic',  # 插值模式为双三次
#         align_corners=True  # 对齐角点，以减少边界效应
#     ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)  # 调整维度顺序并恢复到原始形状
#
#     return new_weight
#
#
# class DynamicFilter(nn.Module):
#     """
#     DynamicFilter 实现了一个动态滤波器，可以根据输入动态调整滤波器的权重。
#     该模块首先通过一组全连接层和激活函数生成复数权重，然后利用这些权重对输入特征进行频域的滤波。
#     """
#
#     def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
#                  act1_layer=StarReLU, act2_layer=nn.Identity,
#                  bias=False, num_filters=4, size=14, weight_resize=False,
#                  **kwargs):
#         from timm.layers import to_2tuple
#         super().__init__()
#         self.size = to_2tuple(size)[0]  # 图像高度和宽度
#         self.filter_size = to_2tuple(size)[1] // 2 + 1  # 根据 FFT 的性质计算滤波器大小
#         self.num_filters = num_filters  # 滤波器数量
#         self.med_channels = int(expansion_ratio * dim)  # 中间层通道数
#         self.weight_resize = weight_resize  # 是否根据输入大小调整权重
#
#         # 第一阶段：利用全连接层和激活函数处理输入特征
#         self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
#         self.act1 = act1_layer()
#         # 对特征进行重加权，用于生成路由权重
#         self.reweight = ddformerMlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
#         # 初始化复数权重参数
#         self.complex_weights = nn.Parameter(torch.randn(self.size, self.filter_size, num_filters, 2) * 0.02)
#         self.act2 = act2_layer()
#         # 第二阶段：使用另一个全连接层将处理后的特征映射回原始特征空间
#         self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)
#
#     def forward(self, x):
#         B, H, W, _ = x.shape  # 输入特征的维度
#         # 计算路由权重
#         routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)
#         x = self.pwconv1(x)  # 应用第一个全连接层
#         x = self.act1(x)  # 应用第一个激活函数
#         x = x.to(torch.float32)  # 转换数据类型，准备进行 FFT
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # 对特征进行二维快速傅里叶变换（FFT）
#
#         # 根据是否需要调整权重大小，选择相应的操作
#         if self.weight_resize:
#             complex_weights = resize_complex_weight(self.complex_weights, x.shape[1], x.shape[2])
#             complex_weights = torch.view_as_complex(complex_weights.contiguous())
#         else:
#             complex_weights = torch.view_as_complex(self.complex_weights)
#         routeing = routeing.to(torch.complex64)  # 转换路由权重的数据类型
#         # 根据路由权重和复数权重计算滤波后的特征
#         weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
#         if self.weight_resize:
#             weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
#         else:
#             weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
#         x = x * weight  # 应用滤波器
#         x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')  # 进行逆 FFT
#
#         x = self.act2(x)  # 应用第二个激活函数
#         x = self.pwconv2(x)  # 应用第二个全连接层，完成滤波过程
#         return x

# class ChannelAttention(nn.Module):
#     """Channel attention used in RCAN.
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         squeeze_factor (int): Channel squeeze factor. Default: 16.
#     """
#
#     def __init__(self, num_feat, squeeze_factor=16):
#         super(ChannelAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
#             nn.Sigmoid())
#
#     def forward(self, x):
#         y = self.attention(x)
#         return x * y
# class CAB(nn.Module):#输入输出是
#
#     def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
#         super(CAB, self).__init__()
#
#         self.cab = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
#             ChannelAttention(num_feat, squeeze_factor)
#             )
#
#     def forward(self, x):
#         return self.cab(x)
# #Wavelet Convolutions for Large Receptive Fields [ECCV 2024]
# #https://arxiv.org/pdf/2407.05848
# """
# 小波滤波器创建：
#
# create_wavelet_filter(wave, in_size, out_size, type=torch.float):
# 该函数创建小波分解和重构滤波器。这些滤波器用于将输入数据转换为小波系数并从这些系数重构数据。
# 小波变换函数：
#
# wavelet_transform(x, filters):
# 该函数使用提供的滤波器执行小波变换。
# inverse_wavelet_transform(x, filters):
# 该函数执行逆小波变换以重构数据。
# 小波变换初始化函数：
#
# WTConv2d:
# 这个类定义了一个集成了小波变换的卷积层。它有几个关键组件：
# self.wt_filter 和 self.iwt_filter：小波分解和重构滤波器。
# self.wt_function 和 self.iwt_function：应用小波和逆小波变换的函数。
# self.base_conv：基本卷积层。
# self.wavelet_convs 和 self.wavelet_scale：用于每个小波系数层级的卷积层和缩放模块的列表。
# self.do_stride：如果需要，对输入应用步幅的函数。
# 缩放模块 (_ScaleModule)：
#
# _ScaleModule:
# 这个类定义了一个通过学习的权重来缩放输入张量的模块。它用于对卷积层的输出应用一个学习到的缩放因子。
# """
# def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
#     dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
#
#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
#
#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
#     rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
#
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
#
#     return dec_filters, rec_filters
#
# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x
#
#
# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x
#
# #Wavelet Transform Conv(WTConv2d)
# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
#         super(WTConv2d, self).__init__()
#
#         assert in_channels == out_channels
#
#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride
#         self.dilation = 1
#
#         self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
#         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
#
#         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
#         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
#
#         self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
#                                    groups=in_channels, bias=bias)
#         self.base_scale = _ScaleModule([1, in_channels, 1, 1])
#
#         self.wavelet_convs = nn.ModuleList(
#             [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
#                        groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
#         )
#         self.wavelet_scale = nn.ModuleList(
#             [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
#         )
#
#         if self.stride > 1:
#             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
#             self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
#                                                    groups=in_channels)
#         else:
#             self.do_stride = None
#
#     def forward(self, x):
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#
#         curr_x_ll = x
#
#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
#                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
#                 curr_x_ll = F.pad(curr_x_ll, curr_pads)
#
#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:, :, 0, :, :]
#
#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)
#
#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
#
#         next_x_ll = 0
#
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()
#
#             curr_x_ll = curr_x_ll + next_x_ll
#
#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = self.iwt_function(curr_x)
#
#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
#
#         x_tag = next_x_ll
#         assert len(x_ll_in_levels) == 0
#
#         x = self.base_scale(self.base_conv(x))
#         x = x + x_tag
#
#         if self.do_stride is not None:
#             x = self.do_stride(x)
#
#         return x
#
#
# class _ScaleModule(nn.Module):
#     def __init__(self, dims, init_scale=1.0, init_bias=0):
#         super(_ScaleModule, self).__init__()
#         self.dims = dims
#         self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
#         self.bias = None
#
#     def forward(self, x):
#         return torch.mul(self.weight, x)
#
# class DepthwiseSeparableConvWithWTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(DepthwiseSeparableConvWithWTConv2d, self).__init__()
#
#         # 深度卷积：使用 WTConv2d 替换 3x3 卷积
#         self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)
#
#         # 逐点卷积：使用 1x1 卷积
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
# class haarBlock(nn.Module):
#     """Residual Dense Block.
#
#     Used in RRDB block in ESRGAN.
#
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_grow_ch (int): Channels for each growth.
#     """
#
#     def __init__(self, num_feat=180):
#         super(haarBlock, self).__init__()
#         self.conv1 = nn.Conv2d(num_feat, num_feat // 2, 3,1,1)
#         self.conv2 = DepthwiseSeparableConvWithWTConv2d(num_feat // 2,num_feat // 2, 3)
#         self.conv3 = nn.Conv2d(num_feat,num_feat, 3, 1, 1)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         out =  self.conv3(torch.cat((x1, x2), 1))
#         # x3 = self.conv3(torch.cat((x, x1, x2), 1))
#
#         # Empirically, we use 0.2 to scale the residual for better performance
#         return out
# class AdaptiveConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding=0):
#         super(AdaptiveConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x):
#
#         weight_1x1 = self.weight.sum(dim=[2, 3]).unsqueeze(-1).unsqueeze(-1)
#         x_1x1 = nn.functional.conv2d(x, weight_1x1, bias=None, stride=1, padding=0)
#         weight_3x3 = self.weight
#         x_3x3 = nn.functional.conv2d(x, weight_3x3, bias=None, stride=1, padding=self.padding)
#         x_out = x_1x1 + x_3x3 + self.bias.view(1, -1, 1, 1).expand_as(x_1x1)
#         return x_out
# class SpectralGatingNetwork(nn.Module):
#     def __init__(self, dim, h=9, w=7):
#         super().__init__()
#         # 初始化复数权重参数，使用较小的随机数初始化
#         self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
#         self.w = w
#         self.h = h
#
#     def forward(self, x, spatial_size=None):
#         B, N, C = x.shape  # 获取输入的批次大小、元素数量和通道数
#         if spatial_size is None:
#             # 如果没有指定空间尺寸，则假设输入是正方形
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size  # 使用给定的空间尺寸
#
#         x = x.view(B, a, b, C)  # 将输入调整为四维张量以进行FFT
#         x = x.to(torch.float32)  # 确保数据类型为float32以兼容FFT操作
#         # print(x.shape)
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # 对输入进行二维实数快速傅里叶变换
#         # print(x.shape)
#         weight = torch.view_as_complex(self.complex_weight)  # 将复数权重转换为复数形式
#         # print(weight.shape)
#         x = x * weight  # 应用频谱权重
#         x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')  # 进行二维逆傅里叶变换以返回到空间域
#         x = x.reshape(B, N, C)  # 将输出调整回原始形状
#
#         return x


# class SSL(nn.Module):
#     def __init__(self, channels):
#         super(SSL, self).__init__()
#
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
#         self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 5, dilation=5)
#         self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 7, dilation=7)
#         self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 9, dilation=9)
#
#         self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)
#
#     def forward(self, x):
#
#         aa =  DWTForward(J=1, mode='zero', wave='db3').cuda()
#         yl, yh = aa(x)
#
#         yh_out = yh[0]
#         ylh = yh_out[:,:,0,:,:]
#         yhl = yh_out[:,:,1,:,:]
#         yhh = yh_out[:,:,2,:,:]
#
#         conv_rec1 = self.conv5(yl)
#         conv_rec5 = self.conv5(ylh)
#         conv_rec7 = self.conv7(yhl)
#         conv_rec9 = self.conv9(yhh)
#
#         cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9),dim=2)
#         rec_yh = []
#         rec_yh.append(cat_all)
#
#
#         ifm = DWTInverse(wave='db3', mode='zero').cuda()
#         Y = ifm((conv_rec1, rec_yh))
#
#         return Y
# class Simam_module(torch.nn.Module):
#     """
#     SimAM注意力模块，简单且无需额外参数。
#     e_lambda：用于稳定分母的小正数，防止除零错误。
#     """
#     def __init__(self, e_lambda=1e-4):
#         super(Simam_module, self).__init__()
#         self.act = nn.Sigmoid()  # 使用Sigmoid函数作为激活函数
#         self.e_lambda = e_lambda  # 稳定性参数
#
#     def forward(self, x):
#         b, c, h, w = x.size()  # 输入张量的尺寸
#         n = w * h - 1  # 计算特征图中除去一个元素外的总元素数
#         # 计算输入特征图与其均值之差的平方
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         # 根据SimAM公式计算每个位置的注意力权重
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#         # 应用Sigmoid激活函数并与输入特征图相乘，实现特征重标定
#         return x * self.act(y)
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda()  #

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


class CurveCALayer(nn.Module):
    def __init__(self, channel):
        super(CurveCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_curve = 3
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
        return x


##---------- Curved Wavelet Attention (CWA) Blocks ----------
class CWA(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(CWA, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        modules_body = \
            [
                conv(n_feat * 2, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat * 2, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.CurCA = CurveCALayer(n_feat * 2)

        self.conv1x1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)  # 256 to 128
        self.conv3x3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)

        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_curveca_2 = self.CurCA(res)
        res = torch.cat([branch_sa, branch_curveca_2], dim=1)
        res = self.conv1x1(res) + x_dwt

        wavelet_path = self.iwt(res)
        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # self.haar = CWA(n_feat=dim)
        # self.Simam = Simam_module()
        # self.wavelet = SSL(dim)
        # self.haarBlock = haarBlock(dim)
        # self.conv_block = CAB(num_feat=dim, compress_ratio=3, squeeze_factor=30)
        # self.ddformer = DynamicFilter(dim, size=128).cuda()
        self.WTA = WTAttention(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)
        # self.Spectral = SpectralGatingNetwork(180, 128, 65).cuda()

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # 注意力
        # Spectral = self.Spectral(x)

        x = x.view(b, h, w, c)

        # # WTA
        WTA = self.WTA(x.permute(0, 3, 1, 2))
        WTA = WTA.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # ddfomer
        # ddformer = self.ddformer(x)
        # ddformer = ddformer.contiguous().view(b, h * w, c)

        # # Conv_X
        # conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        # conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # haar
        # ha =  self.haarBlock(x.permute(0, 3, 1, 2))
        # ha = ha.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # simam
        # simam = self.Simam(x.permute(0, 3, 1, 2))
        # simam = simam.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # haar
        # haar = self.haar(x.permute(0, 3, 1, 2))
        # haar = haar.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # haar
        # haar = self.haar(x.permute(0, 3, 1, 2))
        # haar = haar.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # #wavelet
        # wavelet = self.wavelet(x.permute(0, 3, 1, 2))
        # wavelet = wavelet.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x) + WTA * 0.01
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w
        # W-MSA/SW-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.haar = CWA(n_feat=dim)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)

        # 全局注意力加这(x.shape = (b, h*w ,c))
        h, w = x_size
        b, _, c = x.shape

        x = x.view(b, h, w, c)
        x = self.haar(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class SwinIRhaar(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(SwinIRhaar, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        # self.conv_first = AdaptiveConv2d(num_in_ch, embed_dim, 3, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(
        upscale=2,
        img_size=(256, 256),
        window_size=window_size,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
