# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
# from ..mamba.mamba_ssm import Mamba
import torch.nn.functional as F 
from model.cross_relation_atten import CrossAttention


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        B, C = x1.shape[:2]
        x1_skip = x1
        x2_skip = x2
        x3_skip = x3
        assert x1.shape[1] == self.dim
        assert x2.shape[1] == self.dim
        assert x3.shape[1] == self.dim

        n_tokens = x1.shape[2:].numel()

        x_flat1 = x1.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat2 = x2.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat3 = x3.reshape(B, C, n_tokens).transpose(-1, -2)
        # print("x_flat1",x_flat1.shape)

        x_norm1 = self.norm(x_flat1)
        x_norm2 = self.norm(x_flat2)
        x_norm3 = self.norm(x_flat3)

        if self.training:
            p = 0.50
            mask = torch.rand((B, n_tokens), device=x.device) < p
            mask = mask.unsqueeze(-1)

            mask2 = torch.rand((B, n_tokens), device=x.device) < p
            mask2 = mask2.unsqueeze(-1)

            mask3 = torch.rand((B, n_tokens), device=x.device) < p
            mask3 = mask3.unsqueeze(-1)

            x_norm1 = x_norm1.masked_fill(mask, 0.0)
            x_norm2 = x_norm2.masked_fill(mask2, 0.0)
            x_norm3 = x_norm3.masked_fill(mask3, 0.0)

        x_combined = torch.cat((x_norm1, x_norm2, x_norm3), dim=-2)

        x_stacked = torch.stack((x_norm1, x_norm2, x_norm3), dim=2)

        x_combined2 = x_stacked.view(B, n_tokens * 3, C)
        # print(x_combined2.shape)

        x_mamba = self.mamba(x_combined)
        x_mamba2 = self.mamba(x_combined2)
        # print("x_mamba",x_mamba.shape)

        # reshape
        out1 = x_mamba[:, :n_tokens, :].transpose(-1, -2).reshape(B, C, *x1.shape[2:])
        out2 = x_mamba[:, n_tokens:2 * n_tokens, :].transpose(-1, -2).reshape(B, C, *x2.shape[2:])
        out3 = x_mamba[:, 2 * n_tokens:, :].transpose(-1, -2).reshape(B, C, *x3.shape[2:])

        x_stacked = x_mamba2.view(B, 3, n_tokens, C)

        out11 = x_stacked[:, 0, :, :].transpose(-1, -2).reshape(B, C, *x1.shape[2:])
        out12 = x_stacked[:, 1, :, :].transpose(-1, -2).reshape(B, C, *x2.shape[2:])
        out13 = x_stacked[:, 2, :, :].transpose(-1, -2).reshape(B, C, *x3.shape[2:])

        out1 = out1 + x1_skip + out11
        out2 = out2 + x2_skip + out12
        out3 = out3 + x3_skip + out13
        out = torch.cat((out1, out2, out3), dim=1)
        return out

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cross_attn = CrossAttention(heads=4,d_model=64,tumor_dim=1488,node_dim=1488)

        total_features = self.hidden_size + sum(self.feat_size[:4])
        # self.classifier = nn.Linear(total_features*2, 2)
        self.classifier = nn.Linear(5952, 2)
        
    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in, y_in):
        outs = self.vit(x_in)
        
        # Encoder
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        
        enc_hidden = self.encoder5(outs[3])
        
        pooled_enc1 = self.global_pool(enc1).view(enc1.size(0), -1)
        pooled_enc2 = self.global_pool(enc2).view(enc2.size(0), -1)
        pooled_enc3 = self.global_pool(enc3).view(enc3.size(0), -1)
        pooled_enc4 = self.global_pool(enc4).view(enc4.size(0), -1)
        pooled_hidden = self.global_pool(enc_hidden).view(enc_hidden.size(0), -1)

        outs1 = self.vit(y_in)
        
        enc11 = self.encoder1(y_in)
        enc12 = self.encoder2(outs1[0])
        enc13 = self.encoder3(outs1[1])
        enc14 = self.encoder4(outs1[2])
        
        enc_hidden1 = self.encoder5(outs1[3])
        
        pooled_enc11 = self.global_pool(enc11).view(enc1.size(0), -1)
        pooled_enc12 = self.global_pool(enc12).view(enc2.size(0), -1)
        pooled_enc13 = self.global_pool(enc13).view(enc3.size(0), -1)
        pooled_enc14 = self.global_pool(enc14).view(enc4.size(0), -1)
        pooled_hidden1 = self.global_pool(enc_hidden1).view(enc_hidden1.size(0), -1)

        concatenated_features1 = torch.cat([pooled_enc1, pooled_enc2, pooled_enc3, pooled_enc4, pooled_hidden], dim=1)
        # print("1:",concatenated_features1.size()) #[1,1488]
        concatenated_features2 = torch.cat([pooled_enc11, pooled_enc12, pooled_enc13, pooled_enc14, pooled_hidden1], dim=1)
        # print("2:",concatenated_features2.size()) #[1,1488]
        concatenated_features = torch.cat([concatenated_features1,concatenated_features2])

        t_um,t_m,n_um,n_m,t_s,n_s = self.cross_attn(concatenated_features1,concatenated_features2)
        t_emb = torch.concat((t_um, t_s), dim=-1)
        n_emb = torch.concat((n_um, n_s), dim=-1)

        t_emb = torch.flatten(t_emb, 1)
        # print("t_emb",t_emb.size())
        n_emb = torch.flatten(n_emb, 1)
        # print("n_emb", n_emb.size())
        x = torch.cat((t_emb, n_emb),dim=-1)
        # x = torch.flatten(x,1)
        # print(x.size())  # torch.Size([1, 2048])

        out = self.classifier(x)
        
        return out
