from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import trunc_normal_
from .Linear_super import LinearSuper
from naslib.search_spaces.core.primitives import AbstractPrimitive
from model.module.layernorm_super import LayerNormSuper


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LinearEmb(AbstractPrimitive):
    def __init__(self, layer, emb_choice, num_classes):
        super(LinearEmb, self).__init__(locals())
        self.layer = layer
        self.in_dim = emb_choice
        self.out_dim = num_classes

    def set_sample_config(self):
        self.layer.set_sample_config({},
                                     sample_in_dim=self.in_dim,
                                     sample_out_dim=self.out_dim)

    def forward(self, x, edge_data):
        self.set_sample_config()
        x = self.layer(x[:, :, :self.in_dim])
        return x

    def get_embedded_ops(self):
        return None


class LinearSuper_Emb_Ratio_Combi(AbstractPrimitive):
    def __init__(self,
                 fc1,
                 super_ffn_embed_dim_this_layer,
                 super_embed_dim,
                 super_mlp_ratio,
                 sampled_emb_dim,
                 sampled_mlp_ratio,
                 reverse=False,
                 scale=False):
        super(LinearSuper_Emb_Ratio_Combi, self).__init__(locals())
        self.reverse = reverse
        self.sampled_in_dim = sampled_emb_dim
        self.super_mlp_ratio = super_mlp_ratio
        self.sampled_mlp_ratio = sampled_mlp_ratio
        self.super_embed_dim = super_embed_dim
        self.sample_ffn_embed_dim_this_layer = int(sampled_emb_dim *
                                                   sampled_mlp_ratio)
        if reverse == True:
            tmp = self.sampled_in_dim
            self.sampled_in_dim = self.sample_ffn_embed_dim_this_layer
            self.sample_ffn_embed_dim_this_layer = tmp
        self.super_ffn_embed_dim_this_layer = super_ffn_embed_dim_this_layer
        self.fc1 = fc1
        self.activation_fn = gelu

    def set_sample_config(self):
        self.fc1.set_sample_config(
            {},
            sample_in_dim=self.sampled_in_dim,
            sample_out_dim=self.sample_ffn_embed_dim_this_layer)

    def forward(self, x, edge_data):
        self.set_sample_config()
        if self.reverse == False:
            print(self.fc1)
            x = self.activation_fn(self.fc1(x[:, :, :self.sampled_in_dim]))
            output = torch.zeros(
                [x.shape[0], x.shape[1], self.super_ffn_embed_dim_this_layer])
        else:
            x = self.fc1(x[:, :, :self.sampled_in_dim])
            output = torch.zeros(
                [x.shape[0], x.shape[1], self.super_embed_dim])
        output[:, :, :x.shape[-1]] = x
        print(output)
        print("Fc shape", output.shape)
        return output

    def get_embedded_ops(self):
        return None


class Norm_embed_choice(AbstractPrimitive):
    def __init__(self, layer_norm, embed_choice, super_embed_dim):
        super(Norm_embed_choice, self).__init__(locals())
        self.sampled_in_dim = embed_choice
        self.layer_norm = layer_norm
        self.super_embed_dim = super_embed_dim

    def set_sample_config(self):
        self.layer_norm.set_sample_config(sample_embed_dim=self.sampled_in_dim)

    def forward(self, x, edge_data):
        self.set_sample_config()
        x = self.layer_norm(x[:, :, :self.sampled_in_dim])
        output = torch.zeros([x.shape[0], x.shape[1], self.super_embed_dim])
        output[:, :, :x.shape[-1]] = x
        print("Norm out", x.shape)
        print(output)
        return output

    def get_embedded_ops(self):
        return None


class AttnFfnNorm_embed_choice(AbstractPrimitive):
    def __init__(self,
                 attn_layer_norm,
                 embed_choice,
                 super_embed,
                 pre_norm,
                 before=False,
                 after=False):
        super(AttnFfnNorm_embed_choice, self).__init__(locals())
        self.sampled_in_dim = embed_choice
        self.normalize_before = pre_norm
        self.attn_layer_norm = attn_layer_norm
        self.super_embed_dim = super_embed
        self.before = before
        self.after = after

    def set_sample_config(self):
        self.attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sampled_in_dim)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x, edge_data):
        self.set_sample_config()
        x = self.maybe_layer_norm(self.attn_layer_norm,
                                  x[:, :, :self.sampled_in_dim],
                                  after=self.after,
                                  before=self.before)
        output = torch.zeros([x.shape[0], x.shape[1], self.super_embed_dim])
        output[:, :, :x.shape[-1]] = x
        print("Norm out", x.shape)
        print(output)
        return output

    def get_embedded_ops(self):
        return None


class Dropout(AbstractPrimitive):
    def __init__(self, drop_rate):
        super(Dropout, self).__init__(locals())
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, edge_data):
        return self.dropout(x)

    def get_embedded_ops(self):
        return None


class Scale(AbstractPrimitive):
    def __init__(self, super_mlp_ratio, super_embed_dim, sampled_mlp_ratio,
                 sampled_embed_dim):
        super(Scale, self).__init__(locals())
        self.super_mlp_ratio = super_mlp_ratio
        self.sampled_mlp_ratio = sampled_mlp_ratio
        self.sampled_embed_dim = sampled_embed_dim
        self.super_embed_dim = super_embed_dim

    def forward(self, x, edge_data):
        x = x[:, :, :self.sampled_embed_dim] * (self.super_mlp_ratio /
                                                self.sampled_mlp_ratio)
        output = torch.zeros([x.shape[0], x.shape[1], self.super_embed_dim])
        output[:, :, :x.shape[-1]] = x
        print(output)
        return output

    def get_embedded_ops(self):
        return None


class RelativePosition2D_super(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None
        print("Relative position_h", self.embeddings_table_h.shape)
        print("Relative position_v", self.embeddings_table_v.shape)

    def set_sample_config(self, sample_head_dim):
        self.sample_head_dim = sample_head_dim
        self.sample_embeddings_table_h = self.embeddings_table_h[:, :
                                                                 sample_head_dim]
        self.sample_embeddings_table_v = self.embeddings_table_v[:, :
                                                                 sample_head_dim]

    def calc_sampled_param_num(self):
        return self.sample_embeddings_table_h.numel(
        ) + self.sample_embeddings_table_v.numel()

    def forward(self, length_q, length_k):
        print("Relative position_h", self.sample_embeddings_table_h.shape)
        print("Relative position_v", self.sample_embeddings_table_v.shape)
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (range_vec_k[None, :] // int(length_q**0.5) -
                          range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (range_vec_k[None, :] % int(length_q**0.5) -
                          range_vec_q[:, None] % int(length_q**0.5))
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              "constant", 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              "constant", 0)

        final_mat_v = torch.LongTensor(final_mat_v)  #.cuda()
        final_mat_h = torch.LongTensor(final_mat_h)  #.cuda()
        # get the embeddings with the corresponding distance
        embeddings = self.sample_embeddings_table_v[
            final_mat_v] + self.sample_embeddings_table_h[final_mat_h]

        return embeddings


class Dropout_emb_choice(AbstractPrimitive):
    def __init__(self, embed_choice, super_attn_dropout, super_embed_dim):
        super(Dropout_emb_choice, self).__init__(locals())
        self.sampled_in_dim = embed_choice
        self.sample_attn_dropout = calc_dropout(super_attn_dropout,
                                                self.sampled_in_dim,
                                                super_embed_dim)

    def set_sample_config(self):
        pass

    def forward(self, x, edge_data):
        #print(x.sum(dim=(0,1)))
        output = torch.zeros_like(x)
        x = F.dropout(x[:, :, x.sum(dim=(0, 1)) != 0],
                      p=self.sample_attn_dropout,
                      training=self.training)
        output[:, :, :x.shape[-1]] = x
        print(output)
        return output

    def get_embedded_ops(self):
        return None


class Proj_emb_choice(AbstractPrimitive):
    def __init__(self, proj, embed_choice, super_emb):
        super(Proj_emb_choice, self).__init__(locals())
        self.proj = proj
        self.sampled_in_dim = embed_choice
        self.super_embed_dim = super_emb

    def set_sample_config(self):
        self.proj.sample_weight = self.proj.sample_weight[:self.
                                                          sampled_in_dim, :]
        print("Weight shape", self.proj.sample_weight.shape)
        self.sample_scale = self.super_embed_dim / self.sampled_in_dim
        if self.proj.bias is not None:
            self.proj.sample_bias = self.proj.bias[:self.sampled_in_dim]

    def forward(self, x, edge_data):
        self.set_sample_config()
        print("X shape", x.shape)
        x = F.linear(x[:, :, x.sum(dim=(0, 1)) != 0], self.proj.sample_weight,
                     self.proj.sample_bias) * (self.sample_scale
                                               if self.proj.scale else 1)
        output = torch.zeros([x.shape[0], x.shape[1], self.super_embed_dim])
        output[:, :, :x.shape[-1]] = x
        print(output)
        return output

    def get_embedded_ops(self):
        return None


class QKV_super_embed_choice(AbstractPrimitive):
    def __init__(self, qkv_super, attn_layer_norm, embed_choice, super_emb,
                 pre_norm):
        super(QKV_super_embed_choice, self).__init__(locals())
        self.qkv_super = qkv_super
        self.sampled_in_dim = embed_choice
        self.super_embed_dim = super_emb
        self.normalize_before = pre_norm
        self.attn_layer_norm = attn_layer_norm

    def set_sample_config(self):
        self.qkv_super.sample_weight = self.qkv_super.weight[:, :self.
                                                             sampled_in_dim]
        self.attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sampled_in_dim)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x, edge_data):
        self.set_sample_config()
        x = self.maybe_layer_norm(self.attn_layer_norm,
                                  x[:, :, :self.sampled_in_dim],
                                  before=True)
        output = torch.zeros([x.shape[0], x.shape[1], self.super_embed_dim])
        output[:, :, :x.shape[-1]] = x
        print(output)
        return x

    def get_embedded_ops(self):
        return None


class QKV_super_head_choice(AbstractPrimitive):
    def __init__(self, qkv_super, rel_pos_embed_k, rel_pos_embed_v, proj,
                 head_choice, attn_drop, super_emb, super_head_emb_dim):
        super(QKV_super_head_choice, self).__init__(locals())
        self.super_embed_dim = super_emb
        self.qkv_super = qkv_super
        self.rel_pos_embed_k = rel_pos_embed_k
        self.rel_pos_embed_v = rel_pos_embed_v
        self.proj = proj
        self.sampled_out_dim = head_choice * 64 * 3
        self.sample_num_heads = head_choice
        self.sample_scale = (head_choice * 64 // head_choice)**-0.5
        self.super_head_emb_dim = super_head_emb_dim

        self.attn_drop = nn.Dropout(attn_drop)

    def set_sample_config(self):
        self.qkv_super.sample_weight = torch.cat([
            self.qkv_super.sample_weight[i:self.sampled_out_dim:3, :]
            for i in range(3)
        ],
                                                 dim=0)
        self.qkv_super.sample_bias = self.qkv_super.bias
        if self.qkv_super.bias is not None:
            self.qkv_super.sample_bias = self.qkv_super.bias[:self.
                                                             sampled_out_dim]
        self.qkv_super.sample_out_dim = self.sampled_out_dim
        self.rel_pos_embed_k.set_sample_config(64 * 3 * self.sample_num_heads)
        self.rel_pos_embed_v.set_sample_config(64 * 3 * self.sample_num_heads)
        self.proj.sample_weight = self.proj.weight[:, :64 *
                                                   self.sample_num_heads]

    def forward(self, x, edge_data):
        self.set_sample_config()
        B, N, C = x.shape
        print("QKV in shape", x.shape)
        qkv = self.qkv_super(x).reshape(B, N, 3, self.sample_num_heads,
                                        -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)
        print("Q shape", q.shape)
        print("K shape", k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.sample_scale
        r_p_k = self.rel_pos_embed_k(N, N)
        print(r_p_k.shape)
        attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.sample_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        r_p_v = self.rel_pos_embed_v(N, N)
        attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.sample_num_heads,
                                                  -1)
        x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(
            B, self.sample_num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)
        x = x * (self.super_embed_dim / self.sampled_out_dim)
        print("head emb dim", self.super_head_emb_dim)
        output = torch.zeros([x.shape[0], x.shape[1], self.super_head_emb_dim])
        print("QKV out shape", x.shape)
        output[:, :, :x.shape[-1]] = x
        #print(output.shape)
        #print(x.shape)
        print(output)
        return output

    def get_embedded_ops(self):
        return None


class qkv_super(nn.Linear):
    def __init__(self,
                 super_in_dim,
                 super_out_dim,
                 bias=True,
                 uniform_=None,
                 non_linear='linear',
                 scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim,
                                               self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        #self.sample_parameters()
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        #print(self.sample_weight.shape)
        #print(x.shape)
        return F.linear(
            x[:, :, :self.sample_weight.shape[-1]], self.sample_weight,
            self.sample_bias) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


def sample_weight(weight, sample_in_dim, sample_out_dim):

    sample_weight = weight[:, :sample_in_dim]
    sample_weight = torch.cat(
        [sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim=0)

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias
