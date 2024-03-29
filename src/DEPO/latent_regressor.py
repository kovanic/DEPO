# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Copy paste from https://github.com/facebookresearch/detr/blob/main/models/transformer.py
LatentTransformerRegressor and make_fixed_pe are added.

'''
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
'''
"""

import copy
from typing import Optional, List
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.linalg import norm
import math


class LayerNormTranspose(nn.Module):
    def __init__(self, dim):
        super(LayerNormTranspose, self).__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """:param: x, torch.tensor(B, dim, N)"""
        return self.norm(x.transpose(2, 1)).transpose(2, 1)

    
class LatentTransformerRegressor(nn.Module):
    def __init__(self, num_queries=100, d_model=128, d_compressed=64,
                 num_decoder_layers=6, nhead=8, dim_feedforward=2048,
                 dropout=0.0, activation='relu', normalize_before=False,
                 return_intermediate_dec=False, H=60, W=80, use_pos_embed=True):
        super(LatentTransformerRegressor, self).__init__()
        
        d_c = d_compressed * num_queries
        assert d_c % 128 == 0, "d_compressed * num_queries should be divisible by 64"
        
        if use_pos_embed:
            self.register_buffer("pos_embed", make_fixed_pe(H, W, d_model // 2).unsqueeze(0))
        else:
            self.pos_embed = None
            
        self.geometry_patterns = nn.Embedding(num_queries, d_model)
    
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
                    
        self.pose_decoder = nn.Sequential(
            nn.Conv1d(num_queries * d_model, num_queries * d_model, 1, groups=num_queries),
            nn.LeakyReLU(0.1),
            nn.Conv1d(num_queries * d_model, d_c, 1, groups=num_queries),
            nn.LeakyReLU(0.1),
            nn.Conv1d(d_c, d_c // 4, 1, groups=2), #one half of tokens for t, another for q
            nn.LeakyReLU(0.1),
            nn.Conv1d(d_c // 4, d_c // 16, 1, groups=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(d_c // 16, d_c // 32, 1, groups=2),
            nn.LeakyReLU(0.1))

        self.final_dim = d_c // 64
        self.q_proj = nn.Conv1d(self.final_dim, 4, 1)
        self.t_proj = nn.Conv1d(self.final_dim, 3, 1)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    
    def forward(self, geometry):
        # flatten (B x d_model x H x W) -> (HW x B x d_model)
        B, _, H, W = geometry.shape
        geometry = geometry.flatten(2).permute(2, 0, 1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.repeat(B, 1, 1, 1).flatten(2).permute(2, 0, 1) 
        else:
            pos_embed = None
        query_embed = self.geometry_patterns.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_embed)
        decoded_patterns = self.decoder(tgt, geometry, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed) # (num_queries x B x d_model)

        decoded_patterns = decoded_patterns.permute(1, 0, 2).flatten(1).unsqueeze(2) # (B x d_model * num_queries x 1)
        decoded_patterns = self.pose_decoder(decoded_patterns)
        
        t = self.t_proj(decoded_patterns[:, :self.final_dim, ...]).squeeze(2)
        q = self.q_proj(decoded_patterns[:, self.final_dim:, ...]).squeeze(2) #(B x 4 x 1) -> (B x 4)
        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / norm(q, ord=2, dim=1, keepdim=True)
        
        return q, t
        
    

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if 'leaky_relu' in activation:
        slope = float(activation.split(':')[1])
        return partial(F.leaky_relu, negative_slope=0.1)
    
    raise RuntimeError(F"activation should be relu/gelu/leaky_relu, not {activation}.")

    

def make_fixed_pe(H, W, dim, scale=2*math.pi, temperature = 10_000):

    h = torch.linspace(0, 1, H)[:, None, None].repeat(1, W, dim)  # [0, scale]
    w = torch.linspace(0, 1, W)[None, :, None].repeat(H, 1, dim)

    dim_t = torch.arange(0, dim, 2).repeat_interleave(2)
    dim_t = temperature ** (dim_t / dim)

    h /= dim_t
    w /= dim_t

    h = torch.stack([h[:, :, 0::2].sin(), h[:, :, 1::2].cos()], dim=3).flatten(2)
    w = torch.stack([w[:, :, 0::2].sin(), w[:, :, 1::2].cos()], dim=3).flatten(2)

    pe = torch.cat((h, w), dim=2)
    return pe.permute(2, 0, 1)