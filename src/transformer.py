import torch
import torch.nn as nn
from einops import rearrange

import numpy as np

from collections import OrderedDict
from typing import Tuple


def generate_pe_matrix(embedding_dim: int, n_input_tokens: int) -> torch.Tensor:
    """Generate the positional encoding matrix for the input tokens"""
    pe_denominator = 2 * torch.arange(embedding_dim) / embedding_dim
    pe_denominator = torch.pow(1e4, pe_denominator)

    input_length_count = torch.arange(n_input_tokens)
    ratio_matrix = input_length_count.view(-1, 1) / pe_denominator

    sin_matrix = torch.sin(ratio_matrix)
    cos_matrix = torch.cos(ratio_matrix)
    interleaved_matrix = torch.zeros_like(ratio_matrix)

    for i in range(embedding_dim):
        if i % 2 == 0:
            interleaved_matrix[:, i] = sin_matrix[:, i // 2]
        else:
            interleaved_matrix[:, i] = cos_matrix[:, i // 2]

    return interleaved_matrix


class Embedding(nn.Module):
    """Embedding layer for the encoder transformer model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_input_tokens: int,
        model_dim: int,
        padding_idx: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.n_input_tokens = n_input_tokens
        self.padding_idx = padding_idx

        # token_embedding: n_batch * n_input_tokens * embedding_dim
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
        )

        # positional_embedding: n_input_tokens * embedding_dim
        self.register_buffer(
            "pe_matrix",
            generate_pe_matrix(
                embedding_dim=embedding_dim, n_input_tokens=n_input_tokens
            ),
        )

        self.linear_model_dim = nn.Linear(embedding_dim, model_dim)
        nn.init.xavier_normal_(self.linear_model_dim.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() in [1, 2]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] == self.n_input_tokens
        assert x.max() < self.vocab_size
        assert x.min() >= 0

        # x: b,s
        x = self.token_embedding(x) + self.pe_matrix
        # x: b,s,e
        x = self.linear_model_dim(x)
        # x: b,s,m
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """Multi-head self-attention module for the encoder transformer model."""

    def __init__(
        self,
        sequence_length: int,
        model_dim: int,
        k_dim: int,
        v_dim: int,
        n_heads: int,
        dropout: float = 0,
        use_custom_mhsa: bool = True,
    ):
        super().__init__()
        self.use_custom_mhsa = use_custom_mhsa
        self.n_heads = n_heads
        self.k_dim = k_dim

        self.linear_m_qk = nn.Linear(model_dim, n_heads * k_dim * 2)
        nn.init.xavier_normal_(self.linear_m_qk.weight)
        self.linear_m_v = nn.Linear(model_dim, n_heads * v_dim)
        nn.init.xavier_normal_(self.linear_m_v.weight)

        self.softmax_k = nn.Softmax(dim=3)

        if self.use_custom_mhsa:
            self.linear_v_m = nn.Linear(n_heads * v_dim, model_dim)
            nn.init.xavier_normal_(self.linear_v_m.weight)
        else:
            self.mhsa = nn.MultiheadAttention(
                embed_dim=model_dim,
                num_heads=n_heads,
                kdim=k_dim * n_heads,  # kdim is for all heads not per
                vdim=v_dim * n_heads,  # vdim is for all heads not per
                dropout=dropout,
                batch_first=True,
            )

        self.layer_norm = nn.LayerNorm([sequence_length, model_dim])

    def forward(self, x_attn_tuple: Tuple[torch.Tensor]) -> torch.Tensor:
        x, attention_mask = x_attn_tuple
        # x: b,s,m
        # attention_mask: b,s

        ### Scaled Attention
        if self.use_custom_mhsa:
            ## Generate the query and key tensors
            qk = self.linear_m_qk(x)
            # qk: b,s,h*k*2
            qk = rearrange(qk, "b s (k n h) -> h b s k n", n=2, h=self.n_heads)
            # qk: h,b,s,k,2
            q, k = torch.chunk(qk, chunks=2, dim=-1)
            # q,k: h,b,s,k

            ## Generate the value tensor
            v = self.linear_m_v(x)
            # v: b,s,v*h
            v = rearrange(v, "b s (v h) -> h b s v", h=self.n_heads)
            # v: h,b,s,v

            attn = torch.einsum("hbimz,hbjmz->hbij", [q, k]) / np.sqrt(self.k_dim)
            # attn: h,b,s_q,s_v
            attn = attn.masked_fill(
                attention_mask.unsqueeze(0).unsqueeze(2), float("-inf")
            )
            attn = self.softmax_k(attn)  # Softmax along dim 3 (s_v)

            ## Apply the attention weights to the values
            attn = torch.einsum("hbij,hbjv->hbiv", [attn, v])
            # out: h,b,s,v

            ## Stack the output, map to d_model
            attn = rearrange(attn, "h b s v -> b s (h v)")
            # out: b,s,h*v
            attn = self.linear_v_m(attn)
            # out: b,s,m

        else:  # Built-in MultiheadAttention, actually seems ~10% slower
            ## Generate the query and key tensors
            qk = self.linear_m_qk(x)
            # qk: b,s,h*k*2
            qk = rearrange(qk, "b s (k n) -> b s k n", n=2)
            # qk: b,s,k,2
            q, k = torch.chunk(qk, chunks=2, dim=-1)
            # q,k: b,s,k,1

            ## Generate the value tensor
            v = self.linear_m_v(x)
            # v: b,s,v*h

            attn, _ = self.mhsa(
                q.squeeze(3), k.squeeze(3), v, key_padding_mask=attention_mask
            )
            # out: b,s,m

        ### Create residual connection to the input, then layer norm
        attn = self.layer_norm(attn + x)
        # out: b,s,m

        return attn, attention_mask


class FeedForward(nn.Module):
    """
    Feed forward module for the encoder transformer model.
    model_dim -> ff_dim -> ReLU -> model_dim
    """

    def __init__(self, model_dim: int, ff_dim: int, dropout: int = 0):
        super().__init__()

        self.linear_1 = nn.Linear(model_dim, ff_dim)
        nn.init.xavier_normal_(self.linear_1.weight)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        nn.init.xavier_normal_(self.linear_2.weight)
        self.layer_norm = nn.LayerNorm([model_dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + x)
        return out


class MHSAFeedForward(FeedForward):
    def forward(self, x_attn_tuple: Tuple[torch.Tensor]) -> torch.Tensor:
        x, attention_mask = x_attn_tuple
        out = super().forward(x)
        return out, attention_mask


class EncoderTransformer(nn.Module):
    """Encoder transformer model."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        n_layers: int,
        embedding_dim: int,
        model_dim: int,
        k_dim: int,
        v_dim: int,
        n_heads: int,
        ff_dim: int,
        padding_idx: int,
        dropout_ff: int = 0,
        dropout_mhsa: int = 0,
        use_custom_mhsa: bool = True,
    ):
        super().__init__()

        ## Token and positional embedding layer
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_input_tokens=sequence_length,
            model_dim=model_dim,
            padding_idx=padding_idx,
        )

        ## Multi-head self-attention and feed forward modules for each layer
        mhsa_modules = []
        for i in range(n_layers):
            mhsa_modules.append(
                (
                    f"mhsa_{i}",
                    MultiHeadSelfAttentionModule(
                        sequence_length=sequence_length,
                        model_dim=model_dim,
                        k_dim=k_dim,
                        v_dim=v_dim,
                        n_heads=n_heads,
                        dropout=dropout_mhsa,
                        use_custom_mhsa=use_custom_mhsa,
                    ),
                )
            )
            mhsa_modules.append(
                (
                    f"ff_{i}",
                    MHSAFeedForward(
                        model_dim=model_dim, ff_dim=ff_dim, dropout=dropout_ff
                    ),
                )
            )
        self.mhsa = nn.Sequential(OrderedDict(mhsa_modules))

        ## Output layer - likelihood of each token at each position
        self.output = nn.Sequential(
            nn.Linear(model_dim, vocab_size),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for the encoder transformer model.
        attention_mask is a tensor of shape (batch_size, sequence_length) with 0s in the positions of padding tokens.
        """
        # x: b,s
        x = self.embedding(x)
        # x: b,s,m
        x_attn_tuple = self.mhsa((x, attention_mask == 0))
        x = x_attn_tuple[0]
        # x: b,s,m
        x = self.output(x)
        # x: b,s,voc

        return x


# class EncoderTransformerPrebuilt(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         sequence_length: int,
#         n_layers: int,
#         embedding_dim: int,
#         model_dim: int,
#         k_dim: int,
#         v_dim: int,
#         n_heads: int,
#         ff_dim: int,
#         padding_idx: int,
#         dropout_ratio: int = 0,
#     ):

#         ## Token and positional embedding layer
#         self.embedding = Embedding(
#             vocab_size=vocab_size,
#             embedding_dim=embedding_dim,
#             n_input_tokens=sequence_length,
#             model_dim=model_dim,
#             padding_idx=padding_idx,
#         )

#         ## Multi-head self-attention and feed forward modules for each layer
#         mhsa_modules = []
#         for i in range(n_layers):
#             # Use torch.nn.MultiheadAttention
#             mhsa_modules.append(
#                 (
#                     f"mhsa_{i}",
#                     nn.MultiheadAttention(
#                         embed_dim=model_dim, num_heads=n_heads, dropout=dropout_ratio
#                     ),
#                 )
#             )
#         self.mhsa = nn.Sequential(OrderedDict(mhsa_modules))

#         ## Output layer - likelihood of each token at each position
#         self.output = nn.Sequential(
#             nn.Linear(model_dim, vocab_size),
#         )
