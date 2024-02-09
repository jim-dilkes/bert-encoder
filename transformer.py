import torch
import torch.nn as nn
from einops import rearrange

import numpy as np

from collections import OrderedDict


def generate_pe_matrix(embedding_dim:int, n_input_tokens:int) -> torch.Tensor:
    """Generate the positional encoding matrix for the input tokens"""
    pe_denominator = 2*torch.arange(embedding_dim)/embedding_dim
    pe_denominator = torch.pow(1e4,pe_denominator)

    input_length_count = torch.arange(n_input_tokens)
    ratio_matrix = input_length_count.view(-1,1)/pe_denominator

    sin_matrix = torch.sin(ratio_matrix)
    cos_matrix = torch.cos(ratio_matrix)
    interleaved_matrix = torch.zeros_like(ratio_matrix)

    for i in range(embedding_dim):
        if i % 2 == 0:
            interleaved_matrix[:, i] = sin_matrix[:, i // 2]
        else:
            interleaved_matrix[:, i] = cos_matrix[:, i // 2]

    return interleaved_matrix


class EncoderEmbedding(nn.Module):
    """Embedding layer for the encoder transformer model."""
    def __init__(self, vocab_size:int, embedding_dim:int, n_input_tokens:int, model_dim:int, padding_idx:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.n_input_tokens = n_input_tokens
        self.padding_idx = padding_idx

        # token_embedding: n_batch * n_input_tokens * embedding_dim
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=self.padding_idx)

        # positional_embedding: n_input_tokens * embedding_dim
        self.register_buffer('pe_matrix', generate_pe_matrix(embedding_dim=embedding_dim, n_input_tokens=n_input_tokens))

        self.linear_model_dim = nn.Linear(embedding_dim, model_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.dim() in [1,2]
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
    

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for the encoder transformer model."""
    def __init__(self, sequence_length:int, model_dim:int, k_dim:int, v_dim:int, n_heads:int):
        super().__init__()
        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_heads = n_heads
    
        self.linear_m_qk = nn.Linear(self.model_dim, self.n_heads*self.k_dim*2)
        self.linear_m_v = nn.Linear(self.model_dim, self.n_heads*self.v_dim)

        self.softmax_k = nn.Softmax(dim=3)

        self.linear_v_m = nn.Linear(self.n_heads*self.v_dim, self.model_dim)

        self.layer_norm = nn.LayerNorm([self.sequence_length,self.model_dim])

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: b,s,m

        ## Scaled Attention
        qk = self.linear_m_qk(x)
        # qk: b,s,h*k*2
        qk = rearrange(qk, 'b s (k n h) -> h b s k n', n=2, h=self.n_heads)
        # qk: h,b,s,k,2
        q, k = torch.split(qk, split_size_or_sections=1, dim=4)
        # q,k: h,b,s,k,1
        attn = torch.einsum('hbimz,hbjmz->hbij', [q,k]) / np.sqrt(self.k_dim)
        # attn: h,b,s_q,s_v
        attn = self.softmax_k(attn) # Softmax along dim 3 (s_v)
        
        ## Generate the value tensor
        v = self.linear_m_v(x)
        # v: b,s,v*h
        v = rearrange(v,'b s (v h) -> h b s v', h=self.n_heads)
        # v: h,b,s,v
        
        ## Apply the attention weights to the values
        out = torch.einsum('hbij,hbjv->hbiv', [attn,v])
        # out: h,b,s,v
        
        ## Stack the output, map to d_model
        out = rearrange(out, 'h b s v -> b s (h v)')
        # out: b,s,h*v
        out = self.linear_v_m(out)
        # out: b,s,m

        ## Create residual connection to the input, then layer norm
        out = self.layer_norm(out + x)
        # out: b,s,m

        return out
    

class EndoderFeedForward(nn.Module):
    """
    Feed forward module for the encoder transformer model.
    model_dim -> ff_dim -> ReLU -> model_dim
    """
    def __init__(self, model_dim:int, ff_dim:int):
        super().__init__()

        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.layer_norm = nn.LayerNorm([model_dim])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.layer_norm(out + x)
        return out


class EncoderTransformer(nn.Module):
    """Encoder transformer model."""
    def __init__(self, vocab_size:int, sequence_length:int, n_layers:int, embedding_dim:int, 
                 model_dim:int, k_dim:int, v_dim:int, n_heads:int, ff_dim:int, padding_idx:int):
        super().__init__()      

        ## Token and positional embedding layer
        self.embedding = EncoderEmbedding(vocab_size=vocab_size, 
                                                 embedding_dim=embedding_dim, 
                                                 n_input_tokens=sequence_length, 
                                                 model_dim=model_dim,
                                                 padding_idx=padding_idx)

        ## Multi-head self-attention and feed forward modules for each layer
        mhsa_modules = []
        for i in range(n_layers):
            mhsa_modules.append((f'mhsa_{i}', MultiHeadSelfAttention(sequence_length=sequence_length,
                                                model_dim=model_dim, 
                                                n_heads=n_heads,
                                                k_dim=k_dim,
                                                v_dim=v_dim)
                                                ))
            mhsa_modules.append((f'ff_{i}', EndoderFeedForward(model_dim=model_dim, ff_dim=ff_dim)))
        self.mhsa = nn.Sequential(OrderedDict(mhsa_modules))

        ## Output layer - likelihood of each token at each position
        self.output = nn.Sequential(nn.Linear(model_dim, vocab_size),
                                    nn.Softmax(dim=2))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: b,s
        x = self.embedding(x)
        # x: b,s,m
        x = self.mhsa(x)
        # x: b,s,m
        x = self.output(x)
        # x: b,s,voc

        return x
    