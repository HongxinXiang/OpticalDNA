from __future__ import annotations

from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from typing_extensions import Literal


class MultiPageSelfAttentionFusion(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        proj_in: bool = True,
        proj_out: bool = True,
        reduce: Literal["mean", "sum", "none"] = "mean",
        num_attn_layers: int = 1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.proj_in = proj_in
        self.proj_out = proj_out
        self.reduce = reduce
        self.num_attn_layers = num_attn_layers

        self.in_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(self.num_attn_layers)
        ])

        self._ensure_proj(d_in=self.d_model)

        if reduce not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduce mode: {reduce}")

    @staticmethod
    @torch.no_grad()
    def _init_linear_like_mha(linear: nn.Linear) -> None:
        xavier_uniform_(linear.weight)
        if linear.bias is not None:
            constant_(linear.bias, 0.0)

    def _ensure_proj(self, d_in: int) -> Tuple[Optional[nn.Linear], Optional[nn.Linear]]:
        ref = self.attn[0].in_proj_weight                  
        device, dtype = ref.device, ref.dtype

        if self.proj_in and self.in_proj is None:
            self.in_proj = nn.Linear(d_in, self.d_model, bias=True).to(device=device, dtype=dtype)
            self._init_linear_like_mha(self.in_proj)

        if self.proj_out and self.out_proj is None:
            self.out_proj = nn.Linear(self.d_model, d_in, bias=True).to(device=device, dtype=dtype)
            self._init_linear_like_mha(self.out_proj)

        return self.in_proj, self.out_proj

    @staticmethod
    def _normalize_page_mask(page_mask: Optional[torch.Tensor], B: int, P: int, device, dtype) -> Optional[torch.Tensor]:
        if page_mask is None:
            return None
        if page_mask.dim() != 2 or page_mask.shape[0] != B or page_mask.shape[1] != P:
            raise ValueError(f"page_mask must be [B, P], got {page_mask.shape} (B={B}, P={P})")
        if page_mask.dtype != torch.bool:
            page_mask = page_mask.to(torch.bool)
        return page_mask.to(device=device)

    def forward(self, x: torch.Tensor, page_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x to be 4D [B, P, T, D], got {x.shape}")

        B, P, T, D_in = x.shape
        in_proj, out_proj = self._ensure_proj(D_in)
        page_mask = self._normalize_page_mask(page_mask, B=B, P=P, device=x.device, dtype=x.dtype)

                                        
                                        
        if P == 1:
            if in_proj is not None:
                x_proj = in_proj(x)                      
            else:
                x_proj = x                      

                       
            if self.reduce == "none":
                x_fused = x_proj.permute(0, 2, 1, 3).contiguous()                
            else:
                x_fused = x_proj[:, 0]             

            if out_proj is not None:
                x_fused = out_proj(x_fused)

            return x_fused

                                        
                                        
        if in_proj is not None:
            x = in_proj(x)                      

                                        
                                        
        x_bt = x.permute(0, 2, 1, 3).contiguous().view(B * T, P, x.shape[-1])

        key_padding_mask = None
        if page_mask is not None:
            kpm = ~page_mask          
            key_padding_mask = kpm.unsqueeze(1).expand(B, T, P).reshape(B * T, P)

        h = x_bt
        for attn_layer in self.attn:
            h, _ = attn_layer(
                h, h, h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        x_attn = h                     

                                        
                                        
        if self.reduce == "none":
            x_fused = x_attn.view(B, T, P, x_attn.shape[-1])                      
        else:
            if page_mask is None:
                if self.reduce == "mean":
                    x_fused = x_attn.mean(dim=1)                  
                elif self.reduce == "sum":
                    x_fused = x_attn.sum(dim=1)                   
                else:
                    raise ValueError
            else:
                valid = page_mask.unsqueeze(1).expand(B, T, P).reshape(B * T, P)            
                valid_f = valid.to(dtype=x_attn.dtype).unsqueeze(-1)                           

                x_attn_masked = x_attn * valid_f              

                if self.reduce == "sum":
                    x_fused = x_attn_masked.sum(dim=1)                  
                elif self.reduce == "mean":
                    denom = valid_f.sum(dim=1).clamp_min(1.0)            
                    x_fused = x_attn_masked.sum(dim=1) / denom
                else:
                    raise ValueError

            x_fused = x_fused.view(B, T, x_attn.shape[-1])                   

                                        
                                        
        if out_proj is not None:
            x_fused = out_proj(x_fused)

        return x_fused


class MultiPageAttentionPooling(nn.Module):

    def __init__(
        self,
        d_hidden: int,
        proj_in: bool = False,
        proj_out: bool = False,
        activation: Literal["tanh", "relu"] = "tanh",
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.proj_in = proj_in
        self.proj_out = proj_out

        self.in_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None

        self.fc: Optional[nn.Linear] = None     
        self.v: Optional[nn.Linear] = None        

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @staticmethod
    def _normalize_page_mask(page_mask: Optional[torch.Tensor], B: int, P: int, device) -> Optional[torch.Tensor]:
        if page_mask is None:
            return None
        if page_mask.dim() != 2 or page_mask.shape[0] != B or page_mask.shape[1] != P:
            raise ValueError(f"page_mask must be [B, P], got {page_mask.shape} (B={B}, P={P})")
        if page_mask.dtype != torch.bool:
            page_mask = page_mask.to(torch.bool)
        return page_mask.to(device=device)

    def _ensure_layers(self, d_in: int) -> None:
        if self.proj_in and self.in_proj is None:
            self.in_proj = nn.Linear(d_in, self.d_hidden)

        if self.fc is None:
            dim_for_score = self.d_hidden if self.in_proj is not None else d_in
            self.fc = nn.Linear(dim_for_score, self.d_hidden)
            self.v = nn.Linear(self.d_hidden, 1)

        if self.proj_out and self.out_proj is None:
            self.out_proj = nn.Linear(d_in, d_in)

    def forward(self, x: torch.Tensor, page_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x to be 4D [B, P, T, D], got {x.shape}")

        B, P, T, D_in = x.shape
        self._ensure_layers(D_in)
        page_mask = self._normalize_page_mask(page_mask, B=B, P=P, device=x.device)

                                        
                                        
        if P == 1:
            x_fused = x[:, 0]                
            if self.out_proj is not None:
                x_fused = self.out_proj(x_fused)
            return x_fused

                                        
                                        
        if self.in_proj is not None:
            x_score = self.in_proj(x)                       
        else:
            x_score = x                                

                                        
                                        
        h = self.activation(self.fc(x_score))                       
        score = self.v(h)                                    

                                        
                                        
        if page_mask is not None:
            invalid = (~page_mask).unsqueeze(-1).unsqueeze(-1)
            score = score.masked_fill(invalid, float("-inf"))

        attn = torch.softmax(score, dim=1)                

        attn = torch.nan_to_num(attn, nan=0.0)

                                        
                                        
        x_fused = (attn * x).sum(dim=1)                

                                        
                                        
        if self.out_proj is not None:
            x_fused = self.out_proj(x_fused)                

        return x_fused
