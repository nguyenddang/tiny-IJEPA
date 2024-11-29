'''
Full definition of ViT Encoder and Vit Predictor for JEPA. Refrence Andrej Karpathy's nanoGPT.
'''
from dataclasses import dataclass
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config, n_embd):
        super().__init__()
        assert n_embd % config.n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.n_embd = n_embd

    def forward(self, x):
        B, N, C = x.size() 
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, N, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, N, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, N, self.n_head, C // self.n_head).transpose(1, 2) 
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, N, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config, n_embd):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, n_embd):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd, bias=config.bias)
        self.mlp = MLP(config, n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class PatchConv(nn.Module):
        
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(config.in_channels, config.n_embd, kernel_size=config.patch_size, stride=config.patch_size, bias=config.bias)
        
    def forward(self, x):
        B, N, C, H, W = x.size()
        x = self.proj(x.reshape(-1, self.in_channels, self.patch_size, self.patch_size )).view(B, N, -1)
        return x  
    
class Vit(nn.Module):
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.patchify = PatchConv(config)
        self.wpe = nn.Embedding((config.img_size//config.patch_size)**2, config.n_embd) # num positional embeddings = num patches
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config, n_embd=config.n_embd) for _ in range(config.n_layer)])   
        self.ln_out = nn.LayerNorm(config.n_embd, bias=config.bias)
        
    def forward(self, x, mask=None, return_embd_mean=False):
        x = self.patchify(x)
        B, N, C = x.size()
        pos_emb = self.wpe(mask) if mask is not None else self.wpe(torch.arange(N, device=x.device))
        x = self.dropout(x + pos_emb)
        
        if return_embd_mean:
            # when training linear probe, return the concatenated mean embedding of all layers: (B, n_embd * n_layer)
            embd_mean = torch.zeros(x.size(0), self.config.n_embd * 4, device=x.device)
            for r, block in enumerate(self.h):
                x = block(x)
                if r > 7:
                    embd_mean[:, (r-8) * self.config.n_embd:(r - 8 + 1) * self.config.n_embd] = x.mean(dim=1)
            return embd_mean
        else:
            for block in self.h:
                x = block(x)
            return self.ln_out(x)

class VitPredictor(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.ln = nn.LayerNorm(config.n_embd, bias=config.bias)
        # projection from encoder embedding size to predictor embedding size
        self.proj = nn.Linear(config.n_embd, config.pred_embd, bias=config.bias)
        self.h = nn.ModuleList([Block(config, n_embd=config.pred_embd) for _ in range(config.pred_layer)])
        self.ln_f = nn.LayerNorm(config.pred_embd, bias=config.bias)
        # project back to n_embd
        self.lm_head = nn.Linear(config.pred_embd, config.n_embd, bias=config.bias)
        
    def forward(self, x):
        x = self.ln(x)
        x = self.proj(x)
        for block in self.h:
            x = block(x)
        x = self.lm_head(self.ln_f(x))
        return x
    
@dataclass
class IJEPAConfig:
    # encoders and preditor
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    pred_layer: int = 6
    pred_embd: int = 384
    bias: bool = True
    dropout: float = 0.0
    # input and block scaling
    img_size: int = 64
    in_channels: int = 3
    patch_size: int = 8
    # target and context scaling
    n_targets: int = 6
    context_scale: tuple[float, float] = (0.85, 1.0)
    target_aspect_ratio: tuple[float, float] = (0.75, 1.5)   
    target_scale: tuple[float, float] = (0.15, 0.2)
    # default num class in tiny imagenet
    n_classes: int = 200
    
class IJEPA(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.img_size % config.patch_size == 0, 'Image size must be divisible by dimension of a patch'
        self.config = config
        self.n_targets = config.n_targets
        self.n_embd = config.n_embd
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.n_embd)) # make token for prediction
        # encoders and predictor defined here
        self.context_encoder = Vit(config)
        self.predictor = VitPredictor(config)
        # initialize weights
        self.apply(self._init_weights)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        print(f'Number of parameters: {self.get_num_params()/1e6:.2f}M')
        
    def get_num_params(self):
        # exclude target_encoder
        context_params = sum(p.numel() for p in self.context_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        return context_params + predictor_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, x, context_idx, target_idx):
        B, N, C, H, W = x.size()
        # forward target encoder
        with torch.no_grad():
            target_rep = self.target_encoder(x) # (B, (img_size//config.patch_size)**2, n_embd)
            targets = torch.stack([target_rep[:, idx, :] for idx in target_idx], dim=1) 
        # forward context encoder
        context_rep = self.context_encoder(x[:, context_idx]) # (B, context_idx.size(0), n_embd)
        target_pred = torch.zeros(B, target_idx.size(0), target_idx.size(1), self.n_embd, device=x.device)
        for i in range(self.n_targets):
            mask = self.mask_token.expand(B, target_idx.size(1), -1)
            mask = mask + self.context_encoder.wpe(target_idx[i])
            pred_x = torch.cat([context_rep, mask], dim=1)
            pred_x = self.predictor(pred_x)
            target_pred[:, i, :, :] = pred_x[:, -target_idx.size(1):, :]
        loss = F.smooth_l1_loss(target_pred.view(-1, self.n_embd), targets.view(-1, self.n_embd).detach())
        return target_pred, loss
    
    @torch.no_grad()
    def update_target_encoder(self, momentum):
        for param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.mul_(momentum).add_((1.-momentum)*param.detach().data) # EMA formula: target_param = momentum * target_param + (1-momentum) * context_param
    
    @torch.no_grad()
    def generate(self, x):
        out = self.target_encoder(x, return_embd_mean=True)
        return out
    
class LinearProbe(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln = nn.LayerNorm(config.n_embd*4, bias=True)
        self.lm = nn.Linear(config.n_embd*4, config.n_classes, bias=True)
        self.apply(self._init_weights)
        
    def forward(self, x, target):
        x = self.ln(x)
        logits = self.lm(x)
        loss = F.cross_entropy(logits, target)
        return logits, loss
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
