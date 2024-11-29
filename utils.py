'''3 classess here:

- IJEPA_Dataloader:  Target block are chosen randomly (usually 4 blocks, each block contains multiple patches). Context block are chosen  from original image with the target blocks removed. Both target and context blocks scales and aspect ratios are chosen randomly from a specified range. 
- CosineScheduler
- LinearScheduler
'''

import numpy as np
import torch
import math
import os
import torch.nn.functional as F

class IJEPA_Dataloader():
    
    def __init__(self, data, config, batch_size, device):
        assert config.img_size % config.patch_size == 0, 'Image size must be divisible by dimension of a patch'
        self.data = data
        self.device = device
        
        self.context_scale = config.context_scale
        self.target_aspect_ratio = config.target_aspect_ratio
        self.target_scale = config.target_scale
        
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.n_patches = (config.img_size // config.patch_size)**2
        self.n_patch_per_side = config.img_size // config.patch_size
        self.patch_size = config.patch_size
        self.n_targets = config.n_targets
        
        self.train_len = len(np.memmap(os.path.join(self.data, 'train_label.bin'), dtype=np.int32, mode='r'))
        self.valid_len = len(np.memmap(os.path.join(self.data, 'valid_label.bin'), dtype=np.int32, mode='r'))
        self.batch_size = batch_size
        
        # different RNG for context, target and batch
        self.context_rng = np.random.default_rng(1234)
        self.target_rng = np.random.default_rng(4321)
        self.batch_rng = torch.manual_seed(2030)
        
    def get_batch(self, classify=False):
        length = self.train_len if not classify else self.valid_len
        split = 'train' if not classify else 'valid'
        idx = torch.randint(0, length, (self.batch_size,), generator=self.batch_rng)
        data = np.memmap(os.path.join(self.data, f'{split}_img.bin'), dtype=np.float32, mode='r', shape=(length, self.in_channels, self.img_size, self.img_size))
        labels = np.memmap(os.path.join(self.data, f'{split}_label.bin'), dtype=np.int32, mode='r')
        img = torch.from_numpy(data[idx].astype(np.float32))
        y = torch.from_numpy(labels[idx]).long()
        # patchify the image
        img = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size).view(self.batch_size, self.in_channels, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        if classify:
            # evaluating JEPA with classification tasks. Return the image and the label
            if self.device == 'cuda':
                return img.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
            return img.to(self.device), y.to(self.device)
        else:
            # training JEPA. Return the image, target and context blocks indices
            target_idx = self.get_target_idx()
            context_idx = self.get_context_idx(target_idx)
            if self.device == 'cuda':
                return img.pin_memory().to(self.device, non_blocking=True), target_idx.pin_memory().to(self.device, non_blocking=True), context_idx.pin_memory().to(self.device, non_blocking=True)
            return img.to(self.device), target_idx.to(self.device), context_idx.to(self.device)
            
    def get_target_idx(self):
        # choose random target aspect ratio and scale
        target_aspect_ratio = self.target_rng.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = self.target_rng.uniform(self.target_scale[0], self.target_scale[1])
        target_n_patches = int(self.n_patches * target_scale)
        target_hw = int(math.sqrt(target_n_patches / target_aspect_ratio)) # target height and width assuming h=w
        # choose random starting index within range
        patch_idx = torch.arange(self.n_patches).view(self.n_patch_per_side, self.n_patch_per_side)[:self.n_patch_per_side - target_hw, :self.n_patch_per_side - target_hw].flatten()
        start_idx = patch_idx[self.target_rng.choice(len(patch_idx), size=self.n_targets, replace=False)] # whats the starting index of the target
        # get the rest of the target idx
        target_idx = []
        for sidx in start_idx:
            idx_list = []
            for i in range(target_hw):
                for j in range(target_hw):
                    idx_list.append(sidx.item() + i * self.n_patch_per_side + j)
            target_idx.append(idx_list)
        return torch.tensor(target_idx)

    def get_context_idx(self, target_idx):
        context_scale = self.context_rng.uniform(self.context_scale[0], self.context_scale[1])
        context_n_patches = int(self.n_patches * context_scale)
        context_hw = int(math.sqrt(context_n_patches))
        if context_hw == self.n_patch_per_side:
            start_idx = 0
        else:
            start_idx = self.context_rng.integers(0, self.n_patch_per_side - context_hw, (1,)).item()
        context_idx = []
        for i in range(context_hw):
            for j in range(context_hw):
                idx = start_idx + i * self.n_patch_per_side + j
                if all(idx not in target for target in target_idx):
                    # filter out target idx
                    context_idx.append(idx)
        return torch.tensor(context_idx)
    
class CosineScheduler():
    
    def __init__(self, warmup_iters, max_iters, start_value, end_value, max_value, value_decay_iters):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.start_value = start_value
        self.end_value = end_value
        self.max_value = max_value
        self.value_decay_iters = value_decay_iters
        
    def step(self, iter):
        if iter < self.warmup_iters:
            lr = self.start_value + (self.max_value - self.start_value) * iter / self.warmup_iters
        elif iter > self.value_decay_iters:
            lr = self.end_value
        else:
            decay_ratio = (iter - self.warmup_iters) / (self.value_decay_iters - self.warmup_iters)
            ceoff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.end_value  + (self.max_value - self.end_value) * ceoff
        return lr

class LinearScheduler():
    
    def __init__(self, max_iters, min_value, max_value):
        self.max_iters = max_iters
        self.min_value = min_value
        self.max_value = max_value
        
    def step(self, iter):
        return self.min_value + (self.max_value - self.min_value) * iter / self.max_iters
    


        