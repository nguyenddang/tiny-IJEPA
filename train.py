'''
Training script for IJEPA. I onnly have a GeoForce RTX 4060 on a PC :( so there's no DDP here.
'''

import os
import time
from tqdm import tqdm
import yaml
import argparse
import ast

import torch 
import torch.nn as nn

from model import IJEPA, IJEPAConfig, LinearProbe
from utils import IJEPA_Dataloader, CosineScheduler, LinearScheduler

# -------------------------------------------------------------------------------------------
# boring stuff here
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file')
parser.add_argument('--overrides', type=str, nargs='*', help='Overrides for config file')
args = parser.parse_args()
#--------------------------------------------------------------------------------------------
# I/O
out_dir = 'test'
init_from = "scratch" # or "resume"
# system
device = 'cuda'
dtype = 'float16'
compile = False # set to True to compile the model
# linear probe
lnprobe_split = 'valid' # 'train' or 'valid'. When evaluating post-training, set to 'train'. 
lnprobe_epochs = 50 # how long to train linear probe
lnprobe_interval = 10
lnprobe_only = False # if True, exit after first evaluation
lnprobe_grad_accumulation_steps = 1
lnprobe_learning_rate = 1e-2
lnprobe_decay_epochs = 30
# model. 
n_layer = 12
n_head = 4
n_embd = 256
pred_layer = 6
pred_embd = 256
bias = True
dropout = 0.0
img_size = 64
in_channels = 3
patch_size = 8
n_targets = 4
context_scale = (0.85, 1.0)
target_aspect_ratio = (0.75, 1.5)
target_scale = (0.15, 0.2)
n_classes = 200
# data
dataset = 'tiny_imagenet'
batch_size = 192
grad_accumulation_steps = 4
# optimizer
start_lr = 3e-4
learning_rate = 6e-4
final_lr = 3e-4
max_epochs= 80
warmup_epochs = 3
lr_decay_epochs = max_epochs # should be ~ max_epochs
weight_decay = 0.1
min_weight_decay = weight_decay * 0.1
grad_clip = 1.0 
# momentum
momentum = 0.99
max_momentum = 0.99 # for smaller models, setting to momentum seems to work better
#-------------------------------------------------------------------------------------------
# read in config file from command line
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
#  overrides
if args.overrides:
    for override in args.overrides:
        key, value = override.split('=')
        try:
            config[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            config[key] = value
# update globals
globals().update(config)
# -------------------------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -------------------------------------------------------------------------------------------
# some inits
os.makedirs(out_dir, exist_ok=True)
data_dir = os.path.join('data', dataset)
torch.manual_seed(2207)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.autocast(device_type=device, dtype=ptdtype)
scaler = torch.GradScaler(enabled=(dtype == 'float16'))
model_args = {
    'n_layer': n_layer,
    'n_head': n_head,
    'n_embd': n_embd,
    'pred_layer': pred_layer,
    'pred_embd': pred_embd,
    'bias': bias,
    'dropout': dropout,
    'img_size': img_size,
    'in_channels': in_channels,
    'patch_size': patch_size,
    'n_targets': n_targets,
    'context_scale': context_scale,
    'target_aspect_ratio': target_aspect_ratio,
    'target_scale': target_scale,
    'n_classes': n_classes
}
iter_num = 0

# model init
print('Initializing model...')
if init_from == 'scratch':
    ijepa_config = IJEPAConfig(**model_args)
    model = IJEPA(ijepa_config)
else:
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in model_args:
        model_args[k] = checkpoint_model_args[k]
    ijepa_config = IJEPAConfig(**model_args)
    model = IJEPA(ijepa_config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    print(f"Resuming from iteration {out_dir} iter {iter_num}")
    if lnprobe_only:
        iter_num = 0
model.to(device)
# dataloader init
print('Initializing dataloader...')
loader = IJEPA_Dataloader(data_dir, ijepa_config, batch_size, device)
iters_per_epoch = {'train': loader.train_len//batch_size, 'valid': loader.valid_len//batch_size}
max_iters = (max_epochs * iters_per_epoch['train'])//grad_accumulation_steps
warmup_iters = (warmup_epochs * iters_per_epoch['train'])//grad_accumulation_steps
lr_decay_iters = (lr_decay_epochs * iters_per_epoch['train'])//grad_accumulation_steps
lnprobe_train_iters = int(lnprobe_epochs * iters_per_epoch[lnprobe_split]//lnprobe_grad_accumulation_steps)
lnprobe_interval = int(lnprobe_interval * iters_per_epoch['train']//grad_accumulation_steps)
lnprobe_decay_iters = int(lnprobe_decay_epochs * iters_per_epoch[lnprobe_split]//lnprobe_grad_accumulation_steps)
# open log file and clear it
train_log = os.path.join(out_dir, 'log_train.txt')
eval_log = os.path.join(out_dir, 'log_val.txt')
if init_from == 'scratch':
    with open(train_log, 'w') as f:
        f.write('iter,loss,mag,c%,t%\n')
    with open(eval_log, 'w') as f:
        f.write('loss,acc\n')
    
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile model
if compile:
    model = torch.compile( model)

# lr, momentum, weight decay schedule
lr_scheduler = CosineScheduler(warmup_iters, max_iters,start_lr, final_lr, learning_rate, lr_decay_iters)
mom_scheduler = LinearScheduler(max_iters, momentum, max_momentum)
wd_scheduler = LinearScheduler(max_iters, min_weight_decay, weight_decay)

def evaluate():
    model.eval()
    lm_probe = LinearProbe(ijepa_config) # a single linear layer (with batchnorm)
    lm_probe.to(device)
    optimizer = torch.optim.AdamW(lm_probe.parameters(), lr=3e-4, weight_decay=0.0)
    lrs  = [lnprobe_learning_rate*10**(-(i//lnprobe_decay_iters)) for i in range(lnprobe_train_iters)]
    def train():
        lm_probe.train()
        img, y = loader.get_batch(classify=True)
        for iter_num in (range(lnprobe_train_iters)):
            lr = lrs[iter_num]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            acc = 0
            for _ in range(lnprobe_grad_accumulation_steps):
                with ctx:
                    logits, loss = lm_probe(model.generate(img), y)
                    acc += (torch.argmax(logits, dim=1) == y).float().mean().item()/lnprobe_grad_accumulation_steps
                    loss = loss / lnprobe_grad_accumulation_steps
                img, y = loader.get_batch(classify=True)
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(lm_probe.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            print(f"Epoch {iter_num//(lnprobe_train_iters/lnprobe_epochs):.0f} iter {iter_num}: loss {loss.item()*lnprobe_grad_accumulation_steps:.4f} acc {acc:.4f} lr {lr:.4e} norm {norm:.4f}")
            
    @torch.no_grad()
    def val():
        lm_probe.eval()
        length = iters_per_epoch[lnprobe_split]
        losses = torch.zeros(length)
        accs = torch.zeros(length)
        for iter_num in range(length):
            img, y = loader.get_batch(classify=True)
            with ctx:
                logits, loss = lm_probe(model.generate(img), y)
                acc = (torch.argmax(logits, dim=1) == y).float()
            losses[iter_num] = loss.detach()
            accs[iter_num] = acc.mean()
        print(f"Validation: loss {losses.mean().item():.4f} acc {accs.mean().item():.4f}")
        if not lnprobe_only:
            with open(eval_log, 'a') as f:
                f.write(f"{losses.mean().item()},{accs.mean().item()}\n")
    train()
    val()
    model.train()

# training loop
img, target_idx, context_idx= loader.get_batch() # get first batch
while True:
    if iter_num  % lnprobe_interval==0: # add iter_num > 0 if don't want to evaluate at the start. Recommended to evaluate at the start to see if the model is learning anything at all.
        evaluate()
        if lnprobe_only:
            break
        checkpoint = {
            'model': model.state_dict(),
            'model_args': model_args,
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num+1,
            'config': config
        } 
        print(f"Saving checkpoint at iteration {iter_num} to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    t0 = time.time()
    lr = lr_scheduler.step(iter_num)
    mom = mom_scheduler.step(iter_num)
    decay = wd_scheduler.step(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay
    # forward, backward, update target encoder using EMA
    mags = 0
    context_percentage = 0
    target_percentage = 0
    for micro_steps in range(grad_accumulation_steps):
        with ctx:
            target_pred, loss = model(img, context_idx, target_idx)
            loss = loss / grad_accumulation_steps
        img, target_idx, context_idx = loader.get_batch()
        scaler.scale(loss).backward()
        mags += torch.norm(target_pred, p=2, dim=-1).mean().item()/grad_accumulation_steps # keep track of the magnitude of the target predictions, if converging to 0... bad bad
        context_percentage += context_idx.size(0)/(img_size/patch_size)**2/grad_accumulation_steps
        target_percentage += torch.unique(target_idx.view(-1)).size(0)/(img_size/patch_size)**2/grad_accumulation_steps
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    model.update_target_encoder(mom)
    torch.cuda.synchronize()
    t1 = time.time()
    # logging
    dt = (t1 - t0) 
    img_per_sec = batch_size*grad_accumulation_steps / dt
    with open(train_log, 'a') as f:
        f.write(f"{iter_num},{loss.item()*grad_accumulation_steps},{mags},{context_percentage},{target_percentage}\n")
    print(f"Epoch {iter_num//(iters_per_epoch['train']/grad_accumulation_steps):.0f} iter {iter_num} loss {loss.item()*grad_accumulation_steps:.4f} target_mag {mags:.2f} lr {lr:.4e} norm {norm:.4f} mom {mom:.4e} dt {dt*1000:.2f} ms img/s {img_per_sec:.2f} context_% {context_percentage*100:.0f} target_% {target_percentage*100:.0f}")
    iter_num += 1
    if iter_num > max_iters:
        break
