import torch
from pathlib import Path
ckpt_dir = '/shared/s1/lab08/ugyeong/llama-dl/30B'
world_size = 4
max_seq_len = 512
max_batch_size = 12
checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
ckpt_values = []
for i in range(world_size):
    ckpt_path= checkpoints[i]
    print(i)
    ckpt_values.append(torch.load(ckpt_path, map_location="cpu"))

def merge_params(name):
    end0 = ['wq.weight', 'wk.weight', 'wv.weight', 'w1.weight', 'w3.weight']
    end1 = ['wo.weight', 'w2.weight']
    start0 = ['output']
    start1 = ['tok_embeddings']
    if name[:6] in start0 or name[-9:] in end0:
        for i in range(4):
            return torch.cat([a[name] for a in ckpt_values], dim=0)
    if name[:14] in start1 or name[-9:] in end1:
        for i in range(4):
            return torch.cat([a[name] for a in ckpt_values], dim=1)
    return ckpt_values[0][name]
merged = {}
for k in ckpt_values[0].keys():
    merged[k] = merge_params(k)
for k,v in merged.items():
    print(k, v.shape)
model = {}
start = [ 'layers.'+str(i) for i in range(6,15)]
start.extend([ 'layers.'+str(i) +'.' for i in range(6)])
start.append('tok')
for k,v in merged.items():
    if any([k.startswith(st) for st in start]):
        model[k] = v
for k,v in model.items():
    print(k,v.shape)
torch.save(model, './param0.pth')
model = {}
start = [ 'layers.'+str(i) for i in range(15,30)]
for k,v in merged.items():
    if any([k.startswith(st) for st in start]):
        model[k] = v
for k,v in model.items():
    print(k,v.shape)
torch.save(model, './param1.pth')
start = [ 'layers.'+str(i) for i in range(30,45)]
model = {}
for k,v in merged.items():
    if any([k.startswith(st) for st in start]):
        model[k] = v
for k,v in model.items():
    print(k,v.shape)
torch.save(model, './param2.pth')
start = [ 'layers.'+str(i) for i in range(45,60)]
start.append('norm')
start.append('output')
model = {}
for k,v in merged.items():
    if any([k.startswith(st) for st in start]):
        model[k] = v
for k,v in model.items():
    print(k,v.shape)
torch.save(model, './param3.pth')
