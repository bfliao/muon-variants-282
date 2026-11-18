"""Control: the reference repo's own experiment (its MLP, batch 1024, 5 epochs, linear-decay lr,
all layers constrained, no momentum, 100 dual steps), run with OUR ManifoldMuon class, on a
50k-image CIFAR-5m subset standing in for CIFAR-10. Diagnostic only (1 seed, prints JSON rows;
nothing written to results/). The real-CIFAR-10 oracle with the unmodified vendored code is
scripts/oracle_tml.py (Phase 1). See docs/manifold_audit.md."""
import json, sys, numpy as np, torch, torch.nn as nn
from muonarch.manifold import ManifoldMuon
from muonarch.optim import MuonFamily
from muonarch.data import normalize
from pathlib import Path
P=str(Path(__file__).resolve().parent.parent/"data/cifar5m")+"/"
dev=torch.device("mps")
xtr=torch.from_numpy(np.array(np.load(P+"train.npy",mmap_mode="r")[:50000])).to(dev)
ytr=torch.from_numpy(np.array(np.load(P+"train_labels.npy",mmap_mode="r")[:50000])).to(dev)
xte=torch.from_numpy(np.load(P+"test.npy")).to(dev); yte=torch.from_numpy(np.load(P+"test_labels.npy")).to(dev)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(3072,128,bias=False); s.fc2=nn.Linear(128,64,bias=False); s.fc3=nn.Linear(64,10,bias=False)
    def forward(s,x): return s.fc3(torch.relu(s.fc2(torch.relu(s.fc1(x)))))
def acc(m,x,y):
    with torch.no_grad():
        c=0
        for i in range(0,len(x),1024): c+=(m(normalize(x[i:i+1024])).argmax(1)==y[i:i+1024]).sum().item()
    return 100*c/len(x)
def run(kind,lr,seed=42,epochs=5,**kw):
    torch.manual_seed(seed); m=MLP().to(dev); ps=list(m.parameters())
    if kind=="manifold":
        opt=ManifoldMuon(ps,lr=lr,dual_cap=kw.get("cap",100),momentum=kw.get("momentum",0.0)); ManifoldMuon.project_(ps)
    elif kind=="muon": opt=MuonFamily(ps,lr=lr,wd=0.0,momentum=0.95,rule="keller")
    else: opt=torch.optim.AdamW(ps,lr=lr,weight_decay=kw.get("wd",0.0))
    nb=(50000+1023)//1024; steps=epochs*nb; step=0; g=torch.Generator().manual_seed(seed); last=0
    for ep in range(epochs):
        perm=torch.randperm(50000,generator=g).to(dev); tot=0
        for b in range(nb):
            idx=perm[b*1024:(b+1)*1024]
            loss=nn.functional.cross_entropy(m(normalize(xtr[idx])),ytr[idx])
            opt.zero_grad(); loss.backward()
            for pg in opt.param_groups: pg["lr"]=lr*(1-step/steps)
            opt.step(); step+=1; tot+=loss.item()
        last=tot/nb
    row={"kind":kind,"lr":lr,**kw,"seed":seed,"final_epoch_loss":round(last,4),"test_acc":acc(m,xte,yte),"train_acc":round(acc(m,xtr,ytr),2)}
    print(json.dumps(row),flush=True)
for lr in (0.03,0.1,0.3): run("manifold",lr)
run("manifold",0.1,cap=5)
run("manifold",0.1,momentum=0.95)
for lr in (3e-4,1e-3,3e-3,1e-2): run("adamw",lr)
for lr in (1e-3,3e-3): run("adamw",lr,wd=0.01)
for lr in (0.01,0.03,0.1): run("muon",lr)
print("REF DONE")
