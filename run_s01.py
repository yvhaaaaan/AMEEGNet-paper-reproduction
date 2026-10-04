import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ameegnet_ours import AMEEGNet, load_subject_npz, session_standardize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--no-bn-first", action="store_true",
                   help="remove the BatchNorm immediately after temporal convolutions")
    p.add_argument("--elu-before-bn", action="store_true",
                   help="use ELU before BN in depthwise and separable blocks")
    p.add_argument("--paper-pooling", action="store_true",
                   help="retain standard EEGNet average pooling and dropout")
    p.add_argument("--no-fusion", action="store_true",
                   help="disable the two paper-described fusion transmissions")
    p.add_argument("--no-eca", action="store_true",
                   help="disable ECA before the classification block")
    p.add_argument("--out", default="results/s01.json")
    a = p.parse_args()
    out = Path(a.out)
    artifacts = [out, out.with_suffix('.pt'), out.with_suffix('.npz')]
    if any(path.exists() for path in artifacts):
        raise FileExistsError(f'Refusing to overwrite experiment artifacts: {out}')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    xtr, ytr, xte, yte = load_subject_npz(a.data)
    if not a.strict:
        xtr, xte = session_standardize(xtr, xte)
    if a.strict:
        fit, val = np.arange(len(ytr)), None
    else:
        fit, val = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                    stratify=ytr, random_state=a.seed)
    use_pool = a.paper_pooling or not a.strict
    model = AMEEGNet(pool=use_pool, dropout=0.25 if a.paper_pooling else (0.0 if a.strict else 0.25),
                     bn_first=not a.no_bn_first,
                     norm_then_activation=not a.elu_before_bn,
                     fusion=not a.no_fusion, eca=not a.no_eca).to(device)
    opt = (torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
           if a.strict else torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1 if not a.strict else 0.0)
    gen = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr[fit]), torch.from_numpy(ytr[fit])),
                        batch_size=64, shuffle=True, generator=gen)
    best = (-1.0, None, 0); history=[]; t0=time.perf_counter()
    for ep in range(1, a.epochs + 1):
        model.train(); total=0.0; preds=[]; ys=[]
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); opt.zero_grad(set_to_none=True)
            z=model(xb); l=loss_fn(z,yb)
            if not torch.isfinite(l):
                raise FloatingPointError(f'Non-finite loss at epoch {ep}')
            l.backward()
            if not a.strict:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            total += l.item()*len(yb); preds.extend(z.argmax(1).detach().cpu().numpy()); ys.extend(yb.cpu().numpy())
        model.eval()
        with torch.no_grad():
            zt=model(torch.from_numpy(xte).to(device))
        te=float(accuracy_score(yte,zt.argmax(1).cpu().numpy()))
        va=None
        if val is not None:
            with torch.no_grad():
                zv=model(torch.from_numpy(xtr[val]).to(device))
            va=float(accuracy_score(ytr[val],zv.argmax(1).cpu().numpy()))
        history.append({"epoch":ep,"train_loss":total/len(fit),"train_acc":float(accuracy_score(ys,preds)),"val_acc":va,"test_acc":te})
        if va is not None and va > best[0]:
            best=(va,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep)
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad(): pred=model(torch.from_numpy(xte).to(device)).argmax(1).cpu().numpy()
    result={"strict":a.strict,"paper_pooling":a.paper_pooling,"fusion":not a.no_fusion,"eca":not a.no_eca,"bn_first":not a.no_bn_first,"norm_then_activation":not a.elu_before_bn,"seed":a.seed,"epochs":a.epochs,"device":str(device),"final_test_acc":float(accuracy_score(yte,pred)),"best_val_acc":None if a.strict else best[0],"best_epoch":None if a.strict else best[2],"seconds":time.perf_counter()-t0,"history":history}
    result['training_protocol'] = {'optimizer':type(opt).__name__, 'lr':1e-3,
        'weight_decay':opt.defaults['weight_decay'], 'betas':opt.defaults['betas'],
        'eps':opt.defaults['eps'], 'gradient_clip_max_norm':None if a.strict else 5.0,
        'batch_size':64, 'train_samples':len(fit), 'test_samples':len(yte)}
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(), 'optimizer':opt.state_dict(), 'args':vars(a)}, out.with_suffix('.pt'))
    np.savez(out.with_suffix('.npz'), y_true=yte, y_pred=pred)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "history"},indent=2))

if __name__ == "__main__": main()
