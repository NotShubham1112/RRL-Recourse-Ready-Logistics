import json, time, numpy as np
from sklearn.tree import DecisionTreeClassifier
from rrl_sim import *
t00=time.time()
TRAIN=list(range(1000,1040)); VAL=list(range(1100,1115)); TEST=list(range(0,50))
# ---- 1. collect planner decisions (connected RRL) and distil to tree ----
def collect(seeds):
    log=[]
    for s in seeds: run_episode(s,'RRL',tree=None,log=log,no_blackout=True)
    X=np.array([a for a,_ in log]); y=np.array([b for _,b in log]); return X,y
Xtr,ytr=collect(TRAIN); Xva,yva=collect(VAL)
dsweep={}
trees={}
for d in (1,2,3,4,6):
    tr=DecisionTreeClassifier(max_depth=d,random_state=0).fit(Xtr,ytr); trees[d]=tr
    dsweep[d]=dict(train_acc=float(tr.score(Xtr,ytr)),val_agree=float(tr.score(Xva,yva)))
print('tree sweep',dsweep, 'label dist',np.bincount(ytr)/len(ytr),flush=True)
tree=trees[4]
# ---- 2. main evaluation ----
names=['B1','B2','B4','B3','RRL_noOpt','RRL_noCal','RRL_noEdge','RRL']
R={}
for p in names:
    rows=[]
    for s in TEST:
        m=run_episode(s,p,tree=tree if POLICIES[p].get('edge') else None)
        rows.append(m)
    R[p]=rows; print(p,'done',round(time.time()-t00),flush=True)
orc=[run_episode(s,'RRL',tree=tree,no_blackout=True) for s in TEST]
def agg(rows):
    so=np.array([r['so_days'] for r in rows],float); um=np.array([r['unmet'] for r in rows])
    below=np.array([r['below'] for r in rows],float); hold=np.array([HOLD*r['stock_sum'] for r in rows]); fee=np.array([r['fee'] for r in rows])
    mc=np.concatenate([r['mincover'] for r in rows]); sp=[x for r in rows for x in r['spells']]
    tm=np.concatenate([r['times'] for r in rows]) if rows[0]['times'] else np.array([0.])
    cn=sum(r['cov_n'] for r in rows); ch=sum(r['cov_hit'] for r in rows)
    bso=sum(r['blk_so'] for r in rows); bd=sum(r['blk_days'] for r in rows)
    return dict(so_mean=so.mean(),so_p90=float(np.quantile(so,0.9)),unmet=um.mean(),below=below.mean(),hold=hold.mean(),fee=fee.mean(),total=(hold+fee).mean(),
        p10cover=float(np.quantile(mc,0.1)),spell=float(np.mean(sp)) if sp else 0.,t_med=float(np.median(tm)*1000),t_p95=float(np.quantile(tm,0.95)*1000),
        cov=ch/cn if cn else None, blk_so_rate=bso/bd if bd else None, so_arr=so.tolist())
A={p:agg(R[p]) for p in names}; A['RRL_noBlackout']=agg(orc)
# paired bootstrap CI of reduction
g=np.random.default_rng(0)
def boot(a,b):
    a=np.array(a);b=np.array(b); n=len(a); red=[]
    for _ in range(4000):
        i=g.integers(0,n,n); red.append(1-b[i].sum()/a[i].sum())
    return float(1-b.sum()/a.sum()),float(np.quantile(red,0.025)),float(np.quantile(red,0.975))
CI={p:boot(A[p]['so_arr'],A['RRL']['so_arr']) for p in names if p!='RRL'}
print('main done',round(time.time()-t00),flush=True)
# ---- 3. scenario-count sensitivity (RRL, 20 seeds) ----
sens={}
for S in (5,10,20,40,80):
    rows=[run_episode(s,'RRL',tree=tree,S_override=S) for s in TEST[:20]]
    a=agg(rows); sens[S]=dict(so=a['so_mean'],t_med=a['t_med'],total=a['total'])
    print('S',S,sens[S],flush=True)
# ---- 4. scaling benchmark: single LP solve time vs number of bases ----
import rrl_sim as rs
scal={}
for nb in (6,12,24,48,96):
    rs.NB=nb; rs.HUBOF=np.repeat(np.arange(nb//3),3); rs.MU=np.tile([8.,9.,8.],nb//3); rs.SMIN=2*rs.MU
    gg=np.random.default_rng(1); S=20
    dem=rs.MU[None,:,None]*np.exp(0.25*gg.standard_normal((S,nb,rs.H)))
    o=gg.random((S,nb,rs.H))>0.1; o[:,:,0]=True
    # FLEET per hub constraint in plan loops over (0,1) only -> extend by monkeypatch not needed for timing of size; skip fleet for hubs>=2
    ts=[]
    for rep in range(2):
        t0=time.time(); _,_,_,st=rs.plan(0.*rs.MU+5*rs.MU,np.zeros((nb,rs.H+1)),np.ones(nb,bool),rs.MU,dem,o,np.zeros(nb),1.,True,np.full(nb,8.)); ts.append(time.time()-t0)
    scal[nb]=dict(t=float(np.median(ts)),vars=st['vars'],cons=st['cons'],ok=st['ok'])
    print('scal',nb,scal[nb],flush=True)
out=dict(tree=dsweep,A={k:{kk:vv for kk,vv in v.items() if kk!='so_arr'} for k,v in A.items()},CI=CI,sens=sens,scal=scal,
         label_dist=np.bincount(ytr).tolist())
json.dump(out,open('results.json','w'),indent=1); print('ALL DONE',round(time.time()-t00))
