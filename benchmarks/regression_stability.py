import numpy as np, warnings
warnings.simplefilter('ignore')
from uws.label_model import ContinuousLabelModel
from uws.synthetic import generate_lfs, generate_synthetic_xy
CFG={'legacy':dict(legacy=True),'emp':dict(solver='empirical'),'struct':{},'emp-nc':dict(solver='empirical',center=False),'struct-nc':dict(center=False),'struct-nc-lstsq':dict(center=False,estimator='lstsq')}
res={}
for seed in range(20):
    rng=np.random.default_rng(seed)
    X,Y,v=generate_synthetic_xy(2000,10,rng)
    L,S=generate_lfs(6,2000,Y,v,seed=seed)
    cases={'clean m=6 n=1000':(L,1000),
      'dup+scale':(np.hstack([L, L[:,:3]+1e-4*rng.standard_normal((2000,3)), 100*L[:,3:4]]),1000),
      'm=20 n=60':(generate_lfs(20,2000,Y,v,seed=seed,diag_fact=8)[0],60),
      'm=6 n=30':(L,30)}
    for name,(LL,ntr) in cases.items():
        for kw,d in CFG.items():
            try: e=ContinuousLabelModel(**d).fit(LL[:ntr],v).score(Y[1000:],LL[1000:])
            except Exception as ex: e=np.nan
            res.setdefault(name,{}).setdefault(kw,[]).append(e)
for name,r in res.items():
    print(name); 
    for kw,vals in r.items(): print('   %-13s median %.4f  p90 %.4f  max %.4f  fail %d'%(kw,np.nanmedian(vals),np.nanpercentile(vals,90),np.nanmax(vals),np.isnan(vals).sum()))
