import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(11)
def eq(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return a.shape==b.shape and np.array_equal(a,b,equal_nan=True)
fails=0; cases=0
shapes=[(4096,4096),(512,1000),(100,100,100),(3,5,7,11),(1,4096),(7,)]
for shp in shapes:
    last=len(shp)-1
    for ax in [last,-1]:
        for ddof in [0,1]:
            for kd in [True,False]:
                for kind in ['plain','withnan']:
                    cases+=1
                    x=rng.standard_normal(shp)*rng.uniform(0.1,1e5)
                    if kind=='withnan': x[rng.random(shp)<0.12]=np.nan
                    for fm,fn in [(m.nanstd,np.nanstd),(m.nanvar,np.nanvar)]:
                        try: rn=fn(x,axis=ax,ddof=ddof,keepdims=kd); en=None
                        except Exception as e: rn=None; en=type(e).__name__
                        try: rf=fm(x,axis=ax,ddof=ddof,keepdims=kd); ef=None
                        except Exception as e: rf=None; ef=type(e).__name__
                        ok=(en==ef) if (en or ef) else eq(rn,rf)
                        if not ok:
                            fails+=1
                            if fails<=10: print(f"FAIL shp={shp} ax={ax} ddof={ddof} kd={kd} {kind} {fn.__name__}: shapes np={None if rn is None else np.shape(rn)} fnp={None if rf is None else np.shape(rf)}")
print(f"cases={cases} fails={fails}")
def bench(fn,reps=8):
    for _ in range(3): fn()
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e6
d=rng.standard_normal((4096,4096))
for nm,mf,nf in [("nanstd_kd_ax1",lambda:m.nanstd(d,axis=1,keepdims=True),lambda:np.nanstd(d,axis=1,keepdims=True)),
                 ("nanvar_kd_ax1",lambda:m.nanvar(d,axis=1,keepdims=True),lambda:np.nanvar(d,axis=1,keepdims=True))]:
    tn=bench(nf); tf=bench(mf); print(f"{nm:16s} np={tn:9.1f}us fnp={tf:9.1f}us fnp/np={tf/tn:.3f}")
