import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(9)
fails=0; cases=0
for shp in [(1<<22,),(4096,4096),(100,100,100),(7,),(3,5,7)]:
    for kind in ['plain','withnan','allnan']:
        cases+=1
        x=rng.standard_normal(shp)*100
        if kind=='withnan': x[rng.random(shp)<0.1]=np.nan
        elif kind=='allnan': x[:]=np.nan
        for fm,fn in [(m.nanmax,np.nanmax),(m.nanmin,np.nanmin)]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rn=fn(x,keepdims=True); rf=fm(x,keepdims=True)
            rn=np.asarray(rn); rf=np.asarray(rf)
            ok = rn.shape==rf.shape and str(rn.dtype)==str(rf.dtype) and np.array_equal(rn,rf,equal_nan=True)
            if not ok:
                fails+=1
                if fails<=8: print(f"FAIL shp={shp} {kind} {fn.__name__}: np shape={rn.shape}/{rn.dtype}={np.ravel(rn)[:2]} fnp shape={rf.shape}/{rf.dtype}={np.ravel(rf)[:2]}")
print(f"cases={cases} fails={fails}")
def bench(fn,reps=10):
    for _ in range(3): fn()
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e6
x=rng.standard_normal(1<<22)
for nm,mf,nf in [("nanmin_kd",lambda:m.nanmin(x,keepdims=True),lambda:np.nanmin(x,keepdims=True)),
                 ("nanmax_kd",lambda:m.nanmax(x,keepdims=True),lambda:np.nanmax(x,keepdims=True))]:
    tn=bench(nf); tf=bench(mf); print(f"{nm:10s} np={tn:8.1f}us fnp={tf:8.1f}us fnp/np={tf/tn:.3f}")
