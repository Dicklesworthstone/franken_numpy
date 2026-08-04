import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(2)
fails=0; cases=0
for shp in [(1<<22,),(2048,2048),(64,64,64),(7,),(3,5,7)]:
    for kind in ['plain','withnan','allnan']:
        cases+=1
        x=rng.standard_normal(shp)*10
        if kind=='withnan': x[rng.random(shp)<0.1]=np.nan
        elif kind=='allnan': x[:]=np.nan
        for nm in ['nansum','nanmean','nanprod']:
            fm=getattr(m,nm); fn=getattr(np,nm)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rn=np.asarray(fn(x,keepdims=True)); rf=np.asarray(fm(x,keepdims=True))
            ok = rn.shape==rf.shape and str(rn.dtype)==str(rf.dtype) and np.allclose(rn,rf,rtol=1e-12,atol=0,equal_nan=True)
            if not ok:
                fails+=1
                if fails<=10: print(f"FAIL shp={shp} {kind} {nm}: np {rn.shape}/{rn.dtype}={np.ravel(rn)[:2]} fnp {rf.shape}/{rf.dtype}={np.ravel(rf)[:2]}")
print(f"cases={cases} fails={fails}")
def bench(fn,reps=10):
    for _ in range(3): fn()
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e6
x=rng.standard_normal(1<<22)
for nm in ['nansum','nanmean','nanprod']:
    fm=getattr(m,nm); fn=getattr(np,nm)
    tn=bench(lambda:fn(x,keepdims=True)); tf=bench(lambda:fm(x,keepdims=True))
    print(f"{nm}_kd  np={tn:8.1f}us fnp={tf:8.1f}us fnp/np={tf/tn:.3f}")
