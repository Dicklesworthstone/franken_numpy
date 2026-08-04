import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(5)
fails=0; cases=0
shapes=[(4096,4096),(4097,4095),(512,1000),(100,100,100),(3,5,7,11),(2,4096)]
for shp in shapes:
    last=len(shp)-1
    for ax in [last,-1]:
        for kind in ['plain','ties','dup']:
            cases+=1
            if kind=='ties': x=rng.integers(0,3,shp).astype(np.float64)  # many ties -> first-index parity
            elif kind=='dup': x=np.zeros(shp); 
            else: x=rng.standard_normal(shp)
            for fm,fn in [(m.argmax,np.argmax),(m.argmin,np.argmin)]:
                rn=fn(x,axis=ax); rf=fm(x,axis=ax)
                ok = np.array_equal(np.asarray(rn),np.asarray(rf)) and np.asarray(rn).shape==np.asarray(rf).shape
                if not ok:
                    fails+=1
                    if fails<=10: print(f"FAIL shp={shp} ax={ax} {kind} {fn.__name__}: np={np.ravel(rn)[:4]} fnp={np.ravel(rf)[:4]}")
print(f"cases={cases} fails={fails}")
def bench(fn,reps=10):
    for _ in range(3): fn()
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e6
d=rng.standard_normal((4096,4096))
di=rng.integers(-10**9,10**9,(4096,4096))
for nm,mf,nf in [("argmax_ax1",lambda:m.argmax(d,axis=1),lambda:np.argmax(d,axis=1)),
                 ("argmin_ax1",lambda:m.argmin(d,axis=1),lambda:np.argmin(d,axis=1)),
                 ("argmax_ax1_i64",lambda:m.argmax(di,axis=1),lambda:np.argmax(di,axis=1))]:
    tn=bench(nf); tf=bench(mf); print(f"{nm:16s} np={tn:9.1f}us fnp={tf:9.1f}us fnp/np={tf/tn:.3f}")
