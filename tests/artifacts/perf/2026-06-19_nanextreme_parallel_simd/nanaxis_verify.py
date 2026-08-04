import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(0)
fails=0; cases=0
shapes=[(4096,4096),(4097,4095),(100,100,100),(8,1<<16),(1<<16,8),(3,5,7,11),(1,4096),(4096,1),(2,2,2)]
for shp in shapes:
    for ax in list(range(len(shp)))+[-1,None]:
        for frac in [0.0,0.01,0.5,1.0]:
            cases+=1
            x=rng.standard_normal(shp)
            if frac>0:
                mask=rng.random(shp)<frac; x[mask]=np.nan
            for take,mf,nf in [(1,m.nanmax,np.nanmax),(0,m.nanmin,np.nanmin)]:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try: rn=nf(x,axis=ax)
                    except Exception as e: rn=('ERR',type(e).__name__)
                    try: rf=mf(x,axis=ax)
                    except Exception as e: rf=('ERR',type(e).__name__)
                if isinstance(rn,tuple) or isinstance(rf,tuple):
                    ok = (rn==rf)
                else:
                    rn=np.asarray(rn,dtype=np.float64); rf=np.asarray(rf,dtype=np.float64)
                    ok = rn.shape==rf.shape and (np.array_equal(rn,rf) or np.array_equal(np.nan_to_num(rn,nan=7e99),np.nan_to_num(rf,nan=7e99)))
                if not ok:
                    fails+=1
                    if fails<=12: print(f"FAIL shp={shp} ax={ax} frac={frac} max={take}: np={rn if isinstance(rn,tuple) else rn.ravel()[:3]} fnp={rf if isinstance(rf,tuple) else rf.ravel()[:3]}")
print(f"cases={cases} fails={fails}")
def bench(fn,reps=12):
    for _ in range(4): fn()
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1e6
d=rng.standard_normal((4096,4096))
for nm,mf,nf in [("nanmax_ax0",lambda:m.nanmax(d,axis=0),lambda:np.nanmax(d,axis=0)),
                 ("nanmax_ax1",lambda:m.nanmax(d,axis=1),lambda:np.nanmax(d,axis=1)),
                 ("nanmin_ax0",lambda:m.nanmin(d,axis=0),lambda:np.nanmin(d,axis=0)),
                 ("nanmin_ax1",lambda:m.nanmin(d,axis=1),lambda:np.nanmin(d,axis=1))]:
    tn=bench(nf); tf=bench(mf); print(f"{nm:12s} np={tn:8.1f}us fnp={tf:8.1f}us fnp/np={tf/tn:.3f}")
