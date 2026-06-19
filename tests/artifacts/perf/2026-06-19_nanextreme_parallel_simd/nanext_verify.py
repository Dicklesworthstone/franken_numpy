import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(0)
fails=0; cases=0
for n in [7, 8, 100, 1<<15, (1<<18)+3, 1<<22]:
    for frac in [0.0, 0.001, 0.5, 0.999]:
        for scale in [1.0, 1e300, 1e-300]:
            cases+=1
            x=(rng.standard_normal(n)*scale)
            if frac>0:
                mask=rng.random(n)<frac; x[mask]=np.nan
            for take,mf,nf in [(1,m.nanmax,np.nanmax),(0,m.nanmin,np.nanmin)]:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try: rn=nf(x)
                    except Exception: rn='ERR'
                    try: rf=mf(x)
                    except Exception: rf='ERR'
                ok=False
                if rn=='ERR' and rf=='ERR': ok=True
                elif rn=='ERR' or rf=='ERR': ok=False
                else:
                    rn=np.float64(rn); rf=np.float64(rf)
                    ok = (rn.tobytes()==rf.tobytes()) or (np.isnan(rn) and np.isnan(rf))
                if not ok:
                    fails+=1
                    if fails<=10: print(f"FAIL n={n} frac={frac} scale={scale} max={take}: np={rn} fnp={rf}")
print(f"cases={cases} fails={fails}")
# perf
def bench(fn,reps=15):
    for _ in range(5): fn()
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)*1e6
x=rng.standard_normal(1<<22); xn=x.copy(); xn[::100]=np.nan
for nm,mf,nf in [("nanmax_nonan",lambda:m.nanmax(x),lambda:np.nanmax(x)),
                 ("nanmin_nonan",lambda:m.nanmin(x),lambda:np.nanmin(x)),
                 ("nanmax_nan",lambda:m.nanmax(xn),lambda:np.nanmax(xn)),
                 ("nanmin_nan",lambda:m.nanmin(xn),lambda:np.nanmin(xn))]:
    tn=bench(nf); tf=bench(mf); print(f"{nm:14s} np={tn:8.1f}us fnp={tf:8.1f}us fnp/np={tf/tn:.3f}")
