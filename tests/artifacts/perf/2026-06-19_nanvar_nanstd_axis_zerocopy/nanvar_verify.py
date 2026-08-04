import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(3)
def eq(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    if a.shape!=b.shape: return False
    return np.array_equal(a,b,equal_nan=True)  # BIT-EXACT
fails=0; cases=0; bitfail=0
shapes=[(4096,4096),(4097,4095),(512,1000),(100,100,100),(3,5,7,11),(1,4096),(7,)]
for shp in shapes:
    last=len(shp)-1
    for ax in [last,-1]:  # last-axis fast path
        for ddof in [0,1,2]:
            for kind in ['plain','withnan']:
                cases+=1
                x=rng.standard_normal(shp)*rng.uniform(0.1,1e6)
                if kind=='withnan': x[rng.random(shp)<0.15]=np.nan
                for fn_m,fn_n in [(m.nanstd,np.nanstd),(m.nanvar,np.nanvar)]:
                    try: rn=fn_n(x,axis=ax,ddof=ddof); en=None
                    except Exception as e: rn=None; en=type(e).__name__
                    try: rf=fn_m(x,axis=ax,ddof=ddof); ef=None
                    except Exception as e: rf=None; ef=type(e).__name__
                    if en or ef:
                        ok=(en==ef)
                    else:
                        ok=eq(rn,rf)
                    if not ok:
                        fails+=1
                        if fails<=12: print(f"FAIL shp={shp} ax={ax} ddof={ddof} {kind} {fn_n.__name__}: np={None if rn is None else np.ravel(rn)[:2]}/{en} fnp={None if rf is None else np.ravel(rf)[:2]}/{ef}")
print(f"cases={cases} fails={fails}")
def bench(fn,reps=8):
    for _ in range(3): fn()
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e6
d=rng.standard_normal((4096,4096)); dn=d.copy(); dn[::7,::7]=np.nan
for nm,mf,nf in [("nanstd_ax1",lambda:m.nanstd(dn,axis=1),lambda:np.nanstd(dn,axis=1)),
                 ("nanvar_ax1",lambda:m.nanvar(dn,axis=1),lambda:np.nanvar(dn,axis=1)),
                 ("nanstd_ax1_nonan",lambda:m.nanstd(d,axis=1),lambda:np.nanstd(d,axis=1)),
                 ("nanstd_ax0",lambda:m.nanstd(dn,axis=0),lambda:np.nanstd(dn,axis=0))]:
    tn=bench(nf); tf=bench(mf); print(f"{nm:18s} np={tn:9.1f}us fnp={tf:9.1f}us fnp/np={tf/tn:.3f}")
