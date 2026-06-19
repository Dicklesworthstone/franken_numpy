import sys, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(7)
fails=0; cases=0
shapes=[(4096,4096),(4097,4095),(100,100,100),(8,1<<16),(1<<16,8),(3,5,7,11),(1,4096),(4096,1),(2,2),(5,)]
def eq(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    if a.shape!=b.shape: return False
    return np.array_equal(a,b,equal_nan=True)
for shp in shapes:
    for ax in list(range(len(shp)))+[-1,None]:
        for kind in ['plain','signzero','withnan','allnan']:
            cases+=1
            x=rng.standard_normal(shp)
            if kind=='signzero': x[x>1.0]=0.0; x[x<-1.0]=-0.0
            elif kind=='withnan': x[rng.random(shp)<0.1]=np.nan
            elif kind=='allnan': x[:]=np.nan
            try: rn=np.ptp(x,axis=ax); en=None
            except Exception as e: rn=None; en=type(e).__name__
            try: rf=m.ptp(x,axis=ax); ef=None
            except Exception as e: rf=None; ef=type(e).__name__
            ok = (en==ef) if (en or ef) else eq(rn,rf)
            if not ok:
                fails+=1
                if fails<=15: print(f"FAIL shp={shp} ax={ax} {kind}: np={None if rn is None else np.ravel(rn)[:3]}/{en} fnp={None if rf is None else np.ravel(rf)[:3]}/{ef}")
print(f"cases={cases} fails={fails}")
