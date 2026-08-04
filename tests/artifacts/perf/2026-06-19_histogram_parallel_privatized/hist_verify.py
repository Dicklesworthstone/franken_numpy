import sys, time, warnings
sys.path.insert(0,'/data/projects/franken_numpy/.probe'); warnings.filterwarnings('ignore')
import numpy as np, fnp_python as m
rng=np.random.default_rng(0)
# Correctness across dtypes, bin counts, distributions
fails=0; cases=0
for dt in [np.float64, np.float32, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
    for nb in [1,2,10,50,100,257]:
        for dist in ['normal','uniform','dup','small']:
            cases+=1
            if dist=='normal': base=rng.standard_normal(1<<18)*50
            elif dist=='uniform': base=rng.uniform(-100,100,1<<18)
            elif dist=='dup': base=rng.integers(-5,5,1<<18).astype(float)
            else: base=rng.standard_normal(100)*3
            if np.issubdtype(dt,np.integer):
                info=np.iinfo(dt); base=np.clip(base, info.min/4, info.max/4)
            x=base.astype(dt)
            try:
                cn,ce=m.histogram(x,bins=nb); nn,ne=np.histogram(x,bins=nb)
                if not np.array_equal(cn,nn) or not np.allclose(ce,ne,rtol=0,atol=0,equal_nan=True):
                    if not (np.array_equal(cn,nn) and np.array_equal(ce,ne)):
                        fails+=1
                        if fails<=8: print(f"FAIL dt={dt.__name__} nb={nb} {dist}: counts_eq={np.array_equal(cn,nn)} edges_eq={np.array_equal(ce,ne)} sum_fnp={cn.sum()} sum_np={nn.sum()}")
            except Exception as ex:
                fails+=1
                if fails<=8: print(f"ERR dt={dt.__name__} nb={nb} {dist}: {type(ex).__name__} {str(ex)[:50]}")
print(f"cases={cases} fails={fails}")
# Perf at 4M
def bench(fn,reps=8):
    for _ in range(3): fn()
    b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e6
x=rng.standard_normal(1<<22)
for nb in [10,50,100,256,1000]:
    tn=bench(lambda nb=nb:np.histogram(x,bins=nb)); tf=bench(lambda nb=nb:m.histogram(x,bins=nb))
    print(f"hist bins={nb:5d}: np={tn:9.1f}us fnp={tf:9.1f}us fnp/np={tf/tn:.3f}")
