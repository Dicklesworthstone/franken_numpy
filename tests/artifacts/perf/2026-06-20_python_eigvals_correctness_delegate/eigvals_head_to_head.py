import numpy as np, fnp_python as f, time
def bench(fn, n=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]
def sortc(z):
    z=np.asarray(z); return np.sort_complex(z)
rng=np.random.default_rng(33)
for d in [16,32,64,96,128,200,300,450,600,800]:
    A=rng.standard_normal((d,d))
    tn=bench(lambda:np.linalg.eigvals(A)); tf=bench(lambda:f.eigvals(A))
    try: ok=np.allclose(sortc(np.linalg.eigvals(A)), sortc(f.eigvals(A)), atol=1e-6)
    except Exception as e: ok=f"ERR{e}"
    r=tf/tn; v='WIN' if r<0.95 else('par' if r<1.15 else 'LOSS')
    print(f"d={d:4} np={tn*1e3:8.2f}ms fnp={tf*1e3:8.2f}ms ratio={r:6.2f} {v:>5} match={ok}")
