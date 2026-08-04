import numpy as np, fnp_python as f, time
def bench(fn, n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]
rng=np.random.default_rng(71)
for B,d in [(4000,8),(2000,16),(1000,32),(500,64),(200,100),(64,200),(10000,4)]:
    A=rng.standard_normal((B,d,d)); As=np.einsum('...ij,...kj->...ik',A,A)+d*np.eye(d)
    tn=bench(lambda:np.linalg.cholesky(As)); tf=bench(lambda:f.cholesky(As))
    L=np.asarray(f.cholesky(As)); ok=np.allclose(L@np.swapaxes(L,-1,-2), As, atol=1e-8)
    print(f"B={B:6} d={d:4} np={tn*1e3:7.2f}ms fnp={tf*1e3:7.2f}ms ratio={tf/tn:5.2f} match={ok}")
