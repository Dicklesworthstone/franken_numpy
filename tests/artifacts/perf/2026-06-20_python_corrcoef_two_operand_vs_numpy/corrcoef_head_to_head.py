import numpy as np, fnp_python as f, time
def bench(fn, n=21):
    for _ in range(5): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]
rng=np.random.default_rng(500)
print("=== corrcoef(a,b) two-series PERF (was 5-12x loss) ===")
for nobs in [10000,100000,1000000,4000000]:
    a=rng.standard_normal(nobs); b=rng.standard_normal(nobs)
    tn=bench(lambda:np.corrcoef(a,b)); tf=bench(lambda:f.corrcoef(a,b))
    ok=np.allclose(np.corrcoef(a,b),f.corrcoef(a,b))
    r=tf/tn; v='WIN' if r<0.95 else('par' if r<1.15 else 'LOSS')
    print(f"nobs={nobs:>8} np={tn*1e6:8.0f}us fnp={tf*1e6:8.0f}us ratio={r:5.2f} {v:>5} match={ok}")
print("=== correctness: random + edge ===")
bad=0
for _ in range(40):
    n=int(rng.integers(50,5000)); a=rng.standard_normal(n)*rng.uniform(0.1,1e3)+rng.uniform(-1e4,1e4)
    b=rng.standard_normal(n)*rng.uniform(0.1,1e3)+rng.uniform(-1e4,1e4)
    if not np.allclose(np.corrcoef(a,b),f.corrcoef(a,b),rtol=1e-9,atol=1e-11): bad+=1
print("corrcoef(a,b) random bad:",bad,"/40")
# 2-D m + 1-D y, M+Y
M=rng.standard_normal((3,5000)); Y=rng.standard_normal((2,5000)); b=rng.standard_normal(5000)
print("corrcoef(M,b):", np.allclose(np.corrcoef(M,b),f.corrcoef(M,b)))
print("corrcoef(M,Y):", np.allclose(np.corrcoef(M,Y),f.corrcoef(M,Y)))
print("corrcoef(M) no-y unchanged:", np.allclose(np.corrcoef(M),f.corrcoef(M)))
# regression: cov(a,b) still wins
a=rng.standard_normal(1000000); b=rng.standard_normal(1000000)
print("cov(a,b) still correct:", np.allclose(np.cov(a,b),f.cov(a,b)))
