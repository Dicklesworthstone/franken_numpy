import numpy as np, fnp_python as f, time, platform
def bench(fn, n=21):
    for _ in range(5): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]
print("host",platform.node(),"numpy",np.__version__)
rng=np.random.default_rng(321)
print("=== np.cov(a,b) two-series head-to-head (BEFORE: 4-17x LOSS) ===")
print(f"{'nobs':>9} {'NumPy_us':>10} {'FNP_us':>10} {'FNP/NP':>8} verdict match")
for nobs in [10000,100000,1000000,4000000]:
    a=rng.standard_normal(nobs); b=rng.standard_normal(nobs)
    tn=bench(lambda:np.cov(a,b)); tf=bench(lambda:f.cov(a,b))
    ok=np.allclose(np.cov(a,b),f.cov(a,b))
    r=tf/tn; v='WIN' if r<0.95 else('par' if r<1.15 else 'LOSS')
    print(f"{nobs:>9} {tn*1e6:10.1f} {tf*1e6:10.1f} {r:8.3f} {v:>7} {ok}")
print("=== robust correctness (allclose vs numpy, 40 random cases) ===")
bad=0
for _ in range(40):
    n=rng.integers(50,5000); a=rng.standard_normal(n)*rng.uniform(0.1,1e3)+rng.uniform(-1e4,1e4)
    b=rng.standard_normal(n)*rng.uniform(0.1,1e3)+rng.uniform(-1e4,1e4)
    for dd in [None,0,1,2]:
        kw={} if dd is None else {'ddof':dd}
        if not np.allclose(np.cov(a,b,**kw),f.cov(a,b,**kw),rtol=1e-10,atol=1e-12): bad+=1
print("two-operand cov bad:",bad,"/160 (allclose rtol 1e-10)")
