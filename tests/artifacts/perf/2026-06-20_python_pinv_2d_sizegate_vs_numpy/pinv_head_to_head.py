import numpy as np, fnp_python as f, time, platform
def bench(fn, n=25):
    for _ in range(3): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]
print("host", platform.node(), "numpy", np.__version__)
rng=np.random.default_rng(20)
print(f"{'workload':18} {'NumPy_us':>10} {'FNP_us':>10} {'FNP/NP':>8} {'verdict':>8} match")
for label,sh in [('rect_600x400',(600,400)),('rect_400x600',(400,600)),('sq_128',(128,128)),
                 ('sq_64',(64,64)),('sq_32_native',(32,32)),('sq_8_native',(8,8)),
                 ('tall_5x3_native',(5,3)),('batched_256x8x8',(256,8,8))]:
    M=rng.standard_normal(sh)
    tn=bench(lambda:np.linalg.pinv(M)); tf=bench(lambda:f.pinv(M))
    ok=np.allclose(np.linalg.pinv(M),f.pinv(M),rtol=1e-9,atol=1e-10)
    r=tf/tn; v='WIN' if r<0.97 else('par' if r<1.15 else 'LOSS')
    print(f"{label:18} {tn*1e6:10.1f} {tf*1e6:10.1f} {r:8.3f} {v:>8} {ok}")
