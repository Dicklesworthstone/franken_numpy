import numpy as np, fnp_python as f, time
def bench(fn, n=11):
    for _ in range(4): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]
rng=np.random.default_rng(1000)
# EXHAUSTIVE correctness across modes/sizes/kernels (incl swap, boundaries)
fails=0; tot=0
for mode in ['full','same','valid']:
    for La in [1,2,3,5,8,64,100,257,1000]:
        for Lv in [1,2,3,5,8,16,32,64,100]:
            a=rng.standard_normal(La); v=rng.standard_normal(Lv); tot+=1
            try: e=np.convolve(a,v,mode); g=np.asarray(f.convolve(a,v,mode))
            except Exception as ex: e=None
            if e is not None:
                if e.shape!=g.shape or not np.allclose(e,g,atol=1e-9): fails+=1; print('CONV',mode,La,Lv,e.shape,g.shape)
            # correlate
            try: ce=np.correlate(a,v,mode)
            except Exception: ce=None
            if ce is not None:
                cg=np.asarray(f.correlate(a,v,mode))
                if ce.shape!=cg.shape or not np.allclose(ce,cg,atol=1e-9): fails+=1; print('CORR',mode,La,Lv,ce.shape,cg.shape)
print(f"EXHAUSTIVE conv/corr: {fails} fails / {tot} sizes x (conv+corr) x modes")
print("=== PERF convolve 'same' short kernel ===")
for N in [10000,100000,1000000,2000000]:
    a=rng.standard_normal(N)
    for ks in [3,5,16,32]:
        k=rng.standard_normal(ks)
        tn=bench(lambda a=a,k=k:np.convolve(a,k,'same')); tf=bench(lambda a=a,k=k:f.convolve(a,k,'same'))
        r=tf/tn; v='WIN' if r<0.9 else('par' if r<1.15 else 'LOSS')
        print(f"N={N:>8} k={ks:3} np={tn*1e6:8.0f}us fnp={tf*1e6:8.0f}us ratio={r:6.2f} {v}")
