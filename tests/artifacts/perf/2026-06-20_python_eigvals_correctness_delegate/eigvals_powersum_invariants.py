import numpy as np, fnp_python as f
rng=np.random.default_rng(2024)
tot=0; bad=0; worst=0
worstcase=None
for d in [16,32,64,128]:
    Apow_traces=None
    for s in range(30):
        A=rng.standard_normal((d,d))
        ev=np.asarray(f.eigvals(A)).astype(complex)
        A2=A@A; A3=A2@A
        t1,t2,t3=np.trace(A),np.trace(A2),np.trace(A3)
        s1=ev.sum(); s2=(ev**2).sum(); s3=(ev**3).sum()
        # relative errors
        e1=abs(s1-t1)/(abs(t1)+1); e2=abs(s2-t2)/(abs(t2)+1); e3=abs(s3-t3)/(abs(t3)+1)
        e=max(e1,e2,e3); tot+=1
        if e>1e-6: bad+=1
        if e>worst: worst=e; worstcase=(d,s,float(e1),float(e2),float(e3))
print(f"power-sum invariants k=1,2,3: bad={bad}/{tot} worst_relerr={worst:.2e}")
print("worstcase (d,seed,e1,e2,e3):", worstcase)
