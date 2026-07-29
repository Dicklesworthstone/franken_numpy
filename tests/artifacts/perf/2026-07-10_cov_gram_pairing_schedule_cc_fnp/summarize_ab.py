# Summarize interleave_ab.sh output: per-probe min over rounds for base/cand/np
# plus the per-round gauge spread (a drifting gauge invalidates the run).
import re
import sys
from collections import defaultdict

mins = defaultdict(lambda: defaultdict(lambda: 1e18))
gauges = defaultdict(list)
with open(sys.argv[1]) as fh:
    for line in fh:
        m = re.match(r"(fnp-base|fnp-cand|np) (cov|corrcoef) (\S+): ([0-9.]+)", line)
        if m:
            tag, op, shape, ms = m.group(1), m.group(2), m.group(3), float(m.group(4))
            key = f"{op} {shape}"
            mins[key][tag] = min(mins[key][tag], ms)
        g = re.match(r"(fnp-base|fnp-cand|np) gauge matmul2048: ([0-9.]+)", line)
        if g:
            gauges[g.group(1)].append(float(g.group(2)))

print(f"{'probe':>22} {'base':>8} {'cand':>8} {'self':>6} {'np':>8} {'cand/np':>8}")
for key, row in mins.items():
    b, c, n = row.get("fnp-base"), row.get("fnp-cand"), row.get("np")
    print(
        f"{key:>22} {b:8.2f} {c:8.2f} {b / c:6.2f}x {n:8.2f} {n / c:7.2f}x"
    )
for tag, gs in gauges.items():
    print(f"gauge {tag}: min {min(gs):.1f} max {max(gs):.1f} ms over {len(gs)} rounds")
