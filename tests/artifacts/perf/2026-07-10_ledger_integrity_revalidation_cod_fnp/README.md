# Ledger-integrity revalidation (cod_fnp)

Date: 2026-07-10. Worker: `vmi1149989`. Profile: `release-perf` with frame pointers.
Environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=16`.

This is the single-binary, alternating A/B and profile retry of three historical performance
REJECT rows: f64 median radix-select, float16 sort via f32 widening, and tie-heavy f32 argsort.
Each Criterion `iter_custom` alternates AB/BA order and records the two arms separately. ORIG is
a bench-only `#[inline(never)]` NumPy reference function. Candidate and ORIG therefore ran in one
binary, one process, one RCH invocation, and on one worker.

The required Cargo prefix was:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench ...
```

The target runner wrapped the binary with `perf record -F 199 -e cycles:u`; it then printed
`perf report --stdio --no-children --percent-limit 0.1 --sort overhead,symbol --call-graph none`.
Raw remote profile: `/tmp/fnp_ledger_integrity_cod_20260710.data`, 10,343,704 bytes,
SHA-256 `da92e7e10e44bd3648e537685bfc20e5830d70b55a0b0b368953c0c2067859fb`.
Capture: 77,028 samples, zero lost.

An earlier attempt selected `hz2`, where `perf` was absent; Cargo never launched the benchmark,
so it produced no measurement and is not an A/B attempt. A read-only capability probe confirmed
`perf` on the worker used here.

## Alternating A/B result

Correctness was asserted before timing: exact f64 median bits, and `np.array_equal` for f16 sort
and f32 argsort. CV is the sample standard deviation divided by mean over the final ten normalized
`iter_custom` observations.

| Row | Candidate mean | Candidate CV | ORIG mean | ORIG CV | ORIG/candidate |
|---|---:|---:|---:|---:|---:|
| radix median, f64 normal 16M | 366.795 ms | 41.080% | 200.607 ms | 16.243% | 0.5469x |
| f16 sort via f32 widening, 4M | 37.764 ms | 19.067% | 260.331 ms | 5.090% | 6.8935x |
| f32 argsort, rounded ties 2M | 64.117 ms | 21.242% | 41.628 ms | 10.633% | 0.6492x |

All arms fail the strict `<5%` CV trust gate. The attempt is therefore REJECTED as a keep/reject
measurement. It is still decisive integrity evidence:

- Radix-select really executed, but its fresh loss direction is too noisy to renew the old closed
  claim.
- The f16 route really executed and reversed the historical result from 0.75x to a noisy 6.89x;
  the old `do not retry` row is invalid and reopened.
- Today's real f32 dispatch executed `argsort_sample_has_tie::<f32>` and lost, but the historical
  explanation is stale: NumPy uses index-introsort here, and current FNP performs the sampled tie
  sort twice before delegation. The old “irreducible NumPy SIMD wall” row is invalid and reopened.

## Exact execution markers

The all-symbol report (`--percent-limit 0`, explicit sample field) gives non-zero self samples:

| Samples | Self | Target frame |
|---:|---:|---|
| 1,245 | 0.89% | radix-select parallel byte filter closure |
| 413 | 0.22% | radix-select parallel byte histogram/reduce closure |
| 155 | 0.07% | radix-median f64-to-sortable-key materialization closure |
| 102 | 0.05% | radix-median NaN scan closure |
| 62 | 0.05% | f16 candidate uint16 defer pre-scan closure |
| 17 | 0.02% | f32 sampled-tie quicksort body |
| 10 | 0.01% | `fnp_python::argsort_sample_has_tie::<f32>` (two symbol rows) |
| 13 | <0.01% | `criterion_python_surface::ledger_radix_select_key` body |

## Complete ranked self-time table (all frames >= 0.10%)

The monolithic Criterion target traverses non-selected setup after the requested group. That
pollutes the global ranking with Python setup and other parity calls, but it does not enter the
paired timers. All 167 qualifying frames are retained below in rank order. Very long generic
symbols are capped at 180 characters; no frame is omitted.

```text
rank|self_pct|frame
1|4.70|PyDict_GetItem
2|3.47|PyContextVar_Get
3|2.39|0x000000000003489f
4|1.97|__tls_get_addr
5|1.81|random_standard_normal
6|1.68|0x000000000019869d
7|1.37|rayon bridge: Criterion KDE f64 map/collect
8|1.33|0x0000000000034e83
9|1.19|0x000000000003aada
10|1.18|unresolved kernel 0xffffffffb6400ef0
11|1.05|0x00000000000360f2
12|0.97|rayon bridge: fnp_python::radix_perm_from_keys closure#5
13|0.97|0x00000000007a8784
14|0.94|0x000000000003aa1a
15|0.93|0x00000000001c0e4d
16|0.90|0x00000000001c0d6d
17|0.89|rayon bridge: ledger_radix_select_key byte-filter closure#3
18|0.84|dsyrk_
19|0.83|PyLong_AsSsize_t
20|0.80|0x000000000009c3db
21|0.74|0x00000000001c062c
22|0.72|rayon par_sort recurse: try_native_unique_rows_lexsort_int closure#2
23|0.68|exp
24|0.68|0x00000000001c06dc
25|0.67|0x000000000003aad3
26|0.59|0x000000000003ac7d
27|0.54|0x000000000003a9c7
28|0.53|0x00000000001c0c8d
29|0.51|rayon bridge: try_zerocopy_c128_searchsorted closure#1
30|0.50|0x000000000079b9dc
31|0.49|0x00000000001c146f
32|0.48|0x00000000001c0bbd
33|0.45|0x00000000001c161e
34|0.44|0x00000000007a87bd
35|0.43|0x000000000003a9b4
36|0.43|0x0000000000034897
37|0.40|0x0000000000034890
38|0.38|0x00000000007a8725
39|0.35|__tls_get_addr@plt
40|0.35|0x00000000000348bf
41|0.33|0x0000000000034894
42|0.33|0x000000000009c43a
43|0.30|0x00000000001c113e
44|0.29|random_bounded_uint64_fill
45|0.28|0x000000000009c435
46|0.28|0x00000000001c12d5
47|0.27|0x00000000001c553c
48|0.27|0x00000000007a875a
49|0.27|0x000000000003aac1
50|0.27|0x00000000000348a4
51|0.25|0x0000000000777dd0
52|0.25|0x000000000079b9f3
53|0.25|0x00000000001617b0
54|0.25|0x000000000003a8a6
55|0.24|0x000000000079ba2c
56|0.24|0x00000000001a45e4
57|0.23|0x000000000016179c
58|0.23|0x000000000019dde7
59|0.22|0x0000000000777bb9
60|0.22|0x0000000000161e9f
61|0.22|0x000000000003aaf4
62|0.22|rayon bridge: ledger_radix_select_key histogram/reduce closures#0..2
63|0.21|0x0000000000161dd4
64|0.21|rayon bridge: fnp native i64 output fill
65|0.21|0x0000000000198b85
66|0.21|0x000000000079b9e9
67|0.21|0x00000000001617b4
68|0.20|0x00000000001618c0
69|0.20|rayon bridge: fnp native i64 output fill
70|0.20|random_standard_normal_fill
71|0.20|0x0000000000198b66
72|0.20|0x0000000000776924
73|0.19|0x0000000000008ac4
74|0.19|0x00000000001bc4b8
75|0.19|0x00000000007766ba
76|0.19|0x000000000079ba39
77|0.19|unresolved kernel 0xffffffffb6400b90
78|0.19|0x000000000009c14c
79|0.19|0x0000000000775d25
80|0.19|0x000000000079ba24
81|0.19|0x00000000001bc48c
82|0.18|0x000000000009c25b
83|0.18|0x0000000000198a85
84|0.18|rayon bridge: fnp native i64 output fill
85|0.18|rayon bridge: fnp native i64 output fill
86|0.18|0x000000000078aebc
87|0.18|0x0000000000161e8a
88|0.17|0x000000000019e05c
89|0.17|0x00000000001b0ce7
90|0.17|0x000000000003aa14
91|0.17|0x0000000000161dc8
92|0.16|0x00000000001bc3d8
93|0.16|0x0000000000034a4b
94|0.16|0x0000000000008ac0
95|0.16|0x000000000009c179
96|0.16|0x000000000003aa4c
97|0.16|0x0000000000775f22
98|0.16|0x0000000000161794
99|0.16|rayon par_sort recurse: u64 natural order
100|0.15|0x000000000003aca7
101|0.15|hashbrown::HashMap<&[u8], (), FastIntBuildHasher>::insert
102|0.15|0x00000000001bc3dd
103|0.15|0x00000000001c56ec
104|0.15|0x00000000001bec01
105|0.15|0x000000000003aacb
106|0.15|0x000000000009c15b
107|0.15|0x00000000001c56c0
108|0.15|0x0000000000198699
109|0.15|0x00000000001bc3ac
110|0.15|0x0000000000161e7b
111|0.14|0x00000000001a45a4
112|0.14|0x00000000001ba57b
113|0.14|random_interval
114|0.14|rayon par_sort recurse: (u64,u32) natural order
115|0.14|0x000000000009c163
116|0.14|0x000000000009c43d
117|0.14|rayon par_sort recurse: try_native_unique_rows_lexsort_int_full closure#2
118|0.14|0x000000000009c164
119|0.14|0x00000000001a45c0
120|0.14|0x0000000000161e7e
121|0.14|0x000000000003aa1f
122|0.13|0x00000000001bec95
123|0.13|0x00000000001a45ee
124|0.13|0x00000000001beca1
125|0.13|0x00000000001bc4c7
126|0.13|fnp_python::cov_gram_from_centered closure#2
127|0.13|0x0000000000198a70
128|0.13|PyContextVar_Get@plt
129|0.13|0x00000000001983a4
130|0.13|0x000000000079b9d4
131|0.13|0x00000000001ba585
132|0.13|0x00000000001bc3e7
133|0.13|0x000000000079ba48
134|0.13|0x00000000001bebf5
135|0.13|0x000000000003aa0a
136|0.13|0x000000000009c144
137|0.12|rayon bridge: fnp u64 chunk map/collect
138|0.12|0x00000000001a4623
139|0.12|0x00000000001a45b2
140|0.12|0x000000000019874b
141|0.12|0x0000000000161e82
142|0.12|0x00000000007a87c3
143|0.12|0x0000000000008b20
144|0.12|0x0000000000008b18
145|0.12|0x00000000001ba4b3
146|0.12|0x00000000001a3c25
147|0.12|rayon bridge: fnp native u8 output fill
148|0.12|0x000000000009c251
149|0.11|0x000000000003a92d
150|0.11|rayon par_sort recurse: try_native_unique_rows_lexsort_f32_full closure#3
151|0.11|fnp_python::try_native_c128_intersect_setdiff_dense_integral closure#2
152|0.11|0x00000000001bc4bd
153|0.11|rayon par_sort recurse: try_native_unique_struct_valuelex closure#2
154|0.11|0x0000000000008b24
155|0.11|rayon par_sort recurse: searchsorted_int_merge_typed<i64> closure
156|0.11|0x000000000009c14f
157|0.11|0x000000000019de12
158|0.11|0x000000000009c17c
159|0.11|0x00000000001ba8b3
160|0.10|0x000000000003a874
161|0.10|rayon bridge: fnp native u8 output fill
162|0.10|0x00000000001b0ccc
163|0.10|rayon par_sort recurse: try_native_unique_rows_lexsort_f32 closure#3
164|0.10|0x00000000001a45c3
165|0.10|0x00000000001b0d10
166|0.10|0x0000000000161e8e
167|0.10|rayon par_sort recurse: try_native_lexsort_valuelex closure#4
```

The global top frame is monolithic setup (`PyDict_GetItem`), not a candidate mechanism. No
production optimization is selected from this polluted profile. The targeted non-zero frames
above are used only to validate route execution; each reopened family needs a focused profile
whose target is not diluted by unrelated setup before any new lever can pass a keep/reject gate.
