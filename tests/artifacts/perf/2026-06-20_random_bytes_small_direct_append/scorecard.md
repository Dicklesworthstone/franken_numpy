# fnp-random small PCG bytes direct append fill

Run identity:
- Bead: `franken_numpy-ixs5y.265`
- Agent: `BlackThrush` / `cod-a`
- Subject API: direct Rust `fnp-random` `Generator::bytes(length)`
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42)).bytes(length)` inside the Criterion benchmark harness
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`
- Source worktree: `/data/projects/.scratch/franken_numpy-cod-a-20260619-194850`

Lever:
- For sub-threshold PCG byte requests, append directly into the final `Vec<u8>` from `next_u64` words instead of routing through the scalar `next_uint32` stream or constructing an intermediate word buffer.
- Preserve the existing low/high `next_uint32` half-buffer contract by consuming any pending `u32_buf` first and saving the high 32 bits only when the final direct append consumed the low half of a fresh 64-bit word.
- Keep non-PCG generators and large PCG direct-fill behavior unchanged.

Alien mapping:
- Graveyard primitive: vectorized/direct final-buffer execution, but below the large parallel fill threshold where constants dominate.
- Artifact proof: behavior-preserving byte-stream rewrite with an explicit RNG state invariant.
- Rejected neighboring lever: the prior intermediate `Vec<u64>` transcode from `.257` remains rejected. This lever writes to the final byte vector directly and pays only one output allocation.

Decisive same-worker evidence:

| Worker | Git state | Workload | FrankenNumPy | NumPy | FNP/NumPy ratio | Old-to-new ratio | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `ovh-a` | origin/main baseline | 100k bytes | 87,044 ns | 94,154 ns | 0.925x, 1.08x faster | - | Baseline already slight win on this worker |
| `ovh-a` | candidate | 100k bytes | 32,920 ns | 47,212 ns | 0.697x, 1.43x faster | 0.378x, 2.64x faster | Keep |
| `ovh-a` | origin/main baseline | 1M bytes | 154,618 ns | 429,977 ns | 0.360x, 2.78x faster | - | Baseline win |
| `ovh-a` | candidate | 1M bytes | 122,926 ns | 427,988 ns | 0.287x, 3.48x faster | 0.795x, 1.26x faster | Keep |

Supplemental final-source rows:

| Worker | Workload | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---:|---:|---:|---:|---|
| `hz2` | 100k bytes | 42,322 ns | 58,116 ns | 0.728x, 1.37x faster | Confirmation |
| `hz2` | 1M bytes | 289,734 ns | 569,907 ns | 0.508x, 1.97x faster | Confirmation |
| `hz2` | 100k bytes rerun | 43,131 ns | 59,201 ns | 0.728x, 1.37x faster | Confirmation |
| `hz2` | 1M bytes rerun | 254,030 ns | 597,591 ns | 0.425x, 2.35x faster | Confirmation |
| `vmi1153651` | 100k bytes post-clippy | 85,242 ns | 465,151 ns | 0.183x, 5.46x faster | Noisy final-source confirmation |
| `vmi1153651` | 1M bytes post-clippy | 2,410,257 ns | 4,857,309 ns | 0.496x, 2.02x faster | Noisy final-source confirmation |

Routing evidence:

| Worker | Git state | Workload | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---:|---|
| `ovh-a` | pre-edit current | 100k bytes | 87,111 ns | 46,518 ns | 1.87x slower | Open gap observed |
| `ovh-a` | pre-edit current | 1M bytes | 123,059 ns | 453,671 ns | 0.271x, 3.69x faster | Already win |
| `hz1` | fresh origin/main control | 100k bytes | 131,543 ns | 73,649 ns | 1.79x slower | Open gap reproduced |
| `hz1` | fresh origin/main control | 1M bytes | 277,638 ns | 723,140 ns | 0.384x, 2.60x faster | Already win |

Scorecard:
- Same-worker old-to-new: win/loss/neutral = 2/0/0.
- Candidate vs NumPy decisive rows: win/loss/neutral = 2/0/0.
- Candidate vs NumPy including supplemental rows: win/loss/neutral = 8/0/0.
- Remaining gap status: the 100k PCG `bytes` row moved from observed loss to measured win in the decisive same-worker candidate run.

Validation:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture` passed before and after the clippy-only expression collapse.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_match_live_numpy_oracle_when_available -- --nocapture` passed.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random` passed: 431 unit tests, 12 golden tests, 16 metamorphic tests.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --all-targets` passed.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo clippy -p fnp-random --all-targets -- -D warnings` passed.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build --release -p fnp-random` passed.
- `git diff --check` passed.
- `cargo fmt --check` and `cargo fmt -p fnp-random --check` are blocked by pre-existing broad rustfmt drift in unrelated workspace / `fnp-random` sections; this proof keeps that drift out of scope.

Retry predicate:
- Do not retry intermediate word-vector transcodes for PCG bytes.
- Retry this lane only with a same-worker candidate that preserves the `next_uint32` half-buffer invariant, passes the live NumPy oracle shard, and beats this direct append path at both 100k and 1M rows.
