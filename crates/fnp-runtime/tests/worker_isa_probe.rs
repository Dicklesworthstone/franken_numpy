//! Diagnostic probe for the fleet ISA-baseline question: prints (a) the target features the
//! COMPILER baked into this binary (proves what `.cargo/config.toml`'s rustflags actually
//! emitted on whichever machine built it) and (b) the features the EXECUTING CPU supports at
//! runtime. Run remotely (`rch exec -- cargo test -p fnp-runtime --test worker_isa_probe --
//! --nocapture`) to answer "what do the workers build and support" without shipping perf code.
//! Always passes; the output is the artifact.

#[test]
fn worker_isa_probe() {
    let compiled = [
        ("sse2", cfg!(target_feature = "sse2")),
        ("sse4.2", cfg!(target_feature = "sse4.2")),
        ("avx", cfg!(target_feature = "avx")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("fma", cfg!(target_feature = "fma")),
        ("avx512f", cfg!(target_feature = "avx512f")),
    ];
    #[cfg(target_arch = "x86_64")]
    let runtime = [
        ("sse2", std::arch::is_x86_feature_detected!("sse2")),
        ("sse4.2", std::arch::is_x86_feature_detected!("sse4.2")),
        ("avx", std::arch::is_x86_feature_detected!("avx")),
        ("avx2", std::arch::is_x86_feature_detected!("avx2")),
        ("fma", std::arch::is_x86_feature_detected!("fma")),
        ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
    ];
    println!(
        "ISA_PROBE compiled_target_features: {}",
        compiled
            .iter()
            .map(|(name, on)| format!("{name}={on}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    #[cfg(target_arch = "x86_64")]
    println!(
        "ISA_PROBE runtime_cpu_features: {}",
        runtime
            .iter()
            .map(|(name, on)| format!("{name}={on}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // The project contract: AVX2 baked in (the .cargo/config.toml rustflags), FMA NOT baked in
    // (x86-64-v3 contraction regresses 16 fnp-linalg conformance tests). Fail loudly if a build
    // environment ever drops the flag (e.g. a worker-side RUSTFLAGS override shadowing the
    // project config).
    assert!(
        cfg!(target_feature = "avx2"),
        "ISA_PROBE: this binary was built WITHOUT +avx2 - project rustflags were not applied"
    );
    assert!(
        !cfg!(target_feature = "fma"),
        "ISA_PROBE: this binary was built WITH +fma - bit-parity policy violated"
    );
}
