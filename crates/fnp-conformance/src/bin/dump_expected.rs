use fnp_random::Generator;

fn main() {
    let seed = 12345;
    let seed_msg = "fnp-conformance dump_expected: PCG64 seed is a compile-time constant";

    // maxwell, halfnormal, lomax, levy
    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!("maxwell(1): {:?}", g.maxwell(1.0, 5));

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!("halfnormal(1): {:?}", g.halfnormal(1.0, 5));

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!("lomax(3): {:?}", g.lomax(3.0, 5));

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!("levy(0,1): {:?}", g.levy(0.0, 1.0, 5));

    // advanced
    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!(
        "dirichlet: {:?}",
        g.dirichlet(&[1.0, 2.0, 3.0], 2).expect(
            "fnp-conformance dump_expected: dirichlet with constant alpha vector and size=2"
        )
    );

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!(
        "noncentral_chisquare: {:?}",
        g.noncentral_chisquare(2.0, 3.0, 5)
            .expect("fnp-conformance dump_expected: noncentral_chisquare with df=2, nonc=3")
    );

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!(
        "noncentral_f: {:?}",
        g.noncentral_f(2.0, 3.0, 4.0, 5)
            .expect("fnp-conformance dump_expected: noncentral_f with dfnum=2, dfden=3, nonc=4")
    );

    // remaining
    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!("multinomial: {:?}", g.multinomial(10, &[0.2, 0.3, 0.5], 2));

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!(
        "zipf: {:?}",
        g.zipf(2.0, 5)
            .expect("fnp-conformance dump_expected: zipf with a=2 > 1")
    );

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!(
        "hypergeometric: {:?}",
        g.hypergeometric(10, 20, 5, 5).expect(
            "fnp-conformance dump_expected: hypergeometric with ngood=10, nbad=20, nsample=5"
        )
    );

    let mut g = Generator::from_pcg64_dxsm(seed).expect(seed_msg);
    println!(
        "multivariate_hypergeometric: {:?}",
        g.multivariate_hypergeometric(&[10, 20, 30], 5, 2)
            .expect("fnp-conformance dump_expected: multivariate_hypergeometric with colors=[10,20,30], nsample=5")
    );
}
