use fnp_random::Generator;

fn main() {
    let seed = 12345;

    // maxwell, halfnormal, lomax, levy
    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("maxwell(1): {:?}", g.maxwell(1.0, 5));

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("halfnormal(1): {:?}", g.halfnormal(1.0, 5));

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("lomax(3): {:?}", g.lomax(3.0, 5));

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("levy(0,1): {:?}", g.levy(0.0, 1.0, 5));

    // advanced
    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("dirichlet: {:?}", g.dirichlet(&[1.0, 2.0, 3.0], 2).unwrap());

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!(
        "noncentral_chisquare: {:?}",
        g.noncentral_chisquare(2.0, 3.0, 5).unwrap()
    );

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!(
        "noncentral_f: {:?}",
        g.noncentral_f(2.0, 3.0, 4.0, 5).unwrap()
    );

    // remaining
    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("multinomial: {:?}", g.multinomial(10, &[0.2, 0.3, 0.5], 2));

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("zipf: {:?}", g.zipf(2.0, 5).unwrap());

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!("hypergeometric: {:?}", g.hypergeometric(10, 20, 5, 5).unwrap());

    let mut g = Generator::from_pcg64_dxsm(seed).unwrap();
    println!(
        "multivariate_hypergeometric: {:?}",
        g.multivariate_hypergeometric(&[10, 20, 30], 5, 2)
    );
}
