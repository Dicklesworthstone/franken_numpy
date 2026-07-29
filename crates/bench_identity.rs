use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::Write as _;

pub fn report_bench_identity() {
    let path = std::env::current_exe().expect("bench executable path must be available");
    let bytes = std::fs::read(&path).expect("bench executable must be readable");
    let digest = Sha256::digest(&bytes);
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hash, "{byte:02x}").expect("writing to String cannot fail");
    }
    println!(
        "bench_elf_sha256={hash} ({} bytes) {}",
        bytes.len(),
        path.display()
    );
    std::io::stdout()
        .flush()
        .expect("bench identity line must flush");
}
