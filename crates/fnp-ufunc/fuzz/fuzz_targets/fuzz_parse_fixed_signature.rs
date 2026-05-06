#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FixedSigInput {
    signature: String,
    nin: u8,
    nout: u8,
}

fuzz_target!(|input: FixedSigInput| {
    if input.signature.len() > 512 {
        return;
    }
    let nin = input.nin as usize;
    let nout = (input.nout as usize).max(1);
    let _ = fnp_ufunc::parse_fixed_signature_string(&input.signature, nin, nout);
});
