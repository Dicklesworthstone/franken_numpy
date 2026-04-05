#[test]
fn test_nat_parsing() {
    let result = fnp_ufunc::UFuncArray::from_datetime_strings(vec![1], vec!["NaT".to_string()], None);
    println!("{:?}", result);
}
