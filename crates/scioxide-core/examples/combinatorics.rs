//! Checked counts, borrowed enumeration, indexing, and permutation composition.
use scioxide_core::{
    CombinationEnumerator, Permutation, SciOxideError, checked_binomial, combination_rank,
    combination_unrank_into,
};
fn main() -> Result<(), SciOxideError> {
    let mut indices = [0; 2];
    let mut combinations = CombinationEnumerator::first_into(4, &mut indices)?;
    loop {
        println!(
            "rank {}: {:?}",
            combination_rank(4, combinations.indices())?,
            combinations.indices()
        );
        if !combinations.advance() {
            break;
        }
    }
    combinations.into_inner();
    assert_eq!(checked_binomial(4, 2)?, 6);
    combination_unrank_into(4, 4, &mut indices)?;
    assert_eq!(indices, [1, 3]);
    let p = Permutation::new(vec![2, 0, 1])?;
    let mut gathered = [0.; 3];
    p.apply_into(&[10., 20., 30.], &mut gathered)?;
    assert_eq!(gathered, [30., 10., 20.]);
    assert_eq!(p.compose(&p.inverse()?)?, Permutation::identity(3)?);
    Ok(())
}
