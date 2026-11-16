//! Caller-owned combinatorial operations allocate no heap storage.
#![allow(clippy::unwrap_used)]

use scioxide_core::*;

#[test]
fn borrowed_lifecycle_and_error_paths_execute() {
    let mut permutation = [0; 5];
    let mut selection = [0; 3];
    let mut inverse = [0; 5];
    let mut output = [0; 5];
    let mut validation = [0_u8; 1];
    let mut inverse_validation = [0_u8; 1];
    {
        assert_eq!(checked_factorial(5).unwrap(), 120);
        assert_eq!(checked_permutation_count(5, 3).unwrap(), 60);
        assert_eq!(checked_binomial(5, 3).unwrap(), 10);
        assert_eq!(checked_multiset_count(5, 3).unwrap(), 35);
        let mut permutations = PermutationEnumerator::identity_into(&mut permutation);
        loop {
            let rank = permutation_rank(permutations.as_view()).unwrap();
            permutation_unrank_into(rank, &mut inverse).unwrap();
            assert_eq!(permutations.new_to_old(), inverse);
            let view = permutations.as_view();
            view.inverse_map_into(&mut inverse).unwrap();
            let inverse_view = PermutationView::new(&inverse, &mut inverse_validation).unwrap();
            view.compose_into(inverse_view, &mut output).unwrap();
            assert_eq!(output, [0, 1, 2, 3, 4]);
            if !permutations.advance() {
                break;
            }
        }
        let mut combinations = CombinationEnumerator::first_into(5, &mut selection).unwrap();
        loop {
            let rank = combination_rank(5, combinations.indices()).unwrap();
            combination_unrank_into(5, rank, &mut output[..3]).unwrap();
            assert_eq!(combinations.indices(), &output[..3]);
            if !combinations.advance() {
                break;
            }
        }
        let mut multisets = MultisetEnumerator::first_into(5, &mut selection).unwrap();
        loop {
            let rank = multiset_rank(5, multisets.indices()).unwrap();
            multiset_unrank_into(5, rank, &mut output[..3]).unwrap();
            assert_eq!(multisets.indices(), &output[..3]);
            if !multisets.advance() {
                break;
            }
        }
        assert!(PermutationView::new(&[0, 0], &mut validation).is_err());
        assert!(combination_unrank_into(usize::MAX, 0, &mut selection).is_err());
    };
}

fn gather_roundtrip<T: Copy + PartialEq + core::fmt::Debug>(input: [T; 3]) {
    let mut intermediate = input;
    let mut output = input;
    let mut scratch = [0_u8; 1];
    {
        let view = PermutationView::new(&[2, 0, 1], &mut scratch).unwrap();
        view.apply_into(&input, &mut intermediate).unwrap();
        assert_eq!(intermediate, [input[2], input[0], input[1]]);
        view.inverse_into(&intermediate, &mut output).unwrap();
        assert_eq!(output, input);
    };
}

#[test]
fn real_complex_and_nonnumerical_values_are_supported() {
    gather_roundtrip([1_f32, -0.0, 3.0]);
    gather_roundtrip([1_f64, -0.0, 3.0]);
    gather_roundtrip([
        Complex::new(1_f32, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
    ]);
    gather_roundtrip([
        Complex::new(1_f64, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.0, 4.0),
    ]);
    gather_roundtrip(['a', 'b', 'c']);
}

#[test]
fn trivial_stirling_numbers_execute() {
    {
        assert_eq!(
            checked_stirling_second_kind(usize::MAX, usize::MAX).unwrap(),
            1
        );
        assert_eq!(checked_stirling_first_kind(0, 1).unwrap(), 0);
        assert_eq!(checked_stirling_second_kind(3, usize::MAX).unwrap(), 0);
        assert_eq!(checked_stirling_second_kind(usize::MAX, 1).unwrap(), 1);
    };
}
