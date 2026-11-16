//! Exact combinatorial identities and public failure boundaries.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;

#[test]
fn counts_match_independent_pascal_triangle() {
    let mut row = vec![1_u128];
    for n in 0..=100 {
        for (k, &expected) in row.iter().enumerate() {
            let actual = checked_binomial(n, k);
            if let Ok(expected) = usize::try_from(expected) {
                assert_eq!(actual.unwrap(), expected, "n={n}, k={k}");
            } else {
                assert!(matches!(
                    (actual).as_ref().unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::SizeOverflow { .. }
                ));
            }
        }
        let mut next = vec![1];
        next.extend(row.windows(2).map(|p| p[0] + p[1]));
        next.push(1);
        row = next;
    }
    assert_eq!(checked_binomial(usize::MAX, 1).unwrap(), usize::MAX);
    assert_eq!(
        checked_binomial(usize::MAX, usize::MAX - 1).unwrap(),
        usize::MAX
    );
    assert_eq!(checked_binomial(1, 2).unwrap(), 0);
    assert_eq!(checked_factorial(0).unwrap(), 1);
    assert_eq!(checked_factorial(10).unwrap(), 3_628_800);
    assert!(matches!(
        (checked_factorial(usize::MAX)).as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
    assert_eq!(checked_permutation_count(5, 3).unwrap(), 60);
    assert_eq!(checked_permutation_count(0, 1).unwrap(), 0);
    assert_eq!(checked_multiset_count(0, 0).unwrap(), 1);
    assert_eq!(checked_multiset_count(0, 1).unwrap(), 0);
    assert_eq!(checked_multiset_count(1, usize::MAX).unwrap(), 1);
    assert_eq!(checked_multiset_count(usize::MAX, 1).unwrap(), usize::MAX);
    assert!(matches!(
        (checked_multiset_count(usize::MAX, 2))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn borrowed_and_owned_permutations_share_gather_convention() {
    let p = Permutation::new(vec![2, 0, 1]).unwrap();
    let q = Permutation::new(vec![1, 0, 2]).unwrap();
    let composed = p.compose(&q).unwrap();
    assert_eq!(composed.new_to_old(), &[2, 1, 0]);
    assert_eq!(
        p.compose(&p.inverse().unwrap()).unwrap(),
        Permutation::identity(3).unwrap()
    );
    let mut scratch = [0_u8; 1];
    let view = PermutationView::new(p.new_to_old(), &mut scratch).unwrap();
    let mut inverse = [0; 3];
    view.inverse_map_into(&mut inverse).unwrap();
    assert_eq!(inverse, p.old_to_new());
    let mut output = [9; 2];
    assert!(matches!(
        (p.apply_into(&[1, 2, 3], &mut output))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, [9; 2]);
    for bad in [&[0, 0][..], &[0, 2], &[usize::MAX]] {
        let mut scratch = [0_u8; 1];
        assert!(matches!(
            (PermutationView::new(bad, &mut scratch))
                .as_ref()
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(Permutation::new(bad.to_vec()).is_err());
    }
    let empty = Permutation::identity(0).unwrap();
    assert_eq!(empty.inverse().unwrap(), empty);
}

#[test]
fn enumeration_is_checked_and_terminal_states_are_stable() {
    let mut p = [0, 1, 2];
    let mut permutations = PermutationEnumerator::identity_into(&mut p);
    let mut count = 1;
    while permutations.advance() {
        count += 1;
    }
    assert_eq!(count, 6);
    assert_eq!(permutations.new_to_old(), [2, 1, 0]);
    assert!(!permutations.advance());
    permutations.reset();
    assert_eq!(permutations.new_to_old(), [0, 1, 2]);
    let mut c = [0, 1, 2];
    let mut combinations = CombinationEnumerator::new(5, &mut c).unwrap();
    let mut count = 1;
    while combinations.advance() {
        count += 1;
    }
    assert_eq!(count, checked_binomial(5, 3).unwrap());
    assert_eq!(combinations.indices(), [2, 3, 4]);
    let mut m = [0, 0, 0];
    let mut multisets = MultisetEnumerator::new(5, &mut m).unwrap();
    let mut count = 1;
    while multisets.advance() {
        count += 1;
    }
    assert_eq!(count, checked_multiset_count(5, 3).unwrap());
    assert_eq!(multisets.indices(), [4, 4, 4]);
    assert!(!PermutationEnumerator::identity_into(&mut []).advance());
    assert!(
        !CombinationEnumerator::first_into(0, &mut [])
            .unwrap()
            .advance()
    );
    assert!(
        !MultisetEnumerator::first_into(0, &mut [])
            .unwrap()
            .advance()
    );
    let mut bad = [1, 0];
    assert!(CombinationEnumerator::new(2, &mut bad).is_err());
    assert!(MultisetEnumerator::new(2, &mut bad).is_err());
    assert_eq!(bad, [1, 0]);
    assert!(CombinationEnumerator::first_into(1, &mut bad).is_err());
    assert!(MultisetEnumerator::first_into(0, &mut bad).is_err());
    assert_eq!(bad, [1, 0]);
    let mut huge = [0, 1];
    let mut huge_state = CombinationEnumerator::new(usize::MAX, &mut huge).unwrap();
    assert!(huge_state.advance());
    assert_eq!(huge_state.indices(), [0, 2]);
}

#[test]
fn validation_scratch_is_linear_reusable_and_transactional() {
    assert_eq!(permutation_validation_scratch_bytes(0), 0);
    assert_eq!(permutation_validation_scratch_bytes(1), 1);
    assert_eq!(permutation_validation_scratch_bytes(8), 1);
    assert_eq!(permutation_validation_scratch_bytes(9), 2);
    assert_eq!(
        permutation_validation_scratch_bytes(usize::MAX),
        usize::MAX / 8 + 1
    );
    for n in [1_000, 10_000, 100_000] {
        assert_eq!(permutation_validation_scratch_bytes(n), n / 8);
    }

    let mut state = [2, 0, 1, 4, 3, 7, 6, 5, 8];
    let original = state;
    let mut short = [0_u8; 1];
    assert!(matches!(
        (PermutationEnumerator::new(&mut state, &mut short))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(state, original);

    let mut scratch = [0xff_u8; 2];
    let view = PermutationView::new(&state, &mut scratch).unwrap();
    assert_eq!(view.new_to_old(), original);
    let duplicate = [2, 0, 1, 4, 3, 7, 6, 5, 5];
    assert!(PermutationView::new(&duplicate, &mut scratch).is_err());
    assert!(PermutationView::new(&state, &mut scratch).is_ok());
}

#[test]
fn owned_and_borrowed_states_cover_long_suffixes_resumption_and_reset() {
    let mut permutation = PermutationEnumerator::new(vec![0, 3, 2, 1], &mut [0_u8; 1]).unwrap();
    assert!(permutation.advance());
    assert_eq!(permutation.new_to_old(), &[1, 0, 2, 3]);
    let mut inverse = [0; 4];
    permutation
        .as_view()
        .inverse_map_into(&mut inverse)
        .unwrap();
    for (new, &old) in permutation.new_to_old().iter().enumerate() {
        assert_eq!(inverse[old], new);
    }
    permutation.reset();
    assert_eq!(permutation, PermutationEnumerator::identity(4).unwrap());

    let mut combination = CombinationEnumerator::new(6, vec![0, 4, 5]).unwrap();
    assert!(combination.advance());
    assert_eq!(combination.indices(), &[1, 2, 3]);
    assert!(combination.advance());
    let interrupted = combination.indices().to_vec();
    assert_eq!(interrupted, [1, 2, 4]);
    assert!(combination.advance());
    assert_eq!(combination.indices(), &[1, 2, 5]);
    combination.reset();
    assert_eq!(combination.indices(), &[0, 1, 2]);

    let mut multiset = MultisetEnumerator::new(4, vec![0, 3, 3]).unwrap();
    assert!(multiset.advance());
    assert_eq!(multiset.indices(), &[1, 1, 1]);
    multiset.reset();
    assert_eq!(multiset.indices(), &[0, 0, 0]);

    let mut near_full = CombinationEnumerator::first(5, 4).unwrap();
    let mut seen = 1;
    while near_full.advance() {
        seen += 1;
    }
    assert_eq!(seen, 5);
    assert_eq!(near_full.indices(), &[1, 2, 3, 4]);
    assert!(!near_full.advance());

    assert_eq!(CombinationEnumerator::first(0, 0).unwrap().indices(), &[]);
    assert_eq!(MultisetEnumerator::first(0, 0).unwrap().indices(), &[]);
    assert!(!CombinationEnumerator::first(1, 1).unwrap().advance());
    assert!(!MultisetEnumerator::first(1, 4).unwrap().advance());
}

// Independent exhaustive Cartesian construction, filtered by the mathematical
// definitions. This does not use any production successor or ranking recurrence.
fn words(n: usize, k: usize, prefix: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
    if prefix.len() == k {
        output.push(prefix.clone());
        return;
    }
    for value in 0..n {
        prefix.push(value);
        words(n, k, prefix, output);
        prefix.pop();
    }
}

#[test]
fn independent_enumeration_and_lexicographic_ranks() {
    for n in 0..=5 {
        let mut permutations = Vec::new();
        words(n, n, &mut Vec::new(), &mut permutations);
        permutations.retain(|v| v.iter().enumerate().all(|(i, x)| !v[..i].contains(x)));
        assert_eq!(permutations.len(), checked_factorial(n).unwrap());
        let mut state = vec![0; n];
        let mut state = PermutationEnumerator::identity_into(&mut state);
        for (rank, expected) in permutations.iter().enumerate() {
            assert_eq!(state.new_to_old(), expected);
            assert_eq!(permutation_rank(state.as_view()).unwrap(), rank);
            let mut actual = vec![usize::MAX; n];
            permutation_unrank_into(rank, &mut actual).unwrap();
            assert_eq!(&actual, expected);
            assert_eq!(state.advance(), rank + 1 < permutations.len());
            let p = Permutation::new(expected.clone()).unwrap();
            assert_eq!(
                p.compose(&p.inverse().unwrap()).unwrap(),
                Permutation::identity(n).unwrap()
            );
        }
        for k in 0..=5 {
            let mut candidates = Vec::new();
            words(n, k, &mut Vec::new(), &mut candidates);
            for repeated in [false, true] {
                let expected: Vec<_> = candidates
                    .iter()
                    .filter(|v| {
                        v.windows(2)
                            .all(|p| if repeated { p[0] <= p[1] } else { p[0] < p[1] })
                    })
                    .collect();
                let count = if repeated {
                    checked_multiset_count(n, k)
                } else {
                    checked_binomial(n, k)
                }
                .unwrap();
                assert_eq!(count, expected.len());
                let mut storage = vec![0; k];
                if repeated {
                    let initialized = MultisetEnumerator::first_into(n, &mut storage);
                    assert_eq!(initialized.is_ok(), count != 0);
                    if let Ok(mut state) = initialized {
                        for (rank, word) in expected.iter().enumerate() {
                            assert_eq!(state.indices(), *word);
                            assert_eq!(multiset_rank(n, word).unwrap(), rank);
                            let mut actual = vec![usize::MAX; k];
                            multiset_unrank_into(n, rank, &mut actual).unwrap();
                            assert_eq!(&actual, *word);
                            assert_eq!(state.advance(), rank + 1 < count);
                        }
                    }
                } else {
                    let initialized = CombinationEnumerator::first_into(n, &mut storage);
                    assert_eq!(initialized.is_ok(), count != 0);
                    if let Ok(mut state) = initialized {
                        for (rank, word) in expected.iter().enumerate() {
                            assert_eq!(state.indices(), *word);
                            assert_eq!(combination_rank(n, word).unwrap(), rank);
                            let mut actual = vec![usize::MAX; k];
                            combination_unrank_into(n, rank, &mut actual).unwrap();
                            assert_eq!(&actual, *word);
                            assert_eq!(state.advance(), rank + 1 < count);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rank_boundaries_and_transactional_errors() {
    let mut output = [77; 3];
    assert!(matches!(
        (permutation_unrank_into(6, &mut output))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::OutOfBounds { .. }
    ));
    assert_eq!(output, [77; 3]);
    assert!(combination_unrank_into(2, 0, &mut output).is_err());
    assert_eq!(output, [77; 3]);
    assert!(multiset_unrank_into(0, 0, &mut output).is_err());
    assert_eq!(output, [77; 3]);
    assert!(combination_unrank_into(usize::MAX, 0, &mut output).is_err());
    assert_eq!(output, [77; 3]);
    assert!(multiset_unrank_into(usize::MAX, 0, &mut output).is_err());
    assert_eq!(output, [77; 3]);
    assert!(combination_rank(usize::MAX, &[0, 1, 2]).is_err());
    assert!(multiset_rank(usize::MAX, &[0, 0, 0]).is_err());
    let mut scratch = [0_u8; 1];
    assert!(PermutationView::new(&[0, 0], &mut scratch).is_err());
    assert!(combination_rank(4, &[2, 1]).is_err());
    assert!(multiset_rank(2, &[0, 2]).is_err());
    let mut large = [77; 100];
    assert!(permutation_unrank_into(0, &mut large).is_err());
    assert_eq!(large, [77; 100]);
    let mut singleton = [0];
    for rank in [0, 1, usize::MAX / 2, usize::MAX - 1] {
        combination_unrank_into(usize::MAX, rank, &mut singleton).unwrap();
        assert_eq!(singleton, [rank]);
        assert_eq!(combination_rank(usize::MAX, &singleton).unwrap(), rank);
        multiset_unrank_into(usize::MAX, rank, &mut singleton).unwrap();
        assert_eq!(singleton, [rank]);
        assert_eq!(multiset_rank(usize::MAX, &singleton).unwrap(), rank);
    }
    multiset_unrank_into(1, 0, &mut output).unwrap();
    assert_eq!(output, [0; 3]);
    assert_eq!(multiset_rank(1, &output).unwrap(), 0);
}

#[test]
fn indexing_covers_largest_representable_cardinalities() {
    for n in 0..=70 {
        if let Ok(count) = checked_factorial(n) {
            let mut state = vec![0; n];
            for rank in [0, count / 2, count - 1] {
                permutation_unrank_into(rank, &mut state).unwrap();
                let mut scratch = vec![0; permutation_validation_scratch_bytes(n)];
                let view = PermutationView::new(&state, &mut scratch).unwrap();
                assert_eq!(permutation_rank(view).unwrap(), rank);
            }
        }
        for k in 0..=n {
            let mut state = vec![0; k];
            if let Ok(count) = checked_binomial(n, k) {
                for rank in [0, count / 2, count - 1] {
                    combination_unrank_into(n, rank, &mut state).unwrap();
                    assert_eq!(combination_rank(n, &state).unwrap(), rank);
                }
            }
            if let Ok(count) = checked_multiset_count(n, k) {
                for rank in [0, count / 2, count - 1] {
                    multiset_unrank_into(n, rank, &mut state).unwrap();
                    assert_eq!(multiset_rank(n, &state).unwrap(), rank);
                }
            }
        }
    }
    assert!(matches!(
        (Permutation::identity(usize::MAX))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
    assert!(matches!(
        (CombinationEnumerator::first(usize::MAX, usize::MAX))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
    assert!(matches!(
        (MultisetEnumerator::first(1, usize::MAX))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
    let mut p_scratch = [0_u8; 1];
    let mut q_scratch = [0_u8; 1];
    let p = PermutationView::new(&[1, 0], &mut p_scratch).unwrap();
    let q = PermutationView::new(&[0], &mut q_scratch).unwrap();
    let mut output = [77; 2];
    assert!(matches!(
        p.compose_into(q, &mut output).unwrap_err().view(),
        SciOxideErrorView::DimensionMismatch {  expected, actual }
            if expected == [2] && actual == [1]
    ));
    assert_eq!(output, [77; 2]);
    assert!(matches!(
        p.inverse_map_into(&mut output[..1]).unwrap_err().view(),
        SciOxideErrorView::DimensionMismatch {  expected, actual }
            if expected == [2] && actual == [1]
    ));
    assert_eq!(output, [77; 2]);
    assert!(matches!(
        p.inverse_into(&[1], &mut output).unwrap_err().view(),
        SciOxideErrorView::DimensionMismatch {  expected, actual }
            if expected == [2] && actual == [1]
    ));
    assert_eq!(output, [77; 2]);
}

#[test]
fn exact_factorial_variants_check_final_growth() {
    assert_eq!(checked_double_factorial(0).unwrap(), 1);
    assert_eq!(checked_double_factorial(9).unwrap(), 945);
    assert_eq!(checked_multifactorial(10, 3).unwrap(), 280);
    assert_eq!(
        checked_multifactorial(usize::MAX, usize::MAX).unwrap(),
        usize::MAX
    );
    assert!(checked_multifactorial(1, 0).is_err());
    assert_eq!(checked_rising_factorial(0, usize::MAX).unwrap(), 0);
    assert_eq!(checked_rising_factorial(usize::MAX, 0).unwrap(), 1);
    assert_eq!(checked_rising_factorial(usize::MAX, 1).unwrap(), usize::MAX);
    assert!(checked_rising_factorial(usize::MAX, 2).is_err());
    assert_eq!(checked_rising_factorial(3, 4).unwrap(), 360);
}
#[test]
fn checked_stirling_matches_independent_polynomial_coefficients() {
    let mut coefficients = vec![1_i128];
    for n in 0..=30 {
        for (k, &expected) in coefficients.iter().enumerate() {
            let actual = checked_stirling_first_kind(n, k);
            if let Ok(expected) = isize::try_from(expected) {
                assert_eq!(actual.unwrap(), expected, "{n},{k}");
            } else {
                assert!(matches!(
                    (actual).as_ref().unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::SizeOverflow { .. }
                ));
            }
        }
        // Coefficients of x(x-1)…(x-n): independent polynomial multiplication.
        let mut next = vec![0; coefficients.len() + 1];
        for (k, &v) in coefficients.iter().enumerate() {
            next[k] -= (n as i128) * v;
            next[k + 1] += v;
        }
        coefficients = next;
    }
    assert_eq!(checked_stirling_second_kind(5, 2).unwrap(), 15);
    assert_eq!(checked_stirling_second_kind(10, 3).unwrap(), 9330);
    assert_eq!(checked_stirling_second_kind(40, 39).unwrap(), 780);
    assert_eq!(checked_stirling_first_kind(40, 39).unwrap(), -780);
    assert_eq!(
        checked_stirling_second_kind(usize::MAX, usize::MAX).unwrap(),
        1
    );
    assert_eq!(checked_stirling_first_kind(0, 1).unwrap(), 0);
}

#[test]
fn checked_stirling_second_kind_reaches_the_usize_boundary() {
    // S(n, 2) = 2^(n−1) − 1.
    let bits = usize::BITS as usize;
    assert_eq!(
        checked_stirling_second_kind(bits + 1, 2).unwrap(),
        usize::MAX
    );
    assert!(matches!(
        checked_stirling_second_kind(bits + 2, 2)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}
