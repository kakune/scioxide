//! Sparse reconstruction and manufactured solutions without dense test storage.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, Permutation, RealScalar};
use scioxide_sparse::{
    CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, LdlInertia, SparseBuilder, SparseLdl,
    SparseOrdering, SparseSymmetricLimits, SparseSymmetricSymbolic,
};

fn scalar<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_exact(re).unwrap(),
        S::Real::try_from_f64_exact(im).unwrap(),
    )
}

fn edge<S: NumericScalar>(b: &mut SparseBuilder<S>, row: usize, col: usize, value: S) {
    b.push(row, col, value).unwrap();
    b.push(col, row, value.conj()).unwrap();
}

fn fixture<S: NumericScalar>(n: usize, family: usize) -> CsrMatrix<S> {
    let mut b = SparseBuilder::new(n, n);
    if family == 0 {
        // Strict diagonal dominance fixes the inertia throughout a homotopy
        // from the alternating-sign diagonal to this connected banded matrix.
        for i in 0..n {
            b.push(i, i, scalar(if i % 2 == 0 { 4. } else { -4. }, 0.))
                .unwrap();
            if i > 0 {
                edge(&mut b, i, i - 1, scalar(0.25, 0.125));
            }
        }
    } else {
        // Interleave primal and dual coordinates of [H Bᴴ; B 0]. B is
        // invertible bidiagonal; H is positive definite (KKT) or zero (saddle).
        // Both have inertia (n/2,n/2,0), independently of the factorization.
        for i in 0..n / 2 {
            let primal = 2 * i;
            let dual = primal + 1;
            b.push(
                primal,
                primal,
                scalar(if family == 1 { 4. } else { 0. }, 0.),
            )
            .unwrap();
            b.push(dual, dual, S::ZERO).unwrap();
            edge(&mut b, dual, primal, scalar(1., 0.5));
            if i > 0 {
                edge(&mut b, dual, primal - 2, scalar(0.125, -0.0625));
                if family == 1 {
                    edge(&mut b, primal, primal - 2, scalar(0.25, 0.125));
                }
            }
        }
    }
    b.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap()
}

// Expand each D block's outer products independently, then reduce coordinates
// together with -PAPᵀ. Including absent input coordinates detects spurious fill.
fn reconstruct<S: NumericScalar>(a: &CsrMatrix<S>, f: &SparseLdl<S>) {
    let n = a.view().rows();
    let p = f.permutation();
    let mut inverse = vec![0; n];
    let mut columns: Vec<_> = (0..n).map(|i| vec![(i, S::ONE)]).collect();
    for (i, &old) in p.iter().enumerate() {
        inverse[old] = i;
    }
    for (row, col, value) in f.lower_entries() {
        assert!(row > col);
        columns[col].push((row, value));
    }
    let mut terms = Vec::new();
    for block in f.pivot_blocks() {
        let range = block.checked_range(n).unwrap();
        for col in range.clone() {
            for other in range.clone() {
                let d = if col == other {
                    f.diagonal()[col]
                } else if col > other {
                    f.off_diagonal()[other]
                } else {
                    f.off_diagonal()[col].conj()
                };
                for &(i, l) in &columns[col] {
                    for &(j, r) in &columns[other] {
                        terms.push((i, j, l * d * r.conj()));
                    }
                }
            }
        }
    }
    for (i, j, value) in a.view().iter() {
        terms.push((inverse[i], inverse[j], -value));
    }
    terms.sort_unstable_by_key(|&(i, j, _)| (i, j));
    let epsilon = S::Real::EPSILON.try_to_f64_exact().unwrap();
    let mut start = 0;
    while start < terms.len() {
        let key = (terms[start].0, terms[start].1);
        let mut end = start;
        let mut residual = S::ZERO;
        let mut scale = 0.;
        while end < terms.len() && (terms[end].0, terms[end].1) == key {
            residual = residual + terms[end].2;
            scale += terms[end].2.abs().try_to_f64_exact().unwrap();
            end += 1;
        }
        assert!(
            residual.abs().try_to_f64_exact().unwrap() <= 256. * epsilon * scale.max(1.),
            "reconstruction at {key:?}: {residual:?}, scale={scale}"
        );
        start = end;
    }
}

fn check<S: NumericScalar>(n: usize, family: usize, ordering: SparseOrdering<'_>) {
    let a = fixture::<S>(n, family);
    let plan = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        ordering,
        None,
        SparseSymmetricLimits::new(16 * n, 16 * n, 4096 * n),
    )
    .unwrap();
    let f = SparseLdl::factor(plan, a.view()).unwrap();
    assert_eq!(
        f.inertia().unwrap(),
        LdlInertia::new(n / 2, n / 2, 0).unwrap()
    );
    if family == 2 {
        assert!(f.pivot_blocks().all(|block| block.size() == 2));
    }
    reconstruct(&a, &f);
    let expected: Vec<S> = (0..n)
        .map(|i| scalar(1. + (i % 3) as f64 * 0.125, -0.25))
        .collect();
    let mut rhs = vec![S::ZERO; n];
    for (i, j, value) in a.view().iter() {
        rhs[i] = rhs[i] + value * expected[j];
    }
    let result = f.solve(&rhs).unwrap();
    let epsilon = S::Real::EPSILON.try_to_f64_exact().unwrap();
    for (&actual, &expected) in result.iter().zip(&expected) {
        assert!(
            (actual - expected).abs().try_to_f64_exact().unwrap()
                <= 512. * epsilon * (1. + expected.abs().try_to_f64_exact().unwrap())
        );
    }
}

fn domains<S: NumericScalar>() {
    for family in 0..3 {
        check::<S>(10_000, family, SparseOrdering::Natural);
        check::<S>(64, family, SparseOrdering::FillReducing);
    }
}

#[test]
fn banded_kkt_and_saddle_reconstruct_at_ten_thousand_unknowns() {
    domains::<f32>();
    domains::<f64>();
    domains::<Complex<f32>>();
    domains::<Complex<f64>>();
}

fn pivot_ties<S: NumericScalar>() {
    let mut builder = SparseBuilder::new(4, 4);
    for (i, diagonal) in [0., 3., 0., 0.].into_iter().enumerate() {
        builder.push(i, i, scalar::<S>(diagonal, 0.)).unwrap();
    }
    edge(&mut builder, 2, 0, scalar(2., 1.));
    edge(&mut builder, 3, 0, scalar(2., -1.));
    edge(&mut builder, 3, 2, scalar(1., 0.25));
    let a = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    for order in [vec![0, 1, 2, 3], vec![0, 1, 3, 2]] {
        let expected_second = order[2];
        let order = Permutation::new(order).unwrap();
        let plan = SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            SparseOrdering::Supplied(&order),
            None,
            SparseSymmetricLimits::new(16, 16, 16_384),
        )
        .unwrap();
        let f = SparseLdl::factor(plan.clone(), a.view()).unwrap();
        assert_eq!(&f.permutation()[..2], &[0, expected_second]);
        assert_eq!(
            f.pivot_blocks()
                .map(|block| block.size())
                .collect::<Vec<_>>(),
            [2, 1, 1]
        );
        reconstruct(&a, &f);
    }
}

#[test]
fn rook_ties_use_current_factor_positions_after_ordering() {
    pivot_ties::<f32>();
    pivot_ties::<f64>();
    pivot_ties::<Complex<f32>>();
    pivot_ties::<Complex<f64>>();
}
