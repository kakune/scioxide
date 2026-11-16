//! One-shot constructors and owned forms against their configured forms:
//! the same numbers bit for bit, the capacity rule at its estimate
//! and its bound, empty inputs and the error kinds of the one-shot solves.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AlgebraicScalar, BlockVectorView, Complex, ExecutionContext, NumericScalar, RankThreshold,
    RealScalar, SciOxideErrorKind,
};
use scioxide_sparse::*;

/// Owned solves equal `solve_into` with the default options for each operation.
macro_rules! owned_solves {
    ($factor:expr, $rhs:expr) => {{
        let (factor, rhs) = (&$factor, &$rhs[..]);
        for (operation, owned) in [
            (SparseOperation::Normal, factor.solve(rhs).unwrap()),
            (
                SparseOperation::Transpose,
                factor.solve_transpose(rhs).unwrap(),
            ),
            (SparseOperation::Adjoint, factor.solve_adjoint(rhs).unwrap()),
        ] {
            let mut output = rhs.to_vec();
            factor
                .solve_into(
                    rhs,
                    operation,
                    SparseSolveOptions::new(),
                    &mut output,
                    &mut factor.workspace(1).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(owned, output);
        }
    }};
}

fn scalar<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(re).unwrap(),
        S::Real::try_from_f64_lossy(im).unwrap(),
    )
}

fn csr<S: NumericScalar>(rows: usize, cols: usize, entries: &[(usize, usize, S)]) -> CsrMatrix<S> {
    let (r, c, v): (Vec<_>, Vec<_>, Vec<_>) = entries.iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut r, mut c, mut v), &(i, j, x)| {
            r.push(i);
            c.push(j);
            v.push(x);
            (r, c, v)
        },
    );
    CsrMatrix::from_triplets(rows, cols, &r, &c, &v).unwrap()
}

/// Hermitian positive definite 5-point operator on an `m × m` grid: diagonal 5
/// and neighbours `−1 ± i/4`, so its LU and Cholesky factors fill in.
fn grid<S: NumericScalar>(m: usize) -> CsrMatrix<S> {
    let index = |x: usize, y: usize| x * m + y;
    let mut entries = Vec::new();
    for x in 0..m {
        for y in 0..m {
            let k = index(x, y);
            entries.push((k, k, scalar(5.0, 0.0)));
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx < m && ny < m {
                    let l = index(nx, ny);
                    entries.push((k, l, scalar(-1.0, 0.25)));
                    entries.push((l, k, scalar(-1.0, -0.25)));
                }
            }
        }
    }
    csr(m * m, m * m, &entries)
}

/// Fully dense `rows × cols` matrix of full rank.
fn dense<S: NumericScalar>(rows: usize, cols: usize) -> CsrMatrix<S> {
    let entries: Vec<_> = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .map(|(i, j)| {
            let shift = if i == j { (rows + cols) as f64 } else { 0.0 };
            let re = 1.0 / (i + j + 1) as f64 + shift;
            (i, j, scalar(re, 0.125 * (i as f64 - j as f64)))
        })
        .collect();
    csr(rows, cols, &entries)
}

/// Hermitian matrix of order 8 with a zero diagonal: `LDLᴴ` needs 2×2 pivots
/// and interchanges.
fn saddle<S: NumericScalar>() -> CsrMatrix<S> {
    let mut entries: Vec<(usize, usize, S)> = Vec::new();
    for i in 0..8 {
        for (j, value) in [(i + 1, scalar(1.0, 0.5)), (i + 3, scalar(2.0, -0.25))] {
            if j < 8 {
                entries.push((i, j, value));
                entries.push((j, i, value.conj()));
            }
        }
    }
    csr(8, 8, &entries)
}

fn rhs<S: NumericScalar>(n: usize) -> Vec<S> {
    (0..n)
        .map(|i| scalar((i + 1) as f64 / 3.0, 0.5 - i as f64 / 7.0))
        .collect()
}

fn lu_configured<S: NumericScalar>(a: &CsrMatrix<S>, entries: usize) -> SparseLu<S> {
    let symbolic = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::FillReducing,
        SparseLuLimits::new(entries, usize::MAX),
    )
    .unwrap();
    SparseLu::factor(symbolic, a.view()).unwrap()
}

fn symmetric_analysis(
    pattern: scioxide_core::StructuralPatternView<'_>,
    factor: usize,
    active: usize,
) -> SparseSymmetricSymbolic {
    SparseSymmetricSymbolic::analyze(
        pattern,
        SparseOrdering::FillReducing,
        None,
        SparseSymmetricLimits::new(factor, active, usize::MAX),
    )
    .unwrap()
}

fn same_lu<S: NumericScalar>(a: &SparseLu<S>, b: &SparseLu<S>) {
    assert_eq!(a.row_permutation(), b.row_permutation());
    assert!(a.lower_entries().eq(b.lower_entries()));
    assert!(a.upper_entries().eq(b.upper_entries()));
}

fn same_ldl<S: NumericScalar>(a: &SparseLdl<S>, b: &SparseLdl<S>) {
    assert_eq!(a.permutation(), b.permutation());
    assert_eq!(a.diagonal(), b.diagonal());
    assert_eq!(a.off_diagonal(), b.off_diagonal());
    assert!(a.pivot_blocks().eq(b.pivot_blocks()));
    assert!(a.lower_entries().eq(b.lower_entries()));
}

fn lu_contract<S: NumericScalar>() {
    for a in [grid::<S>(8), dense::<S>(6, 6), dense::<S>(1, 1)] {
        let n = a.rows();
        let b = rhs::<S>(n);
        let one_shot = SparseLu::new(&a).unwrap();
        let configured = lu_configured(&a, n * n);
        same_lu(&one_shot, &configured);
        assert_eq!(solve(&a, &b).unwrap(), configured.solve(&b).unwrap());
        owned_solves!(one_shot, b);
        // The owned factor is the scoped factor with the default options.
        let symbolic = configured.symbolic().try_clone().unwrap();
        let scoped = SparseLu::factor_scoped(
            symbolic,
            a.view(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        same_lu(&scoped, &configured);
    }
    // The 8 × 8 grid fills beyond the first estimate `nnz + n`, so the
    // one-shot factorization above needed a retry.
    let a = grid::<S>(8);
    let symbolic = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::FillReducing,
        SparseLuLimits::new(a.nnz() + a.rows(), usize::MAX),
    )
    .unwrap();
    let error = SparseLu::factor(symbolic, a.view()).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
}

fn symmetric_contract<S: NumericScalar>() {
    let a = grid::<S>(6);
    let n = a.rows();
    let b = rhs::<S>(n);
    let cholesky = SparseCholesky::new(&a).unwrap();
    let analysis = symmetric_analysis(a.structural_pattern(), n * n, 0);
    let scoped = SparseCholesky::factor_scoped(
        analysis.try_clone().unwrap(),
        a.view(),
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let configured = SparseCholesky::factor(analysis, a.view()).unwrap();
    assert!(scoped.lower_entries().eq(configured.lower_entries()));
    owned_solves!(cholesky, b);
    assert_eq!(cholesky.permutation(), configured.permutation());
    assert_eq!(cholesky.diagonal(), configured.diagonal());
    assert!(cholesky.lower_entries().eq(configured.lower_entries()));
    assert_eq!(
        cholesky.factor_entries(),
        cholesky.symbolic().cholesky_entries()
    );
    assert_eq!(cholesky.solve(&b).unwrap(), configured.solve(&b).unwrap());
    let adjoint = cholesky.solve_adjoint(&b).unwrap();
    assert_eq!(adjoint, configured.solve_adjoint(&b).unwrap());
    // `dense` is Hermitian: its real part is symmetric, its imaginary part
    // antisymmetric.
    for a in [
        grid::<S>(6),
        saddle::<S>(),
        dense::<S>(6, 6),
        dense::<S>(1, 1),
    ] {
        assert!(is_hermitian(&a));
        let n = a.rows();
        let b = rhs::<S>(n);
        let one_shot = SparseLdl::new(&a).unwrap();
        // The dense limits of the capacity rule: n(n − 1)/2 factor entries and
        // n(n + 1)/2 active entries.
        let analysis = symmetric_analysis(a.structural_pattern(), n * (n - 1) / 2, n * (n + 1) / 2);
        let scoped = SparseLdl::factor_scoped(
            analysis.try_clone().unwrap(),
            a.view(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let configured = SparseLdl::factor(analysis, a.view()).unwrap();
        same_ldl(&one_shot, &configured);
        same_ldl(&scoped, &configured);
        owned_solves!(one_shot, b);
        let x = one_shot.solve(&b).unwrap();
        assert_eq!(x, configured.solve(&b).unwrap());
        assert_eq!(
            one_shot.solve_transpose(&b).unwrap(),
            configured.solve_transpose(&b).unwrap()
        );
        residual_close(&a, &x, &b, 256.0);
    }
    // The saddle point matrix pivots with 2×2 blocks.
    let ldl = SparseLdl::new(&saddle::<S>()).unwrap();
    assert!(
        ldl.pivot_blocks()
            .any(|block| matches!(block, LdlPivotBlock::Two { .. }))
    );
}

fn is_hermitian<S: NumericScalar>(a: &CsrMatrix<S>) -> bool {
    a.iter()
        .all(|(i, j, v)| a.get(j, i).is_ok_and(|w| w == v.conj()))
}

fn residual_close<S: NumericScalar>(a: &CsrMatrix<S>, x: &[S], b: &[S], multiple: f64) {
    let ax = a.apply(x).unwrap();
    let scale = b
        .iter()
        .fold(S::Real::ONE, |m, v| m.max(S::Real::ONE + v.abs()));
    let factor = S::Real::try_from_f64_lossy(multiple * a.rows() as f64).unwrap();
    let bound = factor * S::Real::EPSILON * scale;
    for (&lhs, &rhs) in ax.iter().zip(b) {
        assert!((lhs - rhs).abs() <= bound);
    }
}

fn qr_contract<S: NumericScalar>() {
    for a in [
        dense::<S>(7, 4),
        dense::<S>(4, 7),
        grid::<S>(4),
        dense::<S>(3, 0),
    ] {
        let (m, n) = (a.rows(), a.cols());
        let k = m.min(n);
        let b = rhs::<S>(m);
        let one_shot = SparseQrMinimumNormPlan::new(&a).unwrap();
        let symbolic = SparseQrSymbolic::analyze(
            a.structural_pattern(),
            SparseOrdering::FillReducing,
            SparseQrLimits::new(k * (m + n), m * n, usize::MAX),
        )
        .unwrap();
        let epsilon = S::Real::EPSILON * S::Real::try_from_f64_lossy(m.max(n) as f64).unwrap();
        let threshold = RankThreshold::new(S::Real::ZERO, epsilon).unwrap();
        let scoped = SparseQr::factor_scoped(
            symbolic.try_clone().unwrap(),
            a.view(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let factor = SparseQr::factor(symbolic, a.view()).unwrap();
        assert!(scoped.r_entries().eq(factor.r_entries()));
        let configured = SparseQrMinimumNormPlan::build(
            factor,
            threshold,
            SparseSolveOptions::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(
            (one_shot.rank(), one_shot.threshold()),
            (configured.rank(), configured.threshold())
        );
        assert!(
            one_shot
                .factor()
                .r_entries()
                .eq(configured.factor().r_entries())
        );
        let (x, report) = one_shot.solve(&b).unwrap();
        let (expected, expected_report) = configured.solve(&b).unwrap();
        assert_eq!(x, expected);
        assert_eq!(report.rank(), expected_report.rank());
        assert_eq!(least_squares(&a, &b).unwrap(), expected);
        let c = rhs::<S>(n);
        assert_eq!(
            one_shot.solve_adjoint(&c).unwrap().0,
            configured.solve_adjoint(&c).unwrap().0
        );
        assert_eq!(
            one_shot.solve_transpose(&c).unwrap().0,
            configured.solve_transpose(&c).unwrap().0
        );
        for (operation, input) in [
            (SparseOperation::Normal, &b),
            (SparseOperation::Transpose, &c),
            (SparseOperation::Adjoint, &c),
        ] {
            let owned = match operation {
                SparseOperation::Normal => one_shot.solve(input),
                SparseOperation::Transpose => one_shot.solve_transpose(input),
                SparseOperation::Adjoint => one_shot.solve_adjoint(input),
            }
            .unwrap();
            let mut output = vec![S::ZERO; owned.0.len()];
            let report = one_shot
                .solve_into(
                    input,
                    operation,
                    SparseSolveOptions::new(),
                    &mut output,
                    &mut one_shot.workspace(1).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!((owned.0, owned.1.rank()), (output, report.rank()));
        }
    }
}

fn incomplete_contract<S: NumericScalar>() {
    let a = grid::<S>(5);
    let b = rhs::<S>(a.rows());
    for (one_shot, method) in [
        (
            IncompleteFactor::ilu0(&a).unwrap(),
            IncompleteMethod::Ilu { level: 0 },
        ),
        (IncompleteFactor::ic0(&a).unwrap(), IncompleteMethod::Ic0),
    ] {
        let symbolic = IncompleteSymbolic::analyze(
            a.structural_pattern(),
            method,
            1,
            IncompleteLimits::new(a.nnz() + a.rows(), usize::MAX),
        )
        .unwrap();
        let scoped = IncompleteFactor::factor_scoped(
            symbolic.try_clone().unwrap(),
            a.view(),
            IncompleteOptions::new(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let configured =
            IncompleteFactor::factor(symbolic, a.view(), IncompleteOptions::new()).unwrap();
        assert!(one_shot.blocks().eq(configured.blocks()));
        assert!(scoped.blocks().eq(configured.blocks()));
        owned_solves!(one_shot, b);
        assert_eq!(one_shot.solve(&b).unwrap(), configured.solve(&b).unwrap());
        assert_eq!(
            one_shot.solve_adjoint(&b).unwrap(),
            configured.solve_adjoint(&b).unwrap()
        );
    }
}

fn amg_contract<S: NumericScalar>() {
    let a = grid::<S>(6);
    let n = a.rows();
    let b = rhs::<S>(n);
    let ones = vec![S::ONE; n];
    let options = SparseSolveOptions::new();
    let hierarchy = AmgHierarchy::build(
        a.view(),
        BlockVectorView::new(&ones, n, 1).unwrap(),
        AmgSetupOptions::new(),
        AmgLimits::default(),
        options,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let configured = AmgPreconditioner::build(
        hierarchy,
        AmgCycleOptions::new(),
        SparseSymmetricLimits::new(n * n, 0, usize::MAX),
        options,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let one_shot = AmgPreconditioner::new(&a).unwrap();
    let mut output = vec![S::ZERO; n];
    configured
        .apply_into(
            &b,
            options,
            &mut output,
            &mut configured.workspace(1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(one_shot.apply(&b).unwrap(), output);
    assert_eq!(configured.apply(&b).unwrap(), output);
}

#[test]
fn one_shot_direct_solvers_equal_their_configured_forms() {
    lu_contract::<f32>();
    lu_contract::<f64>();
    lu_contract::<Complex<f64>>();
    symmetric_contract::<f32>();
    symmetric_contract::<f64>();
    symmetric_contract::<Complex<f64>>();
    qr_contract::<f32>();
    qr_contract::<f64>();
    qr_contract::<Complex<f64>>();
}

#[test]
fn one_shot_preconditioners_equal_their_configured_forms() {
    incomplete_contract::<f32>();
    incomplete_contract::<f64>();
    incomplete_contract::<Complex<f64>>();
    amg_contract::<f32>();
    amg_contract::<f64>();
    amg_contract::<Complex<f64>>();
}

#[test]
fn empty_inputs_have_empty_solutions() {
    let empty = CsrMatrix::<f64>::from_triplets(0, 0, &[], &[], &[]).unwrap();
    assert!(solve(&empty, &[]).unwrap().is_empty());
    assert!(least_squares(&empty, &[]).unwrap().is_empty());
    assert!(
        SparseCholesky::new(&empty)
            .unwrap()
            .solve(&[])
            .unwrap()
            .is_empty()
    );
    assert!(
        SparseLdl::new(&empty)
            .unwrap()
            .solve(&[])
            .unwrap()
            .is_empty()
    );
    let tall = CsrMatrix::<f64>::from_triplets(3, 0, &[], &[], &[]).unwrap();
    assert!(least_squares(&tall, &[1.0, 2.0, 3.0]).unwrap().is_empty());
    let wide = CsrMatrix::<f64>::from_triplets(0, 2, &[], &[], &[]).unwrap();
    assert_eq!(least_squares(&wide, &[]).unwrap(), [0.0, 0.0]);
}

#[test]
fn one_shot_solves_report_each_error_kind() {
    let kind = |result: Result<Vec<f64>, scioxide_core::SciOxideError>| result.unwrap_err().kind();
    let a = csr(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 4.0)]);
    let wide = csr(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]);
    assert_eq!(
        kind(solve(&wide, &[1.0])),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        kind(solve(&a, &[1.0])),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        kind(solve(&a, &[1.0, f64::NAN])),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        kind(least_squares(&a, &[1.0])),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        kind(least_squares(&a, &[f64::INFINITY, 1.0])),
        SciOxideErrorKind::NonFinite
    );
    // Column 1 stores no entry, so it has no pivot candidate.
    let singular = csr(2, 2, &[(0, 0, 1.0), (1, 0, 1.0)]);
    assert_eq!(
        kind(solve(&singular, &[1.0, 1.0])),
        SciOxideErrorKind::Singular
    );
    // [[1, 2], [2, 1]] is indefinite; [[2, 1], [3, 2]] is not Hermitian.
    let indefinite = csr(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 1.0)]);
    let error = SparseCholesky::new(&indefinite).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    let unsymmetric = csr(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 3.0), (1, 1, 2.0)]);
    let error = SparseCholesky::new(&unsymmetric).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(
        SparseLdl::new(&unsymmetric).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        SparseLu::new(&wide).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn rank_deficient_least_squares_is_the_minimum_norm_solution() {
    // Columns 0 and 1 are equal and row 2 is zero: every x with x₀ + x₁ = 2
    // and x₂ = 3 minimizes, and (1, 1, 3) has the least norm.
    let a = csr(
        3,
        3,
        &[
            (0, 0, 1.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 1.0),
            (0, 2, 1.0),
            (2, 2, 0.0),
        ],
    );
    let x = least_squares(&a, &[5.0, 2.0, 0.0]).unwrap();
    // Rows 0 and 1 read x₀ + x₁ + x₂ = 5 and x₀ + x₁ = 2.
    for (value, expected) in x.iter().zip([1.0, 1.0, 3.0]) {
        assert!((value - expected).abs() <= 1e-12, "{x:?}");
    }
    let plan = SparseQrMinimumNormPlan::new(&a).unwrap();
    assert_eq!(plan.rank(), 2);
}
