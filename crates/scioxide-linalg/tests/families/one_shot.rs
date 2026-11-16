//! One-shot functions: accuracy for each scalar type, bitwise equality with
//! the wrapped plans and factorizations, and input errors.
#![allow(clippy::unwrap_used)]

use core::fmt::Debug;

use scioxide_core::{
    Complex, DiagonalKind, ExecutionContext, InvalidInputReason, NumericScalar, RealScalar,
    SciOxideError, SciOxideErrorKind, SciOxideErrorView, Triangle,
};
use scioxide_linalg::*;

/// Converts `value` to the scalar type `S`.
fn scalar<S: NumericScalar>(value: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(value).unwrap())
}

/// Returns `re + i·im` for a complex `S` and `re` for a real one.
fn complex<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(re).unwrap(),
        S::Real::try_from_f64_lossy(im).unwrap(),
    )
}

/// Returns `ω = 1 + 2i` for a complex `S` and `1` for a real one.
///
/// Multiplying the data of a linear problem by `ω` keeps its solution, so one
/// expected value serves every scalar type.
fn weight<S: NumericScalar>() -> S {
    complex(1., 2.)
}

fn weighted<S: NumericScalar>(entries: &[f64], weight: S) -> Vec<S> {
    entries
        .iter()
        .map(|&value| scalar::<S>(value) * weight)
        .collect()
}

fn dense<S: NumericScalar, const C: usize>(rows: &[[f64; C]], weight: S) -> Matrix<S> {
    Matrix::from_vec(rows.len(), C, weighted(rows.as_flattened(), weight)).unwrap()
}

fn column<S: NumericScalar>(entries: &[f64], weight: S) -> Vector<S> {
    Vector::from_vec(weighted(entries, weight))
}

/// Asserts `|cᵢ − eᵢ| ≤ ulps · ε · max(1, |eᵢ|)` for every entry, where `ε` is
/// the machine epsilon of `S`.
fn assert_close<S: NumericScalar + Debug>(computed: &[S], expected: &[S], ulps: f64) {
    assert_eq!(computed.len(), expected.len());
    let epsilon = S::Real::EPSILON.into_f64();
    for (index, (&c, &e)) in computed.iter().zip(expected).enumerate() {
        let error = (c - e).abs().into_f64();
        let bound = ulps * epsilon * e.abs().into_f64().max(1.);
        assert!(
            error <= bound,
            "entry {index}: |{c:?} - {e:?}| = {error:e} > {bound:e}"
        );
    }
}

/// Returns the bit patterns of the real and imaginary parts of each entry.
fn bits<S: NumericScalar>(entries: &[S]) -> Vec<(u64, u64)> {
    entries
        .iter()
        .map(|value| {
            (
                value.real_component().into_f64().to_bits(),
                value.imaginary_component().into_f64().to_bits(),
            )
        })
        .collect()
}

fn error_kind<T: Debug>(result: Result<T, SciOxideError>) -> SciOxideErrorKind {
    result.unwrap_err().kind()
}

fn names_argument(error: &SciOxideError, name: &str) -> bool {
    matches!(
        error.view(),
        SciOxideErrorView::DomainError { argument, .. } if argument.as_str() == name
    )
}

fn solve_case<S: NumericScalar + Debug>(ulps: f64) {
    let w = weight::<S>();
    // 2x + y − z = 8, −3x − y + 2z = −11, −2x + y + 2z = −3 has x = (2, 3, −1).
    let a = dense(&[[2., 1., -1.], [-3., -1., 2.], [-2., 1., 2.]], w);
    let b = column(&[8., -11., -3.], w);
    let x = solve(&a, &b).unwrap();
    assert_close(x.as_slice(), &weighted(&[2., 3., -1.], scalar(1.)), ulps);
    let chain = LuDecomposition::factor(&a).unwrap().solve(&b).unwrap();
    assert_eq!(bits(x.as_slice()), bits(chain.as_slice()));
}

#[test]
fn solve_matches_lu_for_each_scalar_type() {
    solve_case::<f32>(16.);
    solve_case::<f64>(16.);
    solve_case::<Complex<f64>>(16.);
}

#[test]
fn solve_rejects_mismatched_non_finite_and_singular_systems() {
    let a = Matrix::from_rows(&[[4.0_f64, 3.], [6., 3.]]).unwrap();
    let kind = error_kind(solve(&a, &[1., 2., 3.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let tall = Matrix::from_rows(&[[1.0_f64, 2.], [3., 4.], [5., 6.]]).unwrap();
    let kind = error_kind(solve(&tall, &[1., 2., 3.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let nan = Matrix::from_rows(&[[4.0_f64, f64::NAN], [6., 3.]]).unwrap();
    let kind = error_kind(solve(&nan, &[1., 2.][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    let kind = error_kind(solve(&a, &[1., f64::INFINITY][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    let singular = Matrix::from_rows(&[[1.0_f64, 2.], [2., 4.]]).unwrap();
    let kind = error_kind(solve(&singular, &[1., 2.][..]));
    assert_eq!(kind, SciOxideErrorKind::Singular);
}

#[test]
fn solve_handles_orders_zero_and_one() {
    let empty = solve(&Matrix::<f64>::zeros(0, 0).unwrap(), &[0.0_f64; 0][..]).unwrap();
    assert!(empty.is_empty());
    let x = solve(&Matrix::from_rows(&[[4.0_f64]]).unwrap(), &[2.][..]).unwrap();
    assert_eq!(x.as_slice(), &[0.5]);
    let zero = Matrix::from_rows(&[[0.0_f64]]).unwrap();
    let kind = error_kind(solve(&zero, &[1.][..]));
    assert_eq!(kind, SciOxideErrorKind::Singular);
}

fn minimum_norm_chain<S: NumericScalar>(a: &Matrix<S>, b: &Vector<S>) -> Vector<S> {
    HouseholderDecomposition::factor(a, HouseholderOrientation::Qr, HouseholderPivoting::Column)
        .unwrap()
        .into_minimum_norm(&DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
        .unwrap()
        .solve(b)
        .unwrap()
        .into_accepted()
        .unwrap()
}

fn least_squares_case<S: NumericScalar + Debug>(ulps: f64) {
    let w = weight::<S>();
    let one = scalar::<S>(1.);
    let cases = [
        // Tall, full rank: c₀ + c₁ t through (0, 1), (1, 3), (2, 5) is c = (1, 2).
        (
            dense(&[[1., 0.], [1., 1.], [1., 2.]], w),
            column(&[1., 3., 5.], w),
            weighted(&[1., 2.], one),
        ),
        // Wide: the minimum-norm solution of x₁ + x₂ = 2 is (1, 1).
        (
            dense(&[[1., 1.]], w),
            column(&[2.], w),
            weighted(&[1., 1.], one),
        ),
        // Rank one: both rows are x₁ + x₂ = 2.
        (
            dense(&[[1., 1.], [1., 1.]], w),
            column(&[2., 2.], w),
            weighted(&[1., 1.], one),
        ),
        // Inconsistent: x = 1 and x = 3 have the least-squares solution 2.
        (
            dense(&[[1.], [1.]], w),
            column(&[1., 3.], w),
            vec![scalar(2.)],
        ),
    ];
    for (a, b, expected) in cases {
        let x = least_squares(&a, &b).unwrap();
        assert_close(x.as_slice(), &expected, ulps);
        assert_eq!(
            bits(x.as_slice()),
            bits(minimum_norm_chain(&a, &b).as_slice())
        );
    }
}

#[test]
fn least_squares_matches_the_minimum_norm_plan_for_each_scalar_type() {
    least_squares_case::<f32>(16.);
    least_squares_case::<f64>(16.);
    least_squares_case::<Complex<f64>>(16.);
}

#[test]
fn least_squares_rejects_mismatched_and_non_finite_data() {
    let a = Matrix::from_rows(&[[1.0_f64, 0.], [1., 1.], [1., 2.]]).unwrap();
    let kind = error_kind(least_squares(&a, &[1., 3.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let kind = error_kind(least_squares(&a, &[1., f64::NAN, 5.][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    let infinite = Matrix::from_rows(&[[1.0_f64, 0.], [f64::INFINITY, 1.], [1., 2.]]).unwrap();
    let kind = error_kind(least_squares(&infinite, &[1., 3., 5.][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
}

#[test]
fn least_squares_of_a_zero_matrix_is_the_zero_vector() {
    let x = least_squares(&Matrix::<f64>::zeros(3, 2).unwrap(), &[1., 2., 3.][..]).unwrap();
    assert_eq!(x.as_slice(), &[0., 0.]);
}

fn triangular_case<S: NumericScalar + Debug>(ulps: f64) {
    let w = weight::<S>();
    // The entry outside the triangle is never read. Both systems have x = (1, 2).
    let cases = [
        (Triangle::Lower, [[2., 7.], [1., 4.]], [2., 9.]),
        (Triangle::Upper, [[2., 2.], [-5., 4.]], [6., 8.]),
    ];
    for (triangle, rows, rhs) in cases {
        let a = dense(&rows, w);
        let b = column(&rhs, w);
        let x = solve_triangular(&a, triangle, &b).unwrap();
        assert_close(x.as_slice(), &weighted(&[1., 2.], scalar(1.)), ulps);
        let chain = TriangularPlan::from_dense(&a, triangle, DiagonalKind::Explicit)
            .unwrap()
            .solve(StructuredOperation::Normal, &b)
            .unwrap();
        assert_eq!(bits(x.as_slice()), bits(chain.as_slice()));
    }
}

#[test]
fn solve_triangular_matches_the_triangular_plan_for_each_scalar_type() {
    triangular_case::<f32>(4.);
    triangular_case::<f64>(4.);
    triangular_case::<Complex<f64>>(4.);
}

#[test]
fn solve_triangular_reads_only_its_triangle() {
    let a = Matrix::from_rows(&[[2.0_f64, f64::NAN], [1., 4.]]).unwrap();
    let x = solve_triangular(&a, Triangle::Lower, &[2., 9.][..]).unwrap();
    assert_close(x.as_slice(), &[1., 2.], 4.);
    let kind = error_kind(solve_triangular(&a, Triangle::Upper, &[2., 9.][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
}

#[test]
fn solve_triangular_rejects_mismatched_non_finite_and_singular_systems() {
    let a = Matrix::from_rows(&[[2.0_f64, 0.], [1., 4.]]).unwrap();
    let kind = error_kind(solve_triangular(&a, Triangle::Lower, &[2.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let wide = Matrix::from_rows(&[[2.0_f64, 0., 0.], [1., 4., 0.]]).unwrap();
    let kind = error_kind(solve_triangular(&wide, Triangle::Lower, &[2., 9.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let kind = error_kind(solve_triangular(&a, Triangle::Lower, &[f64::NAN, 9.][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    let zero_diagonal = Matrix::from_rows(&[[2.0_f64, 1.], [1., 0.]]).unwrap();
    for triangle in [Triangle::Lower, Triangle::Upper] {
        let kind = error_kind(solve_triangular(&zero_diagonal, triangle, &[2., 9.][..]));
        assert_eq!(kind, SciOxideErrorKind::Singular);
    }
}

fn structured_chain<S: NumericScalar>(
    matrix: StructuredMatrixView<'_, S>,
    b: &Vector<S>,
) -> Vector<S> {
    let plan = StructuredSolvePlan::new(matrix, &StructuredSolveOptions::new()).unwrap();
    let mut x = Vector::zeros(plan.order()).unwrap();
    plan.solve_into(
        StructuredOperation::Normal,
        b,
        x.view_mut(),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )
    .unwrap()
    .termination()
    .require_accepted()
    .unwrap();
    x
}

fn structured_case<S: NumericScalar + Debug>(ulps: f64) {
    let (w, one) = (weight::<S>(), scalar::<S>(1.));
    let toeplitz = (weighted(&[4., 1., -1.], w), weighted(&[4., 2., 0.5], w));
    let hermitian = (
        vec![scalar(4.), complex(1., 1.), complex(0.5, -0.5)],
        vec![scalar(4.), complex(1., -1.), complex(0.5, 0.5)],
    );
    let hankel = (weighted(&[1., 2., 4.], w), weighted(&[4., 0., 1.], w));
    let circulant = weighted(&[2., 1., 0.], w);
    // With x = (1, 1, 1), b holds the row sums of each matrix. The Hermitian
    // Toeplitz matrix takes the Levinson–Durbin path.
    let cases = [
        (
            StructuredMatrixView::toeplitz(&toeplitz.0, &toeplitz.1).unwrap(),
            weighted(&[6.5, 7., 4.], w),
        ),
        (
            StructuredMatrixView::toeplitz(&hermitian.0, &hermitian.1).unwrap(),
            vec![complex(5.5, -0.5), scalar(6.), complex(5.5, 0.5)],
        ),
        (
            StructuredMatrixView::hankel(&hankel.0, &hankel.1).unwrap(),
            weighted(&[7., 6., 5.], w),
        ),
        (
            StructuredMatrixView::circulant(&circulant).unwrap(),
            weighted(&[3., 3., 3.], w),
        ),
    ];
    for (matrix, rhs) in cases {
        let b = Vector::from_vec(rhs);
        let x = solve_structured(matrix, &b).unwrap();
        assert_close(x.as_slice(), &[one; 3], ulps);
        assert_eq!(
            bits(x.as_slice()),
            bits(structured_chain(matrix, &b).as_slice())
        );
    }
}

#[test]
fn solve_structured_matches_the_structured_plan_for_each_scalar_type() {
    structured_case::<f32>(32.);
    structured_case::<f64>(32.);
    structured_case::<Complex<f64>>(32.);
}

#[test]
fn solve_structured_rejects_matrices_without_a_structured_solver() {
    let hilbert = StructuredMatrixView::<f64>::hilbert(3, MatrixForm::Direct).unwrap();
    let error = solve_structured(hilbert, &[1., 1., 1.][..]).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { argument, reason, .. }
            if argument.as_str() == "matrix" && reason == InvalidInputReason::UnsupportedConfiguration
    ));
}

#[test]
fn solve_structured_rejects_mismatched_non_finite_and_singular_systems() {
    let (column, row) = ([4.0_f64, 1., -1.], [4.0_f64, 2., 0.5]);
    let toeplitz = StructuredMatrixView::toeplitz(&column, &row).unwrap();
    let kind = error_kind(solve_structured(toeplitz, &[6.5, 7.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let rectangular = StructuredMatrixView::toeplitz(&column, &row[..2]).unwrap();
    let kind = error_kind(solve_structured(rectangular, &[1., 1., 1.][..]));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let kind = error_kind(solve_structured(toeplitz, &[6.5, f64::NAN, 4.][..]));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    // The circulant with first column (1, −1) has the eigenvalue 1 + (−1) = 0.
    let circulant = StructuredMatrixView::circulant(&[1.0_f64, -1.]).unwrap();
    let kind = error_kind(solve_structured(circulant, &[1., 1.][..]));
    assert_eq!(kind, SciOxideErrorKind::Singular);
    let zero = StructuredMatrixView::toeplitz(&[0.0_f64], &[0.]).unwrap();
    let kind = error_kind(solve_structured(zero, &[1.][..]));
    assert_eq!(kind, SciOxideErrorKind::Singular);
}

fn evaluate_chain<S: NumericScalar>(
    a: &Matrix<S>,
    function: MatrixFunction<'_, S::Real>,
) -> Result<Matrix<S>, SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let plan = MatrixFunctionPlan::new(a, function, &MatrixFunctionOptions::new())?;
    let mut workspace = plan.workspace(&MatrixFunctionWorkspaceRequest::Evaluate)?;
    let n = plan.order();
    let mut output = Matrix::zeros(n, n)?;
    plan.evaluate_into(
        output.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

fn matrix_function_case<S: NumericScalar + Debug>(ulps: f64)
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let (w, zero) = (weight::<S>(), scalar::<S>(0.));
    let (one, fifth) = (scalar::<S>(1.), scalar::<S>(0.2));
    let triangular = |first: f64, last: f64| {
        Matrix::from_vec(2, 2, vec![scalar(first), w, zero, scalar(last)]).unwrap()
    };
    // N = [[0, ω], [0, 0]] is nilpotent, so exp(N) = I + N. The principal
    // square root of [[4, ω], [0, 9]] is [[2, ω/5], [0, 3]].
    let cases = [
        (
            MatrixFunction::Exp,
            triangular(0., 0.),
            vec![one, w, zero, one],
        ),
        (
            MatrixFunction::Sqrt,
            triangular(4., 9.),
            vec![scalar(2.), w * fifth, zero, scalar(3.)],
        ),
    ];
    for (function, a, expected) in cases {
        let value = matrix_function(&a, function).unwrap();
        assert_close(value.as_slice(), &expected, ulps);
        let chain = evaluate_chain(&a, function).unwrap();
        assert_eq!(bits(value.as_slice()), bits(chain.as_slice()));
    }
}

#[test]
fn matrix_function_matches_the_function_plan_for_each_scalar_type() {
    matrix_function_case::<f32>(64.);
    matrix_function_case::<f64>(64.);
    matrix_function_case::<Complex<f64>>(64.);
}

#[test]
fn matrix_function_names_the_matrix_in_domain_errors() {
    let negative = Matrix::from_rows(&[[-1.0_f64]]).unwrap();
    let error = matrix_function(&negative, MatrixFunction::Sqrt).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
    assert!(names_argument(&error, "a"), "{error:?}");
    // The plan reports the complex result on its real output.
    let error = evaluate_chain(&negative, MatrixFunction::Sqrt).unwrap_err();
    assert!(names_argument(&error, "output"), "{error:?}");
    // log(0) is undefined.
    let zero = Matrix::<f64>::zeros(1, 1).unwrap();
    let error = matrix_function(&zero, MatrixFunction::Log).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
    assert!(names_argument(&error, "a"), "{error:?}");
}

#[test]
fn matrix_function_of_a_complex_negative_matrix_is_complex() {
    let negative = Matrix::from_rows(&[[Complex::new(-1.0_f64, 0.)]]).unwrap();
    let root = matrix_function(&negative, MatrixFunction::Sqrt).unwrap();
    assert_close(root.as_slice(), &[Complex::new(0., 1.)], 4.);
}

#[test]
fn matrix_function_rejects_mismatched_non_finite_singular_and_invalid_input() {
    let wide = Matrix::<f64>::zeros(1, 2).unwrap();
    let kind = error_kind(matrix_function(&wide, MatrixFunction::Exp));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let nan = Matrix::from_rows(&[[1.0_f64, f64::NAN], [0., 1.]]).unwrap();
    let kind = error_kind(matrix_function(&nan, MatrixFunction::Exp));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    let zero = Matrix::<f64>::zeros(2, 2).unwrap();
    let kind = error_kind(matrix_function(&zero, MatrixFunction::Power(-1.)));
    assert_eq!(kind, SciOxideErrorKind::Singular);
    let identity = Matrix::<f64>::identity(2).unwrap();
    let error = matrix_function(&identity, MatrixFunction::Power(f64::NAN)).unwrap_err();
    assert!(names_argument(&error, "function"), "{error:?}");
}

fn sylvester_case<S: NumericScalar + Debug>(ulps: f64) {
    let w = weight::<S>();
    let a = dense(&[[3., 1.], [0., 2.]], w);
    let b = dense(&[[1., 0.], [1., 4.]], w);
    // X = [[1, 1], [1, 1]]: A X = [[4, 4], [2, 2]] and X B = [[2, 4], [2, 4]].
    let cases = [
        (SylvesterSign::Plus, [[6., 8.], [4., 6.]]),
        (SylvesterSign::Minus, [[2., 0.], [0., -2.]]),
    ];
    for (sign, rhs) in cases {
        let c = dense(&rhs, w);
        let x = solve_sylvester(&a, &b, sign, &c).unwrap();
        assert_close(x.as_slice(), &[scalar(1.); 4], ulps);
        let chain = SylvesterPlan::new(&a, &b, sign, MatrixEquationOptions::new())
            .unwrap()
            .solve(&c)
            .unwrap()
            .into_accepted()
            .unwrap();
        assert_eq!(bits(x.as_slice()), bits(chain.as_slice()));
    }
}

#[test]
fn solve_sylvester_matches_the_sylvester_plan_for_each_scalar_type() {
    sylvester_case::<f32>(32.);
    sylvester_case::<f64>(32.);
    sylvester_case::<Complex<f64>>(32.);
}

#[test]
fn solve_sylvester_rejects_mismatched_non_finite_and_singular_equations() {
    let a = Matrix::from_rows(&[[3.0_f64, 1.], [0., 2.]]).unwrap();
    let b = Matrix::from_rows(&[[1.0_f64]]).unwrap();
    let row = Matrix::from_rows(&[[5.0_f64, 3.]]).unwrap();
    let kind = error_kind(solve_sylvester(&a, &b, SylvesterSign::Plus, &row));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let kind = error_kind(solve_sylvester(&row, &b, SylvesterSign::Plus, &row));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let nan = Matrix::from_rows(&[[5.0_f64], [f64::NAN]]).unwrap();
    let kind = error_kind(solve_sylvester(&a, &b, SylvesterSign::Plus, &nan));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    // With A = B = 1, A X − X B = 0 for every X.
    let kind = error_kind(solve_sylvester(&b, &b, SylvesterSign::Minus, &b));
    assert_eq!(kind, SciOxideErrorKind::Singular);
}

fn lyapunov_case<S: NumericScalar + Debug>(ulps: f64) {
    let (zero, one) = (scalar::<S>(0.), scalar::<S>(1.));
    let x = weighted(&[1., 1., 1., 2.], one);
    // For A = diag(a₁, a₂) and Hermitian X, the right-hand side is
    // qᵢⱼ = xᵢⱼ (aᵢ + conj(aⱼ)) (continuous) or qᵢⱼ = xᵢⱼ (1 − aᵢ conj(aⱼ))
    // (discrete); all products are exact.
    let cases = [
        (
            LyapunovKind::Continuous,
            [complex(1., 1.), complex(2., -1.)],
        ),
        (
            LyapunovKind::Discrete,
            [complex(0.5, 0.25), complex(-0.25, 0.5)],
        ),
    ];
    for (equation, diagonal) in cases {
        let a = Matrix::from_vec(2, 2, vec![diagonal[0], zero, zero, diagonal[1]]).unwrap();
        let entries = (0..4)
            .map(|k| {
                let (left, right) = (diagonal[k / 2], diagonal[k % 2].conj());
                x[k] * match equation {
                    LyapunovKind::Continuous => left + right,
                    LyapunovKind::Discrete => one - left * right,
                }
            })
            .collect();
        let q = Matrix::from_vec(2, 2, entries).unwrap();
        let solution = solve_lyapunov(&a, equation, &q).unwrap();
        assert_close(solution.as_slice(), &x, ulps);
        let chain = LyapunovPlan::new(&a, equation, MatrixEquationOptions::new())
            .unwrap()
            .solve(&q)
            .unwrap()
            .into_accepted()
            .unwrap();
        assert_eq!(bits(solution.as_slice()), bits(chain.as_slice()));
    }
}

#[test]
fn solve_lyapunov_matches_the_lyapunov_plan_for_each_scalar_type() {
    lyapunov_case::<f32>(32.);
    lyapunov_case::<f64>(32.);
    lyapunov_case::<Complex<f64>>(32.);
}

#[test]
fn solve_lyapunov_rejects_mismatched_non_finite_and_singular_equations() {
    let a = Matrix::from_rows(&[[1.0_f64, 0.], [0., 2.]]).unwrap();
    let small = Matrix::from_rows(&[[2.0_f64]]).unwrap();
    let kind = error_kind(solve_lyapunov(&a, LyapunovKind::Continuous, &small));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let wide = Matrix::from_rows(&[[1.0_f64, 0.]]).unwrap();
    let kind = error_kind(solve_lyapunov(&wide, LyapunovKind::Continuous, &small));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let nan = Matrix::from_rows(&[[2.0_f64, 3.], [3., f64::NAN]]).unwrap();
    let kind = error_kind(solve_lyapunov(&a, LyapunovKind::Continuous, &nan));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    // A = 0 makes A X + X Aᴴ = 0, and A = 1 makes A X Aᴴ − X = 0, for every X.
    let zero = Matrix::from_rows(&[[0.0_f64]]).unwrap();
    let kind = error_kind(solve_lyapunov(&zero, LyapunovKind::Continuous, &small));
    assert_eq!(kind, SciOxideErrorKind::Singular);
    let one = Matrix::from_rows(&[[1.0_f64]]).unwrap();
    let kind = error_kind(solve_lyapunov(&one, LyapunovKind::Discrete, &small));
    assert_eq!(kind, SciOxideErrorKind::Singular);
}

fn riccati_chain<S: NumericScalar>(
    problem: RiccatiProblem<'_, S>,
    kind: RiccatiKind,
) -> (Matrix<S>, Matrix<S>) {
    let options = RiccatiOptions::new().with_closed_loop_verification(Estimation::Estimate);
    let (solution, report) = RiccatiPlan::new(problem, kind, options)
        .unwrap()
        .solve()
        .unwrap()
        .into_parts();
    assert_eq!(report.termination(), DirectSolveTermination::Accepted);
    solution.unwrap()
}

fn riccati_case<S: NumericScalar + Debug>(ulps: f64) {
    let (zero, one) = (scalar::<S>(0.), scalar::<S>(1.));
    let square = |n: usize, entries: Vec<S>| Matrix::from_vec(n, n, entries).unwrap();
    let eye = vec![one, zero, zero, one];
    let cases = [
        // CARE 2 Re(a) x − x² + 3 = 0 with Re(a) = −1: x = 1 and K = b x / r = 1.
        (
            RiccatiKind::Continuous,
            square(1, vec![complex(-1., 1.)]),
            square(1, vec![one]),
            square(1, vec![scalar(3.)]),
            square(1, vec![one]),
            vec![one],
            vec![one],
        ),
        // DARE a² x − x − a² x² / (1 + x) + 7/8 = 0 with a = ½: x = 1 and
        // K = (1 + x)⁻¹ x a = ¼.
        (
            RiccatiKind::Discrete,
            square(1, vec![scalar(0.5)]),
            square(1, vec![one]),
            square(1, vec![scalar(0.875)]),
            square(1, vec![one]),
            vec![one],
            vec![scalar(0.25)],
        ),
        // Two decoupled CAREs with Re(aᵢ) = −1, 0 and qᵢ = 3, 1: X = K = I.
        (
            RiccatiKind::Continuous,
            square(2, vec![complex(-1., 1.), zero, zero, complex(0., 2.)]),
            square(2, eye.clone()),
            square(2, vec![scalar(3.), zero, zero, one]),
            square(2, eye.clone()),
            eye.clone(),
            eye,
        ),
    ];
    for (kind, a, b, q, r, x_expected, k_expected) in cases {
        let problem = RiccatiProblem {
            a: a.view(),
            b: b.view(),
            q: q.view(),
            r: r.view(),
            e: None,
            s: None,
        };
        let (x, k) = solve_riccati(problem, kind).unwrap();
        assert_close(x.as_slice(), &x_expected, ulps);
        assert_close(k.as_slice(), &k_expected, ulps);
        let (x_plan, k_plan) = riccati_chain(problem, kind);
        assert_eq!(bits(x.as_slice()), bits(x_plan.as_slice()));
        assert_eq!(bits(k.as_slice()), bits(k_plan.as_slice()));
    }
}

#[test]
fn solve_riccati_matches_the_riccati_plan_for_each_scalar_type() {
    riccati_case::<f32>(64.);
    riccati_case::<f64>(64.);
    riccati_case::<Complex<f64>>(64.);
}

#[test]
fn solve_riccati_rejects_mismatched_non_finite_and_singular_problems() {
    let a = Matrix::from_rows(&[[-1.0_f64]]).unwrap();
    let one = Matrix::from_rows(&[[1.0_f64]]).unwrap();
    let q = Matrix::from_rows(&[[3.0_f64]]).unwrap();
    let problem = RiccatiProblem {
        a: a.view(),
        b: one.view(),
        q: q.view(),
        r: one.view(),
        e: None,
        s: None,
    };
    let tall = Matrix::from_rows(&[[1.0_f64], [1.]]).unwrap();
    let mismatched = RiccatiProblem {
        b: tall.view(),
        ..problem
    };
    let kind = error_kind(solve_riccati(mismatched, RiccatiKind::Continuous));
    assert_eq!(kind, SciOxideErrorKind::DimensionMismatch);
    let nan = Matrix::from_rows(&[[f64::NAN]]).unwrap();
    let non_finite = RiccatiProblem {
        q: nan.view(),
        ..problem
    };
    let kind = error_kind(solve_riccati(non_finite, RiccatiKind::Continuous));
    assert_eq!(kind, SciOxideErrorKind::NonFinite);
    let zero = Matrix::from_rows(&[[0.0_f64]]).unwrap();
    let singular = RiccatiProblem {
        e: Some(zero.view()),
        ..problem
    };
    let kind = error_kind(solve_riccati(singular, RiccatiKind::Continuous));
    assert_eq!(kind, SciOxideErrorKind::Singular);
}
