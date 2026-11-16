//! One-shot functions for the primary entry points of the crate.
//!
//! Each function takes the problem data and nothing else, runs on
//! [`ExecutionContext::sequential_native`] with the default options of the
//! wrapped plan or factorization, allocates its result and scratch, and turns a
//! rejected solve into an error (see [Execution](crate#execution)).

use scioxide_core::{
    Complex, DiagonalKind, ExecutionContext, NumericScalar, SciOxideError, SciOxideErrorView,
    Triangle,
};

use crate::{
    DirectSolveOptions, HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting,
    LinalgWorkspace, LuDecomposition, LyapunovKind, LyapunovPlan, Matrix, MatrixEquationOptions,
    MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan, MatrixFunctionWorkspaceRequest,
    MatrixView, RankPolicy, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem,
    StructuredMatrixView, StructuredOperation, StructuredSolveOptions, StructuredSolvePlan,
    SylvesterPlan, SylvesterSign, TriangularPlan, Vector, VectorView,
};

/// Solves the square linear system `A x = b`.
///
/// The method is LU factorization with partial pivoting, `P A = L U`, followed
/// by forward and back substitution; it costs about `2n³/3` flops for an
/// `n × n` matrix. The solve uses [`RankPolicy::Exact`], so only an exactly
/// zero pivot rejects it. The computed solution is the exact solution of
/// `(A + ΔA) x = b` with `‖ΔA‖∞ ≤ c n ε ‖|L| |U|‖∞` for a modest constant `c`,
/// where `ε` is the machine epsilon of the scalar type.
///
/// To solve several right-hand sides with one factorization, or to estimate
/// the condition number, use [`LuDecomposition::factor`] and its methods.
///
/// ```
/// use scioxide_linalg::{Matrix, Vector, solve};
/// // 2x + y − z = 8, −3x − y + 2z = −11, −2x + y + 2z = −3.
/// let a = Matrix::from_rows(&[[2.0_f64, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])?;
/// let x = solve(&a, &Vector::from_vec(vec![8.0, -11.0, -3.0]))?;
/// for (computed, exact) in x.as_slice().iter().zip([2.0, 3.0, -1.0]) {
///     assert!((computed - exact).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `a` is not square or `b` does not have length `n`.
/// - `NonFinite` if `a` or `b` contains NaN or infinity.
/// - `Singular` if a pivot is exactly zero.
/// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
/// - `AllocationFailure` if the factor or the solution cannot be allocated.
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §3.2–3.4.
pub fn solve<'a, 'b, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<VectorView<'b, S>>,
) -> Result<Vector<S>, SciOxideError> {
    LuDecomposition::factor(a)?.solve(b)
}

/// Returns the minimum-norm least-squares solution of `A x ≈ b`.
///
/// The result is the `x` of smallest 2-norm among the minimizers of
/// `‖A_r x − b‖₂`, where `A_r` is `A` truncated to its numerical rank `r`. `A`
/// may be tall, square or wide. The method is Householder QR with column
/// pivoting, `A P = Q R`. The numerical rank is the number of diagonal entries
/// of `R` above `τ = min(max(m, n) ε, 1/2) · max |rᵢᵢ|`
/// ([`RankPolicy::DimensionScaledEpsilon`]). A rank-deficient `R` is reduced by a complete
/// orthogonal decomposition before the solve. The cost is about
/// `2mn² − 2n³/3` flops for `m ≥ n`.
///
/// For another rank policy or a condition threshold, use
/// [`HouseholderDecomposition::into_minimum_norm`].
///
/// ```
/// use scioxide_linalg::{Matrix, least_squares};
/// // Fit c₀ + c₁ t to (0, 1), (1, 3), (2, 5): the exact fit is c = (1, 2).
/// let a = Matrix::from_rows(&[[1.0_f64, 0.0], [1.0, 1.0], [1.0, 2.0]])?;
/// let c = least_squares(&a, &[1.0, 3.0, 5.0][..])?;
/// assert!((c.as_slice()[0] - 1.0).abs() < 1e-12 && (c.as_slice()[1] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `b` does not have length `m`.
/// - `NonFinite` if `a` or `b` contains NaN or infinity.
/// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
/// - `AllocationFailure` if the factor or the solution cannot be allocated.
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §5.4–5.5.
pub fn least_squares<'a, 'b, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<VectorView<'b, S>>,
) -> Result<Vector<S>, SciOxideError> {
    HouseholderDecomposition::factor(a, HouseholderOrientation::Qr, HouseholderPivoting::Column)?
        .into_minimum_norm(&DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))?
        .solve(b)?
        .into_accepted()
}

/// Solves the triangular system `T x = b`, where `T` is the `triangle` of `a`.
///
/// Only the entries of the selected triangle, including the diagonal, are
/// read. The method is forward substitution for [`Triangle::Lower`] and back
/// substitution for [`Triangle::Upper`], `n²` flops for an `n × n` matrix. The
/// solve uses [`RankPolicy::Exact`], so only an exactly zero diagonal entry
/// rejects it. Triangular substitution is backward stable: the computed `x`
/// solves `(T + ΔT) x = b` with `|ΔT| ≤ γₙ |T|`, `γₙ = n u / (1 − n u)` and `u`
/// the unit roundoff.
///
/// For a unit diagonal, a banded or packed triangle, a transposed system or
/// repeated solves, use [`TriangularPlan`].
///
/// ```
/// use scioxide_core::Triangle;
/// use scioxide_linalg::{Matrix, solve_triangular};
/// // [2 0; 1 4] x = [2, 9] has the solution x = (1, 2).
/// let a = Matrix::from_rows(&[[2.0_f64, 0.0], [1.0, 4.0]])?;
/// let x = solve_triangular(&a, Triangle::Lower, &[2.0, 9.0][..])?;
/// assert!((x.as_slice()[0] - 1.0).abs() < 1e-15 && (x.as_slice()[1] - 2.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `a` is not square or `b` does not have length `n`.
/// - `NonFinite` if the triangle of `a` or `b` contains NaN or infinity.
/// - `Singular` if a diagonal entry is exactly zero.
/// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
///
/// # References
///
/// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
/// SIAM, 2002, chapter 8.
pub fn solve_triangular<'a, 'b, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    triangle: Triangle,
    b: impl Into<VectorView<'b, S>>,
) -> Result<Vector<S>, SciOxideError> {
    TriangularPlan::from_dense(a, triangle, DiagonalKind::Explicit)?
        .solve(StructuredOperation::Normal, b)
}

/// Solves `A x = b` for a square Toeplitz, Hankel or circulant matrix `A`.
///
/// The method follows the structure of `matrix`: Levinson–Durbin recursion for
/// a Hermitian Toeplitz matrix, Gohberg–Kailath–Olshevsky elimination with
/// partial pivoting for other Toeplitz and Hankel matrices, and diagonalization
/// by the discrete Fourier transform for a circulant. A Toeplitz or Hankel
/// solve costs `O(n²)` operations and needs `O(n)` storage. No rank policy
/// applies: the solution is accepted when its normwise backward error is at
/// most `2√ε`, the default tolerance of [`StructuredSolveOptions`].
///
/// For several right-hand sides, other tolerances or the dense LU fallback,
/// use [`StructuredSolvePlan`].
///
/// ```
/// use scioxide_linalg::{StructuredMatrixView, solve_structured};
/// // Toeplitz A = [4 2 0.5; 1 4 2; −1 1 4] with A [1, 1, 1]ᵀ = [6.5, 7, 4]ᵀ.
/// let matrix = StructuredMatrixView::toeplitz(&[4.0_f64, 1.0, -1.0], &[4.0, 2.0, 0.5])?;
/// let x = solve_structured(matrix, &[6.5, 7.0, 4.0][..])?;
/// assert!(x.as_slice().iter().all(|value| (value - 1.0).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `matrix` is not square or `b` does not have
///   length `n`.
/// - `InvalidInput` (`UnsupportedConfiguration`, argument `matrix`) if
///   `matrix` is not a Toeplitz, Hankel or circulant matrix.
/// - `NonFinite` if the entries of `matrix` or `b` contain NaN or infinity.
/// - `Singular` if a circulant eigenvalue, a pivot of the
///   Gohberg–Kailath–Olshevsky elimination or the entry of a `1 × 1` matrix is
///   exactly zero.
/// - `NumericalInstability` (`LossOfSignificance`) if the backward error
///   exceeds the tolerance, or `NumericalInstability` if no structured method
///   produced a finite solution.
/// - `AllocationFailure` if the plan, the workspace or the solution cannot be
///   allocated.
///
/// # References
///
/// - N. Levinson, "The Wiener RMS (root mean square) error criterion in filter
///   design and prediction", *J. Math. Phys.* 25 (1947), 261–278.
/// - I. Gohberg, T. Kailath and V. Olshevsky, "Fast Gaussian elimination with
///   partial pivoting for matrices with displacement structure", *Math. Comp.*
///   64 (1995), 1557–1576.
pub fn solve_structured<'b, T: NumericScalar>(
    matrix: StructuredMatrixView<'_, T>,
    b: impl Into<VectorView<'b, T>>,
) -> Result<Vector<T>, SciOxideError> {
    let plan = StructuredSolvePlan::new(matrix, &StructuredSolveOptions::new())?;
    let mut x = Vector::zeros(plan.order())?;
    plan.solve_into(
        StructuredOperation::Normal,
        b,
        x.view_mut(),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?
    .termination()
    .require_accepted()?;
    Ok(x)
}

/// Computes the matrix function `f(A)` of a square matrix.
///
/// `function` selects `f`, for example [`MatrixFunction::Exp`] for `exp(A)`.
/// The result has the scalar type of `a`. `exp` uses scaling and squaring with
/// Padé approximants; the other functions start from the Schur form
/// `A = Q T Qᴴ` and evaluate `f(T)` by the Schur–Parlett recurrence or a
/// function-specific triangular method. The cost is `O(n³)` flops. A real
/// matrix whose `f(A)` is complex, such as the square root of a matrix with a
/// negative eigenvalue, is a `DomainError`; evaluate a complex view instead or
/// use [`MatrixFunctionPlan`] with complex output.
///
/// ```
/// use scioxide_linalg::{Matrix, MatrixFunction, matrix_function};
/// // exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]].
/// let a = Matrix::from_rows(&[[0.0_f64, 1.0], [0.0, 0.0]])?;
/// let exp = matrix_function(&a, MatrixFunction::Exp)?;
/// for (computed, exact) in exp.as_slice().iter().zip([1.0, 1.0, 0.0, 1.0]) {
///     assert!((computed - exact).abs() < 1e-14);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `a` is not square.
/// - `NonFinite` if `a` contains NaN or infinity.
/// - `DomainError` (argument `a`) if `f(A)` is complex for a real `a`, or an
///   eigenvalue of `A` lies where `f` is not defined, such as `(−∞, 0]` for
///   [`MatrixFunction::Log`].
/// - `DomainError` (argument `function`) if the parameters of `function` are
///   invalid.
/// - `Singular` if [`MatrixFunction::Tan`], [`MatrixFunction::Tanh`] or a
///   negative integer [`MatrixFunction::Power`] meets an exactly singular
///   factor.
/// - `ConvergenceFailure` if the Schur reduction or a series does not
///   converge within its limits.
/// - `NumericalInstability` if `f(A)` overflows.
/// - `AllocationFailure` if the plan, the workspace or the result cannot be
///   allocated.
///
/// # References
///
/// N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.
pub fn matrix_function<'a, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    function: MatrixFunction<'_, S::Real>,
) -> Result<Matrix<S>, SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let evaluate = || {
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
    };
    evaluate().map_err(matrix_domain)
}

/// Names the matrix in the domain errors that a plan reports on its output or
/// on itself.
fn matrix_domain(error: SciOxideError) -> SciOxideError {
    match error.view() {
        SciOxideErrorView::DomainError {
            argument, reason, ..
        } if argument.as_str() == "output" => SciOxideError::domain_error(
            scioxide_core::argument!("a"),
            reason,
            "function has a complex result for this real matrix",
        ),
        SciOxideErrorView::DomainError {
            argument, reason, ..
        } if argument.as_str() == "self" => SciOxideError::domain_error(
            scioxide_core::argument!("a"),
            reason,
            "matrix function is not defined for this matrix",
        ),
        _ => error,
    }
}

/// Solves the Sylvester equation `A X + s X B = C`.
///
/// `A` is `m × m`, `B` is `n × n`, `C` and `X` are `m × n`, and `s = ±1` is
/// given by `sign`. The solution is unique if and only if no eigenvalue `λ` of
/// `A` and `μ` of `B` satisfy `λ + s μ = 0`. The method is Bartels–Stewart:
/// Schur forms of `A` and `B`, a quasi-triangular solve and a residual check of
/// the original equation, in `O(m³ + n³)` flops. No rank policy applies.
///
/// For several right-hand sides with the same coefficients, use
/// [`SylvesterPlan`].
///
/// ```
/// use scioxide_linalg::{Matrix, SylvesterSign, solve_sylvester};
/// // (A + I) x = c is [[4, 1], [0, 3]] x = [5, 3], so x = [1, 1].
/// let a = Matrix::from_rows(&[[3.0_f64, 1.0], [0.0, 2.0]])?;
/// let b = Matrix::from_rows(&[[1.0_f64]])?;
/// let c = Matrix::from_rows(&[[5.0_f64], [3.0]])?;
/// let x = solve_sylvester(&a, &b, SylvesterSign::Plus, &c)?;
/// assert!(x.as_slice().iter().all(|value| (value - 1.0).abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - The errors of [`SylvesterPlan::new`] and [`SylvesterPlan::solve`].
/// - `NumericalInstability` (`LossOfSignificance`) if the residual of the
///   computed `X` exceeds the tolerance of [`MatrixEquationOptions`].
///
/// # References
///
/// R. H. Bartels and G. W. Stewart, "Algorithm 432: Solution of the matrix
/// equation AX + XB = C", *Communications of the ACM* 15(9), 1972.
pub fn solve_sylvester<'a, 'b, 'c, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<MatrixView<'b, S>>,
    sign: SylvesterSign,
    c: impl Into<MatrixView<'c, S>>,
) -> Result<Matrix<S>, SciOxideError> {
    SylvesterPlan::new(a, b, sign, MatrixEquationOptions::new())?
        .solve(c)?
        .into_accepted()
}

/// Solves the Lyapunov equation of `kind` for the Hermitian right-hand side `Q`.
///
/// [`LyapunovKind`] states the continuous and discrete equations. The method
/// is Bartels–Stewart on the Schur form of `A`, `O(n³)` flops, followed by a
/// residual check of the original equation. No rank policy applies.
///
/// For several right-hand sides with the same coefficient, use
/// [`LyapunovPlan`].
///
/// ```
/// use scioxide_linalg::{LyapunovKind, Matrix, solve_lyapunov};
/// // A X + X Aᴴ = Q with A = diag(1, 2): xᵢⱼ (aᵢ + aⱼ) = qᵢⱼ.
/// let a = Matrix::from_rows(&[[1.0_f64, 0.0], [0.0, 2.0]])?;
/// let q = Matrix::from_rows(&[[2.0_f64, 3.0], [3.0, 8.0]])?;
/// let x = solve_lyapunov(&a, LyapunovKind::Continuous, &q)?;
/// for (computed, exact) in x.as_slice().iter().zip([1.0, 1.0, 1.0, 2.0]) {
///     assert!((computed - exact).abs() < 1e-14);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - The errors of [`LyapunovPlan::new`] and [`LyapunovPlan::solve`].
/// - `NumericalInstability` (`LossOfSignificance`) if the residual of the
///   computed `X` exceeds the tolerance of [`MatrixEquationOptions`].
///
/// # References
///
/// R. H. Bartels and G. W. Stewart, "Algorithm 432: Solution of the matrix
/// equation AX + XB = C", *Communications of the ACM* 15(9), 1972.
pub fn solve_lyapunov<'a, 'q, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    kind: LyapunovKind,
    q: impl Into<MatrixView<'q, S>>,
) -> Result<Matrix<S>, SciOxideError> {
    LyapunovPlan::new(a, kind, MatrixEquationOptions::new())?
        .solve(q)?
        .into_accepted()
}

/// Returns the stabilizing solution `X` and the feedback gain `K` of an
/// algebraic Riccati equation.
///
/// [`RiccatiKind`] states the continuous (CARE) and discrete (DARE) equations
/// and the gain. The method orders the QZ decomposition of the extended
/// pencil so that its stable deflating subspace comes first and recovers `X`
/// from that subspace, `O(n³)` flops for `n` states. The solution is accepted
/// when its residual is within the tolerance of [`RiccatiOptions`] and the
/// closed loop is stable: unlike the default of [`RiccatiOptions`], this
/// function computes the eigenvalues of the closed-loop pencil `(A − B K, E)`
/// with the QZ algorithm, another `O(n³)` flops, and requires each stability
/// margin to exceed the stability tolerance. No rank policy applies.
///
/// ```
/// use scioxide_linalg::{Matrix, RiccatiKind, RiccatiProblem, solve_riccati};
/// // Scalar CARE −2x − x² + 3 = 0 has the stabilizing root x = 1, so K = 1.
/// let a = Matrix::from_rows(&[[-1.0_f64]])?;
/// let one = Matrix::from_rows(&[[1.0_f64]])?;
/// let q = Matrix::from_rows(&[[3.0_f64]])?;
/// let (b, r) = (one.view(), one.view());
/// let problem = RiccatiProblem { a: a.view(), b, q: q.view(), r, e: None, s: None };
/// let (x, k) = solve_riccati(problem, RiccatiKind::Continuous)?;
/// assert!((x.as_slice()[0] - 1.0).abs() < 1e-12 && (k.as_slice()[0] - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - The errors of [`RiccatiPlan::new`] and [`RiccatiPlan::solve`].
/// - `NumericalInstability` with `IllConditioned` if the closed loop is not
///   stable within the tolerance, or with `LossOfSignificance` if the residual
///   exceeds the tolerance.
///
/// # References
///
/// W. F. Arnold and A. J. Laub, "Generalized eigenproblem algorithms and
/// software for algebraic Riccati equations", *Proc. IEEE* 72(12), 1984.
pub fn solve_riccati<S: NumericScalar>(
    problem: RiccatiProblem<'_, S>,
    kind: RiccatiKind,
) -> Result<(Matrix<S>, Matrix<S>), SciOxideError> {
    let options = RiccatiOptions::new().with_closed_loop_verification(crate::Estimation::Estimate);
    RiccatiPlan::new(problem, kind, options)?
        .solve()?
        .into_accepted()
}
