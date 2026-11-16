//! Dense matrix functions `f(A)`, their Fréchet derivatives and condition estimates.
//!
//! A [`MatrixFunctionPlan`] stores the input matrix for the exponential, a real
//! Schur form for real logarithms and square roots, and a complex Schur form
//! otherwise. Fréchet derivatives are read from `f` applied to a 2n × 2n block
//! upper triangular matrix.
use crate::workspace::{RealRegion, ScalarRegion};
mod dense;
mod kernel;
mod prepared;
use crate::{
    LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix,
    MatrixView, MatrixViewMut, SchurDecomposition, SchurOptions,
};
use kernel::{Kernel, SLOTS, domain};
use prepared::{Prepared, Principal};
pub(crate) use prepared::{PreparedRef, ValidatedFunction, real_eigenvalues};
use scioxide_core::{
    AlgebraicScalar, Budget, Complex, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar,
    SciOxideError, SolveContext, Tolerance,
};
use scioxide_core::{Scaled, ScaledValue};

/// Matrix function `f` evaluated by a [`MatrixFunctionPlan`].
///
/// Each variant is the primary matrix function of its scalar function: if
/// `A = Z J Z⁻¹` is a Jordan form, `f(A) = Z f(J) Z⁻¹`. Functions with a branch
/// cut use the principal branch, and the plan rejects matrices with an
/// eigenvalue on the cut. "Schur form" below means `A = Q T Qᴴ` with unitary
/// `Q` and upper triangular `T`, or quasi-triangular `T` with real `Q` for real
/// input where noted.
///
/// [`Self::Analytic`] borrows a callback. It must be `Sync` so that the plan
/// can be shared between threads.
///
/// ```
/// use scioxide_linalg::MatrixFunction;
/// let function = MatrixFunction::<f64>::Power(0.5);
/// assert!(matches!(function, MatrixFunction::Power(p) if p == 0.5));
/// assert_eq!(format!("{function:?}"), "Power(0.5)");
/// ```
#[derive(Clone, Copy)]
pub enum MatrixFunction<'f, R: RealScalar> {
    /// Matrix exponential `exp(A)`.
    ///
    /// Computed from `A` itself, without a Schur form, by scaling and squaring
    /// with a Padé approximant of degree 3, 5, 7, 9 or 13 (Al-Mohy & Higham,
    /// 2009, Algorithm 6.1).
    Exp,
    /// Principal logarithm `log(A)`, whose eigenvalues have imaginary parts in
    /// `(−π, π)`.
    ///
    /// No eigenvalue of `A` may lie on the closed negative real axis `(−∞, 0]`.
    /// The method is inverse scaling and squaring on the Schur form (real Schur
    /// form for real input): square roots are taken until `‖T − I‖₁ ≤ 1/4`,
    /// then the Padé approximant of `log(I + X)` is evaluated by 8-point
    /// Gauss–Legendre quadrature (4 points for `f32`). The diagonal blocks are
    /// computed directly from the eigenvalues to avoid cancellation (Al-Mohy &
    /// Higham, 2012).
    Log,
    /// Principal square root `A^(1/2)`, whose eigenvalues have nonnegative real
    /// parts.
    ///
    /// A negative real eigenvalue `λ` maps to the principal scalar value
    /// `i √|λ|`, so the result is complex and needs complex output. The method is
    /// the blocked Schur method on the real or complex Schur form (Björck &
    /// Hammarling, 1983; Higham, 1987; Deadman, Higham & Ralha, 2013), with
    /// tiles of order 32 that never split a 2×2 block.
    Sqrt,
    /// Real power `A^p` with finite exponent `p`.
    ///
    /// An integer `p` uses binary powering of the Schur factor, after inverting
    /// it when `p < 0`, and has no branch cut; an integer `|p| ≥ 2⁶²` is
    /// `m 2ᵏ` with `|m| < 2⁶²` and adds `k` squarings of `A^m`. Other exponents
    /// use `A^p = exp(p log A)` with the principal logarithm and its domain.
    Power(R),
    /// Matrix sign function `sign(A)`.
    ///
    /// Eigenvalues with positive real part map to `+1` and those with negative
    /// real part to `−1`. No eigenvalue may lie on the imaginary axis. The
    /// method is the Parlett recurrence on the complex Schur form, using
    /// `sign(T)² = I` for entries whose diagonal signs agree.
    Sign,
    /// Matrix sine.
    ///
    /// This and the other trigonometric and hyperbolic functions use the
    /// Schur–Parlett method described on [`MatrixFunction::Analytic`] with exact
    /// Taylor coefficients. The Taylor series on a block `T_b = cI + D + N`
    /// (`D` diagonal, `N` strictly upper triangular, order `m`) stops after `s`
    /// terms once the remainder bound
    ///
    /// ```text
    /// e^(γ+μ) Σ_{k<m} (νᵏ / k!) μ^((s−k)₊) / ((s−k)₊)!
    /// ```
    ///
    /// is at most `ε ‖F‖₁`, where `μ = maxᵢ |dᵢ|`, `ν = ‖N‖₁`, `F` is the
    /// current value on the block, and `γ` is `|Im c|` for sin, cos and tan
    /// and `|Re c|` for the hyperbolic functions. A large `ν` from a
    /// non-normal block thus costs only about `m` extra terms (Davies & Higham,
    /// 2003, §2).
    Sin,
    /// Matrix cosine.
    Cos,
    /// Matrix tangent `cos(A)⁻¹ sin(A)`.
    ///
    /// Fails with `Singular` when the computed triangular `cos(T)` has an exactly
    /// zero diagonal entry.
    Tan,
    /// Matrix hyperbolic sine.
    Sinh,
    /// Matrix hyperbolic cosine.
    Cosh,
    /// Matrix hyperbolic tangent `cosh(A)⁻¹ sinh(A)`.
    ///
    /// Fails with `Singular` when the computed triangular `cosh(T)` has an
    /// exactly zero diagonal entry.
    Tanh,
    /// A function you supply, evaluated by the Schur–Parlett method.
    ///
    /// The plan groups the eigenvalues of the complex Schur form into clusters,
    /// the connected components of the relation `|λᵢ − λⱼ| ≤ 0.1`, and moves
    /// each cluster into a contiguous diagonal block. On each block it sums the
    /// Taylor series of `f` about the mean eigenvalue `c` of the block. It
    /// couples the blocks by solving Sylvester equations (Davies & Higham,
    /// 2003). The plan relies on the coefficients, radius and tail bounds of
    /// the callback and cannot check them.
    Analytic(&'f (dyn AnalyticMatrixFunction<R> + Sync)),
}
impl<R: RealScalar> core::fmt::Debug for MatrixFunction<'_, R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Exp => "Exp",
            Self::Log => "Log",
            Self::Sqrt => "Sqrt",
            Self::Power(exponent) => {
                return formatter.debug_tuple("Power").field(exponent).finish();
            }
            Self::Sign => "Sign",
            Self::Sin => "Sin",
            Self::Cos => "Cos",
            Self::Tan => "Tan",
            Self::Sinh => "Sinh",
            Self::Cosh => "Cosh",
            Self::Tanh => "Tanh",
            Self::Analytic(_) => "Analytic(..)",
        };
        formatter.write_str(name)
    }
}

/// Numerical settings of a matrix-function evaluation.
///
/// These settings hold no execution policy, time, memory or iteration limit.
/// Those belong to [`LinalgExecutionOptions`] in [`MatrixFunctionOptions`].
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_linalg::MatrixFunctionSettings;
/// let settings = MatrixFunctionSettings::<f64>::new().with_series_limit(Budget::ONE);
/// assert_eq!(settings.series_limit(), Budget::ONE);
/// ```
///
/// Execution limits cannot be set on these settings:
///
/// ```compile_fail
/// scioxide_linalg::MatrixFunctionSettings::<f64>::new().with_byte_limit(16);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MatrixFunctionSettings<R: RealScalar> {
    tolerance: Tolerance<R>,
    series_limit: Budget,
}
impl<R: RealScalar> Default for MatrixFunctionSettings<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> MatrixFunctionSettings<R> {
    /// Returns the default settings: tolerance `atol = rtol = √ε` and a series
    /// limit of 256.
    pub fn new() -> Self {
        Self {
            tolerance: Tolerance::from_sqrt_epsilon(),
            series_limit: Budget::ONE.saturating_add(255),
        }
    }
    /// Sets the tolerance `atol`, `rtol`.
    ///
    /// It is used in two places. The Taylor series of an
    /// [`MatrixFunction::Analytic`] callback stops when the tail bound is at
    /// most `atol + rtol ‖F‖₁`, where `F` is the current value on the block.
    /// For real output, an evaluation fails if an imaginary part of the
    /// computed `f(A)` exceeds `atol + rtol Σᵢⱼ |fᵢⱼ|`. The built-in
    /// Schur–Parlett functions use their own tolerance `rtol = ε`, `atol = 0`
    /// for the Taylor series.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Sets the largest Taylor degree of the Schur–Parlett method.
    ///
    /// The logarithm also fails if the limit is below its number of quadrature
    /// nodes (8 for `f64`, 4 for `f32`).
    pub const fn with_series_limit(mut self, series_limit: Budget) -> Self {
        self.series_limit = series_limit;
        self
    }
    /// Returns the tolerance.
    pub const fn tolerance(self) -> Tolerance<R> {
        self.tolerance
    }
    /// Returns the series limit.
    pub const fn series_limit(self) -> Budget {
        self.series_limit
    }
}

/// Options of a [`MatrixFunctionPlan`]: numerical settings and execution options.
///
/// An evaluation may use at most 1 000 000 work units, or the iteration budget
/// of the execution options when one is set. A work unit is one block
/// operation, such as a matrix product, a triangular solve, a square root
/// step or one Taylor term. The same budget also limits the QR iteration of
/// the Schur reduction in [`MatrixFunctionPlan::build`], which otherwise may
/// take `30 max(n, 1)` steps and typically takes about two per eigenvalue,
/// so a budget below that fails the build.
///
/// ```
/// use scioxide_linalg::{LinalgExecutionOptions, MatrixFunctionOptions, MatrixFunctionSettings};
/// let options = MatrixFunctionOptions::<f64>::new()
///     .with_function_settings(MatrixFunctionSettings::new())
///     .with_execution_options(LinalgExecutionOptions::new().with_byte_limit(1 << 20));
/// assert_eq!(options.execution_options().byte_limit(), Some(1 << 20));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MatrixFunctionOptions<R: RealScalar> {
    execution: LinalgExecutionOptions,
    settings: MatrixFunctionSettings<R>,
}
impl<R: RealScalar> Default for MatrixFunctionOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> MatrixFunctionOptions<R> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            settings: MatrixFunctionSettings::new(),
        }
    }
    /// Sets the execution options, including the work limit (iteration budget).
    pub const fn with_execution_options(mut self, execution: LinalgExecutionOptions) -> Self {
        self.execution = execution;
        self
    }
    /// Sets the numerical settings.
    pub const fn with_function_settings(mut self, settings: MatrixFunctionSettings<R>) -> Self {
        self.settings = settings;
        self
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }
    /// Returns the numerical settings.
    pub const fn function_settings(self) -> MatrixFunctionSettings<R> {
        self.settings
    }
    pub(crate) const fn series_limit(self) -> Budget {
        self.settings.series_limit
    }
    pub(crate) const fn work_limit(self) -> Budget {
        match self.execution.iteration_budget() {
            Some(limit) => limit,
            None => Budget::ONE.saturating_add(999_999),
        }
    }
}

/// Scalar analytic function for [`MatrixFunction::Analytic`].
///
/// For each diagonal block `T_b` of the reordered Schur form, the plan picks
/// the mean eigenvalue `c` of the block and sums
///
/// ```text
/// f(T_b) = Σₖ cₖ (T_b − c I)ᵏ,    cₖ = f⁽ᵏ⁾(c) / k!,
/// ```
///
/// with `cₖ` from [`Self::coefficient`]. It requires
/// `‖T_b − c I‖₁ < radius(c)` and stops after degree `k` once
/// `tail_bound(c, k, ‖T_b − c I‖₁)` is at most the tolerance of
/// [`MatrixFunctionSettings`]. If all eigenvalues of the block are equal,
/// `T_b − c I` is nilpotent, the sum is exact after as many terms as the block
/// order, and `radius` and `tail_bound` are not called. The plan uses these
/// values as given: wrong coefficients or bounds give a wrong result.
pub trait AnalyticMatrixFunction<R: RealScalar> {
    /// Returns the radius of a disk about `center` in which `f` is analytic.
    ///
    /// A non-finite radius, or one at most `‖T_b − c I‖₁`, makes the
    /// evaluation fail with `DomainError`.
    fn radius(&self, center: Complex<R>) -> R;
    /// Returns the Taylor coefficient `f⁽ᵒʳᵈᵉʳ⁾(center) / order!`.
    ///
    /// # Errors
    /// Return an error if the coefficient is not available. The evaluation
    /// returns that error unchanged.
    fn coefficient(&self, center: Complex<R>, order: usize) -> Result<Complex<R>, SciOxideError>;
    /// Returns an upper bound on `‖Σ_{k > order} cₖ Mᵏ‖₁` that holds for every
    /// matrix `M` with `‖M‖₁ ≤ radius`, where `cₖ` are the Taylor coefficients
    /// about `center`.
    ///
    /// A negative or non-finite bound makes the evaluation fail with
    /// `DomainError`.
    ///
    /// # Errors
    /// Return an error if no bound is available. The evaluation returns that
    /// error unchanged.
    fn tail_bound(&self, center: Complex<R>, order: usize, radius: R) -> Result<R, SciOxideError>;
}

/// Diagnostics of one evaluation of `f(A)` or of a Fréchet derivative.
#[derive(Debug, Clone, Copy)]
pub struct MatrixFunctionReport<R: RealScalar> {
    /// Work units used, including block transforms.
    pub(crate) work: usize,
    /// Minimum spectral distance to the relevant cut/boundary, when applicable.
    pub(crate) branch_distance: Option<R>,
    /// Maximum discarded imaginary component for a real output, zero for complex output.
    pub(crate) imaginary_defect: R,
    /// Entrywise one-norm of the returned value or directional derivative.
    pub(crate) output_norm: Scaled<R>,
}

impl<R: RealScalar> MatrixFunctionReport<R> {
    /// Returns the number of work units used, including the Schur basis transforms.
    pub const fn work(&self) -> usize {
        self.work
    }

    /// Returns the smallest distance from an eigenvalue of `A` to the set where
    /// `f` is not analytic.
    ///
    /// For [`MatrixFunction::Log`], [`MatrixFunction::Sqrt`] and non-integer
    /// [`MatrixFunction::Power`] this set is the closed negative real axis
    /// `(−∞, 0]`, and for [`MatrixFunction::Sign`] it is the imaginary axis. A
    /// small distance means `f(A)` is sensitive to perturbations of `A`. It is
    /// `None` for the other functions, for integer powers and when `A` is empty.
    pub const fn branch_distance(&self) -> Option<R> {
        self.branch_distance
    }

    /// Returns the largest imaginary part dropped when writing real output, or
    /// zero for complex output.
    pub const fn imaginary_defect(&self) -> R {
        self.imaginary_defect
    }

    /// Returns the entrywise one-norm `Σᵢⱼ |fᵢⱼ|` of the computed value or
    /// derivative.
    ///
    /// The value is a [`Scaled`] number because the sum can exceed the range of
    /// `R` even when every entry is finite.
    pub const fn output_norm(&self) -> Scaled<R> {
        self.output_norm
    }
}

/// Condition number estimates of `f` at `A`, from [`MatrixFunctionPlan::condition_into`].
///
/// The norms are entrywise one-norms `‖X‖ = Σᵢⱼ |xᵢⱼ|`. The absolute condition
/// number is the induced norm of the Fréchet derivative,
///
/// ```text
/// cond_abs = max_{E ≠ 0} ‖L_f(A, E)‖ / ‖E‖,
/// cond_rel = cond_abs ‖A‖ / ‖f(A)‖,
/// ```
///
/// which equals the one-norm of the `n² × n²` matrix of `vec(E) ↦ vec(L_f(A, E))`.
#[derive(Debug, Clone, Copy)]
pub struct MatrixFunctionConditionReport<R: RealScalar> {
    /// Estimated absolute condition number.
    pub(crate) absolute: R,
    /// Estimated relative condition number, if `f(A) ≠ 0`.
    pub(crate) relative: Option<R>,
    /// Total work including function evaluation and estimator applications.
    pub(crate) work: usize,
}

impl<R: RealScalar> MatrixFunctionConditionReport<R> {
    /// Returns the estimate of `cond_abs`.
    ///
    /// The estimate comes from Hager's method as refined by Higham, which
    /// applies the Fréchet derivative and its adjoint to at most a few
    /// directions. It is a lower bound on `cond_abs` up to rounding errors, and
    /// exact when `n ≤ 2`.
    pub const fn absolute(&self) -> R {
        self.absolute
    }

    /// Returns the estimate of `cond_rel`, or `None` if `f(A) = 0`.
    pub const fn relative(&self) -> Option<R> {
        self.relative
    }

    /// Returns the number of work units used by the evaluation of `f(A)` and the
    /// Fréchet derivatives of the estimator.
    pub const fn work(&self) -> usize {
        self.work
    }
}

/// Operation for which a [`MatrixFunctionPlan`] workspace is sized.
///
/// A Fréchet derivative works on a matrix of order `2n`, and a condition
/// estimate needs storage for the estimator in addition. A workspace for
/// [`Self::Condition`] is therefore large enough for all three operations.
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{
///     Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan,
///     MatrixFunctionWorkspaceRequest,
/// };
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 0.0, 3.0])?;
/// let plan = MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &MatrixFunctionOptions::new())?;
/// let condition = plan.workspace_requirements(&MatrixFunctionWorkspaceRequest::Condition)?;
/// let workspace = plan.workspace(&MatrixFunctionWorkspaceRequest::Condition)?;
/// let evaluate = plan.workspace_requirements(&MatrixFunctionWorkspaceRequest::Evaluate)?;
/// assert_eq!(workspace.check(&condition)?, Capacity::Sufficient);
/// assert_eq!(workspace.check(&evaluate)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixFunctionWorkspaceRequest {
    /// [`MatrixFunctionPlan::evaluate_into`].
    Evaluate,
    /// [`MatrixFunctionPlan::frechet_into`].
    Frechet,
    /// [`MatrixFunctionPlan::condition_into`].
    Condition,
}

/// Direct scalar, extended and index lengths of one evaluation job.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScratchLengths {
    pub(crate) scalars: usize,
    packed: usize,
    pub(crate) extended: usize,
    pub(crate) indices: usize,
}

fn elements() -> SciOxideError {
    SciOxideError::size_overflow(scioxide_core::quantity!(
        "matrix_function_workspace_elements"
    ))
}

/// Kernel order and scratch lengths of `request` at matrix order `n`.
pub(crate) fn scratch_lengths<R: RealScalar>(
    n: usize,
    request: MatrixFunctionWorkspaceRequest,
) -> Result<ScratchLengths, SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let order = kernel_order(n, request)?;
    let square = n.checked_mul(n).ok_or_else(elements)?;
    let kernel = order.checked_mul(order).ok_or_else(elements)?;
    // Kernel slots, transform and candidate; conditioning adds the estimator
    // direction and the retained f(A).
    let matrices = match request {
        MatrixFunctionWorkspaceRequest::Condition => 4,
        _ => 2,
    };
    let estimate = match request {
        MatrixFunctionWorkspaceRequest::Condition => square,
        _ => 0,
    };
    // Shape-dependent padding differs between precisions (for example the
    // six-row f64 AVX2 tile). Reserve both actual arithmetic domains instead
    // of assuming f32 is always the largest packing geometry.
    let packed = dense::packing::<R>(order)?
        .scalar_len()?
        .max(dense::packing::<Complex<R>>(order)?.scalar_len()?);
    Ok(ScratchLengths {
        packed,
        scalars: kernel
            .checked_mul(SLOTS)
            .and_then(|slots| square.checked_mul(matrices)?.checked_add(slots))
            .and_then(|slots| slots.checked_add(packed))
            .ok_or_else(elements)?,
        extended: kernel.checked_add(estimate).ok_or_else(elements)?,
        indices: order
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(elements)?,
    })
}

fn kernel_order(n: usize, request: MatrixFunctionWorkspaceRequest) -> Result<usize, SciOxideError> {
    match request {
        MatrixFunctionWorkspaceRequest::Evaluate => Ok(n),
        _ => n.checked_mul(2).ok_or_else(elements),
    }
}

/// Requirements of `request` at order `n`, with only direct regions populated.
pub(crate) fn requirements<R: RealScalar>(
    n: usize,
    request: MatrixFunctionWorkspaceRequest,
    byte_limit: Option<usize>,
) -> Result<LinalgRequirements, SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let lengths = scratch_lengths::<R>(n, request)?;
    let mut requirements = LinalgRequirements::new(byte_limit);
    *requirements.scalar_mut(ScalarRegion::Direct) = lengths.scalars;
    requirements.extended = lengths.extended;
    requirements.indices = lengths.indices;
    *requirements.real_mut(RealRegion::Direct) = lengths.scalars;
    requirements.required_bytes::<Complex<R>>()?;
    Ok(requirements)
}

fn prepare_schur<S: NumericScalar>(
    mut t: Matrix<S>,
    execution: &LinalgExecutionOptions,
    scope: LinalgExecutionContext<'_, '_>,
) -> Result<(Matrix<S>, Matrix<S>), SciOxideError> {
    let n = t.rows();
    let square = n.checked_mul(n).ok_or_else(elements)?;
    let kernel = crate::eigen::schur_kernel_lengths::<S>(n, ExecutionPolicy::Sequential)?;
    let mut requirements = LinalgRequirements::new(execution.byte_limit());
    *requirements.scalar_mut(ScalarRegion::Direct) = square
        .checked_mul(2)
        .and_then(|v| v.checked_add(kernel.scalars))
        .ok_or_else(elements)?;
    *requirements.real_mut(RealRegion::Direct) = kernel.reals;
    let mut workspace = LinalgWorkspace::<S>::new();
    workspace.try_reserve(&requirements)?;
    let mut q = Matrix::zeros(n, n)?;
    // Preserve already (quasi-)triangular input in its original units.
    let triangular = (0..n)
        .all(|i| (0..i.saturating_sub(1)).all(|j| t.as_slice()[i * n + j] == S::ZERO))
        && (1..n).all(|i| {
            let c = t.as_slice()[i * n + i - 1];
            if c == S::ZERO {
                return true;
            }
            let b = t.as_slice()[(i - 1) * n + i];
            S::IS_REAL
                && t.as_slice()[i * n + i] == t.as_slice()[(i - 1) * n + i - 1]
                && ((b.real_component() > S::Real::ZERO && c.real_component() < S::Real::ZERO)
                    || (b.real_component() < S::Real::ZERO && c.real_component() > S::Real::ZERO))
                && (i == 1 || t.as_slice()[(i - 1) * n + i - 2] == S::ZERO)
        });
    workspace.run_scope(
        scope,
        identity(n),
        scioxide_core::SolveBudgets::unbounded(),
        |workspace, _, solve| {
            workspace.with_direct(|mut scalars, reals, _, _| {
                let schur_t = crate::workspace::take_mut(&mut scalars, square)?;
                let schur_q = crate::workspace::take_mut(&mut scalars, square)?;
                schur_t.copy_from_slice(t.as_slice());
                if triangular {
                    schur_q.fill(S::ZERO);
                    for i in 0..n {
                        schur_q[i * n + i] = S::ONE;
                    }
                } else {
                    crate::eigen::schur_kernel(
                        schur_t,
                        schur_q,
                        n,
                        scalars,
                        reals,
                        &SchurOptions::new().with_execution_options(
                            execution.with_execution_policy(ExecutionPolicy::Sequential),
                        ),
                        solve,
                    )?;
                }
                solve.execution().check()?;
                t.as_mut_slice().copy_from_slice(schur_t);
                q.as_mut_slice().copy_from_slice(schur_q);
                Ok(())
            })
        },
    )?;
    Ok((t, q))
}
pub(crate) fn complexify_schur<R: RealScalar>(
    t: &[R],
    q: &[R],
    ct: &mut [Complex<R>],
    cq: &mut [Complex<R>],
    n: usize,
) -> Result<(), SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    for ((out, &value), (&basis, out_q)) in ct.iter_mut().zip(t).zip(q.iter().zip(cq.iter_mut())) {
        *out = Complex::from_real(value);
        *out_q = Complex::from_real(basis);
    }
    let mut i = 0;
    while i + 1 < n {
        if t[(i + 1) * n + i] != R::ZERO {
            let b = t[i * n + i + 1];
            let omega = b.abs().sqrt() * t[(i + 1) * n + i].abs().sqrt();
            let scale = b.abs().max(omega);
            let norm = ((b / scale) * (b / scale) + (omega / scale) * (omega / scale)).sqrt();
            let x = Complex::from_real((b / scale) / norm);
            let y = Complex::new(R::ZERO, (omega / scale) / norm);
            for j in i..n {
                let a = ct[i * n + j];
                let b = ct[(i + 1) * n + j];
                ct[i * n + j] = x * a - y * b;
                ct[(i + 1) * n + j] = -y * a + x * b;
            }
            for k in 0..n {
                let a = ct[k * n + i];
                let b = ct[k * n + i + 1];
                ct[k * n + i] = x * a + y * b;
                ct[k * n + i + 1] = y * a + x * b;
                let a = cq[k * n + i];
                let b = cq[k * n + i + 1];
                cq[k * n + i] = x * a + y * b;
                cq[k * n + i + 1] = y * a + x * b;
            }
            ct[(i + 1) * n + i] = Complex::ZERO;
            i += 2;
        } else {
            i += 1;
        }
    }
    for &value in ct.iter().chain(cq.iter()) {
        crate::checked::finite_candidate(value)?;
    }
    Ok(())
}

pub(crate) fn candidate_report<S: NumericScalar>(
    values: &[S],
) -> Result<Candidate<S::Real>, SciOxideError> {
    // Real magnitudes use native compensated accumulation, promoting both
    // parts on overflow. Complex magnitudes retain independent extended scales,
    // so neither a modulus nor the sum needs to fit the scalar range.
    let mut output_norm = scioxide_core::PromotingCompensatedSum::ZERO;
    let mut imaginary_defect = S::Real::ZERO;
    for &value in values {
        crate::checked::finite_candidate(value)?;
        if value.imaginary_component() == S::Real::ZERO {
            output_norm.add_value(value.real_component().abs())?;
        } else {
            output_norm.add_scaled(ScaledValue::from_value(value)?.magnitude()?)?;
        }
        imaginary_defect = imaginary_defect.max(value.imaginary_component().abs());
    }
    Ok(Candidate {
        output_norm: output_norm.total()?,
        imaginary_defect,
    })
}

fn workspace_product<S: NumericScalar>(
    left: MatrixView<'_, S>,
    right: MatrixView<'_, S>,
    output: MatrixViewMut<'_, S>,
    left_op: crate::kernel::GemmOp,
    right_op: crate::kernel::GemmOp,
    packed: &mut [S],
    control: &mut FunctionControl<'_, '_, '_, '_>,
) -> Result<(), SciOxideError> {
    control.step()?;
    let requirements = dense::packing::<S>(
        left.rows()
            .max(left.cols())
            .max(right.rows())
            .max(right.cols()),
    )?;
    let mut workspace = LinalgWorkspace::from_slices(
        crate::LinalgSlices {
            scalars: packed,
            ..crate::LinalgSlices::default()
        },
        &requirements,
    )?;
    crate::kernel::gemm_workspace_with_context(
        crate::kernel::WorkspaceGemmProblem {
            left: crate::FiniteMatrixView::from_workspace(left),
            right: crate::FiniteMatrixView::from_workspace(right),
            output,
            config: crate::kernel::GemmConfig {
                left_op,
                right_op,
                ..crate::kernel::GemmConfig::ordinary()
            },
        },
        ExecutionPolicy::Sequential,
        &mut workspace,
        control.kernels(),
    )
}
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn evaluate_typed<S: NumericScalar, I: NumericScalar<Real = S::Real>>(
    input: &[S],
    q: Option<&[S]>,
    n: usize,
    function: Principal,
    direction: Option<MatrixView<'_, I>>,
    direction_scale: i32,
    imaginary: bool,
    data: &mut [S],
    rhs: &mut [crate::workspace::Extended<S>],
    indices: &mut [usize],
    packed: &mut [S],
    transform: &mut [S],
    candidate: &mut [S],
    options: &MatrixFunctionOptions<S::Real>,
    control: &mut FunctionControl<'_, '_, '_, '_>,
) -> Result<(), SciOxideError> {
    if n == 0 {
        return Ok(());
    }
    use crate::kernel::GemmOp;
    let order = if direction.is_some() { 2 * n } else { n };
    let square = order * order;
    data[..square].fill(S::ZERO);
    for i in 0..n {
        for j in 0..n {
            data[i * order + j] = input[i * n + j];
            if direction.is_some() {
                data[(n + i) * order + n + j] = input[i * n + j];
            }
        }
    }
    if let Some(e) = direction {
        for (index, value) in candidate.iter_mut().take(n * n).enumerate() {
            let z = crate::kernel::standard::scale(
                e.get_element(index / n, index % n),
                direction_scale,
            );
            *value = if S::IS_REAL {
                S::from_real(if imaginary {
                    z.imaginary_component()
                } else {
                    z.real_component()
                })
            } else {
                S::from_components(z.real_component(), z.imaginary_component())
            };
        }
        if let Some(q) = q {
            workspace_product(
                MatrixView::from_slice(n, n, q)?,
                MatrixView::from_slice(n, n, candidate)?,
                MatrixViewMut::from_slice(n, n, transform)?,
                GemmOp::Adjoint,
                GemmOp::None,
                packed,
                control,
            )?;
            workspace_product(
                MatrixView::from_slice(n, n, transform)?,
                MatrixView::from_slice(n, n, q)?,
                MatrixViewMut::from_slice(order, order, &mut data[..square])?
                    .block_mut(0, n, n, n)?,
                GemmOp::None,
                GemmOp::None,
                packed,
                control,
            )?;
        } else {
            for i in 0..n {
                for j in 0..n {
                    data[i * order + n + j] = candidate[i * n + j];
                }
            }
        }
    }
    control.step()?;
    {
        let mut kernel = dense::Dense::new(data, indices, packed, rhs, order)?;
        match function {
            Principal::Exp => kernel.exp(control)?,
            Principal::Log => kernel.log(options, control)?,
            Principal::Sqrt => kernel.sqrt(0, 1, control)?,
        }
    }
    let offset = if direction.is_some() { n } else { 0 };
    let result =
        MatrixView::from_slice(order, order, &data[square..2 * square])?.block(0, offset, n, n)?;
    if let Some(q) = q {
        workspace_product(
            MatrixView::from_slice(n, n, q)?,
            result,
            MatrixViewMut::from_slice(n, n, transform)?,
            GemmOp::None,
            GemmOp::None,
            packed,
            control,
        )?;
        workspace_product(
            MatrixView::from_slice(n, n, transform)?,
            MatrixView::from_slice(n, n, q)?,
            MatrixViewMut::from_slice(n, n, candidate)?,
            GemmOp::None,
            GemmOp::Adjoint,
            packed,
            control,
        )?;
    } else {
        for i in 0..n {
            for j in 0..n {
                candidate[i * n + j] = result.get_element(i, j);
            }
        }
    }
    Ok(())
}
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn evaluate_real<R: RealScalar, I: NumericScalar<Real = R>>(
    input: &[R],
    q: Option<&[R]>,
    n: usize,
    function: Principal,
    direction: Option<MatrixView<'_, I>>,
    direction_scale: i32,
    buffers: &mut Buffers<'_, R>,
    options: &MatrixFunctionOptions<R>,
    control: &mut FunctionControl<'_, '_, '_, '_>,
) -> Result<(), SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let order = if direction.is_some() { 2 * n } else { n };
    let mut real = &mut *buffers.real;
    let data = crate::workspace::take_mut(&mut real, SLOTS * order * order)?;
    let transform = crate::workspace::take_mut(&mut real, n * n)?;
    let candidate = crate::workspace::take_mut(&mut real, n * n)?;
    evaluate_typed(
        input,
        q,
        n,
        function,
        direction,
        direction_scale,
        false,
        data,
        buffers.rhs,
        buffers.blocks,
        real,
        transform,
        candidate,
        options,
        control,
    )?;
    for (out, &value) in buffers.candidate.iter_mut().zip(candidate.iter()) {
        *out = Complex::from_real(value);
    }
    if let Some(e) = direction
        && !I::IS_REAL
        && (0..n).any(|i| (0..n).any(|j| e.get_element(i, j).imaginary_component() != R::ZERO))
    {
        evaluate_typed(
            input,
            q,
            n,
            function,
            Some(e),
            direction_scale,
            true,
            data,
            buffers.rhs,
            buffers.blocks,
            real,
            transform,
            candidate,
            options,
            control,
        )?;
        for (out, &value) in buffers.candidate.iter_mut().zip(candidate.iter()) {
            out.im = value;
        }
    }
    Ok(())
}

type Wide<R> = ScaledValue<scioxide_core::Complex<R>>;

/// Kernel slots and transform staging shared by every evaluation.
pub(crate) struct Buffers<'w, R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    kernel: &'w mut [Complex<R>],
    rhs: &'w mut [Wide<R>],
    blocks: &'w mut [usize],
    packed: &'w mut [Complex<R>],
    real: &'w mut [R],
    transform: &'w mut [Complex<R>],
    pub(crate) candidate: &'w mut [Complex<R>],
}

/// Borrowed scratch of one evaluation job carved from workspace regions.
pub(crate) struct Scratch<'w, R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    pub(crate) buffers: Buffers<'w, R>,
    direction: &'w mut [Complex<R>],
    pub(crate) value: &'w mut [Complex<R>],
    estimate: &'w mut [Wide<R>],
}
impl<'w, R: RealScalar> Scratch<'w, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Real exponential evaluation needs only real kernel storage and a
    /// complex publication candidate. Its Padé solve does not use extended RHS.
    fn real_exponential(
        n: usize,
        mut scalars: &'w mut [Complex<R>],
        reals: &'w mut [R],
        mut indices: &'w mut [usize],
    ) -> Result<Self, SciOxideError> {
        let square = n.checked_mul(n).ok_or_else(elements)?;
        Ok(Self {
            buffers: Buffers {
                kernel: &mut [],
                rhs: &mut [],
                blocks: crate::workspace::take_mut(&mut indices, 4 * n + 1)?,
                packed: &mut [],
                real: reals,
                transform: &mut [],
                candidate: crate::workspace::take_mut(&mut scalars, square)?,
            },
            direction: &mut [],
            value: &mut [],
            estimate: &mut [],
        })
    }

    /// Carve the scratch of `request` at order `n`; short borrowed regions are
    /// `DimensionMismatch`.
    pub(crate) fn carve(
        n: usize,
        request: MatrixFunctionWorkspaceRequest,
        mut scalars: &'w mut [Complex<R>],
        reals: &'w mut [R],
        mut extended: &'w mut [Wide<R>],
        mut indices: &'w mut [usize],
    ) -> Result<Self, SciOxideError> {
        let lengths = scratch_lengths::<R>(n, request)?;
        if lengths.scalars > scalars.len() {
            return Err(SciOxideError::dimension_mismatch(
                [lengths.scalars],
                [scalars.len()],
            ));
        }
        let order = kernel_order(n, request)?;
        let condition = request == MatrixFunctionWorkspaceRequest::Condition;
        let buffers = Buffers {
            kernel: crate::workspace::take_mut(&mut scalars, SLOTS * order * order)?,
            transform: crate::workspace::take_mut(&mut scalars, n * n)?,
            candidate: crate::workspace::take_mut(&mut scalars, n * n)?,
            rhs: crate::workspace::take_mut(&mut extended, order * order)?,
            blocks: crate::workspace::take_mut(&mut indices, 4 * order + 1)?,
            packed: crate::workspace::take_mut(&mut scalars, lengths.packed)?,
            real: reals,
        };
        let condition_len = if condition { n * n } else { 0 };
        Ok(Self {
            buffers,
            direction: crate::workspace::take_mut(&mut scalars, condition_len)?,
            value: crate::workspace::take_mut(&mut scalars, condition_len)?,
            estimate: crate::workspace::take_mut(&mut extended, condition_len)?,
        })
    }
}

/// Work accounting for one evaluation: execution checks, the function work
/// limit and an optional solve context charged once per work unit.
pub(crate) struct FunctionControl<'e, 'a, 'b, 'c> {
    execution: ExecutionContext<'e>,
    kernels: crate::KernelContext<'e>,
    solve: Option<&'b mut SolveContext<'a, 'c>>,
    limit: usize,
    pub(crate) work: usize,
}
impl<'e, 'a, 'b, 'c> FunctionControl<'e, 'a, 'b, 'c> {
    pub(crate) fn new(
        kernels: crate::KernelContext<'e>,
        solve: Option<&'b mut SolveContext<'a, 'c>>,
        limit: usize,
    ) -> Self {
        Self {
            execution: kernels.execution(),
            kernels: kernels.with_accuracy(scioxide_core::Accuracy::Standard),
            solve,
            limit,
            work: 0,
        }
    }
    pub(crate) fn kernels(&self) -> crate::KernelContext<'e> {
        self.kernels
    }
    pub(crate) fn counts(&self) -> scioxide_core::SolveCounts {
        self.solve.as_ref().map_or_else(
            || scioxide_core::EvaluationCounts::from_iterations(self.work).into(),
            |solve| solve.counts(),
        )
    }
    pub(crate) fn step(&mut self) -> Result<(), SciOxideError> {
        self.execution.check()?;
        if self.work == self.limit {
            return Err(limit("function work budget exhausted", self.counts()));
        }
        if let Some(s) = &mut self.solve
            && !s.record_iteration()?.is_granted()
        {
            return Err(limit("shared function work budget exhausted", s.counts()));
        }
        self.work += 1;
        Ok(())
    }
}
pub(crate) fn limit(detail: &'static str, counts: scioxide_core::SolveCounts) -> SciOxideError {
    SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(counts),
        detail,
    )
}

/// Distance from the spectrum to the branch cut or singular set of the
/// principal branch: the negative real axis for log, power and sqrt, the
/// imaginary axis for sign. Eigenvalues on the cut are rejected; integer
/// powers and analytic callbacks have no branch. Derivatives additionally
/// reject square roots on the closed cut, where sqrt is not analytic.
pub(crate) fn branch<R: RealScalar>(
    function: ValidatedFunction<'_, R>,
    eigenvalues: impl Iterator<Item = Complex<R>>,
    derivative: bool,
) -> Result<Option<R>, SciOxideError> {
    let function = function.function();
    if let MatrixFunction::Power(p) = function
        && p.floor() == p
    {
        return Ok(None);
    }
    let mut distance = None::<R>;
    let mut nearest = |value: R| distance = Some(distance.map_or(value, |d| d.min(value)));
    for z in eigenvalues {
        match function {
            MatrixFunction::Log | MatrixFunction::Power(_) => {
                if z.im == R::ZERO && z.re <= R::ZERO {
                    return Err(domain(
                        scioxide_core::argument!("self"),
                        "principal logarithm/power requires spectrum off the closed negative real cut",
                    ));
                }
                nearest(if z.re < R::ZERO { z.im.abs() } else { z.norm() });
            }
            MatrixFunction::Sqrt => {
                if derivative && z.im == R::ZERO && z.re <= R::ZERO {
                    return Err(domain(
                        scioxide_core::argument!("self"),
                        "principal square-root derivative is not analytic on the cut or at zero",
                    ));
                }
                nearest(if z.re < R::ZERO { z.im.abs() } else { z.norm() });
            }
            MatrixFunction::Sign => {
                if z.re == R::ZERO {
                    return Err(domain(
                        scioxide_core::argument!("self"),
                        "matrix sign requires separation from the imaginary axis",
                    ));
                }
                nearest(z.re.abs());
            }
            _ => {}
        }
    }
    Ok(distance)
}

/// Norm diagnostics of a staged candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate<R: RealScalar> {
    imaginary_defect: R,
    output_norm: Scaled<R>,
}

/// Evaluate `f(A)`, or the Fréchet derivative `L_f(A, E)`, into
/// `buffers.candidate` (row-major `n x n`).
pub(crate) fn evaluate_candidate<R: RealScalar, I: NumericScalar<Real = R>>(
    factors: PreparedRef<'_, R>,
    direction: Option<MatrixView<'_, I>>,
    buffers: &mut Buffers<'_, R>,
    options: &MatrixFunctionOptions<R>,
    control: &mut FunctionControl<'_, '_, '_, '_>,
) -> Result<Candidate<R>, SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    use crate::kernel::standard::{exponent, magnitude, restore};
    let n = factors.order();
    // Higham, Functions of Matrices (3.16): L(A, 2^s E) = 2^s L(A, E).
    // One exponent, before any basis transforms, serves every scalar path.
    let shift = if let Some(e) = direction {
        let bound = (0..n)
            .flat_map(|i| (0..n).map(move |j| magnitude(e.get_element(i, j))))
            .fold(R::ZERO, R::max);
        if bound == R::ZERO {
            0
        } else {
            let input = match factors {
                PreparedRef::RealSchur { t, .. } | PreparedRef::DenseReal { a: t, .. } => {
                    t.iter().fold(R::ZERO, |m, &v| m.max(magnitude(v)))
                }
                PreparedRef::ComplexSchur { t, .. } | PreparedRef::DenseComplex { a: t, .. } => {
                    t.iter().fold(R::ZERO, |m, &v| m.max(magnitude(v)))
                }
            };
            exponent(if input == R::ZERO { R::ONE } else { input })? - exponent(bound)?
        }
    } else {
        0
    };
    evaluate_normalized(factors, direction, shift, buffers, options, control)?;
    for value in &mut buffers.candidate[..n * n] {
        *value = restore(*value, -shift)?;
    }
    candidate_report(&buffers.candidate[..n * n])
}

fn evaluate_normalized<R: RealScalar, I: NumericScalar<Real = R>>(
    factors: PreparedRef<'_, R>,
    direction: Option<MatrixView<'_, I>>,
    direction_scale: i32,
    buffers: &mut Buffers<'_, R>,
    options: &MatrixFunctionOptions<R>,
    control: &mut FunctionControl<'_, '_, '_, '_>,
) -> Result<(), SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let (t, q, n, function) = match factors {
        PreparedRef::RealSchur { t, q, n, function } => {
            return evaluate_real(
                t,
                Some(q),
                n,
                function,
                direction,
                direction_scale,
                buffers,
                options,
                control,
            );
        }
        PreparedRef::DenseReal { a, n } => {
            return evaluate_real(
                a,
                None,
                n,
                Principal::Exp,
                direction,
                direction_scale,
                buffers,
                options,
                control,
            );
        }
        PreparedRef::DenseComplex { a, n } => {
            return evaluate_typed(
                a,
                None,
                n,
                Principal::Exp,
                direction,
                direction_scale,
                false,
                buffers.kernel,
                buffers.rhs,
                buffers.blocks,
                buffers.packed,
                buffers.transform,
                buffers.candidate,
                options,
                control,
            );
        }
        PreparedRef::ComplexSchur { t, q, n, function } => (t, q, n, function),
    };
    let function = match function {
        ValidatedFunction::Principal(function) => {
            return evaluate_typed(
                t,
                Some(q),
                n,
                function,
                direction,
                direction_scale,
                false,
                buffers.kernel,
                buffers.rhs,
                buffers.blocks,
                buffers.packed,
                buffers.transform,
                buffers.candidate,
                options,
                control,
            );
        }
        ValidatedFunction::Schur(function) => function,
    };
    if n == 0 {
        return Ok(());
    }
    use crate::kernel::GemmOp;
    let order = if direction.is_some() { 2 * n } else { n };
    let square = order * order;
    buffers.kernel[..square].fill(Complex::ZERO);
    for i in 0..n {
        for j in 0..n {
            buffers.kernel[i * order + j] = t[i * n + j];
            if direction.is_some() {
                buffers.kernel[(n + i) * order + n + j] = t[i * n + j];
            }
        }
    }
    if let Some(e) = direction {
        for (index, value) in buffers.candidate.iter_mut().take(n * n).enumerate() {
            *value = crate::eigen::into_complex(crate::kernel::standard::scale(
                e.get_element(index / n, index % n),
                direction_scale,
            ));
        }
        workspace_product(
            MatrixView::from_slice(n, n, q)?,
            MatrixView::from_slice(n, n, buffers.candidate)?,
            MatrixViewMut::from_slice(n, n, buffers.transform)?,
            GemmOp::Adjoint,
            GemmOp::None,
            buffers.packed,
            control,
        )?;
        workspace_product(
            MatrixView::from_slice(n, n, buffers.transform)?,
            MatrixView::from_slice(n, n, q)?,
            MatrixViewMut::from_slice(order, order, &mut buffers.kernel[..square])?
                .block_mut(0, n, n, n)?,
            GemmOp::None,
            GemmOp::None,
            buffers.packed,
            control,
        )?;
    }
    Kernel::new(
        buffers.kernel,
        buffers.rhs,
        buffers.blocks,
        buffers.packed,
        order,
    )
    .evaluate(function, options, control)?;
    let offset = if direction.is_some() { n } else { 0 };
    let value = MatrixView::from_slice(order, order, &buffers.kernel[square..2 * square])?
        .block(0, offset, n, n)?;
    workspace_product(
        MatrixView::from_slice(n, n, q)?,
        value,
        MatrixViewMut::from_slice(n, n, buffers.transform)?,
        GemmOp::None,
        GemmOp::None,
        buffers.packed,
        control,
    )?;
    workspace_product(
        MatrixView::from_slice(n, n, buffers.transform)?,
        MatrixView::from_slice(n, n, q)?,
        MatrixViewMut::from_slice(n, n, buffers.candidate)?,
        GemmOp::None,
        GemmOp::Adjoint,
        buffers.packed,
        control,
    )?;
    Ok(())
}

/// Evaluate `f(A)` into `scratch.value` and estimate the absolute condition
/// number with the Hager–Higham one-norm estimator.
pub(crate) fn condition_candidate<R: RealScalar>(
    factors: PreparedRef<'_, R>,
    scratch: &mut Scratch<'_, R>,
    options: &MatrixFunctionOptions<R>,
    control: &mut FunctionControl<'_, '_, '_, '_>,
) -> Result<(Candidate<R>, R), SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let n = factors.order();
    let value = evaluate_candidate(
        factors,
        None::<MatrixView<'_, Complex<R>>>,
        &mut scratch.buffers,
        options,
        control,
    )?;
    scratch
        .value
        .copy_from_slice(&scratch.buffers.candidate[..n * n]);
    if n == 0 {
        return Ok((value, R::ZERO));
    }
    // The absolute condition number is the norm of the Frechet derivative
    // `E -> L_f(A, E)` viewed as an n^2 x n^2 operator. It is estimated
    // with the Hager/Higham 1-norm estimator, applying L_f through the
    // Frechet kernel. For the built-in functions (real Taylor coefficients)
    // the adjoint operator is `X -> L_f(A, X^H)^H`, which is what
    // conjugate-transposing the direction and the image implements. Every
    // application is charged against the shared work limit.
    let Scratch {
        buffers,
        direction,
        estimate,
        ..
    } = scratch;
    let inverse = crate::estimate::operator_one_norm(estimate, |values, adjoint| {
        for i in 0..n {
            for j in 0..n {
                let z = values[i * n + j].round()?;
                if adjoint {
                    direction[j * n + i] = z.conj();
                } else {
                    direction[i * n + j] = z;
                }
            }
        }
        evaluate_candidate(
            factors,
            Some(MatrixView::from_slice(n, n, direction)?),
            buffers,
            options,
            control,
        )?;
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = ScaledValue::from_value(if adjoint {
                    buffers.candidate[j * n + i].conj()
                } else {
                    buffers.candidate[i * n + j]
                })?;
            }
        }
        Ok(true)
    })?;
    let absolute = inverse.value.round()?;
    Ok((value, absolute))
}

/// Reject a significant imaginary part for real output; return the reported defect.
pub(crate) fn real_defect<R: RealScalar, O: NumericScalar<Real = R>>(
    candidate: Candidate<R>,
    options: &MatrixFunctionOptions<R>,
) -> Result<R, SciOxideError> {
    if !O::IS_REAL {
        return Ok(R::ZERO);
    }
    let tolerance = options.settings.tolerance;
    let threshold = Scaled::from_value(tolerance.absolute())?
        .add(Scaled::from_value(tolerance.relative())?.mul(candidate.output_norm)?)?;
    if Scaled::from_value(candidate.imaginary_defect)?
        .abs_cmp(threshold)
        .is_gt()
    {
        return Err(domain(
            scioxide_core::argument!("output"),
            "function has a complex result; provide complex output",
        ));
    }
    Ok(candidate.imaginary_defect)
}

fn publish<R: RealScalar, O: NumericScalar<Real = R>>(
    candidate: &[Complex<R>],
    output: &mut MatrixViewMut<'_, O>,
    n: usize,
) {
    for i in 0..n {
        for j in 0..n {
            let z = candidate[i * n + j];
            output.set_element(i, j, O::from_components(z.re, z.im));
        }
    }
}

fn identity(n: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "matrix_function", 1),
        n as u128,
        0,
    )
}

/// Evaluate from a prepared complex Schur factorization with work charged
/// directly to `solve`, for restarted function actions; output is complex and
/// unchanged on failure.
pub(crate) fn evaluate_schur_into<R: RealScalar>(
    schur: &SchurDecomposition<Complex<R>>,
    function: ValidatedFunction<'_, R>,
    mut output: MatrixViewMut<'_, Complex<R>>,
    workspace: &mut LinalgWorkspace<'_, Complex<R>>,
    options: &MatrixFunctionOptions<R>,
    kernels: crate::KernelContext<'_>,
    solve: Option<&mut SolveContext<'_, '_>>,
) -> Result<(), SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    let n = schur.t().rows();
    let factors = PreparedRef::ComplexSchur {
        t: schur.t().as_slice(),
        q: schur.q().as_slice(),
        n,
        function,
    };
    factors.branch(false, false)?;
    workspace.with_direct(|scalars, reals, indices, child| {
        let mut scratch = Scratch::carve(
            n,
            MatrixFunctionWorkspaceRequest::Evaluate,
            scalars,
            reals,
            child.extended_mut(),
            indices,
        )?;
        let mut control = FunctionControl::new(kernels, solve, options.work_limit().get());
        evaluate_candidate(
            factors,
            None::<MatrixView<'_, Complex<R>>>,
            &mut scratch.buffers,
            options,
            &mut control,
        )?;
        kernels.execution().check()?;
        publish(scratch.buffers.candidate, &mut output, n);
        Ok(())
    })
}

/// Matrix function `f(A)` of a fixed square matrix, prepared for repeated evaluation.
///
/// The plan computes `f(A)` for one of the functions of [`MatrixFunction`], its
/// Fréchet derivative `L_f(A, E)` in a direction `E`, and condition number
/// estimates. `A` may be real or complex, and each evaluation may write real or
/// complex output. Real output is accepted only when the imaginary part of the
/// computed result is within the tolerance of [`MatrixFunctionSettings`].
///
/// Construction depends on the function:
///
/// - [`MatrixFunction::Exp`] stores `A` and does no spectral work.
/// - [`MatrixFunction::Log`], and [`MatrixFunction::Sqrt`] of a matrix with no
///   negative real eigenvalue, compute the real Schur form for real `A`.
/// - All other cases compute a complex Schur form `A = Q T Qᴴ`. A real Schur
///   form is converted to complex form by unitary rotations of its 2×2 blocks.
///
/// The Schur reduction costs O(n³) operations once. Each evaluation costs
/// O(n³) operations for the built-in functions. Construction checks the
/// eigenvalues of the Schur form against the branch cut of `f` (see
/// [`MatrixFunctionReport::branch_distance`]).
///
/// The Fréchet derivative is the `(1, 2)` block of `f` applied to a block
/// triangular matrix (Higham, 2008, Chapter 3):
///
/// ```text
/// f([A  E])  =  [f(A)  L_f(A, E)]
///   [0  A]      [0     f(A)     ]
/// ```
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
///
/// # References
///
/// - N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.
/// - A. H. Al-Mohy and N. J. Higham, "A new scaling and squaring algorithm for
///   the matrix exponential", *SIAM J. Matrix Anal. Appl.* 31(3), 2009.
/// - A. H. Al-Mohy and N. J. Higham, "Improved inverse scaling and squaring
///   algorithms for the matrix logarithm", *SIAM J. Sci. Comput.* 34(4), 2012.
/// - Å. Björck and S. Hammarling, "A Schur method for the square root of a
///   matrix", *Linear Algebra Appl.* 52/53, 1983.
/// - N. J. Higham, "Computing real square roots of a real matrix", *Linear
///   Algebra Appl.* 88/89, 1987.
/// - E. Deadman, N. J. Higham and R. Ralha, "Blocked Schur algorithms for
///   computing the matrix square root", *Applied Parallel and Scientific
///   Computing*, LNCS 7782, 2013.
/// - P. I. Davies and N. J. Higham, "A Schur–Parlett algorithm for computing
///   matrix functions", *SIAM J. Matrix Anal. Appl.* 25(2), 2003.
/// - W. W. Hager, "Condition estimates", *SIAM J. Sci. Stat. Comput.* 5(2), 1984.
/// - N. J. Higham, "FORTRAN codes for estimating the one-norm of a real or
///   complex matrix, with applications to condition estimation", *ACM Trans.
///   Math. Software* 14(4), 1988.
#[derive(Debug, Clone)]
pub struct MatrixFunctionPlan<'f, R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    options: MatrixFunctionOptions<R>,
    prepared: Prepared<'f, R>,
    input_norm: Scaled<R>,
}

impl<'f, R: RealScalar> MatrixFunctionPlan<'f, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Prepares `function` on the square matrix `a`.
    ///
    /// This is the owned form of [`Self::build`] (see [execution](crate#execution)).
    ///
    /// # Errors
    /// - The errors of [`Self::build`].
    /// - `CapabilityUnavailable` if the options set a time limit.
    pub fn new<'a, S: NumericScalar<Real = R>>(
        a: impl Into<MatrixView<'a, S>>,
        function: MatrixFunction<'f, R>,
        options: &MatrixFunctionOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Self::build(a, function, options, ExecutionContext::sequential_native())
    }

    /// Prepares `function` on the square matrix `a` under `context`.
    ///
    /// The QR iterations of the Schur reduction count as nested work of
    /// `context`. The plan does not keep `context`.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    /// # Errors
    /// - `DimensionMismatch` if `a` is not square.
    /// - `NonFinite` if `a` has a NaN or infinite entry.
    /// - `DomainError` (argument `function`) if a [`MatrixFunction::Power`]
    ///   exponent is not finite.
    /// - `DomainError` (argument `self`) if an eigenvalue of the Schur form lies
    ///   on `(−∞, 0]` for [`MatrixFunction::Log`] or a non-integer
    ///   [`MatrixFunction::Power`], or on the imaginary axis for
    ///   [`MatrixFunction::Sign`].
    /// - `ConvergenceFailure` if the QR iteration of the Schur reduction does not
    ///   converge.
    /// - `ResourceLimit` or `AllocationFailure` for the plan's storage.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn build<'a, 'scope, 'capability, S: NumericScalar<Real = R>>(
        a: impl Into<MatrixView<'a, S>>,
        function: MatrixFunction<'f, R>,
        options: &MatrixFunctionOptions<R>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let a = a.into();
        let scope = context.into().with_options(&options.execution_options())?;
        let n = a.rows();
        crate::checked::validate_matrix(a, n, n)?;
        let function = ValidatedFunction::new(function)?;
        let maximum = (0..n)
            .flat_map(|i| (0..n).map(move |j| a.get_element(i, j)))
            .fold(R::ZERO, |bound, value| {
                bound.max(crate::kernel::standard::magnitude(value))
            });
        let exponent = crate::kernel::standard::exponent(maximum)?;
        let sum = (0..n)
            .flat_map(|i| (0..n).map(move |j| a.get_element(i, j)))
            .fold(R::ZERO, |sum, value| {
                sum + crate::kernel::standard::scale(value, -exponent).abs()
            });
        let input_norm = Scaled::from_parts(sum, i64::from(exponent))?;
        let prepared = if matches!(function, ValidatedFunction::Principal(Principal::Exp)) {
            // Dense exp retains the input and performs no spectral work.
            let mut admission = LinalgRequirements::new(options.execution.byte_limit());
            *admission.scalar_mut(ScalarRegion::Direct) = n.checked_mul(n).ok_or_else(elements)?;
            admission.required_bytes::<S>()?;
            let mut workspace = LinalgWorkspace::<S>::new();
            workspace.run_scope(
                scope,
                identity(n),
                scioxide_core::SolveBudgets::unbounded(),
                |_, _, solve| {
                    solve.execution().check()?;
                    if S::IS_REAL {
                        let mut value = Matrix::zeros(n, n)?;
                        for i in 0..n {
                            for j in 0..n {
                                value.as_mut_slice()[i * n + j] =
                                    a.get_element(i, j).real_component();
                            }
                        }
                        Ok(Prepared::DenseReal(value))
                    } else {
                        let mut value = Matrix::zeros(n, n)?;
                        for i in 0..n {
                            for j in 0..n {
                                value.as_mut_slice()[i * n + j] =
                                    crate::eigen::into_complex(a.get_element(i, j));
                            }
                        }
                        Ok(Prepared::DenseComplex(value))
                    }
                },
            )?
        } else if S::IS_REAL {
            let mut value = Matrix::<R>::zeros(n, n)?;
            for i in 0..n {
                for j in 0..n {
                    value.as_mut_slice()[i * n + j] = a.get_element(i, j).real_component();
                }
            }
            let (t, q) = prepare_schur(value, &options.execution, scope)?;
            match function {
                ValidatedFunction::Principal(principal)
                    if !matches!(principal, Principal::Sqrt)
                        || !real_eigenvalues(t.as_slice(), n)
                            .any(|z| z.im == R::ZERO && z.re < R::ZERO) =>
                {
                    Prepared::RealSchur {
                        t,
                        q,
                        function: principal,
                    }
                }
                _ => {
                    let mut ct = Matrix::zeros(n, n)?;
                    let mut cq = Matrix::zeros(n, n)?;
                    complexify_schur(
                        t.as_slice(),
                        q.as_slice(),
                        ct.as_mut_slice(),
                        cq.as_mut_slice(),
                        n,
                    )?;
                    Prepared::ComplexSchur {
                        t: ct,
                        q: cq,
                        function,
                    }
                }
            }
        } else {
            let mut value = Matrix::zeros(n, n)?;
            for i in 0..n {
                for j in 0..n {
                    value.as_mut_slice()[i * n + j] =
                        crate::eigen::into_complex(a.get_element(i, j));
                }
            }
            let (t, q) = prepare_schur(value, &options.execution, scope)?;
            Prepared::ComplexSchur { t, q, function }
        };
        let plan = Self {
            options: *options,
            prepared,
            input_norm,
        };
        plan.factors().branch(false, false)?;
        Ok(plan)
    }

    /// Returns the order `n` of `A`.
    pub fn order(&self) -> usize {
        self.prepared.factors().order()
    }

    fn factors(&self) -> PreparedRef<'_, R> {
        self.prepared.factors()
    }

    /// Returns the workspace size needed by the operation `request`.
    ///
    /// The workspace holds about 15 matrices of order `n` for an evaluation and
    /// of order `2n` for a Fréchet derivative or condition estimate. The real
    /// exponential needs the least storage.
    ///
    /// # Errors
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `ResourceLimit` if it exceeds the configured byte limit.
    pub fn workspace_requirements(
        &self,
        request: &MatrixFunctionWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        self.job_requirements(*request)
    }

    fn real_exponential_evaluation(&self, request: MatrixFunctionWorkspaceRequest) -> bool {
        matches!(self.prepared, Prepared::DenseReal(_))
            && request == MatrixFunctionWorkspaceRequest::Evaluate
    }

    fn job_requirements(
        &self,
        request: MatrixFunctionWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let mut requirements = requirements::<R>(self.order(), request, None)?;
        requirements.byte_limit = self.options.execution.byte_limit();
        if self.real_exponential_evaluation(request) {
            *requirements.scalar_mut(ScalarRegion::Direct) = self
                .order()
                .checked_mul(self.order())
                .ok_or_else(elements)?;
            requirements.extended = 0;
        }
        if matches!(
            self.prepared,
            Prepared::DenseComplex(_) | Prepared::ComplexSchur { .. }
        ) {
            *requirements.real_mut(RealRegion::Direct) = 0;
        }
        requirements.required_bytes::<Complex<R>>()?;
        Ok(requirements)
    }

    /// Allocates a workspace for the operation `request`.
    ///
    /// # Errors
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the allocation fails.
    pub fn workspace(
        &self,
        request: &MatrixFunctionWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, Complex<R>>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Computes `f(A)` and writes it to the `n × n` view `output`.
    ///
    /// `output` may be real or complex. Work units count as nested work of
    /// `context`.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    /// # Errors
    /// - `DimensionMismatch` if `output` is not `n × n`.
    /// - `DomainError` (argument `output`) if `output` is real and `f(A)` is
    ///   complex: [`MatrixFunction::Sqrt`] with a negative real eigenvalue, or an
    ///   imaginary part above the tolerance of [`MatrixFunctionSettings`].
    /// - `DomainError` (argument `function`) for [`MatrixFunction::Analytic`] if
    ///   a block does not fit in the disk of analyticity, a tail bound is
    ///   negative or not finite, or the equation coupling two clusters is
    ///   singular (see [`AnalyticMatrixFunction`]).
    /// - `DomainError` (argument `self`) if the square-root recurrence meets an
    ///   inconsistent singular local system.
    /// - `Singular` for [`MatrixFunction::Tan`] or [`MatrixFunction::Tanh`] when
    ///   the triangular denominator is exactly singular, or for a negative
    ///   integer [`MatrixFunction::Power`] of an exactly singular Schur factor.
    /// - `ConvergenceFailure` if the evaluation needs more work units than the
    ///   limit on [`MatrixFunctionOptions`], or more Taylor terms or quadrature
    ///   nodes than the series limit of [`MatrixFunctionSettings`].
    /// - `NumericalInstability` if `f(A)` overflows.
    /// - Errors returned by an [`AnalyticMatrixFunction`] callback, unchanged.
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn evaluate_into<'scope, 'capability, O: NumericScalar<Real = R>>(
        &self,
        output: MatrixViewMut<'_, O>,
        workspace: &mut LinalgWorkspace<'_, Complex<R>>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<MatrixFunctionReport<R>, SciOxideError> {
        self.run(
            None::<MatrixView<'_, Complex<R>>>,
            output,
            workspace,
            context.into(),
        )
    }

    /// Computes the Fréchet derivative `L_f(A, E)` in the direction `E` and
    /// writes it to the `n × n` view `output`.
    ///
    /// `L_f(A, E)` is the linear term of `f(A + E) − f(A)`. It is computed from
    /// `f` of the `2n × 2n` block matrix shown on [`MatrixFunctionPlan`], so it
    /// costs about 8 times as much as an evaluation of `f(A)`. `direction` and
    /// `output` may each be real or complex. The direction is scaled by a power
    /// of two to the magnitude of `A` before the evaluation and the result is
    /// scaled back. This scaling adds no rounding error unless an entry of the
    /// result is subnormal.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    /// # Errors
    /// - The errors of [`Self::evaluate_into`].
    /// - `DimensionMismatch` if `direction` is not `n × n`.
    /// - `NonFinite` if `direction` has a NaN or infinite entry.
    /// - `DomainError` (argument `self`) for [`MatrixFunction::Sqrt`] if an
    ///   eigenvalue lies on `(−∞, 0]`, where the square root is not
    ///   differentiable.
    pub fn frechet_into<'d, 'scope, 'capability, I, O>(
        &self,
        direction: impl Into<MatrixView<'d, I>>,
        output: MatrixViewMut<'_, O>,
        workspace: &mut LinalgWorkspace<'_, Complex<R>>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<MatrixFunctionReport<R>, SciOxideError>
    where
        I: NumericScalar<Real = R>,
        O: NumericScalar<Real = R>,
    {
        self.run(Some(direction.into()), output, workspace, context.into())
    }

    fn output_branch<O: NumericScalar<Real = R>>(
        &self,
        output: &MatrixViewMut<'_, O>,
        derivative: bool,
    ) -> Result<Option<R>, SciOxideError> {
        let n = self.order();
        if (output.rows(), output.cols()) != (n, n) {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(n, n),
                scioxide_core::Dimensions::two(output.rows(), output.cols()),
            ));
        }
        self.factors().branch(derivative, O::IS_REAL)
    }

    fn run<I: NumericScalar<Real = R>, O: NumericScalar<Real = R>>(
        &self,
        direction: Option<MatrixView<'_, I>>,
        mut output: MatrixViewMut<'_, O>,
        workspace: &mut LinalgWorkspace<'_, Complex<R>>,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<MatrixFunctionReport<R>, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.with_options(&self.options.execution_options())?;
        let n = self.order();
        if let Some(e) = direction {
            crate::checked::validate_matrix(e, n, n)?;
        }
        let branch_distance = self.output_branch(&output, direction.is_some())?;
        let request = if direction.is_some() {
            MatrixFunctionWorkspaceRequest::Frechet
        } else {
            MatrixFunctionWorkspaceRequest::Evaluate
        };
        let requirements = self.job_requirements(request)?;
        workspace.try_reserve(&requirements)?;
        workspace.run_scope(
            scope,
            identity(n),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                workspace.with_direct(|scalars, reals, indices, child| {
                    let mut scratch = if self.real_exponential_evaluation(request) {
                        Scratch::real_exponential(n, scalars, reals, indices)?
                    } else {
                        Scratch::carve(n, request, scalars, reals, child.extended_mut(), indices)?
                    };
                    let execution = solve.execution();
                    let mut control =
                        FunctionControl::new(kernels, Some(solve), self.options.work_limit().get());
                    let candidate = evaluate_candidate(
                        self.factors(),
                        direction,
                        &mut scratch.buffers,
                        &self.options,
                        &mut control,
                    )?;
                    let imaginary_defect = real_defect::<R, O>(candidate, &self.options)?;
                    execution.check()?;
                    publish(scratch.buffers.candidate, &mut output, n);
                    Ok(MatrixFunctionReport {
                        work: control.work,
                        branch_distance,
                        imaginary_defect,
                        output_norm: candidate.output_norm,
                    })
                })
            },
        )
    }

    /// Computes `f(A)`, writes it to the `n × n` view `output`, and estimates
    /// the condition numbers of `f` at `A`.
    ///
    /// The condition numbers are defined on [`MatrixFunctionConditionReport`].
    /// The estimator applies the Fréchet derivative and its adjoint to a few
    /// directions, without forming the `n² × n²` derivative matrix. It applies
    /// at most 17 derivatives, independently of `n`. All applications count
    /// toward the work limit of [`MatrixFunctionOptions`].
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    /// # Errors
    /// - The errors of [`Self::evaluate_into`].
    /// - `InvalidInput` (argument `self`) if the function is
    ///   [`MatrixFunction::Analytic`]. The adjoint derivative used by the
    ///   estimator needs real Taylor coefficients, which only the built-in
    ///   functions guarantee.
    /// - `DomainError` (argument `self`) for [`MatrixFunction::Sqrt`] if an
    ///   eigenvalue lies on `(−∞, 0]`.
    pub fn condition_into<'scope, 'capability, O: NumericScalar<Real = R>>(
        &self,
        mut output: MatrixViewMut<'_, O>,
        workspace: &mut LinalgWorkspace<'_, Complex<R>>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<MatrixFunctionConditionReport<R>, SciOxideError> {
        workspace.begin_attempt();
        let scope = context
            .into()
            .with_options(&self.options.execution_options())?;
        if matches!(
            self.factors().function().function(),
            MatrixFunction::Analytic(_)
        ) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "condition estimates require a built-in function",
            ));
        }
        let n = self.order();
        self.output_branch(&output, true)?;
        let request = MatrixFunctionWorkspaceRequest::Condition;
        let requirements = self.job_requirements(request)?;
        workspace.try_reserve(&requirements)?;
        workspace.run_scope(
            scope,
            identity(n),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                workspace.with_direct(|scalars, reals, indices, child| {
                    let mut scratch =
                        Scratch::carve(n, request, scalars, reals, child.extended_mut(), indices)?;
                    let execution = solve.execution();
                    let mut control =
                        FunctionControl::new(kernels, Some(solve), self.options.work_limit().get());
                    let (value, absolute) = condition_candidate(
                        self.factors(),
                        &mut scratch,
                        &self.options,
                        &mut control,
                    )?;
                    real_defect::<R, O>(value, &self.options)?;
                    let relative = if value.output_norm.is_zero() {
                        None
                    } else {
                        Some(
                            Scaled::from_value(absolute)?
                                .mul(self.input_norm)?
                                .div(value.output_norm)?
                                .round()?,
                        )
                    };
                    execution.check()?;
                    publish(scratch.value, &mut output, n);
                    Ok(MatrixFunctionConditionReport {
                        absolute,
                        relative,
                        work: control.work,
                    })
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_lengths_overflow_and_grow_with_the_job() {
        for request in [
            MatrixFunctionWorkspaceRequest::Evaluate,
            MatrixFunctionWorkspaceRequest::Frechet,
            MatrixFunctionWorkspaceRequest::Condition,
        ] {
            assert_eq!(
                scratch_lengths::<f64>(usize::MAX, request)
                    .unwrap_err()
                    .kind(),
                scioxide_core::SciOxideErrorKind::SizeOverflow
            );
        }
        let evaluate = scratch_lengths::<f64>(3, MatrixFunctionWorkspaceRequest::Evaluate).unwrap();
        let frechet = scratch_lengths::<f64>(3, MatrixFunctionWorkspaceRequest::Frechet).unwrap();
        let condition =
            scratch_lengths::<f64>(3, MatrixFunctionWorkspaceRequest::Condition).unwrap();
        assert!(evaluate.scalars < frechet.scalars && frechet.scalars < condition.scalars);
        assert!(evaluate.extended < frechet.extended && frechet.extended < condition.extended);
        assert!(evaluate.indices < frechet.indices);
        assert_eq!(frechet.indices, condition.indices);
    }
    #[test]
    fn real_candidate_norm_preserves_compensation_and_promotes() {
        for values in [
            alloc::vec![1e16, 1.0, 1.0, 1.0],
            alloc::vec![f64::from_bits(1); 8],
            alloc::vec![f64::MAX, f64::MAX, 1.0],
            alloc::vec![-0.0, 0.0],
        ] {
            let mut reference = scioxide_core::ScaledAccumulator::ZERO;
            for &value in &values {
                reference.add_value(value.abs()).unwrap();
            }
            let candidate = candidate_report(&values).unwrap();
            assert_eq!(candidate.output_norm, reference.total().unwrap());
            let complex: alloc::vec::Vec<_> = values.into_iter().map(Complex::from_real).collect();
            assert_eq!(
                candidate_report(&complex).unwrap().output_norm,
                candidate.output_norm
            );
        }
        assert!(candidate_report(&[f64::INFINITY]).is_err());
        assert!(candidate_report(&[Complex::new(1.0, f64::NAN)]).is_err());
    }

    #[test]
    fn preparation_uses_dense_exp_and_real_schur() {
        let a = Matrix::from_vec(2, 2, alloc::vec![0.7_f64, -0.6, 0.6, 0.7]).unwrap();
        let options = MatrixFunctionOptions::new();
        let exp = MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &options).unwrap();
        assert!(matches!(exp.prepared, Prepared::DenseReal(_)));
        for function in [MatrixFunction::Log, MatrixFunction::Sqrt] {
            let plan = MatrixFunctionPlan::new(&a, function, &options).unwrap();
            assert!(matches!(plan.prepared, Prepared::RealSchur { .. }));
        }
        let a = Matrix::from_vec(1, 1, alloc::vec![Complex::new(1.0, 0.5)]).unwrap();
        let exp = MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &options).unwrap();
        assert!(matches!(exp.prepared, Prepared::DenseComplex(_)));
    }
    #[test]
    fn converted_real_schur_reconstructs_nonnormal_input() {
        let n = 4;
        let a = Matrix::from_vec(
            n,
            n,
            alloc::vec![
                1.0_f64, -0.7, 0.2, 0.3, 0.7, 1.0, 0.0, 0.1, 0.2, 0.4, 2.0, -0.5, 0.3, 0.0, 0.5,
                2.0,
            ],
        )
        .unwrap();
        for function in [
            MatrixFunction::Sin,
            MatrixFunction::Sign,
            MatrixFunction::Power(0.3),
        ] {
            let plan =
                MatrixFunctionPlan::new(&a, function, &MatrixFunctionOptions::new()).unwrap();
            let factors = match plan.prepared {
                Prepared::ComplexSchur { t, q, .. } => Some((t, q)),
                _ => None,
            };
            let (t, q) = factors.unwrap();
            for i in 0..n {
                for j in 0..n {
                    let mut value = Complex::ZERO;
                    let mut orthogonality = Complex::ZERO;
                    for k in 0..n {
                        orthogonality += q.as_slice()[i * n + k] * q.as_slice()[j * n + k].conj();
                        for l in 0..n {
                            value += q.as_slice()[i * n + k]
                                * t.as_slice()[k * n + l]
                                * q.as_slice()[j * n + l].conj();
                        }
                    }
                    assert!(
                        (value - Complex::from_real(a.as_slice()[i * n + j])).norm()
                            < 256.0 * f64::EPSILON
                    );
                    assert!(
                        (orthogonality - Complex::from_real(if i == j { 1.0 } else { 0.0 })).norm()
                            < 256.0 * f64::EPSILON
                    );
                    if i > j {
                        assert_eq!(t.as_slice()[i * n + j], Complex::ZERO);
                    }
                }
            }
        }
    }
    #[test]
    fn real_exponential_uses_compact_borrowed_storage() {
        use crate::LinalgSlices;
        use alloc::vec;

        let a = Matrix::from_vec(2, 2, vec![0.0_f64, 0.25, 0.0, 0.0]).unwrap();
        let p = MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &MatrixFunctionOptions::new())
            .unwrap();
        let mut requirements = p
            .workspace_requirements(&MatrixFunctionWorkspaceRequest::Evaluate)
            .unwrap();
        let mut scalars = vec![Complex::new(0.0, 0.0); requirements.scalar_len().unwrap()];
        let mut reals = vec![0.0; *requirements.real_mut(RealRegion::Direct)];
        let mut indices = vec![0; requirements.indices];
        // No extended storage, and only the publication candidate is complex.
        assert_eq!(scalars.len(), 4);
        let mut workspace = LinalgWorkspace::from_slices(
            LinalgSlices {
                scalars: &mut scalars,
                reals: &mut reals,
                indices: &mut indices,
                ..LinalgSlices::default()
            },
            &requirements,
        )
        .unwrap();
        let mut output = Matrix::<Complex<f64>>::zeros(2, 2).unwrap();
        for cx in [
            ExecutionContext::sequential(),
            ExecutionContext::sequential_native(),
        ] {
            {
                p.evaluate_into(output.view_mut(), &mut workspace, cx)
                    .unwrap();
            };
            for (&actual, expected) in output.as_slice().iter().zip([1.0, 0.25, 0.0, 1.0]) {
                assert!((actual - Complex::new(expected, 0.0)).norm() <= 8.0 * f64::EPSILON);
            }
        }
        // This compact borrowed workspace cannot silently grow for a derivative.
        let before = output.clone();
        assert!(
            p.frechet_into(
                &a,
                output.view_mut(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output.as_slice(), before.as_slice());
    }
}
