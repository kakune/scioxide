//! Rational approximation by the AAA algorithm, with optional AAA–Lawson refinement.

use alloc::vec::Vec;

use scioxide_core::backend::try_reserve;
use scioxide_core::{
    AlgebraicScalar, Capacity, ConvergenceReason, ExecutionContext, ExecutionPolicy, Interval,
    NumericScalar, ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveContext,
    SolveIdentity, SolveScope, SolveState, Tolerance,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix, SvdWorkspaceRequest,
    ThinSvdDecomposition,
};

use super::{
    BarycentricRational, RationalDomain, RationalReport, RationalTermination, Support,
    support_capacity,
};
use crate::checked::intermediate;
use crate::error::instability;
use crate::error::invalid;
use crate::error::shape;
use crate::error::unconverged;
use crate::protocol::ExtrapolationPolicy;
use crate::validate::{validate_count, validate_finite};

/// Default and one-shot limit on the number of support points.
const DEFAULT_MAX_TERMS: usize = 100;

/// Configures [`aaa_with_options`] and [`aaa_into`]: support-point limit,
/// tolerance, Lawson steps, SVD and iteration limits.
///
/// - `max_terms` is the maximum number of support points; it must be at
///   least one and at most the number of samples.
/// - `tolerance` gives the target for the maximum residual at the samples,
///   `maxᵢ |r(zᵢ) − fᵢ| ≤ abs + rel · maxᵢ |fᵢ|`.
/// - `lawson_iterations` is the maximum number of AAA–Lawson steps after
///   the AAA steps; zero selects plain AAA.
/// - The execution policy applies to the SVDs and overrides the policy in the
///   SVD options. The SVD options otherwise set accuracy, sweep and memory
///   limits.
/// - The budgets limit the iterations of each solve; see
///   [`RationalTermination`].
///
/// The defaults are at most 100 support points, the tolerance
/// [`Tolerance::from_sqrt_epsilon`] (`abs = rel = √ε`, about `1.5·10⁻⁸` in
/// `f64` and `3.5·10⁻⁴` in `f32`), plain AAA, sequential execution, default
/// SVD options and no iteration limit.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionPolicy, Tolerance};
/// use scioxide_interp::AaaOptions;
/// let options = AaaOptions::<f64>::new()
///     .with_max_terms(8)
///     .with_tolerance(Tolerance::new(1e-12, 1e-12)?)
///     .with_lawson_iterations(10);
/// assert_eq!((options.max_terms(), options.lawson_iterations()), (8, 10));
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// assert_eq!(AaaOptions::<f64>::default().max_terms(), 100);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct AaaOptions<R: RealScalar> {
    max_terms: usize,
    tolerance: Tolerance<R>,
    lawson_iterations: usize,
    execution: ExecutionPolicy,
    svd: LinalgExecutionOptions,
    budgets: SolveBudgets,
}

impl<R: RealScalar> Default for AaaOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RealScalar> AaaOptions<R> {
    /// Creates the default options: plain AAA with at most 100 support points and tolerance `√ε`.
    pub fn new() -> Self {
        Self {
            max_terms: DEFAULT_MAX_TERMS,
            tolerance: Tolerance::from_sqrt_epsilon(),
            lawson_iterations: 0,
            execution: ExecutionPolicy::Sequential,
            svd: LinalgExecutionOptions::new(),
            budgets: SolveBudgets::unbounded(),
        }
    }

    /// Sets the maximum number of support points; a fit rejects zero and a limit
    /// above the sample count.
    pub const fn with_max_terms(mut self, max_terms: usize) -> Self {
        self.max_terms = max_terms;
        self
    }

    /// Sets the tolerance for the maximum residual at the samples.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of Lawson steps; zero selects plain AAA.
    pub const fn with_lawson_iterations(mut self, iterations: usize) -> Self {
        self.lawson_iterations = iterations;
        self
    }

    /// Sets the execution policy of the SVDs.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }

    /// Sets the SVD accuracy, sweep and memory limits; the execution policy in `options` is ignored.
    pub const fn with_svd_options(mut self, options: LinalgExecutionOptions) -> Self {
        self.svd = options;
        self
    }

    /// Sets the iteration limits of each solve.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Returns the maximum number of support points.
    #[must_use]
    pub const fn max_terms(&self) -> usize {
        self.max_terms
    }

    /// Returns the tolerance for the maximum residual at the samples.
    pub const fn tolerance(&self) -> Tolerance<R> {
        self.tolerance
    }

    /// Returns the maximum number of Lawson steps.
    #[must_use]
    pub const fn lawson_iterations(&self) -> usize {
        self.lawson_iterations
    }

    /// Returns the execution policy of the SVDs.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution
    }

    /// Returns the SVD accuracy, sweep and memory limits.
    pub const fn svd_options(&self) -> LinalgExecutionOptions {
        self.svd
    }

    /// Returns the iteration limits of each solve.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }

    /// SVD options with the configured execution policy.
    const fn effective_svd(&self) -> LinalgExecutionOptions {
        self.svd.with_execution_policy(self.execution)
    }
}

/// Describes one AAA fit for workspace sizing: the number of samples and the options.
///
/// Passed to [`RationalWorkspace::workspace_requirements`] and
/// [`RationalWorkspace::workspace`].
///
/// # Examples
///
/// ```
/// use scioxide_interp::{AaaOptions, RationalRequest};
/// let request = RationalRequest::new(10, AaaOptions::<f64>::new().with_max_terms(4));
/// assert_eq!((request.samples(), request.options().max_terms()), (10, 4));
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RationalRequest<R: RealScalar> {
    samples: usize,
    options: AaaOptions<R>,
}

impl<R: RealScalar> RationalRequest<R> {
    /// Creates a request for fitting `samples` samples with `options`.
    pub const fn new(samples: usize, options: AaaOptions<R>) -> Self {
        Self { samples, options }
    }

    /// Returns the number of samples.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Returns the options of the fit.
    pub const fn options(&self) -> AaaOptions<R> {
        self.options
    }
}

/// Fits a rational function in barycentric form to samples `(zᵢ, fᵢ)` with the AAA
/// algorithm and returns it.
///
/// The samples may be real or complex and must be finite and distinct. AAA
/// starts from the sample mean and adds support points greedily: each step
/// takes the sample with the largest residual as a new support point, and
/// chooses the weights as the right singular vector of the smallest singular
/// value of the Loewner matrix `Aᵢⱼ = (fᵢ − fⱼ) / (zᵢ − zⱼ)`, with rows for
/// the other samples `zᵢ` and columns for the support points `zⱼ`. The result
/// interpolates the data at the support points. Ties go to the earliest
/// sample, so the fit is deterministic.
///
/// The one-shot form uses [`AaaOptions::default`] with at most
/// `min(100, n)` support points for `n` samples, and stops when the maximum
/// residual at the samples satisfies `maxᵢ |r(zᵢ) − fᵢ| ≤ √ε (1 + maxᵢ |fᵢ|)`.
/// Its domain is the box spanned by the nodes: the interval
/// `[minᵢ zᵢ, maxᵢ zᵢ]` when every node is real, otherwise the rectangle of
/// the real and imaginary components. With `m` samples, step `k` computes the
/// SVD of an `m × k` matrix, so `K` support points cost O(m K³) operations.
/// The residual is measured only at the samples; it is not an error bound
/// between them.
///
/// A fit that stops before meeting its tolerance is an error here;
/// [`aaa_with_options`] takes a domain and options, adds AAA–Lawson
/// refinement, and returns the approximant with its [`RationalReport`]
/// whatever the status. [`aaa_into`] reuses caller storage.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `nodes` is empty, and `InvalidInput` if two
///   nodes coincide.
/// - `NonFinite` (`Input`) if a node or value is NaN or infinite, or
///   `NonFinite` if an intermediate overflows.
/// - `DimensionMismatch` if `nodes` and `values` differ in length.
/// - `ConvergenceFailure` if the fit stops before meeting its tolerance:
///   `TruncationLimit` when it reaches `min(100, n)` support points.
/// - The errors of the SVD if it fails.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{Univariate, aaa};
/// let x: Vec<f64> = (0..21).map(|i| -1.0 + f64::from(i) / 10.0).collect();
/// let y: Vec<f64> = x.iter().map(|x| 1.0 / (2.0 + x)).collect();
/// let r = aaa(&x, &y)?;
/// assert!((r.value_at(0.25)? - 1.0 / 2.25).abs() < 1e-12);
/// assert!(r.value_at(1.5).is_err()); // outside [−1, 1]
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - Y. Nakatsukasa, O. Sète and L. N. Trefethen, "The AAA algorithm for
///   rational approximation", *SIAM J. Sci. Comput.* 40(3), 2018, A1494–A1522.
pub fn aaa<C: NumericScalar>(
    nodes: &[C],
    values: &[C],
) -> Result<BarycentricRational<C>, SciOxideError> {
    validate_count(nodes.len(), 1, scioxide_core::argument!("nodes"))?;
    validate_finite(nodes)?;
    let domain = node_domain(nodes)?;
    let options = AaaOptions::default().with_max_terms(DEFAULT_MAX_TERMS.min(nodes.len()));
    let RationalFit {
        approximant,
        report,
        ..
    } = aaa_with_options(nodes, values, domain, options)?;
    let (Some(approximant), Some(report)) = (approximant, report) else {
        return Err(instability(
            scioxide_core::NumericalIssue::Breakdown,
            "AAA returned no approximant",
        ));
    };
    let reason = match report.termination() {
        RationalTermination::Converged => return Ok(approximant),
        RationalTermination::SupportLimit => ConvergenceReason::TruncationLimit,
        RationalTermination::IterationLimit | RationalTermination::BudgetExhausted => {
            ConvergenceReason::IterationLimit
        }
        RationalTermination::Stagnation => ConvergenceReason::Stagnation,
    };
    Err(unconverged(
        reason,
        scioxide_core::algorithm_id!("interp", "aaa", 1),
        report.iterations(),
        "AAA stopped before reaching its residual tolerance",
    ))
}

/// The box spanned by finite nodes: an interval when every node is real.
fn node_domain<C: NumericScalar>(nodes: &[C]) -> Result<RationalDomain<C::Real>, SciOxideError> {
    let span = |component: fn(C) -> C::Real| {
        let mut values = nodes.iter().map(|&z| component(z));
        let first = values.next().unwrap_or(C::Real::ZERO);
        let (lower, upper) = values.fold((first, first), |(lower, upper), value| {
            (lower.min(value), upper.max(value))
        });
        Interval::new(lower, upper)
    };
    let real = span(C::real_component)?;
    if nodes
        .iter()
        .all(|z| z.imaginary_component() == C::Real::ZERO)
    {
        Ok(RationalDomain::Real(real))
    } else {
        Ok(RationalDomain::Complex {
            real,
            imaginary: span(C::imaginary_component)?,
        })
    }
}

/// Fits the samples `(nodes[i], values[i])` on `domain` with the AAA algorithm,
/// configured by `options`, and returns the fit.
///
/// This is the configured form of [`aaa`], whose rustdoc describes the
/// algorithm. The samples must lie inside `domain`, and `options` sets the
/// support-point limit, the tolerance, optional AAA–Lawson steps, the SVDs
/// and the iteration budget. If the tolerance is not met after the AAA steps,
/// the AAA–Lawson steps fit numerator and denominator coefficients separately
/// by iteratively reweighted least squares, moving toward the minimax
/// approximation on the samples. A Lawson step is kept only if it reduces the
/// maximum residual, so the reported residual never increases. The
/// approximant then need not interpolate. A Lawson step with `k` support
/// points computes the SVD of an `m × 2k` matrix.
///
/// Reaching the support-point, Lawson or iteration limit still returns the
/// best approximant: check [`RationalReport::termination`] of
/// [`RationalFit::report`]. It allocates the workspace and the output, and
/// runs [`aaa_into`] with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// The errors of [`RationalWorkspace::workspace_requirements`] and
/// [`aaa_into`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{Interval, Tolerance};
/// use scioxide_interp::{AaaOptions, RationalDomain, RationalTermination, aaa_with_options};
/// let x: Vec<f64> = (0..101).map(|i| -1.0 + f64::from(i) / 50.0).collect();
/// let y: Vec<f64> = x.iter().map(|x| x.exp()).collect();
/// let domain = RationalDomain::Real(Interval::new(-1.0, 1.0)?);
/// let tolerance = Tolerance::new(1e-13, 1e-13)?;
/// let options = AaaOptions::new()
///     .with_max_terms(3)
///     .with_tolerance(tolerance);
/// let plain = aaa_with_options(&x, &y, domain, options)?.report().unwrap();
/// let lawson = options.with_lawson_iterations(20);
/// let refined = aaa_with_options(&x, &y, domain, lawson)?.report().unwrap();
/// assert_eq!(plain.termination(), RationalTermination::SupportLimit);
/// assert!(refined.max_residual() < plain.max_residual());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - Y. Nakatsukasa, O. Sète and L. N. Trefethen, "The AAA algorithm for
///   rational approximation", *SIAM J. Sci. Comput.* 40(3), 2018, A1494–A1522.
/// - Y. Nakatsukasa and L. N. Trefethen, "An algorithm for real and complex
///   rational minimax approximation", *SIAM J. Sci. Comput.* 42(5), 2020,
///   A3157–A3179.
pub fn aaa_with_options<C: NumericScalar>(
    nodes: &[C],
    values: &[C],
    domain: RationalDomain<C::Real>,
    options: AaaOptions<C::Real>,
) -> Result<RationalFit<C>, SciOxideError> {
    let requirements =
        RationalWorkspace::<C>::workspace_requirements(RationalRequest::new(nodes.len(), options))?;
    let mut workspace = RationalWorkspace::empty();
    workspace.try_reserve(&requirements)?;
    let mut fit = RationalFit::new();
    fit.try_reserve(&requirements)?;
    aaa_into(
        nodes,
        values,
        domain,
        options,
        &mut fit,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok(fit)
}

/// Fits the samples `(nodes[i], values[i])` as [`aaa_with_options`] does and writes
/// the approximant and report to `out`.
///
/// `out` and `workspace` must be reserved for
/// [`RationalWorkspace::workspace_requirements`] of the same samples and
/// options; the fit then allocates nothing. `out` changes only when the call
/// returns `Ok`. `scope` is an [`ExecutionContext`] for a top-level solve, or
/// a `&mut SolveContext` for a solve nested in another, whose iterations then
/// count toward the parent (see
/// [solve accounting](scioxide_core#solve-accounting)). After the call,
/// [`RationalWorkspace::residual_history`] holds the residual of every
/// accepted approximant.
///
/// # Errors
///
/// - `DimensionMismatch` if `nodes` and `values` differ in length, or `out`
///   or `workspace` was not reserved for these requirements.
/// - `InvalidInput` if `max_terms` is zero, there are fewer samples than
///   `max_terms`, the nodes are not distinct, or a real domain gets a complex
///   node.
/// - `NonFinite` if a node or value is NaN or infinite, or an intermediate
///   overflows.
/// - `DomainError` if a node lies outside `domain`.
/// - `ExecutionStopped` on cancellation or deadline.
/// - The errors of the SVD workspace query for invalid SVD options, and of
///   the SVD if it fails.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_interp::{AaaOptions, RationalDomain, RationalFit, RationalRequest};
/// use scioxide_interp::{RationalWorkspace, aaa_into};
/// let (x, y) = ([-1.0_f64, -0.5, 0.0, 0.5, 1.0], [1.0, 1.5, 2.0, 2.5, 3.0]);
/// let domain = RationalDomain::Real(Interval::new(-1.0, 1.0)?);
/// let options = AaaOptions::new().with_max_terms(4);
/// let request = RationalRequest::new(x.len(), options);
/// let requirements = RationalWorkspace::<f64>::workspace_requirements(request)?;
/// let mut workspace = RationalWorkspace::<f64>::workspace(request)?;
/// let mut out = RationalFit::new();
/// out.try_reserve(&requirements)?;
/// let cx = ExecutionContext::sequential_native();
/// aaa_into(&x, &y, domain, options, &mut out, &mut workspace, cx)?;
/// let iterations = out.report().unwrap().iterations();
/// assert_eq!(workspace.residual_history().len(), iterations + 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn aaa_into<'scope, 'capability, C: NumericScalar>(
    nodes: &[C],
    values: &[C],
    domain: RationalDomain<C::Real>,
    options: AaaOptions<C::Real>,
    out: &mut RationalFit<C>,
    workspace: &mut RationalWorkspace<C>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    let n = nodes.len();
    if n != values.len() {
        return Err(shape(n, values.len()));
    }
    let requirements =
        RationalWorkspace::<C>::workspace_requirements(RationalRequest::new(n, options))?;
    validate_finite(values)?;
    if workspace.check(&requirements)? != Capacity::Sufficient
        || support_capacity(out.storage(), requirements.terms) != Capacity::Sufficient
    {
        return Err(SciOxideError::dimension_mismatch(
            [requirements.samples, requirements.terms],
            [workspace.residuals.capacity(), out.storage().capacity()],
        ));
    }
    let plane = domain.plane()?;
    let mut scale = C::Real::ZERO;
    for (&z, &f) in nodes.iter().zip(values) {
        domain.resolve(
            z,
            ExtrapolationPolicy::Reject,
            scioxide_core::argument!("nodes"),
        )?;
        scale = scale.max(f.abs());
    }
    super::distinct(
        nodes,
        &mut workspace.node_indices,
        "duplicate nodes or non-finite samples",
    )?;
    // The stopping threshold is relative to the true data magnitude; the
    // working `scale` (used to normalize samples to O(1) in the linear
    // algebra) falls back to one for all-zero data to avoid dividing by zero.
    let threshold = options.tolerance.threshold(scale)?;
    let scale = if scale == C::Real::ZERO {
        C::Real::ONE
    } else {
        scale
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("interp", "aaa", 1),
        n as u128,
        ((options.max_terms as u128) << 64) | options.lawson_iterations as u128,
    );
    let mut state = core::mem::take(&mut workspace.state);
    let result = scope
        .into()
        .run(&mut state, identity, options.budgets, |context| {
            fit(
                &options,
                Samples {
                    nodes,
                    values,
                    scale,
                    threshold,
                },
                workspace,
                context,
            )
        });
    workspace.state = state;
    let report = result?;
    // Publish by swapping: the previous output's buffers become the next
    // candidate, so capacity is retained and publication never allocates.
    let support = core::mem::take(&mut workspace.candidate);
    workspace.candidate = match out.approximant.take() {
        Some(previous) => previous.into_support(),
        None => core::mem::take(&mut out.spare),
    };
    out.approximant = Some(BarycentricRational::from_resolved(support, domain, plane));
    out.report = Some(report);
    Ok(())
}

/// Describes the storage of an AAA fit for a given number of samples and options.
///
/// Returned by [`RationalWorkspace::workspace_requirements`] and accepted by
/// [`RationalWorkspace::check`], [`RationalWorkspace::try_reserve`] and
/// [`RationalFit::try_reserve`]. The requirements include the SVD storage and
/// do not depend on the executor.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{AaaOptions, RationalRequest, RationalWorkspace};
/// let options = AaaOptions::new()
///     .with_max_terms(4)
///     .with_lawson_iterations(3);
/// let request = RationalRequest::new(10, options);
/// let requirements = RationalWorkspace::<f64>::workspace_requirements(request)?;
/// assert_eq!(
///     (requirements.matrix_rows(), requirements.matrix_columns()),
///     (10, 8)
/// );
/// assert_eq!(requirements.history(), 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct RationalRequirements {
    samples: usize,
    terms: usize,
    lawson_iterations: usize,
    matrix_rows: usize,
    matrix_columns: usize,
    history: usize,
    svd: LinalgRequirements,
}

impl RationalRequirements {
    /// Returns the number of samples.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }
    /// Returns the maximum number of support points.
    #[must_use]
    pub const fn terms(&self) -> usize {
        self.terms
    }
    /// Returns the maximum number of Lawson steps.
    #[must_use]
    pub const fn lawson_iterations(&self) -> usize {
        self.lawson_iterations
    }
    /// Returns the maximum number of rows of an SVD, including zero rows that pad it to at least as many rows as columns.
    #[must_use]
    pub const fn matrix_rows(&self) -> usize {
        self.matrix_rows
    }
    /// Returns the maximum number of columns of an SVD.
    #[must_use]
    pub const fn matrix_columns(&self) -> usize {
        self.matrix_columns
    }
    /// Returns the maximum length of the residual history: the initial constant plus every accepted step.
    #[must_use]
    pub const fn history(&self) -> usize {
        self.history
    }
}

/// Holds the scratch storage of [`aaa_into`] and the residual history of the last fit.
///
/// Create it with [`Self::workspace`]. It grows only through
/// [`Self::try_reserve`]. A fit exchanges coefficient buffers with the
/// [`RationalFit`], so reserve the workspace and the fit for the same
/// requirements: a fit reserved for fewer terms passes its smaller buffers on.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_interp::{AaaOptions, RationalRequest, RationalWorkspace};
/// let options = AaaOptions::new().with_max_terms(3);
/// let mut workspace = RationalWorkspace::<f64>::workspace(RationalRequest::new(5, options))?;
/// let request = RationalRequest::new(9, options);
/// let larger = RationalWorkspace::<f64>::workspace_requirements(request)?;
/// assert_eq!(workspace.check(&larger)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&larger)?;
/// assert_eq!(workspace.check(&larger)?, Capacity::Sufficient);
/// assert!(workspace.residual_history().is_empty());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct RationalWorkspace<C: NumericScalar> {
    matrix: Matrix<C>,
    svd: LinalgWorkspace<'static, C>,
    decomposition: ThinSvdDecomposition<C>,
    /// Shape for which SVD output storage is reserved.
    decomposition_shape: [usize; 2],
    candidate: Support<C>,
    trial: Support<C>,
    residuals: Vec<C::Real>,
    lawson_weights: Vec<C::Real>,
    selected: Vec<bool>,
    support_indices: Vec<usize>,
    node_indices: Vec<usize>,
    history: Vec<C::Real>,
    state: SolveState,
}

impl<C: NumericScalar> RationalWorkspace<C> {
    fn empty() -> Self {
        Self {
            matrix: Matrix::empty(),
            svd: LinalgWorkspace::new(),
            decomposition: ThinSvdDecomposition::empty(),
            decomposition_shape: [0; 2],
            candidate: Support::default(),
            trial: Support::default(),
            residuals: Vec::new(),
            lawson_weights: Vec::new(),
            selected: Vec::new(),
            support_indices: Vec::new(),
            node_indices: Vec::new(),
            history: Vec::new(),
            state: SolveState::new(),
        }
    }

    /// Computes the storage needed to fit `request.samples()` samples with
    /// `request.options()`, without allocating.
    ///
    /// The requirements cover every Loewner and Lawson system a fit can form,
    /// for any executor. The options are checked first.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `max_terms` is zero, or the number of samples is
    ///   zero or less than `max_terms`.
    /// - `SizeOverflow` if a dimension overflows `usize`.
    /// - `ResourceLimit` if the SVD exceeds its memory limit.
    /// - The errors of the SVD workspace query for invalid SVD options.
    pub fn workspace_requirements(
        request: RationalRequest<C::Real>,
    ) -> Result<RationalRequirements, SciOxideError> {
        let options = request.options;
        let samples = request.samples;
        if options.max_terms == 0 {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "AAA requires at least one support point",
            ));
        }
        let svd_options = options.effective_svd();
        let thin = |rows: usize, columns: usize| {
            ThinSvdDecomposition::<C>::workspace_requirements(&SvdWorkspaceRequest::new(
                rows,
                columns,
                &svd_options,
            ))
        };
        thin(1, 1)?;
        let terms = options.max_terms;
        if samples == 0 || terms > samples {
            return Err(invalid(
                scioxide_core::argument!("samples"),
                "AAA needs at least as many samples as support points",
            ));
        }
        let lawson = options.lawson_iterations;
        // AAA solves for one weight per support point; Lawson solves for
        // separate numerator and denominator coefficients (2 per point). Rows
        // are padded to at least the column count because the thin SVD only
        // exposes a full right-singular basis (and hence the null space) when
        // rows >= columns; padding rows are zero and do not change the problem.
        let columns = terms
            .checked_mul(if lawson == 0 { 1 } else { 2 })
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("rational_svd_columns"))
            })?;
        let rows = samples.max(columns);
        rows.checked_mul(columns).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("rational_svd_elements"))
        })?;
        let history = terms
            .checked_add(lawson)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("rational_history"))
            })?;
        // Requirements need not be monotone in the shape (tall systems reduce by
        // QR first), so every system a fit may form is covered.
        let mut svd = LinalgRequirements::default();
        for k in 1..=terms {
            svd = svd.max(&thin(samples.max(k), k)?);
            if lawson > 0 {
                svd = svd.max(&thin(samples.max(2 * k), 2 * k)?);
            }
        }
        Ok(RationalRequirements {
            samples,
            terms,
            lawson_iterations: lawson,
            matrix_rows: rows,
            matrix_columns: columns,
            history,
            svd,
        })
    }

    /// Allocates a workspace reserved for [`Self::workspace_requirements`] of `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(request: RationalRequest<C::Real>) -> Result<Self, SciOxideError> {
        let mut workspace = Self::empty();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Checks whether every buffer is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a byte count overflows `usize`.
    pub fn check(&self, requirements: &RationalRequirements) -> Result<Capacity, SciOxideError> {
        let matrix_bytes = scioxide_core::backend::checked_product(
            &[
                requirements.matrix_rows,
                requirements.matrix_columns,
                core::mem::size_of::<C>(),
            ],
            scioxide_core::quantity!("rational_matrix_bytes"),
        )?;
        let sufficient = self.matrix.retained_bytes()? >= matrix_bytes
            && self.decomposition_shape[0] >= requirements.matrix_rows
            && self.decomposition_shape[1] >= requirements.matrix_columns
            && self.svd.check(&requirements.svd)? == Capacity::Sufficient
            && self.candidate.capacity() >= requirements.terms
            && self.trial.capacity() >= requirements.terms
            && self.residuals.capacity() >= requirements.samples
            && self.lawson_weights.capacity() >= requirements.samples
            && self.selected.capacity() >= requirements.samples
            && self.node_indices.capacity() >= requirements.samples
            && self.support_indices.capacity() >= requirements.terms
            && self.history.capacity() >= requirements.history;
        Ok(if sufficient {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Grows every buffer to cover `requirements`; existing capacity is kept.
    ///
    /// A failed call may leave some buffers already grown.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be reserved.
    /// - The errors of reserving the SVD storage.
    pub fn try_reserve(
        &mut self,
        requirements: &RationalRequirements,
    ) -> Result<(), SciOxideError> {
        self.svd.try_reserve(&requirements.svd)?;
        if self.decomposition_shape[0] < requirements.matrix_rows
            || self.decomposition_shape[1] < requirements.matrix_columns
        {
            let rows = self.decomposition_shape[0].max(requirements.matrix_rows);
            let columns = self.decomposition_shape[1].max(requirements.matrix_columns);
            self.matrix.try_resize_flat(rows, columns)?;
            self.decomposition.try_reserve(rows, columns)?;
            self.decomposition_shape = [rows, columns];
        } else {
            let [rows, columns] = self.decomposition_shape;
            self.matrix.try_resize_flat(rows, columns)?;
        }
        self.candidate.try_reserve(requirements.terms)?;
        self.trial.try_reserve(requirements.terms)?;
        reserve(&mut self.residuals, requirements.samples)?;
        reserve(&mut self.lawson_weights, requirements.samples)?;
        reserve(&mut self.selected, requirements.samples)?;
        reserve(&mut self.node_indices, requirements.samples)?;
        reserve(&mut self.support_indices, requirements.terms)?;
        reserve(&mut self.history, requirements.history)
    }

    /// Returns the residual history of the last fit.
    ///
    /// Entry `k` is the maximum residual at the samples of the `k`-th accepted
    /// approximant; entry 0 belongs to the sample mean.
    pub fn residual_history(&self) -> &[C::Real] {
        &self.history
    }
}

/// Holds the result of an AAA fit: the approximant and its report.
///
/// Both are `None` until a fit succeeds, and both are set by every
/// successful fit. Reserve storage with [`Self::try_reserve`] before calling
/// [`aaa_into`].
///
/// # Examples
///
/// ```
/// use scioxide_interp::{AaaOptions, RationalFit, RationalRequest, RationalWorkspace};
/// let request = RationalRequest::new(5, AaaOptions::new().with_max_terms(3));
/// let mut fit = RationalFit::<f64>::new();
/// fit.try_reserve(&RationalWorkspace::<f64>::workspace_requirements(request)?)?;
/// assert!(fit.approximant().is_none() && fit.report().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct RationalFit<C: NumericScalar> {
    approximant: Option<BarycentricRational<C>>,
    /// Coefficient storage reserved before the first fit.
    spare: Support<C>,
    report: Option<RationalReport<C::Real>>,
}

impl<C: NumericScalar> Default for RationalFit<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: NumericScalar> RationalFit<C> {
    /// Creates an empty result without storage.
    pub fn new() -> Self {
        Self {
            approximant: None,
            spare: Support::default(),
            report: None,
        }
    }

    /// Reserves coefficient storage for `requirements`; a stored result is unchanged.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &RationalRequirements,
    ) -> Result<(), SciOxideError> {
        match &mut self.approximant {
            Some(approximant) => approximant.support.try_reserve(requirements.terms),
            None => self.spare.try_reserve(requirements.terms),
        }
    }

    /// Returns the approximant of the last successful fit.
    pub fn approximant(&self) -> Option<&BarycentricRational<C>> {
        self.approximant.as_ref()
    }

    /// Returns the report of the last successful fit.
    pub fn report(&self) -> Option<RationalReport<C::Real>> {
        self.report
    }

    /// Returns the approximant of the last successful fit, consuming the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_interp::{AaaOptions, RationalDomain, Univariate, aaa_with_options};
    /// let (x, y) = ([-1.0_f64, 0.0, 1.0], [1.0, 2.0, 3.0]);
    /// let domain = RationalDomain::Real(Interval::new(-1.0, 1.0)?);
    /// let options = AaaOptions::new().with_max_terms(3);
    /// let fit = aaa_with_options(&x, &y, domain, options)?;
    /// let line = fit.into_approximant().unwrap();
    /// assert!((line.value_at(0.5)? - 2.5).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn into_approximant(self) -> Option<BarycentricRational<C>> {
        self.approximant
    }

    /// Returns the storage the next fit exchanges with the workspace.
    fn storage(&self) -> &Support<C> {
        self.approximant
            .as_ref()
            .map_or(&self.spare, BarycentricRational::support)
    }
}

/// Grow `v` to capacity `n`.
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), SciOxideError> {
    try_reserve(
        v,
        n.saturating_sub(v.len()),
        scioxide_core::quantity!("rational_workspace"),
    )
}

/// Validated samples of one solve.
#[derive(Clone, Copy)]
struct Samples<'a, C: NumericScalar> {
    nodes: &'a [C],
    values: &'a [C],
    /// Maximum sample magnitude, or one for zero data.
    scale: C::Real,
    threshold: C::Real,
}

/// Mutable iterate of one solve.
struct Progress<R> {
    termination: RationalTermination,
    iterations: usize,
    max_residual: R,
    singular_ratio: R,
}

/// AAA and optional Lawson fit; all candidate state stays in the workspace.
fn fit<C: NumericScalar>(
    options: &AaaOptions<C::Real>,
    samples: Samples<'_, C>,
    w: &mut RationalWorkspace<C>,
    context: &mut SolveContext<'_, '_>,
) -> Result<RationalReport<C::Real>, SciOxideError> {
    context.execution().check()?;
    let Samples {
        nodes,
        values,
        scale,
        threshold,
    } = samples;
    let n = nodes.len();
    let m = options.max_terms;
    let svd_options = options.effective_svd();
    w.candidate.clear();
    w.history.clear();
    w.selected.clear();
    w.selected.resize(n, false);
    w.support_indices.clear();
    w.residuals.clear();
    w.residuals.resize(n, C::Real::ZERO);
    // Initial approximant (AAA step 0): the constant sample mean, represented
    // as a one-point barycentric form a/(z - z0) / (1/(z - z0)) = a. The mean
    // is accumulated on normalized values to avoid overflow. Its placeholder
    // support point is discarded when the first real support is chosen.
    let count = C::Real::try_from_usize_exact(n)?;
    let mean = intermediate(
        values.iter().fold(C::ZERO, |s, &f| {
            s + (f / C::from_real(scale)) / C::from_real(count)
        }) * C::from_real(scale),
    )?;
    w.candidate.nodes.push(nodes[0]);
    w.candidate.numerator.push(mean);
    w.candidate.denominator.push(C::ONE);
    let mut progress = Progress {
        termination: RationalTermination::SupportLimit,
        iterations: 0,
        max_residual: residuals(&w.candidate, nodes, values, &mut w.residuals)?,
        singular_ratio: C::Real::ZERO,
    };
    w.history.push(progress.max_residual);
    for k in 1..=m {
        if progress.max_residual <= threshold {
            progress.termination = RationalTermination::Converged;
            break;
        }
        if !crate::error::admit_iteration(context)? {
            progress.termination = RationalTermination::BudgetExhausted;
            break;
        }
        progress.iterations += 1;
        // Greedy AAA step: the next support point is the unused sample with
        // the largest residual. `max_by` keeps the last maximal element, so the
        // reversed index comparison makes ties resolve to the smallest index,
        // giving a deterministic, input-order-defined choice.
        let index = (0..n)
            .filter(|&i| !w.selected[i])
            .max_by(|&a, &b| {
                w.residuals[a]
                    .partial_cmp(&w.residuals[b])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then_with(|| b.cmp(&a))
            })
            .ok_or_else(|| {
                instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "no unused support point",
                )
            })?;
        if k == 1 {
            w.candidate.nodes.clear();
        }
        w.selected[index] = true;
        w.support_indices.push(index);
        w.candidate.nodes.push(nodes[index]);
        // Loewner matrix A_ij = (f_i - f_{s_j}) / (z_i - s_j) over the
        // non-support samples; support rows stay zero because the barycentric
        // form interpolates there by construction. Values are divided by
        // `scale` and every entry is multiplied by the minimum sample/support
        // distance. Both are common factors, so the right singular vectors
        // (and the weights) are unchanged, but entries stay bounded by
        // roughly 2 in magnitude instead of growing like 1/distance.
        w.matrix.try_resize_flat(n.max(k), k)?;
        w.matrix.as_mut_slice().fill(C::ZERO);
        let mut distance = C::Real::MAX;
        for (&z, _) in nodes
            .iter()
            .zip(&w.selected)
            .filter(|(_, selected)| !**selected)
        {
            for &s in &w.candidate.nodes {
                distance = distance.min(intermediate(z - s)?.abs());
            }
        }
        for (i, (&z, &f)) in nodes.iter().zip(values).enumerate() {
            if w.selected[i] {
                continue;
            }
            for (j, &s) in w.candidate.nodes.iter().enumerate() {
                let si = w.support_indices[j];
                w.matrix.view_mut().set(
                    i,
                    j,
                    intermediate(
                        ((f / C::from_real(scale)) - (values[si] / C::from_real(scale)))
                            * (C::from_real(distance) / intermediate(z - s)?),
                    )?,
                )?;
            }
        }
        ThinSvdDecomposition::compute_into(
            &w.matrix,
            &svd_options,
            &mut w.decomposition,
            &mut w.svd,
            context.execution(),
        )?;
        progress.singular_ratio = singular_ratio(&w.decomposition);
        w.candidate.numerator.clear();
        w.candidate.denominator.clear();
        // Interpolatory AAA form: denominator weights w_j from the null vector
        // and numerator weights w_j f_j, so r(s_j) = f_j at every support point.
        for j in 0..w.candidate.nodes.len() {
            let weight = null_weight(&w.decomposition, j)?;
            let si = w.support_indices[j];
            w.candidate.denominator.push(weight);
            w.candidate
                .numerator
                .push(intermediate(weight * values[si])?);
        }
        progress.max_residual = residuals(&w.candidate, nodes, values, &mut w.residuals)?;
        w.history.push(progress.max_residual);
    }
    if progress.max_residual <= threshold {
        progress.termination = RationalTermination::Converged;
    }
    if options.lawson_iterations > 0
        && progress.max_residual > threshold
        && progress.termination != RationalTermination::BudgetExhausted
    {
        lawson(options, samples, &svd_options, w, &mut progress, context)?;
    }
    Ok(RationalReport::new(
        progress.termination,
        progress.iterations,
        progress.max_residual,
        progress.singular_ratio,
    ))
}

/// Deterministic null-space weight.
fn null_weight<C: NumericScalar>(
    svd: &ThinSvdDecomposition<C>,
    column: usize,
) -> Result<C, SciOxideError> {
    // The weights are a unit vector minimizing ||A w||: the right singular
    // vector of the smallest singular value. When the matrix is numerically
    // rank deficient the minimizer is not unique; summing the orthonormal
    // null-space basis vectors (rows `rank..k` of V^H, conjugated to obtain
    // columns of V) and dividing by sqrt(count) yields a deterministic unit
    // vector in that null space instead of an arbitrary basis vector. At
    // least the last row is always used.
    let rank = svd
        .analyze_rank(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)?
        .rank();
    let k = svd.singular_values().len();
    let start = rank.min(k - 1);
    let normalization = C::Real::try_from_usize_exact(k - start)?.sqrt();
    (start..k).try_fold(C::ZERO, |sum, row| {
        intermediate(
            sum + svd.vt_matrix().view().get(row, column)?.conj() / C::from_real(normalization),
        )
    })
}

fn singular_ratio<C: NumericScalar>(svd: &ThinSvdDecomposition<C>) -> C::Real {
    let values = svd.singular_values();
    match (values.first(), values.last()) {
        (Some(&hi), Some(&lo)) if hi > C::Real::ZERO => lo / hi,
        _ => C::Real::ZERO,
    }
}

/// Sample residual magnitudes and their maximum.
fn residuals<C: NumericScalar>(
    support: &Support<C>,
    nodes: &[C],
    values: &[C],
    residuals: &mut [C::Real],
) -> Result<C::Real, SciOxideError> {
    let mut max = C::Real::ZERO;
    for ((&z, &f), r) in nodes.iter().zip(values).zip(residuals) {
        let value = support.evaluate(z)?.0;
        *r = crate::checked::output(intermediate(value - f)?.abs())?;
        max = max.max(*r);
    }
    Ok(max)
}

/// Bounded AAA–Lawson refinement of the accepted AAA iterate.
fn lawson<C: NumericScalar>(
    options: &AaaOptions<C::Real>,
    samples: Samples<'_, C>,
    svd_options: &LinalgExecutionOptions,
    w: &mut RationalWorkspace<C>,
    progress: &mut Progress<C::Real>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let Samples {
        nodes,
        values,
        scale,
        threshold,
    } = samples;
    let n = nodes.len();
    let k = w.candidate.nodes.len();
    let cols = 2 * k;
    w.trial.clear();
    w.trial.nodes.extend_from_slice(&w.candidate.nodes);
    w.trial.numerator.resize(k, C::ZERO);
    w.trial.denominator.resize(k, C::ZERO);
    w.lawson_weights.clear();
    w.lawson_weights.resize(n, C::Real::ONE);
    progress.termination = RationalTermination::IterationLimit;
    // Common column scale for the Cauchy entries, as in the AAA Loewner
    // matrix; capped at one so it never enlarges entries.
    let mut distance = C::Real::ONE;
    for &z in nodes {
        if !w.trial.nodes.contains(&z) {
            for &s in &w.trial.nodes {
                distance = distance.min(intermediate(z - s)?.abs());
            }
        }
    }
    for _ in 0..options.lawson_iterations {
        if !crate::error::admit_iteration(context)? {
            progress.termination = RationalTermination::BudgetExhausted;
            break;
        }
        progress.iterations += 1;
        // Linearized weighted least squares (AAA–Lawson): minimize
        //   sum_i lambda_i |N(z_i) - f_i D(z_i)|^2  over unit-norm [a; b],
        // with N = sum a_j C_ij, D = sum b_j C_ij and Cauchy entries
        // C_ij = 1/(z_i - s_j). Row i is [C_i, -f_i C_i] scaled by sqrt(lambda_i).
        // At a support sample the Cauchy row is singular; it is replaced by its
        // limit after multiplying through by (z_i - s_j), i.e. the unit vector
        // e_j, so that row measures a_j - f_j b_j. Unlike AAA, a and b are
        // independent, so the approximant need not interpolate.
        w.matrix.try_resize_flat(n.max(cols), cols)?;
        w.matrix.as_mut_slice().fill(C::ZERO);
        for (i, (&z, &f)) in nodes.iter().zip(values).enumerate() {
            let support = w.trial.nodes.iter().position(|&s| s == z);
            for (j, &s) in w.trial.nodes.iter().enumerate() {
                let c = match support {
                    Some(index) => {
                        if j == index {
                            C::from_real(distance)
                        } else {
                            C::ZERO
                        }
                    }
                    None => intermediate(C::from_real(distance) / intermediate(z - s)?)?,
                };
                let c = c * C::from_real(w.lawson_weights[i].sqrt());
                w.matrix.view_mut().set(i, j, c)?;
                w.matrix
                    .view_mut()
                    .set(i, j + k, intermediate(-c * (f / C::from_real(scale)))?)?;
            }
        }
        ThinSvdDecomposition::compute_into(
            &w.matrix,
            svd_options,
            &mut w.decomposition,
            &mut w.svd,
            context.execution(),
        )?;
        // The last right singular vector minimizes the weighted residual. The
        // numerator was fitted against normalized values, so undo `scale`.
        for j in 0..k {
            w.trial.numerator[j] = intermediate(
                w.decomposition.vt_matrix().view().get(cols - 1, j)?.conj() * C::from_real(scale),
            )?;
            w.trial.denominator[j] = w
                .decomposition
                .vt_matrix()
                .view()
                .get(cols - 1, k + j)?
                .conj();
        }
        // A trial whose evaluation fails (e.g. a pole on a sample) ends the
        // iteration but keeps the last accepted approximant. Only strictly
        // improving trials are accepted, so the published maximum residual is
        // monotone non-increasing even though Lawson itself is not.
        let Ok(error) = residuals(&w.trial, nodes, values, &mut w.residuals) else {
            progress.termination = RationalTermination::Stagnation;
            break;
        };
        if error < progress.max_residual {
            w.candidate.numerator.copy_from_slice(&w.trial.numerator);
            w.candidate
                .denominator
                .copy_from_slice(&w.trial.denominator);
            progress.max_residual = error;
            progress.singular_ratio = singular_ratio(&w.decomposition);
            w.history.push(error);
        }
        if error <= threshold {
            progress.termination = RationalTermination::Converged;
            break;
        }
        // Lawson update lambda_i <- lambda_i |e_i| / max|e|, shifting weight
        // towards the samples with the largest error (the fixed point is the
        // minimax approximation). Renormalizing to a unit maximum stops the
        // weights drifting towards underflow over many iterations.
        let mut maximum = C::Real::ZERO;
        for (weight, &residual) in w.lawson_weights.iter_mut().zip(&w.residuals) {
            *weight *= residual / error;
            maximum = maximum.max(*weight);
        }
        if maximum == C::Real::ZERO {
            progress.termination = RationalTermination::Stagnation;
            break;
        }
        for weight in &mut w.lawson_weights {
            *weight /= maximum;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::assert_error;
    use scioxide_core::{Complex, Interval};

    fn domain() -> RationalDomain<f64> {
        RationalDomain::Real(Interval::new(-1., 1.).unwrap())
    }

    #[test]
    fn finite_complex_components_do_not_publish_an_infinite_residual() {
        let zero = Complex::new(0.0_f64, 0.);
        let support = Support {
            nodes: alloc::vec![zero],
            numerator: alloc::vec![Complex::new(1.4e308, 1.4e308)],
            denominator: alloc::vec![Complex::new(1., 0.)],
        };
        assert!(support.evaluate(zero).is_ok());
        assert_error(
            &residuals(&support, &[zero], &[zero], &mut [0.]),
            scioxide_core::SciOxideErrorKind::NonFinite,
        );
    }

    fn requirements(
        samples: usize,
        max_terms: usize,
    ) -> Result<RationalRequirements, SciOxideError> {
        RationalWorkspace::<f64>::workspace_requirements(RationalRequest::new(
            samples,
            options(max_terms),
        ))
    }

    fn options(max_terms: usize) -> AaaOptions<f64> {
        AaaOptions::new()
            .with_max_terms(max_terms)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
    }

    #[test]
    fn requirements_validate_options_and_sample_counts() {
        assert_error(
            &requirements(3, 0),
            scioxide_core::SciOxideErrorKind::InvalidInput,
        );
        assert_error(
            &requirements(0, 0),
            scioxide_core::SciOxideErrorKind::InvalidInput,
        );
        assert!(requirements(2, 3).is_err());
        assert!(requirements(0, 3).is_err());
        let requirements = requirements(3, 3).unwrap();
        assert_eq!(requirements.samples(), 3);
        assert_eq!(requirements.terms(), 3);
        assert_eq!(requirements.history(), 4);
    }

    #[test]
    fn insufficient_storage_is_a_dimension_mismatch_and_preserves_output() {
        let x = [-1.0_f64, -0.5, 0., 0.5, 1.];
        let y = x.map(|x| 1. / (2. + x));
        let (small, large) = (options(1), options(4));
        let mut workspace = RationalWorkspace::workspace(RationalRequest::new(5, small)).unwrap();
        let mut out = RationalFit::new();
        out.try_reserve(&requirements(5, 1).unwrap()).unwrap();
        let cx = || ExecutionContext::sequential();
        aaa_into(&x, &y, domain(), small, &mut out, &mut workspace, cx()).unwrap();
        let before = out.report().unwrap();
        let error = aaa_into(&x, &y, domain(), large, &mut out, &mut workspace, cx()).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(out.report().unwrap(), before);
        workspace.try_reserve(&requirements(5, 4).unwrap()).unwrap();
        let error = aaa_into(&x, &y, domain(), large, &mut out, &mut workspace, cx()).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        out.try_reserve(&requirements(5, 4).unwrap()).unwrap();
        aaa_into(&x, &y, domain(), large, &mut out, &mut workspace, cx()).unwrap();
        assert_eq!(
            out.report().unwrap().termination(),
            RationalTermination::Converged
        );
    }

    #[test]
    fn node_domains_are_the_box_of_the_nodes() {
        let real = node_domain(&[0.5_f64, -2.0, 1.0]).unwrap();
        assert_eq!(
            real,
            RationalDomain::Real(Interval::new(-2.0, 1.0).unwrap())
        );
        let axis = node_domain(&[Complex::new(1.0_f64, 0.0), Complex::new(-1.0, -0.0)]).unwrap();
        assert_eq!(
            axis,
            RationalDomain::Real(Interval::new(-1.0, 1.0).unwrap())
        );
        let plane = node_domain(&[Complex::new(1.0_f64, 0.5), Complex::new(-1.0, -0.25)]).unwrap();
        assert_eq!(
            plane,
            RationalDomain::Complex {
                real: Interval::new(-1.0, 1.0).unwrap(),
                imaginary: Interval::new(-0.25, 0.5).unwrap(),
            }
        );
    }
}
