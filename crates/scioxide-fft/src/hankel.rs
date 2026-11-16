//! Prepared Bessel-zero radial quadrature.
use crate::axis::AxisPlan;
use crate::numeric::compensated_term;
use crate::validation::{invalid, length};
use crate::validation::{nonfinite, numerical};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    Budget, Complex, ExecutionContext, RealScalar, SciOxideError, SolveContext, Tolerance,
};

use crate::memory::admit_bytes;
use crate::resource::Work;
/// Sets the storage and work limits of a [`DiscreteHankelPlan`].
///
/// Every limit defaults to unbounded. Exceeding a byte limit is a
/// `ResourceLimit` error, and exhausting `preparation_work` is a
/// `ConvergenceFailure`.
#[derive(Debug, Clone, Copy)]
pub struct HankelLimits {
    /// Maximum bytes live at once during preparation, excluding allocator metadata.
    pub preparation_bytes: usize,
    /// Maximum bytes of an execution workspace.
    pub workspace_bytes: usize,
    /// Maximum preparation work, counted in function calls and Bessel-function
    /// iterations.
    pub preparation_work: Budget,
}
impl Default for HankelLimits {
    fn default() -> Self {
        Self {
            preparation_bytes: usize::MAX,
            workspace_bytes: usize::MAX,
            preparation_work: Budget::ONE.saturating_add(usize::MAX),
        }
    }
}
/// Reports how closely the supplied zeros satisfy `J_ν(z) = 0`, and the preparation work.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteHankelDiagnostics {
    /// Largest residual `|J_ν(z_m)|` over the supplied zeros.
    pub maximum_zero_residual: f64,
    /// Largest Newton correction `|J_ν(z_m) / J_ν′(z_m)|` over the supplied zeros.
    pub maximum_zero_correction: f64,
    /// Work used during preparation, counted as in [`HankelLimits::preparation_work`].
    pub preparation_work: usize,
}
/// Computes the discrete Hankel transform on the zeros of a Bessel function.
///
/// Let `j_1 < j_2 < … < j_(n+1)` be the first `n + 1` positive zeros of `J_ν`,
/// which you supply, and let `R` be the radial extent. The plan samples a
/// function at the nodes `r_m = R j_m / j_(n+1)` and its transform at the
/// frequencies `k_m = j_m / R`, for `m = 1, …, n`. The forward transform is
///
/// ```text
/// F_i = (R / j_(n+1))² Σ_m w_m J_ν(j_i j_m / j_(n+1)) f_m,   w_m = 2 / J_(ν+1)(j_m)²,
/// ```
///
/// a quadrature for `F(k_i) = ∫₀^R f(r) J_ν(k_i r) r dr` with `f_m = f(r_m)`.
/// The inverse transform uses the same sum with the factor `1 / R²` in place of
/// `(R / j_(n+1))²`; it maps samples at `k_m` to values at `r_m`. The two
/// directions are quadratures, not exact inverses of each other. The order must
/// satisfy `ν > −1/2`. Through [`AdjointOperator`](scioxide_core::AdjointOperator)
/// a plan applies the transpose of its own matrix.
///
/// Preparation checks the supplied zeros (see [`Self::new`]), evaluates the
/// `n(n+1)/2` distinct Bessel values of the symmetric kernel, and stores an
/// `n × n` matrix. Each execution costs `O(n²)` operations: a matrix-vector
/// product with compensated summation in a fixed order, so the result does
/// not depend on the number of workers. Rows and inputs carry separate binary
/// exponents, so the scale factors may lie outside the range of the scalar
/// type. Each execution counts one linear-operator evaluation and `n`
/// iterations per transformed lane.
///
/// # Accuracy
///
/// Let `m_ij` be the stored matrix of the plan's direction (its transpose for
/// the adjoint) and `x_j` the real or imaginary parts of the input. Each output
/// component is within `8 ε (n + 2) Σ_j |m_ij| |x_j|`, plus two units of the
/// smallest subnormal, of the compensated dot product of row `i` with `x`
/// evaluated in extended exponent range. The quadrature error itself depends
/// on `f`.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_fft::{DiscreteHankelPlan, DiscreteHankelPlanSpec, FftDirection, dht};
///
/// // The positive zeros of J_{1/2} are kπ; six zeros give five nodes.
/// let zeros: [f64; 6] = core::array::from_fn(|k| (k + 1) as f64 * PI);
/// let spec = DiscreteHankelPlanSpec::new(0.5, 4.0, FftDirection::Forward);
/// let plan = DiscreteHankelPlan::new(&zeros, spec)?;
/// assert!((plan.nodes()[0] - 4.0 * zeros[0] / zeros[5]).abs() < 1e-12);
/// let samples = [1.0, -0.5, 2.0, 0.25, 0.75];
/// let (input, mut output) = (samples.map(|x| Complex::new(x, 0.0)), [Complex::new(0.0, 0.0); 5]);
/// let cx = ExecutionContext::sequential();
/// plan.execute_into(&input, &mut output, &mut plan.workspace(())?, cx)?;
/// let real = dht(&samples, &zeros, 0.5, 4.0)?;
/// assert!(output.iter().zip(real).all(|(z, x)| (z.re - x).abs() < 1e-12 && z.im == 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - H. F. Johnson, "An improved method for computing a discrete Hankel
///   transform", *Comput. Phys. Commun.* 43, 181–202, 1987.
/// - D. Lemoine, "The discrete Bessel transform algorithm", *J. Chem. Phys.*
///   101, 3936–3944, 1994.
/// - NIST DLMF, §10.21 (zeros of Bessel functions). <https://dlmf.nist.gov/10.21>
#[derive(Debug, Clone)]
pub struct DiscreteHankelPlan<S: RealScalar> {
    plan_bytes: usize,
    spec: DiscreteHankelPlanSpec<S>,
    n: usize,
    nodes: Vec<S>,
    frequencies: Vec<S>,
    matrix: Vec<S>,
    weights: Vec<S>,
    row_exponents: Vec<i32>,
    scale_mantissa: S,
    scale_exponent: i32,
    diagnostics: DiscreteHankelDiagnostics,
}
/// Describes a discrete Hankel transform: order, radial extent, direction and zero tolerance.
///
/// The parameters are defined on [`DiscreteHankelPlan`]. A spec checks the
/// supplied zeros with the tolerance `a = r = √ε` of
/// [`Tolerance::from_sqrt_epsilon`], where `ε` is the machine epsilon of `S`
/// (see [`Self::with_tolerance`]), runs sequentially with no iteration budget
/// and unbounded limits, and is not validated until a plan is prepared from it.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_fft::{DiscreteHankelPlan, DiscreteHankelPlanSpec, FftDirection, HankelLimits};
///
/// let spec = DiscreteHankelPlanSpec::new(0.5_f64, 4.0, FftDirection::Forward);
/// assert_eq!(spec.limits().workspace_bytes, usize::MAX);
/// let limits = HankelLimits {
///     workspace_bytes: 64,
///     ..HankelLimits::default()
/// };
/// let zeros: [f64; 6] = core::array::from_fn(|k| (k + 1) as f64 * PI);
/// let spec = spec.with_limits(limits);
/// let error = DiscreteHankelPlan::new(&zeros, spec).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DiscreteHankelPlanSpec<S: RealScalar> {
    direction: crate::FftDirection,
    order: S,
    extent: S,
    tolerance: Tolerance<S>,
    limits: HankelLimits,
    execution_policy: scioxide_core::ExecutionPolicy,
    iteration_budget: Option<Budget>,
}
impl<S: RealScalar> DiscreteHankelPlanSpec<S> {
    /// Returns the transform direction.
    pub const fn direction(&self) -> crate::FftDirection {
        self.direction
    }
    /// Returns the Bessel order `ν`.
    pub const fn order(&self) -> S {
        self.order
    }
    /// Returns the radial extent `R`.
    pub const fn extent(&self) -> S {
        self.extent
    }
    /// Returns the tolerance that the supplied zeros must meet.
    pub const fn tolerance(&self) -> Tolerance<S> {
        self.tolerance
    }
    /// Returns the storage and work limits.
    pub const fn limits(&self) -> HankelLimits {
        self.limits
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> scioxide_core::ExecutionPolicy {
        self.execution_policy
    }
    /// Returns the iteration budget of each execution, if any.
    pub const fn iteration_budget(&self) -> Option<Budget> {
        self.iteration_budget
    }

    /// Creates a spec for Bessel order `ν = order`, radial extent `R = extent` and
    /// the given direction.
    ///
    /// The zero tolerance defaults to [`Tolerance::from_sqrt_epsilon`].
    pub fn new(order: S, extent: S, direction: crate::FftDirection) -> Self {
        Self {
            order,
            extent,
            direction,
            tolerance: Tolerance::from_sqrt_epsilon(),
            limits: HankelLimits::default(),
            execution_policy: scioxide_core::ExecutionPolicy::Sequential,
            iteration_budget: None,
        }
    }
    /// Sets the tolerance that the supplied zeros must meet.
    ///
    /// With `a` and `r` the absolute and relative parts of `tolerance`, the
    /// `k`-th supplied zero `z_k` is accepted when its Newton correction
    /// `c_k = |J_ν(z_k) / J_ν′(z_k)|` is at most `max(a, r z_k)`, `J_ν′(z_k)`
    /// has the sign `(−1)^k` of the `k`-th positive zero, and `z_k` lies within
    /// `max(a, r z_k)` of a bracket of the `k`-th zero that preparation finds
    /// by itself. An accepted zero is stored unchanged, so it moves its node
    /// by the relative amount `c_k / z_k ≤ max(a / z_k, r)`. The default
    /// `a = r = √ε` accepts zeros rounded to working precision, whose
    /// correction is about `ε z_k / 2`; tighten it to require full-precision
    /// zeros.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Limits the iterations of each execution; each transformed lane costs `n`
    /// iterations.
    pub const fn with_iteration_budget(mut self, budget: Budget) -> Self {
        self.iteration_budget = Some(budget);
        self
    }
    /// Sets the preparation and workspace limits.
    pub const fn with_limits(mut self, limits: HankelLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Sets the execution policy of the matrix-vector product.
    pub const fn with_execution_policy(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }
}

impl<S: RealScalar> DiscreteHankelPlan<S> {
    /// Returns the bytes held by the plan, excluding allocator metadata.
    pub const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("fft_plan_bytes");
        [
            crate::memory::vector_bytes(&self.nodes),
            crate::memory::vector_bytes(&self.frequencies),
            crate::memory::vector_bytes(&self.matrix),
            crate::memory::vector_bytes(&self.weights),
            crate::memory::vector_bytes(&self.row_exponents),
        ]
        .into_iter()
        .try_fold(size_of::<Self>(), |a, b| checked_add(a, b, q))
    }

    /// Returns the spec the plan was prepared from.
    pub const fn spec(&self) -> &DiscreteHankelPlanSpec<S> {
        &self.spec
    }
    /// Returns an empty iterator: the discrete Hankel transform uses no FFT.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> {
        core::iter::empty()
    }

    /// Prepares a plan from the first `n + 1` positive zeros of `J_ν`, on the calling thread.
    ///
    /// Each zero must meet the tolerance of the spec (see
    /// [`DiscreteHankelPlanSpec::with_tolerance`]), and the zeros must be the
    /// first `n + 1` in order: preparation brackets the zero of each index with
    /// Bessel-function evaluations and rejects a sequence that skips one. For
    /// `ν = 1/2` the zeros are `kπ`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (attributed to `zeros`) if fewer than two zeros are
    ///   given, or if the zeros are not positive, increasing, accurate and
    ///   consecutive.
    /// - `InvalidInput` if `ν ≤ −1/2` or `R ≤ 0`.
    /// - `NonFinite` if a zero, the order or the extent is NaN or infinite.
    /// - `NumericalInstability` if a node or frequency is not representable, or
    ///   (`Underflow`) if a quadrature weight rounds to zero.
    /// - `ResourceLimit` if storage exceeds [`HankelLimits`], and
    ///   `ConvergenceFailure` if the preparation work is exhausted.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::f64::consts::PI;
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_fft::{DiscreteHankelPlan, DiscreteHankelPlanSpec, FftDirection};
    ///
    /// let mut zeros: [f64; 6] = core::array::from_fn(|k| (k + 1) as f64 * PI);
    /// zeros[2] += 0.01; // 3π + 0.01 is not a zero of J_{1/2}
    /// let spec = DiscreteHankelPlanSpec::new(0.5, 4.0, FftDirection::Forward);
    /// let error = DiscreteHankelPlan::new(&zeros, spec).unwrap_err();
    /// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(zeros: &[S], spec: DiscreteHankelPlanSpec<S>) -> Result<Self, SciOxideError> {
        Self::build(
            zeros,
            spec,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan whose preparation work counts toward `scope`.
    ///
    /// `scope` is an execution context or a parent solve context; see
    /// [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], plus cancellation and deadline errors from
    /// `scope`.
    pub fn build<'scope, 'cap>(
        zeros: &[S],
        spec: DiscreteHankelPlanSpec<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<Self, SciOxideError> {
        let scope = scope.into();
        scope.execution().check()?;
        let mut state = scioxide_core::SolveState::new();
        scope.run(
            &mut state,
            crate::workspace::identity(zeros.len()),
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Iteration,
                spec.limits.preparation_work,
            ),
            |child| Self::prepare(zeros, spec, child),
        )
    }

    // Shared preparation.
    fn prepare(
        zeros: &[S],
        spec: DiscreteHankelPlanSpec<S>,
        parent: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let context = parent.execution();
        let DiscreteHankelPlanSpec {
            order,
            extent,
            tolerance,
            limits,
            ..
        } = spec;
        let zeros_argument = scioxide_core::argument!("zeros");
        crate::validation::nonzero(zeros.len(), zeros_argument)?;
        let n = zeros.len() - 1;
        crate::validation::nonzero(n, zeros_argument)?;
        if zeros.iter().any(|x| !x.is_finite()) {
            return Err(nonfinite());
        }
        if zeros.iter().any(|x| *x <= S::ZERO) || zeros.windows(2).any(|p| p[0] >= p[1]) {
            return Err(invalid(
                scioxide_core::InvalidInputReason::OutOfRange,
                zeros_argument,
                "supplied zeros must be finite, positive and strictly increasing",
            ));
        }
        if !order.is_finite() || !extent.is_finite() {
            return Err(nonfinite());
        }
        if order <= -S::HALF || extent <= S::ZERO {
            return Err(invalid(
                scioxide_core::InvalidInputReason::OutOfRange,
                if order <= -S::HALF {
                    scioxide_core::argument!("spec.order")
                } else {
                    scioxide_core::argument!("spec.extent")
                },
                "positive extent, finite order > -1/2 required",
            ));
        }
        let cells = checked_mul(n, n, scioxide_core::quantity!("dht_matrix_cells"))?;
        let bytes = checked_add(
            checked_mul(
                checked_add(
                    cells,
                    checked_mul(n, 6, scioxide_core::quantity!("dht_vector_entries"))?,
                    scioxide_core::quantity!("dht_storage_entries"),
                )?,
                size_of::<S>().max(8),
                scioxide_core::quantity!("dht_preparation_bytes"),
            )?,
            size_of::<Self>(),
            scioxide_core::quantity!("dht_preparation_bytes"),
        )?;
        admit_bytes(
            bytes,
            limits.preparation_bytes,
            scioxide_core::quantity!("preparation_bytes"),
        )?;
        admit_bytes(
            checked_mul(
                n,
                2 * size_of::<Complex<S>>(),
                scioxide_core::quantity!("dht_workspace_bytes"),
            )?,
            limits.workspace_bytes,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let mut work = Work::new(limits.preparation_work, parent);
        work.check(context)?;
        let nu = order.try_to_f64_exact()?;
        let radius = extent.try_to_f64_exact()?;
        let (roots, max_residual, max_correction) =
            Self::verify_zeros(zeros, nu, tolerance, &mut work, context)?;
        let boundary = roots[n];
        let mut nodes = try_filled(n, S::ZERO, scioxide_core::quantity!("dht_nodes"))?;
        let mut frequencies = try_filled(n, S::ZERO, scioxide_core::quantity!("dht_frequencies"))?;
        for ((node, frequency), &root) in nodes.iter_mut().zip(&mut frequencies).zip(&roots) {
            *node = Scaled::from_value(radius)?
                .mul(Scaled::from_value(root / boundary)?)?
                .round_to()?;
            *frequency = Scaled::from_value(root)?
                .div(Scaled::from_value(radius)?)?
                .round_to()?;
        }
        let weights = Self::prepare_weights(&roots[..n], nu, &mut work, context)?;
        let (matrix, row_exponents) = Self::prepare_rows(&roots, &weights, nu, &mut work, context)?;
        let (scale_mantissa, scale_exponent) =
            Self::prepare_scale(radius, boundary, spec.direction)?;
        drop(roots);
        // A = s K diag(w), K symmetric: A^T z = w A(z / w).
        let mut scalar_weights = try_filled(n, S::ZERO, scioxide_core::quantity!("dht_weights"))?;
        for (out, weight) in scalar_weights.iter_mut().zip(weights) {
            *out = crate::validation::project::<S>(weight)?;
            if *out == S::ZERO {
                return Err(numerical(
                    scioxide_core::NumericalIssue::Underflow,
                    "DHT weight must remain positive",
                ));
            }
        }
        work.check(context)?;
        let mut plan = Self {
            plan_bytes: 0,
            spec,
            n,
            nodes,
            frequencies,
            matrix,
            weights: scalar_weights,
            row_exponents,
            scale_mantissa,
            scale_exponent,
            diagnostics: DiscreteHankelDiagnostics {
                maximum_zero_residual: max_residual,
                maximum_zero_correction: max_correction,
                preparation_work: work.used,
            },
        };
        plan.plan_bytes = plan.retained_bytes()?;
        Ok(plan)
    }
    // Verify the coordinate accuracy and index separately: derivative parity
    // alone cannot detect an even number of omitted roots.
    fn verify_zeros(
        zeros: &[S],
        nu: f64,
        tolerance: Tolerance<S>,
        work: &mut Work<'_, '_, '_>,
        context: ExecutionContext<'_>,
    ) -> Result<(Vec<f64>, f64, f64), SciOxideError> {
        let mut roots = try_filled(zeros.len(), 0.0, scioxide_core::quantity!("dht_zero_roots"))?;
        let mut maximum_residual = 0.0_f64;
        let mut maximum_correction = 0.0_f64;
        let absolute = tolerance.absolute().try_to_f64_exact()?;
        let relative = tolerance.relative().try_to_f64_exact()?;
        let mut brackets = ZeroBrackets::new(nu);
        for (index, (&supplied, root)) in zeros.iter().zip(&mut roots).enumerate() {
            let x = supplied.try_to_f64_exact()?;
            let value = work.bessel(nu, x, context)?;
            let derivative = nu * value / x - work.bessel(nu + 1.0, x, context)?;
            let correction = (value / derivative).abs();
            let threshold = absolute.max(relative * x);
            if !correction.is_finite()
                || correction > threshold
                || (derivative < 0.0) != index.is_multiple_of(2)
            {
                return Err(invalid(
                    scioxide_core::InvalidInputReason::OutOfRange,
                    scioxide_core::argument!("zeros"),
                    "supplied Bessel zero does not meet its coordinate tolerance and derivative parity",
                ));
            }
            let (lower, upper) = if nu == 0.5 {
                let expected = (index + 1) as f64 * core::f64::consts::PI;
                (expected, expected)
            } else {
                brackets.next(nu, work, context)?
            };
            if x < lower - threshold || x > upper + threshold {
                return Err(invalid(
                    scioxide_core::InvalidInputReason::OutOfRange,
                    scioxide_core::argument!("zeros"),
                    "supplied zeros must be the consecutive first positive Bessel zeros",
                ));
            }
            maximum_residual = maximum_residual.max(value.abs());
            maximum_correction = maximum_correction.max(correction);
            *root = x;
        }
        Ok((roots, maximum_residual, maximum_correction))
    }

    fn prepare_weights(
        roots: &[f64],
        nu: f64,
        work: &mut Work<'_, '_, '_>,
        context: ExecutionContext<'_>,
    ) -> Result<Vec<f64>, SciOxideError> {
        let mut weights = try_filled(roots.len(), 0.0, scioxide_core::quantity!("dht_weights"))?;
        for (weight, &root) in weights.iter_mut().zip(roots) {
            let j = Scaled::from_value(work.bessel(nu + 1.0, root, context)?)?;
            *weight = Scaled::from_value(2.0)?.div(j.square()?)?.try_value()?;
        }
        Ok(weights)
    }

    // Johnson/Lemoine: A = s K diag(w), K_ij = J_nu(z_i z_j / boundary).
    // Forward s=(R/boundary)^2; dual-grid inverse s=1/R^2.
    fn prepare_scale(
        radius: f64,
        boundary: f64,
        direction: crate::FftDirection,
    ) -> Result<(S, i32), SciOxideError> {
        let radius = Scaled::from_value(radius)?;
        let scale = match direction {
            crate::FftDirection::Forward => radius.div(Scaled::from_value(boundary)?)?,
            crate::FftDirection::Inverse => Scaled::ONE.div(radius)?,
        }
        .square()?;
        Ok((
            crate::validation::project(scale.significand() * 0.5)?,
            binary_exponent(scale.exponent() + 1)?,
        ))
    }

    fn prepare_rows(
        roots: &[f64],
        weights: &[f64],
        nu: f64,
        work: &mut Work<'_, '_, '_>,
        context: ExecutionContext<'_>,
    ) -> Result<(Vec<S>, Vec<i32>), SciOxideError> {
        let n = weights.len();
        let mut matrix = try_filled(n * n, S::ZERO, scioxide_core::quantity!("dht_matrix"))?;
        let mut exponents = try_filled(n, i32::MIN, scioxide_core::quantity!("dht_row_exponents"))?;
        for (i, &a) in roots[..n].iter().enumerate() {
            for (j, &b) in roots[..=i].iter().enumerate() {
                let kernel = Scaled::from_value(work.bessel(nu, (a / roots[n]) * b, context)?)?;
                insert_coefficient(
                    &mut matrix[i * n..(i + 1) * n],
                    &mut exponents[i],
                    j,
                    kernel.mul(Scaled::from_value(weights[j])?)?,
                )?;
                if i != j {
                    insert_coefficient(
                        &mut matrix[j * n..(j + 1) * n],
                        &mut exponents[j],
                        i,
                        kernel.mul(Scaled::from_value(weights[i])?)?,
                    )?;
                }
            }
        }
        for exponent in &mut exponents {
            if *exponent == i32::MIN {
                *exponent = 0;
            }
        }
        Ok((matrix, exponents))
    }

    /// Returns the radial nodes `r_m = R j_m / j_(n+1)` in increasing order.
    pub fn nodes(&self) -> &[S] {
        &self.nodes
    }
    /// Returns the frequencies `k_m = j_m / R` in increasing order.
    pub fn frequencies(&self) -> &[S] {
        &self.frequencies
    }
    /// Returns the accuracy of the supplied zeros and the preparation work.
    pub const fn diagnostics(&self) -> DiscreteHankelDiagnostics {
        self.diagnostics
    }

    /// Required complex scratch elements, including the staged output.
    const fn scratch_len(&self) -> usize {
        2 * self.n
    }
    /// Returns the workspace size for [`Self::execute_into`] (request `()`) or for
    /// [`Self::execute_axis_into`] (an [`AxisWorkspaceRequest`](crate::AxisWorkspaceRequest)).
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` or `DimensionMismatch` if an axis request does not match
    ///   the plan.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if it exceeds [`HankelLimits::workspace_bytes`].
    pub fn workspace_requirements<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        crate::axis::requirements(self, request.into())
    }
    /// Allocates a workspace for the given request.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request)?)
    }

    /// Transforms `input` in the plan's direction and writes the result to `output`.
    ///
    /// Both slices have length `n`. `scope` is an execution context or a parent
    /// solve context; the call counts one linear-operator evaluation and `n`
    /// iterations.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a slice has the wrong length or the workspace is
    ///   too small.
    /// - `NumericalInstability` (`Overflow`) if a result exceeds the scalar range.
    /// - `ConvergenceFailure` if the iteration or evaluation budget is exhausted.
    /// - `NonFinite` and `ExecutionStopped` as described in the
    ///   [crate errors](crate#errors).
    pub fn execute_into<'scope, 'cap>(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError> {
        crate::axis::execute_radial(self, input, output, workspace, scope.into())
    }

    /// Transforms every lane of `input` along `axis` and writes the results to `output`.
    ///
    /// The other axes are batch axes. Input views may repeat elements (stride
    /// zero); output views may not. Either every lane is written or, on error,
    /// none is. The call counts one linear-operator evaluation if there is at
    /// least one lane.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::execute_into`], plus `OutOfBounds` for an invalid
    /// axis and `DimensionMismatch` for shapes that do not match the plan.
    pub fn execute_axis_into<'scope, 'cap, T: scioxide_core::NumericScalar<Real = S>>(
        &self,
        input: scioxide_core::NdView<'_, '_, T>,
        axis: usize,
        output: scioxide_core::NdViewMut<'_, '_, Complex<S>>,
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError> {
        crate::axis::execute(self, input, axis, output, workspace, scope.into())
    }
    // Shared transactional execution kernel; the result stays in `scratch`.
    fn evaluate<'w>(
        &self,
        input: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        adjoint: bool,
        parent: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        let context = parent.execution();
        length(input.len(), self.n)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        let maximum = crate::scaling::complex_max(input.iter().copied())?;
        let mut input_exponent = crate::scaling::range_exponent(maximum);
        let (normalized, scratch) = scratch.split_at_mut(self.n);
        if adjoint {
            let minimum_exponent = self
                .weights
                .iter()
                .map(|&w| crate::scaling::range_exponent(w))
                .min()
                .unwrap_or(0);
            for ((out, z), &weight) in normalized.iter_mut().zip(input).zip(&self.weights) {
                let exponent = crate::scaling::range_exponent(weight);
                let mantissa = weight.scalbn(-exponent);
                // Divide bounded components by a mantissa in [0.5, 1),
                // then shift down so every weighted component stays <= 1.
                let shift = minimum_exponent - exponent - 1;
                *out = Complex::new(
                    (z.re.scalbn(-input_exponent) / mantissa).scalbn(shift),
                    (z.im.scalbn(-input_exponent) / mantissa).scalbn(shift),
                );
            }
            input_exponent += 1 - minimum_exponent;
        } else {
            normalized.iter_mut().zip(input).for_each(|(out, z)| {
                *out = Complex::new(z.re.scalbn(-input_exponent), z.im.scalbn(-input_exponent));
            });
        }
        let input = &*normalized;
        context.check()?;
        // Admit complete output rows before scheduling. This is deterministic
        // across worker counts and requires no per-worker mutable parent ledger.
        crate::resource::admit_iterations(parent, self.n)?;
        let schedule = context.plan(
            self.spec.execution_policy,
            self.n,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: core::num::NonZeroUsize::new(8).ok_or_else(|| {
                    invalid(
                        scioxide_core::InvalidInputReason::OutOfRange,
                        scioxide_core::argument!("self"),
                        "row granularity",
                    )
                })?,
                worker_limit: core::num::NonZeroUsize::new(64).ok_or_else(|| {
                    invalid(
                        scioxide_core::InvalidInputReason::OutOfRange,
                        scioxide_core::argument!("self"),
                        "worker limit",
                    )
                })?,
            },
        );
        if schedule.worker_count() == 1 {
            for (i, out) in scratch[..self.n].iter_mut().enumerate() {
                context.check()?;
                *out = self.row(i, input, input_exponent, adjoint)?;
            }
        } else {
            // Rows are split into contiguous disjoint chunks of `scratch`, one
            // per partition (at most 64, matching the planned worker limit),
            // transferred through the shared executor protocol boundary.
            use scioxide_core::backend::PartitionSlot;
            let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
            let mut remaining = &mut scratch[..self.n];
            for partition in schedule.partitions() {
                let range = partition.range();
                let (chunk, rest) = remaining.split_at_mut(range.len());
                remaining = rest;
                slots[partition.worker_index()] = PartitionSlot::new(chunk);
            }
            context.execute_partitioned(
                schedule,
                &slots[..schedule.worker_count()],
                &|partition, chunk, scope| {
                    let start = partition.range().start;

                    for (offset, out) in chunk.iter_mut().enumerate() {
                        scope.check()?;
                        *out = self.row(start + offset, input, input_exponent, adjoint)?;
                    }
                    Ok(())
                },
            )?;
        }
        context.check()?;
        Ok(&scratch[..self.n])
    }
    fn row(
        &self,
        i: usize,
        input: &[Complex<S>],
        input_exponent: i32,
        adjoint: bool,
    ) -> Result<Complex<S>, SciOxideError> {
        let row = &self.matrix[i * self.n..(i + 1) * self.n];
        let (mut re, mut im, mut re_correction, mut im_correction) =
            (S::ZERO, S::ZERO, S::ZERO, S::ZERO);
        // Each rounded product has magnitude at most one. Neumaier retains
        // cancellation in fixed column order; rows are never partitioned.
        for (&m, z) in row.iter().zip(input) {
            compensated_term(&mut re, &mut re_correction, m * z.re);
            compensated_term(&mut im, &mut im_correction, m * z.im);
        }
        let mut exponent = self.row_exponents[i] + input_exponent + self.scale_exponent;
        let mut factor = self.scale_mantissa;
        if adjoint {
            let weight_exponent = crate::scaling::range_exponent(self.weights[i]);
            factor *= self.weights[i].scalbn(-weight_exponent);
            exponent += weight_exponent;
        }
        let out = crate::scaling::BinaryScale::new(exponent).restore_complex(Complex::new(
            (re + re_correction) * factor,
            (im + im_correction) * factor,
        ));
        if !out.is_finite() {
            return Err(crate::validation::arithmetic_overflow(
                "DHT result exceeds scalar range",
            ));
        }
        Ok(out)
    }
}

fn binary_exponent(exponent: i64) -> Result<i32, SciOxideError> {
    i32::try_from(exponent).map_err(|_| {
        crate::validation::arithmetic_overflow("DHT binary exponent exceeds supported range")
    })
}

// Grow a row's binary frame before rounding any coefficient into S. Changes
// of frame are powers of two, so already stored normal values shift exactly.
// No second n² buffer or repeated kernel evaluations are needed.
fn insert_coefficient<S: RealScalar>(
    row: &mut [S],
    exponent: &mut i32,
    column: usize,
    coefficient: Scaled<f64>,
) -> Result<(), SciOxideError> {
    if coefficient.is_zero() {
        return Ok(());
    }
    let needed = binary_exponent(coefficient.exponent() + 1)?;
    if *exponent == i32::MIN {
        *exponent = needed;
    } else if needed > *exponent {
        row.iter_mut()
            .for_each(|value| *value = value.scalbn(*exponent - needed));
        *exponent = needed;
    }
    row[column] = coefficient.shifted(-i64::from(*exponent))?.round_to()?;
    Ok(())
}

struct ZeroBrackets {
    cursor: f64,
    positive: bool,
}
impl ZeroBrackets {
    fn new(nu: f64) -> Self {
        // j_(nu,1)>nu for nu>=0 and j_(nu,1)>pi/2 for nu>-1/2.
        Self {
            cursor: nu.max(core::f64::consts::FRAC_PI_4),
            positive: true,
        }
    }

    fn next(
        &mut self,
        nu: f64,
        work: &mut Work<'_, '_, '_>,
        context: ExecutionContext<'_>,
    ) -> Result<(f64, f64), SciOxideError> {
        // For u=sqrt(x)J_nu(x), u''+[1+(1/4-nu²)/x²]u=0
        // (DLMF 10.2.1). On x>=max(nu,pi/4), that coefficient is
        // <=1+4/pi²<4. Sturm comparison gives separation >pi/2.
        // Thus a scan step cannot conceal two roots, even for nu<1/2.
        // See also DLMF 10.21(i),(iv) for simplicity and monotonicity.
        let (mut lower, mut upper, left_positive) = loop {
            let next = self.cursor + core::f64::consts::FRAC_PI_2;
            if !next.is_finite() || next <= self.cursor {
                return Err(invalid(
                    scioxide_core::InvalidInputReason::OutOfRange,
                    scioxide_core::argument!("zeros"),
                    "Bessel zero brackets cannot advance at this coordinate precision",
                ));
            }
            let value = work.bessel(nu, next, context)?;
            let lower = self.cursor;
            self.cursor = next;
            if value == 0.0 || (value > 0.0) != self.positive {
                let left_positive = self.positive;
                self.positive = !self.positive;
                if value == 0.0 {
                    return Ok((next, next));
                }
                break (lower, next, left_positive);
            }
        };
        // Refine independently of the supplied coordinate and tolerance.
        // The original upper endpoint remains the start of the next scan.
        loop {
            let middle = lower + (upper - lower) * 0.5;
            if middle == lower || middle == upper {
                break;
            }
            let value = work.bessel(nu, middle, context)?;
            if value == 0.0 {
                return Ok((middle, middle));
            }
            if (value > 0.0) == left_positive {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        Ok((lower, upper))
    }
}

impl<S: RealScalar> AxisPlan<S> for DiscreteHankelPlan<S> {
    fn lane_lens(&self) -> (usize, usize) {
        (self.n, self.n)
    }
    fn scratch(&self) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        crate::FftWorkspaceRequirements::new::<S>(self.scratch_len())?
            .with_limit(self.spec.limits.workspace_bytes)
    }
    fn workspace_bytes(&self) -> usize {
        self.spec.limits.workspace_bytes
    }
    fn iteration_budget(&self) -> Option<Budget> {
        self.spec.iteration_budget
    }
    fn admit(&self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        crate::resource::admit_operator(context)
    }
    fn lane<'w>(
        &self,
        source: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        self.evaluate(source, scratch, false, context)
    }
}

impl<S: RealScalar> crate::axis::RadialPlan<S> for DiscreteHankelPlan<S> {
    fn evaluate<'w>(
        &self,
        input: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        adjoint: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        self.evaluate(input, scratch, adjoint, context)
    }
}
crate::axis::impl_radial_operators!(DiscreteHankelPlan);

#[cfg(test)]
mod range_tests {
    use super::*;
    use scioxide_core::ScaledAccumulator;

    #[cfg(feature = "std")]
    #[test]
    fn dht_component_bits_match_for_one_two_and_four_workers() {
        let n = 16;
        let zeros: Vec<_> = (1..=n + 1)
            .map(|j| j as f64 * core::f64::consts::PI)
            .collect();
        let input: Vec<_> = (0..n)
            .map(|j| Complex::new((j as f64 * 0.7).sin(), (j as f64 * 0.3).cos()))
            .collect();
        let mut expected = None;
        for count in [1, 2, 4] {
            let workers = core::num::NonZeroUsize::new(count).unwrap();
            let executor = scioxide_core::ThreadExecutor::new(workers);
            let mut bits = Vec::new();
            for direction in [crate::FftDirection::Forward, crate::FftDirection::Inverse] {
                let plan = DiscreteHankelPlan::new(
                    &zeros,
                    DiscreteHankelPlanSpec::new(0.5, 4.0, direction)
                        .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
                        .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers)),
                )
                .unwrap();
                let mut scratch = alloc::vec![Complex::new(0.0, 0.0); plan.scratch_len()];
                let mut state = scioxide_core::SolveState::new();
                let mut budgets = scioxide_core::SolveBudgets::unbounded();
                let mut context = state
                    .begin_solve(
                        ExecutionContext::new(&executor),
                        crate::workspace::identity(n),
                        &mut budgets,
                    )
                    .unwrap();
                for adjoint in [false, true] {
                    let output = plan
                        .evaluate(&input, &mut scratch, adjoint, &mut context)
                        .unwrap();
                    bits.extend(output.iter().map(|z| (z.re.to_bits(), z.im.to_bits())));
                }
            }
            if let Some(expected) = &expected {
                assert_eq!(&bits, expected);
            } else {
                expected = Some(bits);
            }
        }
    }

    fn compensated_reference<S: RealScalar>(exponents: &[i32], subnormal: S) {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let n = 16;
        let zeros: Vec<_> = (1..=n + 1)
            .map(|j| scalar(j as f64 * core::f64::consts::PI))
            .collect();
        let tolerance = scalar(16.0) * S::EPSILON;
        for direction in [crate::FftDirection::Forward, crate::FftDirection::Inverse] {
            let plan = DiscreteHankelPlan::new(
                &zeros,
                DiscreteHankelPlanSpec::new(scalar(0.5), scalar(4.0), direction)
                    .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap()),
            )
            .unwrap();
            let mut scratch = alloc::vec![Complex::new(S::ZERO, S::ZERO); plan.scratch_len()];
            let mut state = scioxide_core::SolveState::new();
            let mut budgets = scioxide_core::SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential_native(),
                    crate::workspace::identity(n),
                    &mut budgets,
                )
                .unwrap();
            for &exponent in exponents {
                let input: Vec<_> = (0..n)
                    .map(|j| {
                        Complex::new(
                            scalar((j % 3) as f64 - 1.0).scalbn(exponent),
                            scalar((j % 5) as f64 * 0.25 - 0.5).scalbn(exponent),
                        )
                    })
                    .collect();
                for adjoint in [false, true] {
                    let actual = plan
                        .evaluate(&input, &mut scratch, adjoint, &mut context)
                        .unwrap();
                    let scale = plan
                        .scale_mantissa
                        .scalbn(plan.scale_exponent)
                        .try_to_f64_exact()
                        .unwrap();
                    for (i, out) in actual.iter().enumerate() {
                        for imaginary in [false, true] {
                            let mut reference = ScaledAccumulator::ZERO;
                            let mut mass = 0.0;
                            for (j, z) in input.iter().enumerate() {
                                // Index the actual transpose independently of the weight identity.
                                let (row, column) = if adjoint { (j, i) } else { (i, j) };
                                let m = plan.matrix[row * n + column]
                                    .try_to_f64_exact()
                                    .unwrap()
                                    .scalbn(plan.row_exponents[row]);
                                let x = if imaginary { z.im } else { z.re }
                                    .try_to_f64_exact()
                                    .unwrap();
                                reference
                                    .add_product(
                                        Scaled::from_value(m).unwrap(),
                                        Scaled::from_value(x).unwrap(),
                                    )
                                    .unwrap();
                                mass += m.abs() * x.scalbn(-exponent).abs();
                            }
                            let expected = reference
                                .shifted(-i64::from(exponent))
                                .unwrap()
                                .total()
                                .unwrap()
                                .mul(Scaled::from_value(scale).unwrap())
                                .unwrap()
                                .round()
                                .unwrap();
                            let value = if imaginary { out.im } else { out.re }
                                .try_to_f64_exact()
                                .unwrap()
                                .scalbn(-exponent);
                            let bound = 8.0
                                * S::EPSILON.try_to_f64_exact().unwrap()
                                * (n + 2) as f64
                                * mass
                                * scale
                                + 2.0 * subnormal.try_to_f64_exact().unwrap().scalbn(-exponent);
                            assert!(
                                (value - expected).abs() <= bound,
                                "exponent={exponent}, direction={direction:?}, adjoint={adjoint}, row={i}: {value} vs {expected}, bound={bound}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compensated_rows_match_extended_exponent_reference_f64() {
        compensated_reference(&[1000, 0, -995, -1070], f64::from_bits(1));
    }

    #[test]
    fn compensated_rows_match_extended_exponent_reference_f32() {
        compensated_reference(&[120, 0, -120, -145], f32::from_bits(1));
    }
}
