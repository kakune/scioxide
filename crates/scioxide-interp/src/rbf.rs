//! Radial basis function kernels and global RBF interpolation of scattered
//! data in the plane.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Argument, Capacity, ExecutionContext, Interval, NonFiniteOrigin,
    NumericScalar, PartialResultTrust, ProblemToken, RankThreshold, RealScalar, Scaled,
    SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState, write_value,
};
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyWorkspaceRequest, LinalgExecutionOptions, LinalgWorkspace,
    LuDecomposition, LuWorkspaceRequest, Matrix, SvdWorkspaceRequest, ThinSvdDecomposition, Vector,
};

use crate::checked::{add, mul, output};
use crate::error::require_proposal;
use crate::factor::{SystemKind, factor, solve_requirements};
use crate::geometry::{bounds, distance, extended_distance_parts};
use crate::protocol::{ExtrapolationPolicy, Interpolant, NodeSeparation};
use crate::radial::{project, split_product, squared_decay};
use crate::validate::{validate_count, validate_distinct_points, validate_finite, validate_length};
use crate::values::{QueryWorkspace, flatten, gather, scatter, zeroed};
use crate::{FitConditionNorm, FitFactorization};

/// Selects the radial basis function `φ(r)` of an [`RbfInterpolator`].
///
/// `r ≥ 0` is the Euclidean distance between two points and `c > 0` is the
/// shape parameter carried by the variant. A larger `c` gives a narrower
/// kernel. The Gaussian and inverse multiquadric kernels are positive definite,
/// so their interpolation matrices are symmetric positive definite for distinct
/// points. Shape parameters are checked wherever a kernel is used.
///
/// ```
/// use scioxide_interp::RbfKernel;
/// assert_eq!(RbfKernel::Gaussian(1.0_f64).evaluate(0.0)?, 1.0);
/// assert_eq!(RbfKernel::<f64>::ThinPlateSpline.evaluate(0.0)?, 0.0);
/// assert!(RbfKernel::Gaussian(0.0_f64).evaluate(1.0).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel<S: RealScalar> {
    /// Gaussian `φ(r) = exp(−c²r²)`.
    Gaussian(S),
    /// Multiquadric `φ(r) = √(1 + c²r²)`.
    Multiquadric(S),
    /// Inverse multiquadric `φ(r) = 1/√(1 + c²r²)`.
    InverseMultiquadric(S),
    /// Thin-plate spline `φ(r) = r² ln r`, with `φ(0) = 0`.
    ThinPlateSpline,
}

impl<S: RealScalar> RbfKernel<S> {
    /// Evaluates `φ(r)` at a distance `r ≥ 0`.
    ///
    /// Intermediate results that would overflow or underflow are computed with
    /// an extended exponent range, so `φ(r)` is accurate whenever it is
    /// representable. A Gaussian tail below the smallest subnormal returns zero.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the shape parameter is not positive or `r < 0`.
    /// - `NonFinite` if the shape parameter or `r` is NaN or infinite, or
    ///   `φ(r)` overflows.
    pub fn evaluate(&self, r: S) -> Result<S, SciOxideError> {
        self.validate(scioxide_core::argument!("self"))?;
        self.evaluate_validated(r)
    }

    /// Evaluates a kernel whose shape was checked when the interpolant was built.
    pub(crate) fn evaluate_validated(&self, r: S) -> Result<S, SciOxideError> {
        if !r.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "RBF distance must be finite",
            ));
        }
        if r < S::ZERO {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("r"),
                "RBF distance must be non-negative",
            ));
        }
        if let Some(value) = self.ordinary(r) {
            return Ok(value);
        }
        self.evaluate_extended(Scaled::from_value(r)?)
    }

    /// Evaluates in plain floating-point arithmetic; `None` when the value or an
    /// intermediate leaves the normal range and needs the extended-range path.
    fn ordinary(&self, r: S) -> Option<S> {
        let value = match *self {
            Self::Gaussian(c) => {
                let scaled = c * r;
                (-(scaled * scaled)).exp()
            }
            Self::Multiquadric(c) => S::ONE.hypot(c * r),
            Self::InverseMultiquadric(c) => S::ONE / S::ONE.hypot(c * r),
            Self::ThinPlateSpline => {
                if r == S::ZERO || r == S::ONE {
                    return Some(S::ZERO);
                }
                let square = r * r;
                if !square.is_normal() {
                    return None;
                }
                square * r.ln()
            }
        };
        value.is_normal().then_some(value)
    }

    /// Evaluates the kernel between two validated points, even when their
    /// Euclidean distance overflows before the shape parameter is applied.
    pub(crate) fn evaluate_between<const D: usize>(
        &self,
        a: &[S; D],
        b: &[S; D],
    ) -> Result<S, SciOxideError> {
        if let Ok(r) = distance(a, b)
            && (r.is_normal() || r == S::ZERO)
            && let Some(value) = self.ordinary(r)
        {
            return Ok(value);
        }
        let [r, correction] = extended_distance_parts(a, b, &[S::ONE; D])?.parts();
        self.evaluate_extended_parts(r, correction)
    }

    fn evaluate_extended(&self, r: Scaled<S>) -> Result<S, SciOxideError> {
        self.evaluate_extended_parts(r, Scaled::ZERO)
    }

    fn evaluate_extended_parts(
        &self,
        r: Scaled<S>,
        correction: Scaled<S>,
    ) -> Result<S, SciOxideError> {
        match *self {
            Self::Gaussian(c) => {
                let shape = Scaled::from_value(c)?;
                let (radius, residual) = split_product(shape, r)?;
                squared_decay(
                    Scaled::ONE,
                    radius,
                    residual.add(shape.mul(correction)?)?,
                    false,
                )
            }
            Self::Multiquadric(c) | Self::InverseMultiquadric(c) => {
                let magnitude = Scaled::ONE
                    .add(Scaled::from_value(c)?.mul(r)?.square()?)?
                    .sqrt()?;
                project(if matches!(self, Self::InverseMultiquadric(_)) {
                    Scaled::ONE.div(magnitude)?
                } else {
                    magnitude
                })
            }
            Self::ThinPlateSpline => {
                if r.is_zero() {
                    Ok(S::ZERO)
                } else {
                    project(r.square()?.mul(Scaled::from_value(r.ln_abs()?)?)?)
                }
            }
        }
    }

    /// Validates the kernel shape, naming the kernel parameter `argument` in errors.
    pub(crate) fn validate(&self, argument: Argument) -> Result<(), SciOxideError> {
        match *self {
            Self::Gaussian(shape)
            | Self::Multiquadric(shape)
            | Self::InverseMultiquadric(shape) => {
                if !shape.is_finite() {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Input,
                        "RBF shape parameter must be finite",
                    ));
                }
                if shape <= S::ZERO {
                    return Err(crate::error::out_of_range(
                        argument,
                        "RBF shape parameter must be positive",
                    ));
                }
                Ok(())
            }
            Self::ThinPlateSpline => Ok(()),
        }
    }
}

/// Selects whether the RBF kernel matrix is regularized.
///
/// With `Diagonal(λ)` the weights solve `(A + λI) w = v` instead of `A w = v`,
/// where `A` is the kernel matrix. The result then smooths the data instead of
/// passing through it, and the system is better conditioned.
///
/// ```
/// use scioxide_interp::RbfRegularization;
/// let smoothing = RbfRegularization::Diagonal(1e-6_f64);
/// assert_ne!(smoothing, RbfRegularization::None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum RbfRegularization<S: RealScalar> {
    /// Solves the interpolation system `A w = v` without regularization.
    None,
    /// Adds the positive value `λ` to every diagonal entry of `A`.
    Diagonal(S),
}

/// Holds the regularization, separation and rank tolerances and the resource
/// limits of an [`RbfInterpolator`].
///
/// The defaults are no regularization, [`NodeSeparation::Exact`], the rank
/// threshold `atol = rtol = ε` (the machine epsilon of `S`), no condition
/// limit, the `Auto` execution policy and no byte limit. The byte limit bounds
/// an estimate of the memory that construction needs: the dense kernel matrix,
/// the direct or SVD factor, the workspaces for factoring and solving, and the
/// weights. [`RbfInterpolator::build`] validates the options.
///
/// ```
/// use scioxide_core::{RankThreshold, Tolerance};
/// use scioxide_interp::{RbfOptions, RbfRegularization};
/// let options = RbfOptions::<f64>::default()
///     .with_regularization(RbfRegularization::Diagonal(1e-6))
///     .with_maximum_bytes(1 << 20);
/// assert_eq!(options.maximum_bytes(), 1 << 20);
/// assert_eq!(options.regularization(), RbfRegularization::Diagonal(1e-6));
/// let epsilon = RankThreshold::from_tolerance(Tolerance::from_machine_epsilon());
/// assert_eq!(options.rank_threshold(), epsilon);
/// ```
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins University Press, 2013, section 8.6.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RbfOptions<S: RealScalar> {
    regularization: RbfRegularization<S>,
    node_separation: NodeSeparation<S>,
    rank_threshold: RankThreshold<S>,
    minimum_rcond: S,
    execution_policy: scioxide_core::ExecutionPolicy,
    maximum_bytes: usize,
}

impl<S: RealScalar> Default for RbfOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> RbfOptions<S> {
    /// Returns the default options.
    ///
    /// The rank threshold `ε` rejects only kernel matrices that are singular
    /// at working precision: a backward-stable factorization is exact for a
    /// matrix within a small multiple of `ε‖A‖₂` of `A`, which moves each
    /// singular value by at most that much, so a reciprocal condition number
    /// below `ε` (`2.2·10⁻¹⁶` in `f64`, `1.2·10⁻⁷` in `f32`) cannot be told apart
    /// from zero.
    pub fn new() -> Self {
        Self {
            regularization: RbfRegularization::None,
            node_separation: NodeSeparation::Exact,
            rank_threshold: RankThreshold::from_tolerance(
                scioxide_core::Tolerance::from_machine_epsilon(),
            ),
            minimum_rcond: S::ZERO,
            execution_policy: scioxide_core::ExecutionPolicy::Auto,
            maximum_bytes: usize::MAX,
        }
    }

    /// Sets the regularization of the kernel matrix; the default is
    /// [`RbfRegularization::None`].
    ///
    /// A diagonal `λ` must be finite and positive.
    pub const fn with_regularization(mut self, regularization: RbfRegularization<S>) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets when two points coincide; the default is [`NodeSeparation::Exact`].
    ///
    /// With [`NodeSeparation::Tolerant`], points `p` and `q` coincide when
    /// `‖p − q‖₂ ≤ atol + rtol · max(‖p‖∞, ‖q‖∞)`.
    pub const fn with_node_separation(mut self, separation: NodeSeparation<S>) -> Self {
        self.node_separation = separation;
        self
    }

    /// Sets the tolerance that decides the numerical rank of the kernel
    /// matrix; the default is `atol = rtol = ε`.
    ///
    /// The Cholesky or LU factor is checked first, on its scaled pivots and its
    /// reciprocal condition estimate. If that check fails, an SVD counts the
    /// singular values above `atol + rtol · σ_max`.
    pub const fn with_rank_threshold(mut self, threshold: RankThreshold<S>) -> Self {
        self.rank_threshold = threshold;
        self
    }

    /// Requires the reciprocal condition estimate of the kernel matrix to be at
    /// least `rcond`, a value in `[0, 1]`.
    ///
    /// The estimate uses the norm reported by
    /// [`RbfConditioning::condition_norm`]. The default is zero, which rejects
    /// only matrices that are rank deficient under the rank threshold.
    pub const fn with_minimum_rcond(mut self, rcond: S) -> Self {
        self.minimum_rcond = rcond;
        self
    }

    /// Returns the smallest accepted reciprocal condition estimate.
    pub const fn minimum_rcond(&self) -> S {
        self.minimum_rcond
    }

    /// Returns the diagonal regularization, in the units of the kernel values.
    pub const fn regularization(&self) -> RbfRegularization<S> {
        self.regularization
    }

    /// Returns the policy that rejects coincident or nearly coincident points.
    pub const fn node_separation(&self) -> NodeSeparation<S> {
        self.node_separation
    }

    /// Returns the tolerance that decides the numerical rank.
    pub const fn rank_threshold(&self) -> RankThreshold<S> {
        self.rank_threshold
    }

    /// Sets how many workers factorization may use; the default is `Auto`.
    ///
    /// The workers come from the executor of the scope passed to
    /// [`RbfInterpolator::build`]. [`RbfInterpolator::new`] and
    /// [`RbfInterpolator::with_options`] run on the calling thread.
    pub const fn with_execution_policy(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }

    /// Returns the execution policy of factorization and condition estimation.
    pub const fn execution_policy(&self) -> scioxide_core::ExecutionPolicy {
        self.execution_policy
    }

    /// Limits the memory estimate of construction to `bytes`.
    pub const fn with_maximum_bytes(mut self, bytes: usize) -> Self {
        self.maximum_bytes = bytes;
        self
    }

    /// Returns the byte limit of construction.
    pub const fn maximum_bytes(&self) -> usize {
        self.maximum_bytes
    }

    /// Checks the regularization and the condition limit, before any other
    /// work of construction.
    fn validate(&self) -> Result<(), SciOxideError> {
        if let RbfRegularization::Diagonal(lambda) = self.regularization {
            if !lambda.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "RBF regularization must be finite",
                ));
            }
            if lambda <= S::ZERO {
                return Err(crate::error::out_of_range(
                    scioxide_core::argument!("options.regularization"),
                    "diagonal RBF regularization must be positive",
                ));
            }
        }
        let rcond = self.minimum_rcond;
        if !rcond.is_finite() || rcond < S::ZERO || rcond > S::ONE {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("options.minimum_rcond"),
                "minimum reciprocal condition must be finite and in [0, 1]",
            ));
        }
        Ok(())
    }
}

/// Reports the rank and conditioning of the kernel matrix of an
/// [`RbfInterpolator`].
///
/// After a Cholesky or LU factorization the reciprocal condition number is an
/// estimate in the 1-norm. After the SVD it is `σ_min/σ_max`, exact up to
/// rounding in the 2-norm. The 1-norm value is an estimate, not a bound.
///
/// ```
/// use scioxide_interp::{RbfInterpolator, RbfKernel};
/// let points = vec![[0.0_f64, 0.0], [1.0, 0.0]];
/// let rbf = RbfInterpolator::new(points, vec![1.0, 2.0], RbfKernel::Gaussian(1.0))?;
/// let conditioning = rbf.conditioning();
/// assert_eq!(
///     (conditioning.matrix_size(), conditioning.numerical_rank()),
///     (2, 2)
/// );
/// assert!(conditioning.reciprocal_condition_number() > 0.0);
/// assert_eq!(
///     conditioning.factorization(),
///     scioxide_interp::FitFactorization::Cholesky
/// );
/// assert_eq!(
///     conditioning.condition_norm(),
///     scioxide_interp::FitConditionNorm::One
/// );
/// assert!(!conditioning.used_fallback());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RbfConditioning<S: RealScalar> {
    matrix_size: usize,
    numerical_rank: usize,
    rank_threshold: RankThreshold<S>,
    reciprocal_condition_number: S,
    factorization: FitFactorization,
    condition_norm: FitConditionNorm,
}

impl<S: RealScalar> RbfConditioning<S> {
    /// Returns the order `n` of the kernel matrix, the number of points.
    pub const fn matrix_size(&self) -> usize {
        self.matrix_size
    }

    /// Returns the numerical rank under [`Self::rank_threshold`].
    pub const fn numerical_rank(&self) -> usize {
        self.numerical_rank
    }

    /// Returns the rank tolerance from [`RbfOptions::rank_threshold`].
    pub const fn rank_threshold(&self) -> RankThreshold<S> {
        self.rank_threshold
    }

    /// Returns the factorization used to solve for the weights.
    pub const fn factorization(&self) -> FitFactorization {
        self.factorization
    }

    /// Returns whether the SVD replaced the Cholesky or LU factorization.
    pub fn used_fallback(&self) -> bool {
        self.factorization == FitFactorization::Svd
    }

    /// Returns the norm of [`Self::reciprocal_condition_number`].
    pub const fn condition_norm(&self) -> FitConditionNorm {
        self.condition_norm
    }

    /// Returns the reciprocal condition number of the matrix that was solved,
    /// including any regularization.
    pub const fn reciprocal_condition_number(&self) -> S {
        self.reciprocal_condition_number
    }
}

/// Interpolates scattered data in the plane by a global radial basis function
/// expansion.
///
/// Given distinct points `pᵢ` with values `vᵢ`, `i = 1, …, n`, the interpolant is
///
/// ```text
/// f(p) = Σᵢ wᵢ φ(‖p − pᵢ‖),
/// ```
///
/// where `φ` is the [`RbfKernel`] and `‖·‖` the Euclidean norm. The weights
/// solve `A w = v` with `Aᵢⱼ = φ(‖pᵢ − pⱼ‖)`, or `(A + λI) w = v` under
/// [`RbfRegularization::Diagonal`]. There is no polynomial term. Without
/// regularization `f` passes through the data up to rounding.
///
/// The Gaussian and inverse multiquadric kernels factor `A` by Cholesky, the
/// others by LU with partial pivoting. If that factorization breaks down, or
/// fails the rank or condition check of [`RbfOptions`], the SVD is used
/// instead. [`Self::conditioning`] reports which. The factor is computed once
/// and solved for each real component of the values. Construction costs O(n³)
/// time and O(n²) memory, and each query costs O(n) per component.
///
/// The domain for [`ExtrapolationPolicy`](crate::ExtrapolationPolicy) is the
/// bounding rectangle of the points, not their convex hull.
///
/// # Errors
///
/// [`Self::build`] lists the construction errors. Queries fail as described in
/// the [crate docs](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{ExtrapolationPolicy, Interpolant, RbfInterpolator, RbfKernel};
/// let kernel = RbfKernel::Multiquadric(1.0_f64);
/// let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let rbf = RbfInterpolator::new(points.clone(), vec![1.0, 2.0, 3.0], kernel)?;
/// assert!((rbf.evaluate([0.0, 0.0], ExtrapolationPolicy::Reject)? - 1.0).abs() < 1e-10);
///
/// let values = vec![Complex::new(1.0, -1.0), Complex::new(2.0, 0.0), Complex::new(3.0, 1.0)];
/// let complex = RbfInterpolator::new(points, values, kernel)?;
/// let at_node = complex.evaluate([1.0, 0.0], ExtrapolationPolicy::Reject)?;
/// assert!((at_node.re - 2.0).abs() < 1e-10 && at_node.im.abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. Wendland, *Scattered Data Approximation*, Cambridge University Press,
/// 2005, chapters 6 and 8.
#[derive(Debug, Clone)]
#[must_use]
pub struct RbfInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    template: V,
    dimension: usize,
    points: Vec<[S; 2]>,
    /// Weights indexed `[point][component]`.
    weights: Vec<S>,
    kernel: RbfKernel<S>,
    conditioning: RbfConditioning<S>,
    domains: [Interval<S>; 2],
}

impl<S: RealScalar + NumericScalar<Real = S>, V: AffineValue<Scale = S>> RbfInterpolator<S, V> {
    /// Builds the interpolant with kernel `kernel` and the default
    /// [`RbfOptions`] on the calling thread.
    ///
    /// The defaults solve the interpolation system without regularization,
    /// reject only coincident points, and accept a kernel matrix whose
    /// numerical rank is full under the threshold `atol = rtol = ε`. This is
    /// [`Self::with_options`] with [`RbfOptions::default`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn new(
        points: Vec<[S; 2]>,
        values: Vec<V>,
        kernel: RbfKernel<S>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(points, values, kernel, RbfOptions::default())
    }

    /// Builds the interpolant with `kernel` and `options` on the calling thread.
    ///
    /// Runs [`Self::build`] with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn with_options(
        points: Vec<[S; 2]>,
        values: Vec<V>,
        kernel: RbfKernel<S>,
        options: RbfOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            points,
            values,
            kernel,
            options,
            ExecutionContext::sequential_native(),
        )
    }

    /// Builds the interpolant through `(points[i], values[i])` with `kernel`
    /// and `options` in `scope`.
    ///
    /// `scope` is an [`ExecutionContext`] or the `&mut SolveContext` of an
    /// enclosing solve (see [solve accounting](scioxide_core#solve-accounting)).
    /// The construction counts as one proposal. Its executor runs the
    /// factorization, and cancellation and deadlines are checked between rows
    /// and components. The kernel and the options are validated first, then
    /// shapes, finiteness and the byte limit, before any other work starts.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the kernel shape parameter or a
    ///   diagonal regularization is not positive, or the minimum reciprocal
    ///   condition is outside `[0, 1]`.
    /// - `InvalidInput` if `points` is empty, or two points coincide under
    ///   [`RbfOptions::node_separation`].
    /// - `DimensionMismatch` if `points` and `values` differ in length or the
    ///   values differ in shape.
    /// - `NonFinite` if the kernel shape parameter, the regularization, a
    ///   coordinate or a value is NaN or infinite, or a kernel value or weight
    ///   overflows.
    /// - `ResourceLimit` if the memory estimate exceeds
    ///   [`RbfOptions::maximum_bytes`].
    /// - `ConvergenceFailure` if the proposal limit of `scope` is exhausted.
    /// - `Singular` if the matrix is rank deficient under
    ///   [`RbfOptions::rank_threshold`], or its reciprocal condition number is
    ///   below [`RbfOptions::minimum_rcond`].
    /// - `ExecutionStopped` on cancellation or a deadline.
    pub fn build<'scope, 'capability>(
        points: Vec<[S; 2]>,
        values: Vec<V>,
        kernel: RbfKernel<S>,
        options: RbfOptions<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        kernel.validate(scioxide_core::argument!("kernel"))?;
        options.validate()?;
        let n = points.len();
        validate_count(n, 1, scioxide_core::argument!("points"))?;
        validate_length(n, values.len())?;
        validate_finite(points.as_flattened())?;
        let flat = flatten(values, n)?;
        let bytes = preparation_bytes::<S>(n, flat.dimension, &options)?;
        if bytes > options.maximum_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("rbf_preparation_bytes"),
                bytes,
                options.maximum_bytes,
            ));
        }
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "rbf", 1),
            ((n as u128) << 64) | flat.dimension as u128,
            0,
        );
        let mut state = SolveState::new();
        scope
            .into()
            .run(&mut state, identity, SolveBudgets::unbounded(), |context| {
                let solved = solve(
                    &points,
                    &flat.components,
                    flat.dimension,
                    kernel,
                    &options,
                    context,
                )?;
                let domains = bounds(points.iter().copied(), scioxide_core::argument!("points"))?;
                context.execution().check()?;
                context.commit_accepted(PartialResultTrust::Admissible)?;
                Ok(Self {
                    template: flat.template,
                    dimension: flat.dimension,
                    points,
                    weights: solved.0,
                    kernel,
                    conditioning: solved.1,
                    domains,
                })
            })
    }

    /// Returns the rank and conditioning of the kernel matrix.
    pub const fn conditioning(&self) -> RbfConditioning<S> {
        self.conditioning
    }
}

/// Validates separation, assembles and factors the kernel matrix, and solves for
/// every component.
fn solve<S: RealScalar + NumericScalar<Real = S>>(
    points: &[[S; 2]],
    components: &[S],
    dimension: usize,
    kernel: RbfKernel<S>,
    options: &RbfOptions<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(Vec<S>, RbfConditioning<S>), SciOxideError> {
    let n = points.len();
    context.execution().check()?;
    // Coincident points would give two identical kernel-matrix rows, so the
    // matrix would be singular. They are rejected up front with an input error
    // rather than surfacing later as a rank failure. This O(n^2) pairwise check
    // matches the cost of assembling the matrix.
    validate_distinct_points(
        points,
        options.node_separation,
        scioxide_core::argument!("points"),
        || context.execution().check(),
    )?;
    require_proposal(context, "proposal budget denied RBF construction")?;
    // Build the kernel matrix A[i,j] = kernel(||p_i - p_j||).
    let matrix_entries = checked_mul(n, n, scioxide_core::quantity!("rbf_kernel_matrix"))?;
    let mut data = try_with_capacity(
        matrix_entries,
        scioxide_core::quantity!("rbf_kernel_matrix"),
    )?;
    for (i, &left) in points.iter().enumerate() {
        context.execution().check()?;
        for (j, &right) in points.iter().enumerate() {
            let mut value = kernel.evaluate_between(&left, &right)?;
            if i == j
                && let RbfRegularization::Diagonal(lambda) = options.regularization
            {
                value = add(value, lambda)?;
            }
            data.push(value);
        }
    }
    let matrix = Matrix::from_vec(n, n, data)?;
    let kind = match kernel {
        RbfKernel::Gaussian(_) | RbfKernel::InverseMultiquadric(_) => SystemKind::PositiveDefinite,
        RbfKernel::Multiquadric(_) | RbfKernel::ThinPlateSpline => SystemKind::Symmetric,
    };
    let factored = factor(
        &matrix,
        kind,
        options.rank_threshold,
        options.minimum_rcond,
        LinalgExecutionOptions::new().with_execution_policy(options.execution_policy),
        context.execution(),
    )?;
    if factored.rank != n || factored.reciprocal_condition < options.minimum_rcond {
        return Err(SciOxideError::singular(None));
    }
    let mut scratch = LinalgWorkspace::new();
    scratch.try_reserve(&solve_requirements::<S>(n, n)?)?;
    let mut rhs = Vector::zeros(n)?;
    let mut solution = Vector::zeros(n)?;
    let mut weights = try_filled(
        checked_mul(n, dimension, scioxide_core::quantity!("rbf_coefficients"))?,
        S::ZERO,
        scioxide_core::quantity!("rbf_coefficients"),
    )?;
    for component in 0..dimension {
        context.execution().check()?;
        gather(components, dimension, component, rhs.as_mut_slice());
        factored.solve_into(&rhs, solution.view_mut(), &mut scratch, context.execution())?;
        scioxide_core::validate_value(solution.as_slice())?;
        scatter(solution.as_slice(), dimension, component, &mut weights);
    }
    Ok((
        weights,
        RbfConditioning {
            matrix_size: n,
            numerical_rank: factored.rank,
            rank_threshold: options.rank_threshold,
            reciprocal_condition_number: factored.reciprocal_condition,
            factorization: factored.method(),
            condition_norm: factored.condition_norm(),
        },
    ))
}

/// Upper bound on the heap memory of construction: inputs, direct and SVD
/// factors, solve scratch and weights. Allocator overhead is excluded.
fn preparation_bytes<S: RealScalar + NumericScalar<Real = S>>(
    n: usize,
    dimension: usize,
    options: &RbfOptions<S>,
) -> Result<usize, SciOxideError> {
    let quantity = scioxide_core::quantity!("rbf_preparation_bytes");
    let square = checked_mul(n, n, quantity)?;
    let scalar_bytes = core::mem::size_of::<S>();
    let vectors = checked_mul(n, checked_add(dimension, 2, quantity)?, quantity)?;
    let inputs = checked_mul(
        checked_add(square, vectors, quantity)?,
        scalar_bytes,
        quantity,
    )?;
    let svd_factor = checked_mul(
        checked_add(checked_mul(square, 2, quantity)?, n, quantity)?,
        scalar_bytes,
        quantity,
    )?;
    let retained = svd_factor
        .max(CholeskyDecomposition::<S>::factor_storage_bytes(n)?)
        .max(LuDecomposition::<S>::factor_storage_bytes(n)?);
    let linalg = LinalgExecutionOptions::new().with_execution_policy(options.execution_policy);
    let scratch = ThinSvdDecomposition::<S>::workspace_requirements(&SvdWorkspaceRequest::new(
        n, n, &linalg,
    ))?
    .max(&CholeskyDecomposition::<S>::workspace_requirements(
        &CholeskyWorkspaceRequest::factor(n, &linalg),
    )?)
    .max(&LuDecomposition::<S>::workspace_requirements(
        &LuWorkspaceRequest::factor(n, &linalg),
    )?)
    .max(&solve_requirements::<S>(n, n)?)
    .required_bytes::<S>()?;
    checked_add(inputs, checked_add(retained, scratch, quantity)?, quantity)
}

impl<S: RealScalar + NumericScalar<Real = S>, V: AffineValue<Scale = S> + Sync> Interpolant<S, 2>
    for RbfInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 2] {
        self.domains
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.dimension)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.dimension)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        let query = policy.resolve(self.domains, point)?.effective();
        // Points are the outer loop so each kernel value is evaluated once for
        // every component; each component still sums points in index order.
        staged.fill(S::ZERO);
        for (point, weights) in self
            .points
            .iter()
            .zip(self.weights.chunks_exact(self.dimension.max(1)))
        {
            let basis = self.kernel.evaluate_between(&query, point)?;
            for (slot, &weight) in staged.iter_mut().zip(weights) {
                *slot = add(*slot, mul(weight, basis)?)?;
            }
        }
        for slot in staged.iter_mut() {
            *slot = output(*slot)?;
        }
        write_value(staged, out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::assert_close;
    use scioxide_core::{Complex, SciOxideErrorView};

    fn options() -> RbfOptions<f64> {
        RbfOptions::new().with_rank_threshold(RankThreshold::from_tolerance(
            scioxide_core::Tolerance::from_sqrt_epsilon(),
        ))
    }

    fn reproduces(points: &[[f64; 2]], values: &[f64], kernel: RbfKernel<f64>, tol: f64) {
        let interp =
            RbfInterpolator::with_options(points.to_vec(), values.to_vec(), kernel, options())
                .unwrap();
        for (point, &expected) in points.iter().zip(values) {
            let value = interp
                .evaluate(*point, ExtrapolationPolicy::Reject)
                .unwrap();
            assert_close(value, expected, (tol) * 0.5, (tol) * 0.5);
        }
    }

    #[test]
    fn kernels_reproduce_data() {
        let square = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        reproduces(
            &square,
            &[1.0, 2.0, 3.0, 4.0],
            RbfKernel::Multiquadric(1.0),
            1e-10,
        );
        reproduces(
            &[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            &[1.0, 2.0, 3.0],
            RbfKernel::Gaussian(0.5),
            1e-10,
        );
        let five = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        reproduces(
            &five,
            &[0.0, 1.0, 1.0, 2.0, 1.0],
            RbfKernel::ThinPlateSpline,
            1e-8,
        );
        let interp = RbfInterpolator::with_options(
            square.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0],
            RbfKernel::Multiquadric(1.0),
            options(),
        )
        .unwrap();
        let conditioning = interp.conditioning();
        assert_eq!(conditioning.matrix_size(), 4);
        assert_eq!(conditioning.numerical_rank(), 4);
        assert!(
            conditioning
                .rank_threshold()
                .threshold(1.0)
                .unwrap()
                .is_finite()
        );
        assert!(conditioning.reciprocal_condition_number() > 0.0);
    }

    #[test]
    fn construction_errors_are_structured() {
        let result = RbfInterpolator::<f64>::with_options(
            Vec::new(),
            Vec::new(),
            RbfKernel::Multiquadric(1.0),
            options(),
        );
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { .. }))
        );
        let result = RbfInterpolator::with_options(
            vec![[0.0, 0.0], [1.0, 0.0]],
            vec![1.0],
            RbfKernel::Multiquadric(1.0),
            options(),
        );
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }))
        );
        let result = RbfInterpolator::with_options(
            vec![[0.0, f64::NAN]],
            vec![1.0],
            RbfKernel::Multiquadric(1.0),
            options(),
        );
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::NonFinite { .. }))
        );
        let points = vec![[0.0, 0.0], [1.0, 0.0]];
        let result = RbfInterpolator::new(points.clone(), vec![1.0, 2.0], RbfKernel::Gaussian(0.0));
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument.as_str() == "kernel"))
        );
        let result = RbfInterpolator::with_options(
            points,
            vec![1.0, 2.0],
            RbfKernel::Gaussian(1.0),
            options().with_regularization(RbfRegularization::Diagonal(0.0)),
        );
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument.as_str() == "options.regularization"))
        );
    }

    #[test]
    fn rank_and_separation_policies_are_enforced() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let rejecting = RbfOptions::new()
            .with_regularization(RbfRegularization::Diagonal(1e-8))
            .with_rank_threshold(RankThreshold::new(2.0, 0.0).unwrap());
        let gaussian = RbfKernel::Gaussian(1.0);
        let result =
            RbfInterpolator::with_options(points, vec![1.0, 2.0, 3.0], gaussian, rejecting);
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::Singular { .. }))
        );
        let tolerant = RbfOptions::new()
            .with_node_separation(NodeSeparation::Tolerant(
                scioxide_core::Tolerance::new(0.0, 2e-6).unwrap(),
            ))
            .with_rank_threshold(RankThreshold::new(0.0, 1e-12).unwrap());
        let result = RbfInterpolator::with_options(
            vec![[1e6, 0.0], [1e6 + 0.1, 0.0], [0.0, 1.0]],
            vec![1.0, 2.0, 3.0],
            gaussian,
            tolerant,
        );
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { .. }))
        );
    }

    #[test]
    fn regularization_smooths_the_diagonal() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let smoothing = RbfOptions::new().with_regularization(RbfRegularization::Diagonal(0.5));
        let smoothed = RbfInterpolator::with_options(
            points,
            vec![1.0, 2.0, 3.0],
            RbfKernel::Gaussian(1.0),
            smoothing,
        )
        .unwrap();
        let value = smoothed
            .evaluate([0.0, 0.0], ExtrapolationPolicy::Reject)
            .unwrap();
        assert!((value - 1.0).abs() > 1e-3);
    }

    #[test]
    fn out_of_box_queries_follow_policy() {
        let interp = RbfInterpolator::with_options(
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![1.0, 2.0, 3.0],
            RbfKernel::Multiquadric(1.0),
            options(),
        )
        .unwrap();
        let result = interp.evaluate([1.2, 0.2], ExtrapolationPolicy::Reject);
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::DomainError { .. }))
        );
    }

    #[test]
    fn extreme_coordinates_use_scaled_distances() {
        let base = 1e200_f64;
        let gap = 1e190_f64;
        let points = vec![[base, base], [base + gap, base], [base, base + gap]];
        let interp = RbfInterpolator::new(
            points.clone(),
            vec![1.0, 2.0, 3.0],
            RbfKernel::Gaussian(1e-190),
        )
        .unwrap();
        let value = interp
            .evaluate(points[1], ExtrapolationPolicy::Reject)
            .unwrap();
        assert!((value - 2.0).abs() < 1e-10);
        assert!(
            matches!(crate::geometry::distance(&[-f64::MAX, 0.0], &[f64::MAX, 0.0]), Err(error) if matches!(error.view(), SciOxideErrorView::NonFinite { origin: NonFiniteOrigin::Intermediate, .. }))
        );
        assert!(
            matches!(RbfKernel::Multiquadric(f64::MAX).evaluate(f64::MAX), Err(error) if matches!(error.view(), SciOxideErrorView::NonFinite { origin: NonFiniteOrigin::Intermediate, .. }))
        );
    }

    #[test]
    fn affine_values_equal_component_evaluation_in_both_widths() {
        let points = vec![[0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let kernel = RbfKernel::Gaussian(1.0_f32);
        let options = RbfOptions::new()
            .with_regularization(RbfRegularization::Diagonal(1e-5))
            .with_rank_threshold(RankThreshold::new(0.0, 1e-5).unwrap());
        let complex = vec![
            Complex::new(1.0_f32, -1.0),
            Complex::new(2.0, 0.5),
            Complex::new(3.0, 2.0),
        ];
        let model = RbfInterpolator::with_options(points.clone(), complex.clone(), kernel, options)
            .unwrap();
        assert_eq!(model.conditioning().numerical_rank(), 3);
        let real = RbfInterpolator::with_options(
            points.clone(),
            complex.iter().map(|v| v.re).collect(),
            kernel,
            options,
        )
        .unwrap();
        let imaginary = RbfInterpolator::with_options(
            points,
            complex.iter().map(|v| v.im).collect(),
            kernel,
            options,
        )
        .unwrap();
        let query = [0.25_f32, 0.25];
        let value = model.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
        assert_eq!(
            value.re,
            real.evaluate(query, ExtrapolationPolicy::Reject).unwrap()
        );
        assert_eq!(
            value.im,
            imaginary
                .evaluate(query, ExtrapolationPolicy::Reject)
                .unwrap()
        );
    }

    #[test]
    fn byte_limit_is_a_resource_limit() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let gaussian = RbfKernel::Gaussian(1.0);
        let limited = options().with_maximum_bytes(64);
        let result =
            RbfInterpolator::with_options(points.clone(), vec![1.0, 2.0, 3.0], gaussian, limited);
        assert!(
            matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
        );
        let required = preparation_bytes::<f64>(3, 1, &limited).unwrap();
        let exact = limited.with_maximum_bytes(required);
        assert!(
            RbfInterpolator::with_options(points, vec![1.0, 2.0, 3.0], gaussian, exact).is_ok()
        );
    }
}
