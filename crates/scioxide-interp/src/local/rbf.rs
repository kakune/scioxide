//! Radial basis function interpolation on the nearest samples of a center.

use alloc::vec::Vec;

use crate::factor::SystemKind;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AffineValueMut, Capacity, ExecutionContext, Interval, NumericScalar, ProblemToken, RealScalar,
    SciOxideError, SolveIdentity, SolveScope, SolveState,
};
use scioxide_linalg::Matrix;

use super::PROPOSAL_DENIED;
use super::polynomial::{basis, neighborhood_scale, powers};
use super::{
    LocalFitReport, LocalModelOptions, LocalModelPlan, LocalPredictionWorkspace, Result, bounds,
    checked_factor, coefficients, evaluate_coefficients, intermediate, invalid, require, resolve,
};
use crate::error::require_proposal;
use crate::protocol::{ExtrapolationPolicy, Interpolant};
use crate::rbf::{RbfKernel, RbfRegularization};
use crate::validate::validate_finite;

/// Sets the kernel, neighborhood size and regularization of a
/// [`LocalRbfModel`].
///
/// The kernel decides the polynomial part of the model: the thin-plate spline
/// adds a polynomial of degree one, the multiquadric a constant, and the
/// positive definite Gaussian and inverse multiquadric kernels none.
///
/// ```
/// use scioxide_interp::{LocalRbfOptions, RbfKernel, RbfRegularization};
/// let options = LocalRbfOptions::new(8, RbfKernel::ThinPlateSpline)?
///     .with_regularization(RbfRegularization::Diagonal(1e-8_f64));
/// assert_eq!(options.neighbors(), 8);
/// assert_eq!(options.regularization(), RbfRegularization::Diagonal(1e-8));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LocalRbfOptions<S: RealScalar> {
    neighbors: usize,
    kernel: RbfKernel<S>,
    regularization: RbfRegularization<S>,
    numerical: LocalModelOptions<S>,
}

impl<S: RealScalar> LocalRbfOptions<S> {
    /// Creates options for `neighbors` nearest samples and `kernel`, without
    /// regularization and with the default [`LocalModelOptions`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the kernel shape parameter is not positive.
    /// - `NonFinite` if the kernel shape parameter is NaN or infinite.
    pub fn new(neighbors: usize, kernel: RbfKernel<S>) -> Result<Self> {
        kernel.validate(scioxide_core::argument!("kernel"))?;
        Ok(Self {
            neighbors,
            kernel,
            regularization: RbfRegularization::None,
            numerical: LocalModelOptions::new(),
        })
    }

    /// Sets the diagonal regularization `λ` added to the kernel block; none is
    /// added unless set here.
    pub const fn with_regularization(mut self, regularization: RbfRegularization<S>) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets the rank, conditioning, size and work limits.
    pub const fn with_numerical(mut self, numerical: LocalModelOptions<S>) -> Self {
        self.numerical = numerical;
        self
    }

    /// Returns the number of nearest samples.
    pub const fn neighbors(&self) -> usize {
        self.neighbors
    }

    /// Returns the kernel.
    pub const fn kernel(&self) -> RbfKernel<S> {
        self.kernel
    }

    /// Returns the diagonal regularization.
    pub const fn regularization(&self) -> RbfRegularization<S> {
        self.regularization
    }

    /// Returns the rank, conditioning, size and work limits.
    pub const fn numerical(&self) -> LocalModelOptions<S> {
        self.numerical
    }
}

/// Interpolates the nearest samples of a center by radial basis functions
/// with a polynomial term.
///
/// With the `k` selected samples `xᵢ`, the model is
///
/// ```text
/// f(x) = Σᵢ wᵢ φ(‖x − xᵢ‖) + Σ_α c_α ((x − x₀)/h)^α,
/// ```
///
/// where `φ` is the [`RbfKernel`], `x₀` the center, `h` the distance to the
/// farthest selected sample, and the polynomial part is chosen by the kernel
/// as described in [`LocalRbfOptions`]. The coefficients solve the symmetric
/// saddle-point system
///
/// ```text
/// [ K + λI  P ] [ w ]   [ y ]
/// [ Pᵀ      0 ] [ c ] = [ 0 ],
/// ```
///
/// where `Kᵢⱼ = φ(‖xᵢ − xⱼ‖)`, `P` holds the polynomial basis at the samples and
/// `λ` is the regularization. Without regularization `f` passes through the
/// selected samples. Without a polynomial part the system is factored by
/// Cholesky, otherwise by LU with partial pivoting. If that fails the rank or
/// condition limit, an SVD is used instead. Construction costs O((k + t)³) for
/// `t` polynomial terms, and a query costs O(k + t) per component.
///
/// The neighborhood is fixed when the model is built: every query uses the
/// same expansion. [`Interpolant::evaluate`] returns a `Vec` of
/// [`LocalModelPlan::components`] entries. The domain for
/// [`ExtrapolationPolicy`] is the bounding box of the selected samples.
///
/// # Errors
///
/// [`Self::build`] lists the construction errors. Queries fail as described in
/// the [crate docs](crate#errors).
///
/// ```
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, LocalModelPlan, LocalRbfModel, LocalRbfOptions,
///     LocalSamples, RbfKernel,
/// };
/// let points = [
///     [-1.0_f64, -1.0],
///     [-1.0, 1.0],
///     [1.0, -1.0],
///     [1.0, 1.0],
///     [0.0, 0.0],
/// ];
/// let values = points.map(|[x, y]| 2.0 + x - 3.0 * y);
/// let samples = LocalSamples::new(&points, &values)?;
/// let set = samples.points()?;
/// let plan = LocalModelPlan::<f64, 2>::new(&set, samples.values()?)?;
/// let model = LocalRbfModel::new(
///     &plan,
///     [0.0, 0.0],
///     LocalRbfOptions::new(5, RbfKernel::ThinPlateSpline)?,
/// )?;
/// // Thin plate splines reproduce affine data.
/// let value = model.evaluate([0.2, -0.3], ExtrapolationPolicy::Reject)?;
/// assert!((value[0] - 3.1).abs() < 1e-10);
/// assert_eq!(model.sample_indices().len(), 5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. Wendland, *Scattered Data Approximation*, Cambridge University Press,
/// 2005, chapter 8.
#[derive(Debug, Clone)]
#[must_use]
pub struct LocalRbfModel<S: RealScalar, const D: usize> {
    points: Vec<[S; D]>,
    indices: Vec<usize>,
    kernel: RbfKernel<S>,
    center: [S; D],
    scale: S,
    powers: Vec<[usize; D]>,
    coefficients: Vec<S>,
    components: usize,
    domains: [Interval<S>; D],
    report: LocalFitReport<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> LocalRbfModel<S, D> {
    /// Fits the model on the calling thread.
    ///
    /// Runs [`Self::build`] with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn new(
        plan: &LocalModelPlan<'_, '_, S, D>,
        center: [S; D],
        options: LocalRbfOptions<S>,
    ) -> Result<Self> {
        Self::build(plan, center, options, ExecutionContext::sequential_native())
    }

    /// Selects the nearest samples of `center` and fits the model in `scope`.
    ///
    /// `scope` is an [`ExecutionContext`] or the `&mut SolveContext` of an
    /// enclosing solve. The fit runs with [`LocalModelOptions::budgets`] and
    /// counts as one proposal (see
    /// [solve accounting](scioxide_core#solve-accounting)). The options, the
    /// center and the byte limit are checked before any work starts.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `center` has a NaN or infinite coordinate.
    /// - `InvalidInput` if the regularization is not finite and positive, the
    ///   number of neighbors is zero or exceeds the number of samples, the
    ///   system order exceeds [`LocalModelOptions::max_system_order`], or the
    ///   minimum reciprocal condition number is outside `[0, 1]`.
    /// - `ResourceLimit` if the memory bound exceeds
    ///   [`LocalModelOptions::maximum_bytes`].
    /// - `Singular` if two selected samples coincide without regularization, the
    ///   samples do not determine the polynomial part, or the system fails the
    ///   rank or condition limit.
    /// - `ConvergenceFailure` with reason `ResourceLimit(Proposal)` if the
    ///   proposal limit is exhausted.
    /// - `ExecutionStopped` on cancellation or a deadline.
    pub fn build<'scope, 'capability>(
        plan: &LocalModelPlan<'_, '_, S, D>,
        center: [S; D],
        options: LocalRbfOptions<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        validate_finite(&center)?;
        let lambda = match options.regularization {
            RbfRegularization::None => S::ZERO,
            RbfRegularization::Diagonal(x) if x.is_finite() && x > S::ZERO => x,
            RbfRegularization::Diagonal(_) => {
                return Err(invalid(
                    scioxide_core::argument!("options"),
                    "RBF smoothing must be finite and positive",
                ));
            }
        };
        let numerical = options.numerical;
        // Polynomial augmentation follows each kernel's conditional positive
        // definiteness order: thin plate splines need affine terms,
        // multiquadrics constants, while Gaussian and inverse multiquadric
        // kernels are positive definite and need none.
        let degree = match options.kernel {
            RbfKernel::ThinPlateSpline => Some(1),
            RbfKernel::Multiquadric(_) => Some(0),
            RbfKernel::Gaussian(_) | RbfKernel::InverseMultiquadric(_) => None,
        };
        let n = options.neighbors;
        let term_count = match degree {
            Some(d) => scioxide_core::checked_binomial(
                checked_add(D, d, scioxide_core::quantity!("polynomial_terms"))?,
                D,
            )?,
            None => 0,
        };
        let order = checked_add(n, term_count, scioxide_core::quantity!("system_order"))?;
        numerical.preflight::<D>(
            order,
            order,
            plan.components,
            scioxide_core::argument!("options"),
        )?;
        let powers = match degree {
            Some(d) => powers(d, numerical.max_system_order)?,
            None => Vec::new(),
        };
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "local-rbf", 1),
            ((n as u128) << 64) | D as u128,
            0,
        );
        let mut state = SolveState::new();
        scope
            .into()
            .run(&mut state, identity, numerical.budgets, |context| {
                require_proposal(context, PROPOSAL_DENIED)?;
                let data = plan.select(
                    center,
                    n,
                    context.execution(),
                    scioxide_core::argument!("options"),
                )?;
                let scale = neighborhood_scale(&data.points, center)?;
                // Saddle-point system [[K + lambda I, P], [P^T, 0]] [w; c] = [f; 0]:
                // the lower-right block stays zero from initialization, and the
                // zero right-hand side rows (left by `coefficients`, which only
                // fills the first n rows) impose the moment conditions
                // P^T w = 0. Without smoothing, coincident neighbours make two
                // rows identical, which is reported as singular immediately.
                let mut entries = try_filled(
                    checked_mul(order, order, scioxide_core::quantity!("system_entries"))?,
                    S::ZERO,
                    scioxide_core::quantity!("system_entries"),
                )?;
                let mut polynomial = try_filled(
                    powers.len(),
                    S::ZERO,
                    scioxide_core::quantity!("polynomial_basis"),
                )?;
                for (i, p) in data.points.iter().enumerate() {
                    context.execution().check()?;
                    for (j, q) in data.points.iter().enumerate() {
                        if i != j && p == q && lambda == S::ZERO {
                            return Err(SciOxideError::singular(None));
                        }
                        entries[i * order + j] = intermediate(
                            options.kernel.evaluate_between(p, q)?
                                + if i == j { lambda } else { S::ZERO },
                        )?;
                    }
                    basis(
                        *p,
                        center,
                        scale,
                        &powers,
                        &mut polynomial,
                        scioxide_core::argument!("options"),
                    )?;
                    for (j, &v) in polynomial.iter().enumerate() {
                        entries[i * order + n + j] = v;
                        entries[(n + j) * order + i] = v;
                    }
                }
                let matrix = Matrix::from_vec(order, order, entries)?;
                let kind = if degree.is_none() {
                    SystemKind::PositiveDefinite
                } else {
                    SystemKind::Symmetric
                };
                let factor = checked_factor(&matrix, kind, &numerical, context.execution())?;
                let (coefficients, residual) =
                    coefficients(&data, &matrix, &factor, context.execution())?;
                context.execution().check()?;
                let model = Self {
                    domains: bounds(
                        data.points.iter().copied(),
                        scioxide_core::argument!("plan"),
                    )?,
                    points: data.points,
                    indices: data.indices,
                    kernel: options.kernel,
                    center,
                    scale,
                    powers,
                    coefficients,
                    components: data.components,
                    report: LocalFitReport {
                        neighbors: n,
                        degree,
                        rank: order,
                        reciprocal_condition: factor.reciprocal_condition,
                        factorization: factor.method(),
                        condition_norm: factor.condition_norm(),
                        residual,
                        attempts: 1,
                    },
                };
                context.commit_accepted(scioxide_core::PartialResultTrust::Admissible)?;
                Ok(model)
            })
    }

    /// Returns the conditioning of the full system and the residual.
    pub const fn fit_report(&self) -> LocalFitReport<S> {
        self.report
    }

    /// Returns the indices of the selected samples, ordered by distance from
    /// the center and then by index.
    #[must_use]
    pub fn sample_indices(&self) -> &[usize] {
        &self.indices
    }

    /// Number of radial plus polynomial basis functions.
    fn width(&self) -> usize {
        self.points.len() + self.powers.len()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> Interpolant<S, D>
    for LocalRbfModel<S, D>
{
    type Value = Vec<S>;
    type Workspace = LocalPredictionWorkspace<S>;

    fn domains(&self) -> [Interval<S>; D] {
        self.domains
    }
    fn value_dimension(&self) -> usize {
        self.components
    }
    fn zero_value(&self) -> Result<Vec<S>> {
        try_filled(
            self.components,
            S::ZERO,
            scioxide_core::quantity!("local_value"),
        )
    }
    fn workspace(&self) -> Result<LocalPredictionWorkspace<S>> {
        LocalPredictionWorkspace::new(0, self.width(), self.components)
    }
    fn workspace_capacity(&self, workspace: &LocalPredictionWorkspace<S>) -> Capacity {
        workspace.covers(self.width(), self.components)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut LocalPredictionWorkspace<S>,
    ) -> Result<()> {
        scioxide_core::check_value_shape(out, self.components)?;
        require(
            self.workspace_capacity(workspace),
            self.width(),
            workspace.basis.len(),
        )?;
        let point = resolve(self.domains, point, policy)?;
        let terms = &mut workspace.basis[..self.width()];
        let (radial, polynomial) = terms.split_at_mut(self.points.len());
        for (v, q) in radial.iter_mut().zip(&self.points) {
            *v = self.kernel.evaluate_between(&point, q)?;
        }
        basis(
            point,
            self.center,
            self.scale,
            &self.powers,
            polynomial,
            scioxide_core::argument!("workspace"),
        )?;
        let staged = &mut workspace.output[..self.components];
        evaluate_coefficients(terms, &self.coefficients, staged)?;
        scioxide_core::write_value(staged, out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::LocalSamples;
    use super::*;
    use scioxide_core::SciOxideErrorView;

    #[test]
    fn options_are_validated_before_preparation() {
        let samples = LocalSamples::new(&[[0.0_f64], [1.0], [2.0]], &[1.0, 2.0, 3.0]).unwrap();
        let points = samples.points().unwrap();
        let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values().unwrap()).unwrap();
        let gaussian = LocalRbfOptions::new(3, RbfKernel::Gaussian(1.0)).unwrap();
        assert!(LocalRbfModel::new(&plan, [f64::INFINITY], gaussian).is_err());
        assert!(LocalRbfOptions::new(3, RbfKernel::Gaussian(0.0)).is_err());
        assert!(
            LocalRbfModel::new(
                &plan,
                [0.0],
                gaussian.with_regularization(RbfRegularization::Diagonal(-1.0))
            )
            .is_err()
        );
        assert!(
            LocalRbfModel::new(
                &plan,
                [0.0],
                LocalRbfOptions::new(4, RbfKernel::Gaussian(1.0)).unwrap()
            )
            .is_err()
        );
        let limited = gaussian.with_numerical(LocalModelOptions::new().with_maximum_bytes(1));
        assert!(
            matches!(LocalRbfModel::new(&plan, [0.0], limited), Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
        );
        let model = LocalRbfModel::new(&plan, [0.0], gaussian).unwrap();
        assert_eq!(model.fit_report().degree(), None);
        assert_eq!(model.width(), 3);
    }
}
