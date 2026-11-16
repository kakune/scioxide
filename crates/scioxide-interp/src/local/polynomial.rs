//! Local least-squares polynomial fits with adaptive neighborhood and degree.

use alloc::vec::Vec;

use crate::factor::SystemKind;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValueMut, Argument, Capacity, ExecutionContext, Interval, NumericScalar, ProblemToken,
    RealScalar, SciOxideError, SolveIdentity, SolveScope, SolveState,
};
use scioxide_linalg::Matrix;

use super::PROPOSAL_DENIED;
use super::{
    LocalFitReport, LocalModelOptions, LocalModelPlan, LocalPredictionWorkspace, Result, bounds,
    checked_factor, coefficients, distance, evaluate_coefficients, intermediate, invalid, require,
    resolve,
};
use crate::error::require_proposal;
use crate::protocol::{ExtrapolationPolicy, Interpolant};
use crate::validate::validate_finite;

/// Sets the degree and neighborhood size of a [`LocalPolynomialModel`].
///
/// A fit first tries the requested total degree with `minimum_neighbors`
/// samples and doubles the number of samples, up to `maximum_neighbors`, until
/// the least-squares system has full rank and passes the condition limit. Only
/// if the largest neighborhood fails is the degree lowered by one, down to
/// `minimum_degree` (zero by default), and the search repeats.
///
/// ```
/// use scioxide_interp::{LocalModelOptions, LocalPolynomialOptions};
/// let options = LocalPolynomialOptions::<f64>::new(2, 6, 12)
///     .with_minimum_degree(1)
///     .with_numerical(LocalModelOptions::new().with_max_system_order(64));
/// assert_eq!((options.degree(), options.minimum_degree()), (2, 1));
/// assert_eq!(
///     (options.minimum_neighbors(), options.maximum_neighbors()),
///     (6, 12)
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LocalPolynomialOptions<S: RealScalar> {
    degree: usize,
    minimum_degree: usize,
    minimum_neighbors: usize,
    maximum_neighbors: usize,
    numerical: LocalModelOptions<S>,
}

impl<S: RealScalar> LocalPolynomialOptions<S> {
    /// Creates options for total degree `degree` (zero is a constant) on between
    /// `minimum_neighbors` and `maximum_neighbors` nearest samples, with the
    /// default [`LocalModelOptions`].
    ///
    /// The values are checked when a fit starts.
    pub fn new(degree: usize, minimum_neighbors: usize, maximum_neighbors: usize) -> Self {
        Self {
            degree,
            minimum_degree: 0,
            minimum_neighbors,
            maximum_neighbors,
            numerical: LocalModelOptions::new(),
        }
    }

    /// Sets the lowest degree to try when the largest neighborhood fails.
    pub const fn with_minimum_degree(mut self, degree: usize) -> Self {
        self.minimum_degree = degree;
        self
    }

    /// Sets the rank, conditioning, size and work limits.
    pub const fn with_numerical(mut self, numerical: LocalModelOptions<S>) -> Self {
        self.numerical = numerical;
        self
    }

    /// Returns the total degree tried first.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the lowest degree tried.
    pub const fn minimum_degree(&self) -> usize {
        self.minimum_degree
    }

    /// Returns the number of nearest samples tried first.
    pub const fn minimum_neighbors(&self) -> usize {
        self.minimum_neighbors
    }

    /// Returns the largest number of nearest samples tried.
    pub const fn maximum_neighbors(&self) -> usize {
        self.maximum_neighbors
    }

    /// Returns the rank, conditioning, size and work limits.
    pub const fn numerical(&self) -> LocalModelOptions<S> {
        self.numerical
    }
}

/// Returns all exponent vectors of total degree at most `degree`; `limit` is
/// the largest accepted number of terms.
pub(super) fn powers<const D: usize>(degree: usize, limit: usize) -> Result<Vec<[usize; D]>> {
    // Enumerate every exponent vector with total degree <= `degree` (the full
    // total-degree space, C(degree + D, D) terms) by depth-first assignment
    // of the remaining degree budget axis by axis. The count is checked
    // against `limit` before allocating; the in-recursion guard is a backstop.
    fn visit<const D: usize>(
        axis: usize,
        remaining: usize,
        p: &mut [usize; D],
        out: &mut Vec<[usize; D]>,
        limit: usize,
    ) -> Result<()> {
        if axis == D {
            if out.len() == limit {
                return Err(invalid(
                    scioxide_core::argument!("options"),
                    "polynomial term budget exceeded",
                ));
            }
            out.push(*p);
            return Ok(());
        }
        for exponent in 0..=remaining {
            p[axis] = exponent;
            visit(axis + 1, remaining - exponent, p, out, limit)?;
        }
        Ok(())
    }
    let count = scioxide_core::checked_binomial(
        checked_add(degree, D, scioxide_core::quantity!("polynomial_terms"))?,
        D,
    )?;
    if count > limit {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "polynomial term budget exceeded",
        ));
    }
    let mut out = try_with_capacity(count, scioxide_core::quantity!("polynomial_terms"))?;
    visit(0, degree, &mut [0; D], &mut out, limit)?;
    Ok(out)
}

/// Evaluates the scaled monomial basis, naming `argument` for a buffer of the
/// wrong length.
pub(super) fn basis<S: RealScalar, const D: usize>(
    query: [S; D],
    center: [S; D],
    scale: S,
    powers: &[[usize; D]],
    out: &mut [S],
    argument: Argument,
) -> Result<()> {
    if out.len() != powers.len() {
        return Err(invalid(argument, "polynomial basis workspace mismatch"));
    }
    for (value, p) in out.iter_mut().zip(powers) {
        let mut product = S::ONE;
        for axis in 0..D {
            let x = intermediate(intermediate(query[axis] - center[axis])? / scale)?;
            for _ in 0..p[axis] {
                product = intermediate(product * x)?;
            }
        }
        *value = product;
    }
    Ok(())
}

/// Returns the largest distance from `center`, or one when every point equals it.
pub(super) fn neighborhood_scale<S: RealScalar, const D: usize>(
    points: &[[S; D]],
    center: [S; D],
) -> Result<S> {
    let scale = points.iter().try_fold(S::ZERO, |s, p| {
        Ok::<_, SciOxideError>(s.max(distance(p, &center)?))
    })?;
    Ok(if scale == S::ZERO { S::ONE } else { scale })
}

/// Fits a polynomial by least squares to the nearest samples of a center.
///
/// The model is the polynomial of total degree `p` in `D` variables,
///
/// ```text
/// f(x) = Σ_α c_α ((x − x₀)/h)^α,   |α| ≤ p,
/// ```
///
/// where `x₀` is the center, `h` the distance from `x₀` to the farthest selected
/// sample, and `α` runs over the `C(p + D, D)` exponent vectors. The scaling
/// makes the conditioning independent of the units and position of the data.
/// The coefficients minimize the unweighted sum of squared residuals at the
/// selected samples, computed with column-pivoted QR (SVD if QR fails the rank
/// or condition limit). [`LocalPolynomialOptions`] describes how the degree
/// and the number of samples are chosen; [`Self::fit_report`] reports the
/// result.
///
/// The neighborhood is fixed when the model is built: every query uses the
/// same polynomial. [`Interpolant::evaluate`] returns a `Vec` of
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
///     ExtrapolationPolicy, Interpolant, LocalModelPlan, LocalPolynomialModel,
///     LocalPolynomialOptions, LocalSamples,
/// };
/// let points = [
///     [-1.0_f64, 0.0],
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
///     [0.0, -1.0],
/// ];
/// let values = points.map(|[x, y]| 1.0 + 2.0 * x - y);
/// let samples = LocalSamples::new(&points, &values)?;
/// let set = samples.points()?;
/// let plan = LocalModelPlan::<f64, 2>::new(&set, samples.values()?)?;
/// let model = LocalPolynomialModel::new(&plan, [0.0, 0.0], LocalPolynomialOptions::new(1, 3, 6))?;
/// let value = model.evaluate([0.5, 0.25], ExtrapolationPolicy::Reject)?;
/// assert!((value[0] - 1.75).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct LocalPolynomialModel<S: RealScalar, const D: usize> {
    center: [S; D],
    scale: S,
    powers: Vec<[usize; D]>,
    coefficients: Vec<S>,
    components: usize,
    domains: [Interval<S>; D],
    indices: Vec<usize>,
    report: LocalFitReport<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> LocalPolynomialModel<S, D> {
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
        options: LocalPolynomialOptions<S>,
    ) -> Result<Self> {
        Self::build(plan, center, options, ExecutionContext::sequential_native())
    }

    /// Fits the model around `center` in `scope`.
    ///
    /// `scope` is an [`ExecutionContext`] or the `&mut SolveContext` of an
    /// enclosing solve. The fit runs with [`LocalModelOptions::budgets`] and
    /// counts one proposal for each neighborhood and degree it tries (see
    /// [solve accounting](scioxide_core#solve-accounting)). The options, the
    /// center and the byte limit are checked before any work starts.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `center` has a NaN or infinite coordinate.
    /// - `InvalidInput` if `minimum_degree > degree`, `minimum_neighbors` is
    ///   zero or exceeds `maximum_neighbors`, `maximum_neighbors` exceeds the
    ///   number of samples, the degree or number of terms exceeds
    ///   [`LocalModelOptions::max_system_order`], or the minimum reciprocal
    ///   condition number is outside `[0, 1]`.
    /// - `ResourceLimit` if the memory bound exceeds
    ///   [`LocalModelOptions::maximum_bytes`].
    /// - `Singular` if no permitted degree and neighborhood gives a system with
    ///   full rank that passes the condition limit.
    /// - `ConvergenceFailure` with reason `ResourceLimit(Proposal)` if the
    ///   proposal limit stops the search.
    /// - `ExecutionStopped` on cancellation or a deadline.
    pub fn build<'scope, 'capability>(
        plan: &LocalModelPlan<'_, '_, S, D>,
        center: [S; D],
        options: LocalPolynomialOptions<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        validate_finite(&center)?;
        let numerical = options.numerical;
        if options.minimum_degree > options.degree
            || options.minimum_neighbors == 0
            || options.minimum_neighbors > options.maximum_neighbors
            || options.maximum_neighbors > plan.tree.len()
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "invalid polynomial degree/neighborhood range",
            ));
        }
        if options.degree >= numerical.max_system_order {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "polynomial degree exceeds term budget",
            ));
        }
        numerical.preflight::<D>(
            options.maximum_neighbors,
            scioxide_core::checked_binomial(
                checked_add(
                    options.degree,
                    D,
                    scioxide_core::quantity!("polynomial_terms"),
                )?,
                D,
            )?,
            plan.components,
            scioxide_core::argument!("options"),
        )?;
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "local-polynomial", 1),
            ((options.maximum_neighbors as u128) << 64) | D as u128,
            options.degree as u128,
        );
        let mut state = SolveState::new();
        scope
            .into()
            .run(&mut state, identity, numerical.budgets, |context| {
                // Adaptation order: for the highest permitted degree first,
                // double the neighbourhood until the least-squares system has at
                // least as many rows as terms and passes the rank/conditioning
                // checks; only when the largest neighbourhood still fails is the
                // degree lowered. Rank and conditioning failures (`Singular`)
                // mean "try the next configuration"; every other error aborts.
                let mut attempts = 0;
                for degree in (options.minimum_degree..=options.degree).rev() {
                    let terms = powers::<D>(degree, numerical.max_system_order)?;
                    let mut count = options.minimum_neighbors;
                    loop {
                        require_proposal(context, PROPOSAL_DENIED)?;
                        attempts += 1;
                        let data = plan.select(
                            center,
                            count,
                            context.execution(),
                            scioxide_core::argument!("options"),
                        )?;
                        let scale = neighborhood_scale(&data.points, center)?;
                        if count >= terms.len() {
                            let mut entries = try_filled(
                                checked_mul(
                                    count,
                                    terms.len(),
                                    scioxide_core::quantity!("design_entries"),
                                )?,
                                S::ZERO,
                                scioxide_core::quantity!("design_entries"),
                            )?;
                            for (p, row) in data
                                .points
                                .iter()
                                .zip(entries.chunks_exact_mut(terms.len()))
                            {
                                basis(
                                    *p,
                                    center,
                                    scale,
                                    &terms,
                                    row,
                                    scioxide_core::argument!("options"),
                                )?;
                            }
                            let matrix = Matrix::from_vec(count, terms.len(), entries)?;
                            match checked_factor(
                                &matrix,
                                SystemKind::LeastSquares,
                                &numerical,
                                context.execution(),
                            ) {
                                Ok(factor) => {
                                    let (coefficients, residual) =
                                        coefficients(&data, &matrix, &factor, context.execution())?;
                                    context.execution().check()?;
                                    let model = Self {
                                        center,
                                        scale,
                                        domains: bounds(
                                            data.points.iter().copied(),
                                            scioxide_core::argument!("plan"),
                                        )?,
                                        components: data.components,
                                        indices: data.indices,
                                        report: LocalFitReport {
                                            neighbors: count,
                                            degree: Some(degree),
                                            rank: terms.len(),
                                            reciprocal_condition: factor.reciprocal_condition,
                                            factorization: factor.method(),
                                            condition_norm: factor.condition_norm(),
                                            residual,
                                            attempts,
                                        },
                                        powers: terms,
                                        coefficients,
                                    };
                                    context.commit_accepted(
                                        scioxide_core::PartialResultTrust::Admissible,
                                    )?;
                                    return Ok(model);
                                }
                                Err(error)
                                    if error.kind()
                                        == scioxide_core::SciOxideErrorKind::Singular => {}
                                Err(error) => return Err(error),
                            }
                        }
                        if count == options.maximum_neighbors {
                            break;
                        }
                        count = count.saturating_mul(2).min(options.maximum_neighbors);
                    }
                }
                Err(SciOxideError::singular(None))
            })
    }

    /// Returns the chosen degree and neighborhood, the conditioning and the
    /// residual.
    pub const fn fit_report(&self) -> LocalFitReport<S> {
        self.report
    }

    /// Returns the indices of the selected samples, ordered by distance from
    /// the center and then by index.
    #[must_use]
    pub fn sample_indices(&self) -> &[usize] {
        &self.indices
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> Interpolant<S, D>
    for LocalPolynomialModel<S, D>
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
        LocalPredictionWorkspace::new(0, self.powers.len(), self.components)
    }
    fn workspace_capacity(&self, workspace: &LocalPredictionWorkspace<S>) -> Capacity {
        workspace.covers(self.powers.len(), self.components)
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
            self.powers.len(),
            workspace.basis.len(),
        )?;
        let point = resolve(self.domains, point, policy)?;
        let terms = &mut workspace.basis[..self.powers.len()];
        basis(
            point,
            self.center,
            self.scale,
            &self.powers,
            terms,
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

    #[test]
    fn exponents_enumerate_total_degree_and_respect_limits() {
        let terms = powers::<2>(2, 64).unwrap();
        assert_eq!(terms.len(), 6);
        assert!(terms.iter().all(|p| p[0] + p[1] <= 2));
        assert!(powers::<2>(2, 5).is_err());
        let mut out = [0.0; 3];
        basis(
            [2.0, 3.0],
            [1.0, 1.0],
            2.0,
            &powers::<2>(1, 8).unwrap(),
            &mut out,
            scioxide_core::argument!("out"),
        )
        .unwrap();
        assert_eq!(out, [1.0, 1.0, 0.5]);
        assert!(neighborhood_scale(&[[1.0_f64]], [1.0]).unwrap() == 1.0);
    }

    #[test]
    fn invalid_ranges_and_centers_are_rejected_before_preparation() {
        let samples = LocalSamples::new(&[[0.0_f64], [1.0], [2.0]], &[1.0, 2.0, 3.0]).unwrap();
        let points = samples.points().unwrap();
        let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values().unwrap()).unwrap();
        assert!(
            LocalPolynomialModel::new(&plan, [f64::NAN], LocalPolynomialOptions::new(1, 2, 3))
                .is_err()
        );
        assert!(
            LocalPolynomialModel::new(&plan, [0.0], LocalPolynomialOptions::new(1, 3, 2)).is_err()
        );
        assert!(
            LocalPolynomialModel::new(&plan, [0.0], LocalPolynomialOptions::new(1, 2, 4)).is_err()
        );
        assert!(
            LocalPolynomialModel::new(
                &plan,
                [0.0],
                LocalPolynomialOptions::new(1, 2, 3).with_minimum_degree(2)
            )
            .is_err()
        );
    }
}
