//! Complex contour integrals over a parameterized path.
use scioxide_core::{AlgorithmId, Complex, Interval, RealScalar, SciOxideError, SolveScope};

use super::{WeightedQuadraturePlan, WeightedQuadratureWorkspace};
use crate::SubdivisionOptions;
use crate::report::IntegrationReport;
use crate::workspace::IntegrationRequirements;

/// Adaptive quadrature of a complex contour integral over a parameterized path.
///
/// Construction prepares a 15-point Gauss–Kronrod rule and a 25 × 25 cosine table, using O(25²) work and storage.
///
/// ```text
/// ∫_C f(z) dz = ∫ₐᵇ f(z(t)) z′(t) dt,
/// ```
///
/// where `z: [a, b] → ℂ` parameterizes the contour `C` and `z′ = dz/dt`. The map
/// callback returns `(z(t), z′(t))`, and the integrand callback returns `f(z)`.
/// The map sets the orientation; a closed or reversed contour needs no special
/// treatment. Isolated zeros of `z′` are allowed. The plan does not choose
/// branches of multivalued functions: `f` must be continuous along `C`.
///
/// The plan integrates the real and imaginary parts of `f(z(t)) z′(t)` by
/// adaptive 15-point Gauss–Kronrod quadrature with the given breakpoints,
/// under its [`SubdivisionOptions`]. Each node calls the map and then `f`. Both
/// calls count toward the evaluation limit; the report counts calls of `f`. The
/// error estimate does not account for errors in `z′`.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
/// use scioxide_core::{Complex, Interval};
/// use scioxide_integrate::{ContourPlan, SubdivisionOptions};
///
/// // ∮ conj(z) dz around the unit circle, z(t) = e^(it), is 2πi.
/// let interval = Interval::new(0.0, 2.0 * PI)?;
/// let plan = ContourPlan::new(interval, &[], SubdivisionOptions::default())?;
/// let circle = |t: f64| {
///     let z = Complex::new(t.cos(), t.sin());
///     Ok((z, Complex::new(-z.im, z.re))) // z′ = i z
/// };
/// let (value, report) = plan.integrate(&|z: Complex<f64>| Ok(z.conj()), &circle)?;
/// assert!(report.is_success() && value.re.abs() < 1e-10);
/// assert!((value.im - 2.0 * PI).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ContourPlan<S: RealScalar> {
    quadrature: WeightedQuadraturePlan<S>,
}

impl<S: RealScalar> ContourPlan<S> {
    /// Prepares the contour integral over the parameter interval `parameter`,
    /// split at `breakpoints` where the path or `f` is not smooth.
    ///
    /// The map and the integrand are passed to each call, so one plan serves
    /// any contour with this parameter interval.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a breakpoint is NaN or infinite.
    /// - `InvalidInput` if a breakpoint lies outside `parameter`.
    /// - `SizeOverflow` or `AllocationFailure` if the region storage cannot be
    ///   sized or reserved.
    pub fn new(
        parameter: Interval<S>,
        breakpoints: &[S],
        options: SubdivisionOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            quadrature: WeightedQuadraturePlan::breakpoints(
                parameter,
                breakpoints,
                super::WeightedQuadratureOptions::new(options),
            )?,
        })
    }

    /// Returns the options.
    pub const fn options(&self) -> &SubdivisionOptions<S> {
        self.quadrature.options().subdivision_options()
    }

    /// Returns the identifier of the method, as reported in diagnostics.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "ContourPlan", 1)
    }

    /// Returns the nonempty parameter segments in ascending order.
    pub fn segments(&self) -> &[Interval<S>] {
        self.quadrature.segments()
    }

    /// Returns the storage that a workspace needs for the two real components
    /// of a complex value.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size is not representable.
    pub fn workspace_requirements(&self) -> Result<IntegrationRequirements, SciOxideError> {
        self.quadrature.workspace_requirements(2)
    }

    /// Creates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    pub fn workspace(&self) -> Result<WeightedQuadratureWorkspace<S>, SciOxideError> {
        self.quadrature.workspace(2)
    }

    /// Integrates `f(z(t)) z′(t)` over the parameter interval, with `map`
    /// returning `(z(t), z′(t))`, and writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// The workspace's component errors are ordered `[re, im]`. The nodes are
    /// evaluated under the execution policy of the options; the result and the
    /// evaluation count do not depend on it. See [Termination](crate#termination)
    /// for results that end before the tolerance is met.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a per-component tolerance does not have two
    ///   components.
    /// - `InvalidInput` if the workspace does not have two components or holds
    ///   fewer regions than the plan needs.
    /// - `NonFinite` if the map returns a NaN or infinite `z` or `z′`.
    /// - The callback, non-finite and stop errors listed under
    ///   [Errors](crate#errors).
    pub fn integrate_into<'scope, 'capability, M, F>(
        &self,
        f: F,
        map: M,
        out: &mut Complex<S>,
        workspace: &mut WeightedQuadratureWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        M: Fn(S) -> Result<(Complex<S>, Complex<S>), SciOxideError> + Sync,
        F: Fn(Complex<S>) -> Result<Complex<S>, SciOxideError> + Sync,
    {
        let map = &map;
        let f = &f;
        workspace.published_errors = false;
        if let Err(error) = scioxide_core::check_value_shape(out, 2)
            .and_then(|()| crate::workspace::check_tolerance(self.options().tolerance(), 2))
            .and_then(|()| {
                if workspace.check(&self.workspace_requirements()?)?
                    == scioxide_core::Capacity::NeedsGrowth
                {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("workspace"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "insufficient contour storage",
                    ));
                }
                Ok(())
            })
        {
            return workspace.attempts.reject(error);
        }
        let mut attempts = core::mem::take(&mut workspace.attempts);
        let mut errors = core::mem::take(&mut workspace.errors);
        let tolerance = self.options().tolerance();
        let result = attempts.run(
            scope.into(),
            crate::workspace::identity(self.algorithm_id(), 2),
            self.options().work_allowance(),
            |context| {
                let progress = self.quadrature.weighting.run_controlled(
                    self.options(),
                    out,
                    &mut errors,
                    workspace,
                    context,
                    &mut |points, rows, cx| {
                        crate::evaluation::sample_rows_counted(
                            points.len(),
                            2,
                            rows,
                            self.options().execution_policy(),
                            cx,
                            &|index, row, counts| node(map, f, points[index], row, counts),
                        )
                    },
                    S::ZERO,
                    true,
                    2,
                    0,
                    &crate::adaptive::regional::RegionalPolicy {
                        extrapolation: false,
                        norm: &|e, v| tolerance.normalized_norm(e, v),
                        refine: true,
                        offsets: None,
                    },
                )?;
                if !progress.measured {
                    return Err(crate::workspace::unpublished(
                        progress.termination,
                        self.algorithm_id(),
                        context.counts(),
                    ));
                }
                Ok(IntegrationReport::new(
                    progress.termination,
                    progress.num_evals,
                    progress.subdivisions,
                    crate::IntegrationErrorComponents::envelope(errors.iter()),
                ))
            },
        );
        workspace.attempts = attempts;
        workspace.errors = errors;
        workspace.published_errors = result.is_ok();
        result
    }

    /// Integrates `f` along the contour given by `map` and returns the value
    /// with its report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Uses a new workspace and
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::integrate_into`].
    pub fn integrate<M, F>(
        &self,
        f: F,
        map: M,
    ) -> Result<(Complex<S>, IntegrationReport<S>), SciOxideError>
    where
        M: Fn(S) -> Result<(Complex<S>, Complex<S>), SciOxideError> + Sync,
        F: Fn(Complex<S>) -> Result<Complex<S>, SciOxideError> + Sync,
    {
        let map = &map;
        let f = &f;
        crate::execution::owned(
            Complex::new(S::ZERO, S::ZERO),
            |_| self.workspace(),
            |out, ws, cx| self.integrate_into(f, map, out, ws, cx),
        )
    }
}

/// Writes one `[re, im]` row of `f(z(t)) z'(t)`, checking that both callbacks
/// return finite values.
fn node<S, M, F>(
    map: &M,
    function: &F,
    t: S,
    row: &mut [S],
    counts: &mut [usize; 2],
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    M: Fn(S) -> Result<(Complex<S>, Complex<S>), SciOxideError> + ?Sized,
    F: Fn(Complex<S>) -> Result<Complex<S>, SciOxideError> + ?Sized,
{
    *counts = [1, 0];
    let (z, jacobian) = map(t)?;
    for v in [z.re, z.im, jacobian.re, jacobian.im] {
        crate::kernel::checked_callback_value(v)?;
    }
    *counts = [2, 1];
    let value = function(z)?;
    crate::kernel::checked_callback_value(value.re)?;
    crate::kernel::checked_callback_value(value.im)?;
    row[0] = scioxide_core::checked_difference_of_products(
        value.re,
        jacobian.re,
        value.im,
        jacobian.im,
    )?;
    row[1] =
        scioxide_core::checked_sum_of_products(&[value.re, value.im], &[jacobian.im, jacobian.re])?;
    Ok(())
}
