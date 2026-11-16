//! Result for Newton–Krylov solves.
use super::*;

/// Records one inner linear solve of a [`NewtonKrylovSolver`] and the Newton step
/// that followed it.
///
/// Norms are taken after the [`NonlinearScaling`] row divisors, if any.
#[derive(Debug, Clone, Copy)]
pub struct NewtonKrylovObservation<R: RealScalar> {
    /// Residual norm at linearization.
    pub(crate) residual_norm: R,
    /// Requested relative linear tolerance.
    pub(crate) forcing: R,
    /// Verified linear residual, when available.
    pub(crate) linear_residual: Option<R>,
    /// Inner iterations.
    pub(crate) linear_iterations: usize,
    /// Inner termination status.
    pub(crate) linear_termination: LinearSolveTermination,
    /// Accepted line-search step, or zero when no trial was accepted.
    pub(crate) step_length: R,
}
impl<R: RealScalar> NewtonKrylovObservation<R> {
    /// Returns `‖F(x)‖₂` at the point where the Jacobian was applied.
    pub const fn residual_norm(&self) -> R {
        self.residual_norm
    }
    /// Returns the forcing term `η` requested from the inner solve.
    pub const fn forcing(&self) -> R {
        self.forcing
    }
    /// Returns the linear residual `‖F(x) + J d‖₂` recomputed by the Newton
    /// solver, or `None` if it was not computed.
    pub const fn linear_residual(&self) -> Option<R> {
        self.linear_residual
    }
    /// Returns the number of inner iterations.
    pub const fn linear_iterations(&self) -> usize {
        self.linear_iterations
    }
    /// Returns the termination status of the inner solve.
    pub const fn linear_termination(&self) -> LinearSolveTermination {
        self.linear_termination
    }
    /// Returns the accepted line-search step `α ∈ (0, 1]`, or zero if no trial
    /// point was accepted.
    pub const fn step_length(&self) -> R {
        self.step_length
    }
}
/// Holds the point, residual, status and diagnostics of a [`NewtonKrylovSolver`]
/// solve.
///
/// Check [`termination`](Self::termination) before using the point. An
/// unsuccessful result from a budget, cancellation or deadline stop still holds
/// the last accepted point and its residual. Before the first solve the status is
/// [`RootTermination::NotStarted`] and [`residual_report`](Self::residual_report)
/// is `None`.
#[derive(Debug, Clone)]
pub struct NewtonKrylovResult<T: NumericScalar> {
    pub(crate) point: Vec<T>,
    pub(crate) residual: Vec<T>,
    pub(crate) report: Option<ResidualReport<T::Real>>,
    pub(crate) termination: RootTermination,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) iterations: usize,
    pub(crate) last_linear: Option<NewtonKrylovObservation<T::Real>>,
    pub(crate) history: Vec<NewtonKrylovObservation<T::Real>>,
}
impl<T: NumericScalar> NewtonKrylovResult<T> {
    pub(crate) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        crate::workspace::sum([
            crate::workspace::buffer(&self.point),
            crate::workspace::buffer(&self.residual),
            crate::workspace::buffer(&self.history),
        ])
    }
    pub(crate) fn required_bytes(
        n: usize,
        options: &NewtonKrylovOptions<T::Real>,
    ) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("nonlinear_result_bytes");
        scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_mul(
                scioxide_core::backend::checked_mul(n, 2, q)?,
                core::mem::size_of::<T>(),
                q,
            )?,
            scioxide_core::backend::checked_mul(
                if options.history {
                    options.iterations.get()
                } else {
                    0
                },
                core::mem::size_of::<NewtonKrylovObservation<T::Real>>(),
                q,
            )?,
            q,
        )
    }

    pub(crate) fn has_capacity(&self, n: usize, options: &NewtonKrylovOptions<T::Real>) -> bool {
        self.point.len() == n
            && (!options.history || self.history.capacity() >= options.iterations.get())
    }
    pub(crate) fn empty() -> Self {
        Self {
            point: Vec::new(),
            residual: Vec::new(),
            report: None,
            termination: RootTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
            iterations: 0,
            last_linear: None,
            history: Vec::new(),
        }
    }
    /// Creates an empty result for `dimension` unknowns, to pass to
    /// [`NewtonKrylovSolver::solve_into`] with the same `options`. With
    /// [`NewtonKrylovOptions::with_history`], it reserves one observation per
    /// Newton iteration.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(
        dimension: usize,
        options: &NewtonKrylovOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                InvalidInputReason::Inconsistent,
                "empty nonlinear system",
            ));
        }
        Ok(Self {
            point: scioxide_core::backend::try_filled(
                dimension,
                T::ZERO,
                scioxide_core::quantity!("result.point"),
            )?,
            residual: scioxide_core::backend::try_filled(
                dimension,
                T::ZERO,
                scioxide_core::quantity!("result.residual"),
            )?,
            report: None,
            termination: RootTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
            iterations: 0,
            last_linear: None,
            history: scioxide_core::backend::try_with_capacity(
                if options.history {
                    options.iterations.get()
                } else {
                    0
                },
                scioxide_core::quantity!("result.history"),
            )?,
        })
    }
    /// Resizes this result for `n` unknowns and `options`, and resets it to
    /// [`RootTermination::NotStarted`]. Existing capacity is reused.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds
    ///   [`NewtonKrylovOptions::with_workspace_byte_limit`].
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovResult};
    ///
    /// let options = NewtonKrylovOptions::<f64>::default();
    /// let mut result = NewtonKrylovResult::<f64>::new(4, &options)?;
    /// result.try_resize(2, &options)?;
    /// assert_eq!(result.point().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_resize(
        &mut self,
        n: usize,
        options: &NewtonKrylovOptions<T::Real>,
    ) -> Result<(), SciOxideError> {
        let bytes = self
            .retained_bytes()?
            .max(Self::required_bytes(n, options)?);
        if bytes > options.byte_limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("nonlinear_result_bytes"),
                bytes,
                options.byte_limit,
            ));
        }
        for v in [&mut self.point, &mut self.residual] {
            scioxide_core::backend::try_reserve(
                v,
                n.saturating_sub(v.len()),
                scioxide_core::quantity!("nonlinear_result"),
            )?;
            v.resize(n, T::ZERO);
        }
        let history = if options.history {
            options.iterations.get()
        } else {
            0
        };
        let additional_history = history.saturating_sub(self.history.len());
        scioxide_core::backend::try_reserve(
            &mut self.history,
            additional_history,
            scioxide_core::quantity!("nonlinear_history"),
        )?;
        self.history.clear();
        self.report = None;
        self.termination = RootTermination::NotStarted;
        self.counts = scioxide_core::SolveCounts::new();
        self.iterations = 0;
        self.last_linear = None;
        Ok(())
    }
    /// Returns the last accepted point. It is meaningful only when
    /// [`residual_report`](Self::residual_report) is `Some`.
    pub fn point(&self) -> &[T] {
        &self.point
    }
    /// Returns `F(x)` at [`point`](Self::point), without the row divisors.
    pub fn residual(&self) -> &[T] {
        &self.residual
    }
    /// Returns the residual norm used by the stopping test (after the row divisors,
    /// if any) and its reference: the initial norm, or the scale set by
    /// [`NewtonKrylovOptions::with_residual_scale`]. A small residual does not bound
    /// the error in `x`.
    pub fn residual_report(&self) -> Option<ResidualReport<T::Real>> {
        self.report
    }
    /// Returns the residual norm of the stopping test, or `None` before the first
    /// solve.
    pub fn residual_norm(&self) -> Option<T::Real> {
        self.report.map(|report| report.raw_norm())
    }
    /// Returns why the solve stopped. See [Termination](crate#termination).
    pub fn termination(&self) -> RootTermination {
        self.termination
    }
    /// Returns `true` if the residual passed the residual tolerance.
    pub fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the iterations and evaluations of the Newton iteration, without the
    /// nested GMRES and derivative work.
    pub fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the Newton work and the nested GMRES and derivative work, counted
    /// separately (see [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of Newton iterations, without GMRES iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the last inner solve, including an unsuccessful one.
    pub fn last_linear(&self) -> Option<NewtonKrylovObservation<T::Real>> {
        self.last_linear
    }
    /// Returns one observation per inner solve if
    /// [`NewtonKrylovOptions::with_history`] is set, and an empty slice otherwise.
    pub fn history(&self) -> &[NewtonKrylovObservation<T::Real>] {
        &self.history
    }
}
