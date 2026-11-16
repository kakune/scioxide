//! The iteration loop of the scalar methods that keep the `N` most recent
//! points: secant, Newton, Halley and Muller, open or safeguarded.
//!
//! The loop counts and checks evaluations, tracks the best point, applies the
//! residual and step tests, detects stagnation, handles limits and builds the
//! result. A method only proposes the next point, or stops with a reason.

use scioxide_core::{
    AlgebraicScalar, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

use super::correction::small_step;
use super::{BracketState, Root, Run, Sample, ScalarRootOptions};
use crate::result::RootTermination;

/// The outcome of one method step.
pub(crate) enum Step<T> {
    /// Evaluate this point next.
    Candidate(T),
    /// End the solve with this unsuccessful reason.
    Stop(RootTermination),
}

impl<T> Step<T> {
    /// A candidate, or stagnation when the correction does not exist.
    pub(crate) fn proposed(candidate: Option<T>) -> Self {
        candidate.map_or(Self::Stop(RootTermination::Stagnation), Self::Candidate)
    }
}

/// One open method over its `N` latest evaluated points, oldest first.
pub(crate) trait OpenMethod<T: NumericScalar, const N: usize> {
    /// Proposes the next point once the iteration has been counted. Derivative
    /// and difference-quotient evaluations go through `run`, so they are
    /// counted.
    fn propose(
        &mut self,
        run: &mut Run<'_, '_, '_, T>,
        points: &[Sample<T>; N],
    ) -> Result<Step<T>, SciOxideError>;

    /// Tests each new `sample` with residual magnitude `residual`, reached by
    /// a step that met the step tolerance when `small`, and returns the reason
    /// and the point to return. Open methods use the residual and step test of
    /// [`ScalarRootOptions`].
    fn assess(
        &mut self,
        sample: Sample<T>,
        residual: T::Real,
        small: bool,
        scale: T::Real,
        options: &ScalarRootOptions<T::Real>,
    ) -> Result<Option<(RootTermination, Sample<T>)>, SciOxideError> {
        Ok(options
            .residual_termination(residual, scale, small, false)?
            .map(|reason| (reason, sample)))
    }

    /// Sign bracket of the current state. Safeguarded methods skip small-step
    /// stagnation: their bracket decides termination.
    fn certificate(&self) -> Option<BracketState<T::Real>> {
        None
    }

    /// Point returned on stagnation or an iteration limit, given the best
    /// observation.
    fn fallback(&self, best: Sample<T>) -> Sample<T> {
        best
    }

    /// An observation of `x` made before the loop, reused for initial points.
    fn known(&self, x: T) -> Option<Sample<T>> {
        let _ = x;
        None
    }

    /// Residual reference fixed before the loop, overriding the initial samples.
    fn scale(&self) -> Option<T::Real> {
        None
    }
}

/// A method given by its proposal alone.
pub(crate) struct Proposal<F>(F);

/// Wrap a proposal closure as an open method.
pub(crate) fn proposal<T, const N: usize, F>(propose: F) -> Proposal<F>
where
    T: NumericScalar,
    F: FnMut(&mut Run<'_, '_, '_, T>, &[Sample<T>; N]) -> Result<Step<T>, SciOxideError>,
{
    Proposal(propose)
}

impl<T, const N: usize, F> OpenMethod<T, N> for Proposal<F>
where
    T: NumericScalar,
    F: FnMut(&mut Run<'_, '_, '_, T>, &[Sample<T>; N]) -> Result<Step<T>, SciOxideError>,
{
    fn propose(
        &mut self,
        run: &mut Run<'_, '_, '_, T>,
        points: &[Sample<T>; N],
    ) -> Result<Step<T>, SciOxideError> {
        (self.0)(run, points)
    }
}

/// Solve with `method` from `init` in `context`, recovering a budget,
/// cancellation or deadline stop as the last accepted state.
///
/// # Errors
/// Returns option, input, callback and numerical errors, and stops before any
/// checked observation.
#[inline(always)]
pub(crate) fn solve<T, M, R, const N: usize>(
    context: &mut SolveContext<'_, '_>,
    options: &ScalarRootOptions<T::Real>,
    init: [T; N],
    callback: impl Fn(T, &mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    method: M,
) -> Result<R, SciOxideError>
where
    T: NumericScalar,
    M: OpenMethod<T, N>,
    R: Root<T>,
{
    let mut run = Run::new(context, options)?;
    let outcome = iterate(&mut run, init, options, &callback, method);
    run.finish(outcome)
}

/// The loop itself, for methods that evaluate a prelude through `run` first.
///
/// Starting points are evaluated in order, reusing equal or known points. An
/// exact zero is accepted at once; the other points are tested once the
/// residual scale is known. Every iteration accepts the newest point, counts
/// itself and the proposal, and evaluates the candidate unless it repeats a
/// stored point. Each new observation is tested immediately. A repeated
/// candidate, or a small step without residual decrease in an open method,
/// stagnates. On stagnation or the iteration limit, an observation that is
/// exactly zero is returned as a success, and otherwise
/// [`OpenMethod::fallback`] of the best observation. Other stops return the
/// newest point.
///
/// # Errors
/// Rejects non-finite starting points before any call; propagates callback,
/// limit and numerical errors.
#[inline(always)]
pub(crate) fn iterate<T, M, R, const N: usize>(
    run: &mut Run<'_, '_, '_, T>,
    init: [T; N],
    options: &ScalarRootOptions<T::Real>,
    callback: &impl Fn(T, &mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    mut method: M,
) -> Result<R, SciOxideError>
where
    T: NumericScalar,
    M: OpenMethod<T, N>,
    R: Root<T>,
{
    if init.iter().any(|x| !x.has_finite_components()) {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "initial points must be finite",
        ));
    }
    let zero = <T::Real as AlgebraicScalar>::ZERO;
    let mut points = [Sample {
        x: T::ZERO,
        value: T::ZERO,
    }; N];
    // Residual magnitudes of `points`, each computed once.
    let mut residuals = [zero; N];
    let fixed = options.residual_scale.or_else(|| method.scale());
    // A stop while later points are evaluated reports against the samples so far.
    run.scale = fixed.unwrap_or(zero);
    for index in 0..N {
        let x = init[index];
        let known = (0..index)
            .find(|&earlier| points[earlier].x == x)
            .map(|earlier| (points[earlier], residuals[earlier]))
            .or_else(|| method.known(x).map(|point| (point, point.residual())));
        let (point, residual) = match known {
            Some(observation) => observation,
            None => run.sample(x, callback)?,
        };
        (points[index], residuals[index]) = (point, residual);
        if fixed.is_none() {
            run.scale = run.scale.max(residual);
        }
        // An exact zero needs no reference: accept it before evaluating more.
        if residual == zero
            && let Some((reason, at)) = method.assess(point, residual, false, run.scale, options)?
        {
            return run.result(at, reason, method.certificate());
        }
    }
    let scale = run.scale;
    for (point, residual) in points.into_iter().zip(residuals) {
        if let Some((reason, at)) = method.assess(point, residual, false, scale, options)? {
            return run.result(at, reason, method.certificate());
        }
    }
    for _ in 0..options.max_iterations.get() {
        let current = points[N - 1];
        run.accept(current, method.certificate())?;
        run.begin_iteration()?;
        let x = match method.propose(run, &points)? {
            Step::Candidate(x) => x,
            Step::Stop(reason) => return publish(run, &method, reason, current),
        };
        let small = small_step(&options.step_tolerance.tolerance(), current.x, x)?;
        let repeated = (0..N).find(|&index| points[index].x == x);
        let (sample, residual) = match repeated {
            Some(index) => (points[index], residuals[index]),
            None => run.sample(x, callback)?,
        };
        if let Some((reason, at)) = method.assess(sample, residual, small, scale, options)? {
            return run.result(at, reason, method.certificate());
        }
        if repeated.is_some()
            || (small && method.certificate().is_none() && residual >= residuals[N - 1])
        {
            return publish(run, &method, RootTermination::Stagnation, current);
        }
        points.rotate_left(1);
        points[N - 1] = sample;
        residuals.rotate_left(1);
        residuals[N - 1] = residual;
    }
    publish(run, &method, RootTermination::IterationLimit, points[N - 1])
}

/// Returns the result for an unsuccessful `reason` reached at `current`.
fn publish<T, M, R, const N: usize>(
    run: &mut Run<'_, '_, '_, T>,
    method: &M,
    reason: RootTermination,
    current: Sample<T>,
) -> Result<R, SciOxideError>
where
    T: NumericScalar,
    M: OpenMethod<T, N>,
    R: Root<T>,
{
    let point = match reason {
        RootTermination::Stagnation | RootTermination::IterationLimit => {
            let best = run.best_sample(current);
            // Any observation, including a derivative probe, that is an exact
            // zero is a root.
            if best.residual() == <T::Real as AlgebraicScalar>::ZERO {
                return run.result(
                    best,
                    RootTermination::ResidualTolerance,
                    method.certificate(),
                );
            }
            method.fallback(best)
        }
        _ => current,
    };
    run.result(point, reason, method.certificate())
}
