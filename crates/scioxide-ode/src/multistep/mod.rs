//! Linear multistep methods of orders 1 to 5 on variable step sizes.
//!
//! [`AdamsBashforth`] is explicit, [`AdamsMoulton`] is an explicit predictor with
//! an Adams–Moulton corrector (PECE), and [`Bdf`] is the implicit backward
//! differentiation formula for stiff problems.

pub(crate) mod adams_bashforth;
pub(crate) mod adams_moulton;
pub(crate) mod bdf;

pub use adams_bashforth::AdamsBashforth;
pub use adams_moulton::AdamsMoulton;
pub use bdf::{Bdf, BdfFormula};

mod adams;
pub(crate) use adams::Prepared as AdamsCoefficients;
pub(crate) mod polynomial;
use crate::result::StepResult;
use crate::safety::{checked_step_endpoint, validate_finite_state};
use crate::stepper::StepRequest;
use crate::traits::{OdeSystem, evaluate_rhs_staged};
use crate::workspace::MultistepWorkspace;
use polynomial::{CAPACITY, coordinate, integral_product};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

/// Selects how a multistep method chooses its order and step size.
///
/// Orders range from 1 to 5. How the first steps fill the history depends on the
/// method; see [`AdamsBashforth`], [`AdamsMoulton`] and [`Bdf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultistepOrder {
    /// Uses this order once the history is filled, at a constant step size.
    Fixed(usize),
    /// Starts at order 1 and adapts the order, up to this maximum, and the step
    /// size to the error tolerance.
    Automatic(usize),
}
impl MultistepOrder {
    pub(crate) const fn maximum(self) -> usize {
        match self {
            Self::Fixed(n) | Self::Automatic(n) => n,
        }
    }
    pub(crate) const fn automatic(self) -> bool {
        matches!(self, Self::Automatic(_))
    }
    /// Validates this policy, reporting errors against `argument`.
    pub(crate) fn validate(self, argument: scioxide_core::Argument) -> Result<Self, SciOxideError> {
        if !(1..=5).contains(&self.maximum()) {
            return Err(SciOxideError::invalid_input(
                argument,
                scioxide_core::InvalidInputReason::OutOfRange,
                "order must be between one and five",
            ));
        }
        Ok(self)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_step<S, T, System>(
    solver_name: &'static str,
    system: &System,
    t: S,
    y: &Vector<T>,
    dt: S,
    workspace: &mut MultistepWorkspace<S, T>,
    policy: MultistepOrder,
    formula: u8,
) -> Result<S, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    validate_finite_state(y, system.dimension(), scioxide_core::NonFiniteOrigin::Input)?;
    let endpoint = checked_step_endpoint(t, dt)?;
    workspace.try_resize(y.len(), policy.maximum() + 1)?;
    if workspace.history_pending.is_some() {
        workspace.discard_step()?;
    }
    // History continues only from the exact accepted endpoint of this epoch.
    if workspace.history_method != Some((solver_name, policy.maximum()))
        || workspace.automatic_order != policy.automatic()
        || workspace.formula_key != formula
        || workspace
            .history_step
            .is_some_and(|h| (h > S::ZERO) != (dt > S::ZERO))
        || workspace.history_endpoint != Some(t)
        || workspace.accepted_endpoint.as_slice() != y.as_slice()
    {
        workspace.reset_epoch()?;
        workspace.history_method = Some((solver_name, policy.maximum()));
        workspace.automatic_order = policy.automatic();
        workspace.formula_key = formula;
        workspace.history_endpoint = Some(t);
        workspace
            .accepted_endpoint
            .as_mut_slice()
            .copy_from_slice(y.as_slice());
    }
    // A continuous trial already matched this exact accepted point above.
    // Keep its snapshot and invariant metadata instead of copying them again.
    workspace.attempt_step = dt;
    workspace.active_order = if policy.automatic() {
        workspace.active_order
    } else {
        policy.maximum()
    }
    .min(workspace.history_count + 1)
    .min(policy.maximum());
    workspace.next_order = workspace.active_order;
    Ok(endpoint)
}

/// Load `f(t, y)` unless the accepted history already holds it or the driver
/// supplies the derivative of its accepted base.
pub(crate) fn evaluate_current<S, T, System>(
    system: &System,
    t: S,
    y: &Vector<T>,
    workspace: &mut MultistepWorkspace<S, T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    if workspace.current_derivative_valid {
        return Ok(());
    }
    if let Some(derivative) = request.initial_rhs {
        workspace
            .current_derivative
            .as_mut_slice()
            .copy_from_slice(derivative);
        workspace.current_derivative_valid = true;
        return Ok(());
    }
    evaluate_rhs_staged(
        system,
        t,
        y,
        &mut workspace.current_derivative,
        &mut workspace.rhs_candidate,
        request,
        cx,
    )?;
    workspace.current_derivative_valid = true;
    Ok(())
}

/// Normalized nodes: current state first, then the requested available history prefix.
pub(crate) fn old_nodes<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
    t: S,
    dt: S,
    workspace: &MultistepWorkspace<S, T>,
    history: usize,
) -> Result<[S; CAPACITY], SciOxideError> {
    let mut nodes = [S::ZERO; CAPACITY];
    for (i, node) in nodes
        .iter_mut()
        .enumerate()
        .skip(1)
        .take(history.min(workspace.history_count))
    {
        *node = coordinate(workspace.history_time(i - 1), t, dt)?;
    }
    Ok(nodes)
}

/// Borrow current, optional endpoint, and older samples from ring-history storage.
pub(crate) fn history_rows<'a, T>(
    current: &'a [T],
    endpoint: Option<&'a [T]>,
    history: &'a [T],
    slots: &[usize; CAPACITY],
    len: usize,
) -> [&'a [T]; CAPACITY] {
    let mut rows: [&[T]; CAPACITY] = [&[]; CAPACITY];
    let offset = usize::from(endpoint.is_some());
    rows[0] = endpoint.unwrap_or(current);
    rows[offset] = current;
    for (j, row) in rows.iter_mut().enumerate().take(len).skip(offset + 1) {
        let start = slots[j - offset - 1] * current.len();
        *row = &history[start..start + current.len()];
    }
    rows
}

/// State-history interpolation provides degree-q dense values and derivatives.
pub(crate) fn finish_step<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
    t: S,
    y: &mut Vector<T>,
    request: StepRequest<'_, S, T>,
    workspace: &mut MultistepWorkspace<S, T>,
) -> Result<(), SciOxideError> {
    let q = workspace.active_order;
    crate::safety::finite(
        workspace.proposed_state.as_slice(),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    crate::safety::finite(
        workspace.errors.as_slice(),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    if request.requires_dense_output() {
        let old = old_nodes(t, workspace.attempt_step, workspace, q.saturating_sub(1))?;
        let mut nodes = [S::ONE; CAPACITY];
        nodes[1..=q].copy_from_slice(&old[..q]);
        let mut weights: [polynomial::Weights<S>; 7] = core::array::from_fn(|_| {
            polynomial::Weights::from_native([S::ZERO; CAPACITY], 0, S::ZERO)
        });
        for (degree, row) in weights[..q].iter_mut().enumerate() {
            *row = polynomial::Weights::new(
                &nodes[..=q],
                polynomial::Functional::Coefficient(degree + 1),
            )?;
        }
        let slots: [usize; 7] = core::array::from_fn(|j| {
            if j < q.saturating_sub(1) {
                workspace.history_slot(j)
            } else {
                0
            }
        });
        let dimension = y.len();
        let history_states = &workspace.history_states;
        let proposed = &workspace.proposed_state;
        let sample = |i, j| {
            if j == 0 {
                proposed.as_slice()[i]
            } else if j == 1 {
                y.as_slice()[i]
            } else {
                history_states[slots[j - 2] * dimension + i]
            }
        };
        workspace.dense.prepare_coefficients(
            crate::dense_output::DenseExtension::Multistep { order: q },
            workspace.attempt_step,
            y,
            |i, imaginary| {
                let mut values = [S::ZERO; 7];
                for (out, row) in values[..q].iter_mut().zip(&weights) {
                    *out = row.native_component(imaginary, |j| sample(i, j))?;
                }
                Some(values)
            },
            |i, imaginary| {
                let mut values = [scioxide_core::Scaled::ZERO; 7];
                for (out, row) in values[..q].iter_mut().zip(&weights) {
                    *out = row.component_scaled(imaginary, &|j| sample(i, j))?;
                }
                Ok(values)
            },
        )?;
        workspace.dense.finish(
            crate::dense_output::DenseExtension::Multistep { order: q },
            t,
            workspace.attempt_step,
            y,
            &workspace.proposed_state,
        )?;
    }
    workspace.push_current(t, y.as_slice());
    y.as_mut_slice()
        .copy_from_slice(workspace.proposed_state.as_slice());
    Ok(())
}

pub(crate) fn step_result<
    'a,
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
>(
    request: StepRequest<'_, S, T>,
    workspace: &'a MultistepWorkspace<S, T>,
) -> Result<StepResult<'a, S, T>, SciOxideError> {
    let dense = request.requires_dense_output().then_some(&workspace.dense);
    let mut result = StepResult::from_checked_parts(
        workspace
            .automatic_order
            .then_some(workspace.errors.as_slice()),
        dense,
    );

    result.error_order = workspace
        .automatic_order
        .then_some(workspace.active_order + 1);
    result.maximum_step_growth = Some(S::TWO);
    Ok(result)
}

/// Compare adjacent-order local-error estimates on the same trial. Keep the
/// current order on ties; limit changes to one and require another history point.
/// Keep an order for q+1 accepted steps before comparing adjacent estimates,
/// so startup differences cannot immediately undo a successful order change.
pub(crate) fn choose_order<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
    policy: MultistepOrder,
    estimates: [S; 3],
    workspace: &mut MultistepWorkspace<S, T>,
) -> Result<(), SciOxideError> {
    if policy.automatic() {
        workspace.next_order = select_order(
            workspace.active_order,
            policy.maximum(),
            workspace.history_count,
            workspace.accepted_at_order,
            estimates,
        )?;
    }
    Ok(())
}

/// Adjacent-order comparison shared by residual and normalized BDF equations.
/// Keep ties, require enough history and accepted steps at the current order, and
/// never raise the order after a failure.
pub(crate) fn select_order<S: RealScalar>(
    q: usize,
    maximum: usize,
    history: usize,
    residence: usize,
    estimates: [S; 3],
) -> Result<usize, SciOxideError> {
    if residence < q {
        return Ok(q);
    }
    let mut chosen = q;
    let mut best_error = estimates[1];
    for (slot, order) in [(0, q.saturating_sub(1)), (2, q + 1)] {
        if order == 0 || order > maximum || order > history {
            continue;
        }
        // Compare predicted step gains without fractional powers. Raising
        // both positive scores to (order+1)*(chosen+1) preserves ordering.
        if order_advantage(estimates[slot], order, best_error, chosen)? {
            best_error = estimates[slot];
            chosen = order;
        }
    }
    Ok(if estimates[1] > S::ONE {
        chosen.min(q)
    } else {
        chosen
    })
}

fn order_advantage<S: RealScalar>(
    candidate: S,
    order: usize,
    current: S,
    current_order: usize,
) -> Result<bool, SciOxideError> {
    let candidate = candidate.max(S::EPSILON);
    let current = current.max(S::EPSILON);
    let margin = S::try_from_f64_lossy(1.2)?;
    let left = positive_power(candidate, current_order + 1)
        * positive_power(margin, (order + 1) * (current_order + 1));
    let right = positive_power(current, order + 1);
    if left.is_normal() && right.is_normal() {
        return Ok(right > left);
    }
    // Extreme estimates can overflow/underflow integer powers even though
    // their fractional-power scores remain representable.
    let score = |error: S, degree| -> Result<S, SciOxideError> {
        Ok(error.powf(-S::ONE / S::try_from_usize_exact(degree + 1)?))
    };
    Ok(score(candidate, order)? > margin * score(current, current_order)?)
}

fn positive_power<S: RealScalar>(mut base: S, mut exponent: usize) -> S {
    let mut value = S::ONE;
    while exponent != 0 {
        if exponent & 1 != 0 {
            value *= base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base * base;
        }
    }
    value
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_precision_loss)]
mod order_evidence {
    use super::*;

    use crate::stepper::testing::with_context;
    use crate::stepper::{StepTrace, Stepper};
    use alloc::vec::Vec;

    #[test]
    fn unused_distant_history_does_not_overflow_node_preparation() {
        let mut workspace = MultistepWorkspace::<f64>::new();
        workspace.try_resize(1, 3).unwrap();
        for time in [-1e308, -1e-300] {
            workspace.push_current(time, &[1.0]);
            workspace.accept_step().unwrap();
        }
        let nodes = old_nodes(0.0, 1e-300, &workspace, 1).unwrap();
        assert!((nodes[1] + 1.0).abs() <= f64::EPSILON);
        assert!(old_nodes(0.0, 1e-300, &workspace, 2).is_err());
    }

    #[test]
    fn integer_order_scores_preserve_fractional_power_comparisons() {
        fn check<S: RealScalar>() {
            let values = [0.0, 1e-30, 1e-6, 0.1, 0.8, 1.0, 10.0, 1e30];
            let margin = S::try_from_f64_lossy(1.2).unwrap();
            for order in 1..=5 {
                for current_order in 1..=5 {
                    for candidate in values {
                        for current in values {
                            let candidate =
                                S::try_from_f64_lossy(candidate).unwrap().max(S::EPSILON);
                            let current = S::try_from_f64_lossy(current).unwrap().max(S::EPSILON);
                            let candidate_score = candidate
                                .powf(-S::ONE / S::try_from_usize_exact(order + 1).unwrap());
                            let current_score = current.powf(
                                -S::ONE / S::try_from_usize_exact(current_order + 1).unwrap(),
                            );
                            assert_eq!(
                                order_advantage(candidate, order, current, current_order).unwrap(),
                                candidate_score > margin * current_score
                            );
                        }
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    struct Decay;
    impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Decay {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: S,
            y: &[S],
            out: &mut [S],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out[0] = -y[0];
            Ok(())
        }
        fn jacobian(
            &self,
            _: S,
            _: &[S],
            out: &mut scioxide_linalg::MatrixViewMut<'_, S>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            let out_data = out.as_contiguous_slice_mut().ok_or_else(|| {
                scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("out"),
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "row-major output required",
                )
            })?;
            out_data[0] = -<S as scioxide_core::AlgebraicScalar>::ONE;
            Ok(Some(()))
        }
        fn jacobian_is_constant(&self) -> bool {
            true
        }
    }

    /// A multistep workspace sized for `solver` on a scalar state.
    fn workspace<S, Solver>(solver: &Solver) -> MultistepWorkspace<S>
    where
        S: RealScalar + NumericScalar<Real = S>,
        Solver: Stepper<S, S, Workspace = MultistepWorkspace<S>>,
    {
        let mut workspace = MultistepWorkspace::default();
        solver.reserve(1, false, &mut workspace).unwrap();
        workspace
    }

    fn seeded_error<S, Solver>(solver: Solver, steps: usize, alternating: bool, formula: u8) -> f64
    where
        S: RealScalar + NumericScalar<Real = S>,
        Solver: Stepper<S, S, Workspace = MultistepWorkspace<S>>,
    {
        let f = |x| S::try_from_f64_lossy(x).unwrap();
        let h = f(1.0 / steps as f64);
        let mut workspace = workspace(&solver);
        let capacity = workspace.history_capacity();
        // Exact history is test-only setup for coefficient-order evidence, not a
        // production continuation path. Cold-start global order is tested separately.
        for index in (1..=capacity).rev() {
            let t = -h * f(index as f64);
            let value = RealScalar::exp(-t);
            workspace.current_derivative.as_mut_slice()[0] = -value;
            workspace.push_current(t, &[value]);
            workspace.accept_step().unwrap();
        }
        let info = solver.info();
        workspace.history_method = Some((info.name, info.order));
        workspace.formula_key = formula;
        workspace.history_step = Some(h);
        workspace.history_endpoint = Some(f(0.0));
        workspace.accepted_endpoint.as_mut_slice()[0] = f(1.0);
        let mut y = Vector::from_vec(vec![f(1.0)]);
        let mut time = f(0.0);
        let trace = StepTrace::default();
        let request = StepRequest {
            inner_limit: scioxide_core::Budget::try_from(100).unwrap(),
            ..StepRequest::new(&trace)
        };
        with_context(|cx| {
            for index in 0..steps {
                let width = if index + 1 == steps {
                    f(1.0) - time
                } else if alternating {
                    h * f(if index % 2 == 0 { 0.5 } else { 1.5 })
                } else {
                    h
                };
                solver.step(&Decay, time, &mut y, width, request, &mut workspace, cx)?;
                solver.accept(&mut workspace)?;
                time += width;
            }
            Ok(())
        })
        .unwrap();
        NumericScalar::abs(y.as_slice()[0] - RealScalar::exp(-f(1.0)))
            .try_to_f64_exact()
            .unwrap()
    }

    fn check<S, Solver>(factory: impl Fn() -> Solver, order: usize, alternating: bool, formula: u8)
    where
        S: RealScalar + NumericScalar<Real = S>,
        Solver: Stepper<S, S, Workspace = MultistepWorkspace<S>>,
    {
        let floor = 8.0 * S::EPSILON.try_to_f64_exact().unwrap();
        let steps: &[usize] = if S::EPSILON.try_to_f64_exact().unwrap() > 1e-10 {
            &[2, 4, 8, 16, 32]
        } else {
            &[3, 6, 12, 24, 48, 96]
        };
        let errors: Vec<_> = steps
            .iter()
            .copied()
            .map(|steps| seeded_error::<S, _>(factory(), steps, alternating, formula))
            .collect();
        let best = errors
            .windows(2)
            .filter(|pair| pair[0] > pair[1] && pair[1] > floor)
            .map(|pair| (pair[0] / pair[1]).log2())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            best >= order as f64 - 0.5,
            "order={order}, alternating={alternating}, formula={formula}, observed={best}, errors={errors:?}"
        );
    }

    fn all_orders<S: RealScalar + NumericScalar<Real = S>>(maximum: usize) {
        for alternating in [false, true] {
            for order in 1..=maximum {
                check::<S, _>(
                    || AdamsBashforth::new(order).unwrap(),
                    order,
                    alternating,
                    0,
                );
                check::<S, _>(|| AdamsMoulton::new(order).unwrap(), order, alternating, 0);
                check::<S, _>(|| Bdf::new(order).unwrap(), order, alternating, 0);
                check::<S, _>(
                    || Bdf::new(order).unwrap().with_formula(BdfFormula::Ndf),
                    order,
                    alternating,
                    1,
                );
            }
        }
    }
    #[test]
    fn f64_actual_history_retains_all_declared_orders() {
        all_orders::<f64>(5);
    }
    #[test]
    fn f32_actual_history_retains_all_declared_orders() {
        all_orders::<f32>(5);
    }
    fn dense_error(
        solver: impl Stepper<f64, f64, Workspace = MultistepWorkspace<f64>>,
        h: f64,
        formula: u8,
    ) -> (f64, f64) {
        let mut workspace = workspace(&solver);
        for index in (1..=workspace.history_capacity()).rev() {
            let time = -h * (index as f64 + if index % 2 == 0 { 0.25 } else { 0.0 });
            workspace.current_derivative.as_mut_slice()[0] = -(-time).exp();
            workspace.push_current(time, &[(-time).exp()]);
            workspace.accept_step().unwrap();
        }
        let info = solver.info();
        workspace.history_method = Some((info.name, info.order));
        workspace.formula_key = formula;
        workspace.history_step = Some(h);
        workspace.history_endpoint = Some(0.0);
        workspace.accepted_endpoint.as_mut_slice()[0] = 1.0;
        let mut state = Vector::from_vec(vec![1.0]);
        let policy = scioxide_core::ErrorTolerance::scalar(
            1e-13,
            1e-13,
            scioxide_core::NormSelection::Infinity,
        )
        .unwrap();
        let trace = StepTrace::default();
        let request = StepRequest {
            inner_limit: scioxide_core::Budget::try_from(100).unwrap(),
            ..StepRequest::new(&trace)
                .with_dense_output()
                .with_error_tolerance(&policy)
        };
        let mut value = [0.0];
        let mut slope = [0.0];
        with_context(|cx| {
            let result = solver.step(&Decay, 0.0, &mut state, h, request, &mut workspace, cx)?;
            let dense = result.dense_output.unwrap();
            dense.eval_into(0.37 * h, &mut value)?;
            dense.derivative_into(0.37 * h, &mut slope)
        })
        .unwrap();
        (
            (value[0] - (-0.37 * h).exp()).abs(),
            (slope[0] + (-0.37 * h).exp()).abs(),
        )
    }
    fn check_dense<Solver: Stepper<f64, f64, Workspace = MultistepWorkspace<f64>>>(
        factory: impl Fn() -> Solver,
        order: usize,
        formula: u8,
    ) {
        let samples: Vec<_> = [0.4, 0.2, 0.1, 0.05]
            .into_iter()
            .map(|h| dense_error(factory(), h, formula))
            .collect();
        for derivative in [false, true] {
            let errors: Vec<_> = samples
                .iter()
                .map(|pair| if derivative { pair.1 } else { pair.0 })
                .collect();
            let observed = errors
                .windows(2)
                .filter(|p| p[1] > 1e-13)
                .map(|p| (p[0] / p[1]).log2())
                .fold(f64::NEG_INFINITY, f64::max);
            let required = if derivative {
                order as f64
            } else {
                (order + 1) as f64
            };
            assert!(
                observed >= required - 0.5,
                "dense order={order}, derivative={derivative}, observed={observed}, errors={errors:?}"
            );
        }
    }
    #[test]
    fn nonuniform_dense_values_and_derivatives_have_declared_orders() {
        for order in 1..=5 {
            check_dense(|| AdamsBashforth::new(order).unwrap(), order, 0);
            check_dense(|| AdamsMoulton::new(order).unwrap(), order, 0);
            check_dense(|| Bdf::new(order).unwrap(), order, 0);
            check_dense(
                || Bdf::new(order).unwrap().with_formula(BdfFormula::Ndf),
                order,
                1,
            );
        }
    }
}
