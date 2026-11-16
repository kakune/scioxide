//! Shared Adams quadrature and PECE orchestration.
use super::polynomial::{Functional, Weights};
use super::{
    CAPACITY, MultistepOrder, choose_order, evaluate_current, finish_step, history_rows,
    integral_product, old_nodes, prepare_step, step_result,
};
use crate::OdeSystem;
use crate::result::StepResult;
use crate::stepper::StepRequest;
use crate::traits::evaluate_rhs_at_finite_state;
use crate::workspace::MultistepWorkspace;
use scioxide_core::{ErrorTolerance, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

#[derive(Clone, Copy)]
pub(crate) enum Method {
    Bashforth,
    Pece,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step<'a, S, T, System>(
    method: Method,
    name: &'static str,
    policy: MultistepOrder,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut MultistepWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    let endpoint = prepare_step(name, system, t, y, dt, workspace, policy, 0)?;
    evaluate_current(system, t, y, workspace, request, cx)?;
    // Fixed-target startup must retain its high-order cold-start accuracy.
    // RK4 (targets through four) and DP5 (five) avoid low-order startup error.
    if !policy.automatic() && workspace.history_count < policy.maximum() {
        return startup(policy.maximum(), system, t, y, dt, request, workspace, cx);
    }
    let q = workspace.active_order;
    let residence = workspace.accepted_at_order;
    // At maximum order, retain an order-wait interval between comparisons.
    // Startup keeps its original promotion cadence; current errors are always evaluated.
    let compare_orders =
        residence >= q && (q < policy.maximum() || residence.is_multiple_of(q + 1));
    let node_count = prepared_order(q, policy, workspace.history_count, compare_orders);
    let nodes = old_nodes(t, dt, workspace, node_count - 1)?;
    let mut slots = [0; CAPACITY];
    for (offset, slot) in slots.iter_mut().enumerate().take(node_count - 1) {
        *slot = workspace.history_slot(offset);
    }
    workspace.adams_coefficients.prepare(
        method,
        q,
        policy,
        workspace.history_count,
        &nodes,
        compare_orders,
    )?;
    let prepared = &workspace.adams_coefficients;
    let rows = history_rows(
        workspace.current_derivative.as_slice(),
        None,
        &workspace.history_derivatives,
        &slots,
        q,
    );
    prepared.predictor.apply(
        Some(y.as_slice()),
        dt,
        &rows[..q],
        workspace.predictor.as_mut_slice(),
    )?;
    if matches!(method, Method::Pece) {
        evaluate_rhs_at_finite_state(
            system,
            endpoint,
            &workspace.predictor,
            &mut workspace.next,
            &mut workspace.rhs_candidate,
            request,
            cx,
        )?;
        if policy.automatic() {
            // The PECE estimator uses the predicted endpoint. Using the final E
            // can hide a large correction when the corrected derivative returns
            // to its initial value (for example Euler prediction on y' = -y).
            workspace
                .rhs_sum
                .as_mut_slice()
                .copy_from_slice(workspace.next.as_slice());
        }
        let rows = history_rows(
            workspace.current_derivative.as_slice(),
            Some(workspace.next.as_slice()),
            &workspace.history_derivatives,
            &slots,
            q,
        );
        prepared.corrector.apply(
            Some(y.as_slice()),
            dt,
            &rows[..q],
            workspace.corrector.as_mut_slice(),
        )?;
        // The corrected state replaces the predictor for the final evaluation.
        core::mem::swap(&mut workspace.predictor, &mut workspace.corrector);
    }
    if matches!(method, Method::Pece) || policy.automatic() {
        // The final E in PECE is evaluated at the corrected state, not its predictor.
        evaluate_rhs_at_finite_state(
            system,
            endpoint,
            &workspace.predictor,
            &mut workspace.next,
            &mut workspace.rhs_candidate,
            request,
            cx,
        )?;
        workspace.next_derivative_valid = true;
    }
    workspace
        .proposed_state
        .as_mut_slice()
        .copy_from_slice(workspace.predictor.as_slice());
    if policy.automatic() {
        for ((scale, old), new) in workspace
            .scales
            .as_mut_slice()
            .iter_mut()
            .zip(y.as_slice())
            .zip(workspace.proposed_state.as_slice())
        {
            *scale = NumericScalar::abs(*old).max(NumericScalar::abs(*new));
        }
        let error_rows = history_rows(
            workspace.current_derivative.as_slice(),
            Some(if matches!(method, Method::Pece) {
                workspace.rhs_sum.as_slice()
            } else {
                workspace.next.as_slice()
            }),
            &workspace.history_derivatives,
            &slots,
            node_count + 1,
        );
        let default;
        let tolerance = match request.error_tolerance {
            Some(tolerance) => tolerance,
            None => {
                default = ErrorTolerance::default();
                &default
            }
        };
        let norm = tolerance.prepare_norm(
            workspace.scales.as_slice(),
            workspace.error_magnitudes.as_mut_slice(),
        )?;
        let current = prepared.errors[1].as_ref().ok_or_else(|| {
            crate::safety::invalid(
                scioxide_core::argument!("workspace"),
                "missing current error weights",
            )
        })?;
        let mut estimates = [S::ZERO; 3];
        estimates[1] = estimate(
            current,
            dt,
            &error_rows[..=q],
            workspace.errors.as_mut_slice(),
            &norm,
            compare_orders,
        )?
        .unwrap_or(S::ZERO);
        // A rejected trial must still be able to select a lower order immediately.
        let force_comparison =
            !compare_orders && q == policy.maximum() && residence >= q && estimates[1] > S::ONE;
        if force_comparison {
            workspace.adams_coefficients.prepare(
                method,
                q,
                policy,
                workspace.history_count,
                &nodes,
                true,
            )?;
        }
        if compare_orders || force_comparison {
            // Preserve the authoritative current-order error while evaluating neighbors.
            core::mem::swap(&mut workspace.errors, &mut workspace.corrector);
            let prepared = &workspace.adams_coefficients;
            for (slot, order) in [(0, q.saturating_sub(1)), (2, q + 1)] {
                if order == 0 || order > policy.maximum() || order > workspace.history_count {
                    continue;
                }
                let weights = prepared.errors[slot].as_ref().ok_or_else(|| {
                    crate::safety::invalid(
                        scioxide_core::argument!("workspace"),
                        "missing adjacent error weights",
                    )
                })?;
                estimates[slot] = estimate(
                    weights,
                    dt,
                    &error_rows[..=order],
                    workspace.errors.as_mut_slice(),
                    &norm,
                    true,
                )?
                .ok_or_else(|| {
                    crate::safety::invalid(
                        scioxide_core::argument!("workspace"),
                        "missing normalized adjacent estimate",
                    )
                })?;
            }
            core::mem::swap(&mut workspace.errors, &mut workspace.corrector);
            choose_order(policy, estimates, workspace)?;
        }
    }
    finish_step(t, y, request, workspace)?;
    let mut result = step_result(request, workspace)?;
    if policy.automatic() || matches!(method, Method::Pece) {
        result.endpoint_rhs = Some(workspace.next.as_slice());
    }
    Ok(result)
}

#[inline(never)]
fn estimate<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
    weights: &Weights<S>,
    step: S,
    rows: &[&[T]],
    output: &mut [T],
    norm: &scioxide_core::PreparedErrorNorm<'_, S>,
    normalize: bool,
) -> Result<Option<S>, SciOxideError> {
    weights.apply(None, step, rows, output)?;
    if !normalize && norm.is_satisfied_by(|i| output[i].abs())? {
        return Ok(None);
    }
    norm.normalized_norm_by(|i| output[i].abs()).map(Some)
}

fn prepared_order(q: usize, policy: MultistepOrder, history: usize, compare: bool) -> usize {
    if policy.automatic() && compare {
        q.max((q + 1).min(policy.maximum()).min(history))
    } else {
        q
    }
}

/// One actual-node geometry prepares predictor, corrector and all candidate
/// error functionals. Every component shares these immutable coefficients.
#[derive(Debug)]
pub(crate) struct Prepared<S: RealScalar> {
    predictor: Weights<S>,
    corrector: Weights<S>,
    // Lower, current, higher order: the same slots used by order selection.
    errors: [Option<Weights<S>>; 3],
}
impl<S: RealScalar> Default for Prepared<S> {
    fn default() -> Self {
        Self {
            predictor: Weights::from_native([S::ZERO; CAPACITY], 0, S::ONE),
            corrector: Weights::from_native([S::ZERO; CAPACITY], 0, S::ONE),
            errors: core::array::from_fn(|_| None),
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Prepared<S> {
    fn prepare(
        &mut self,
        method: Method,
        q: usize,
        policy: MultistepOrder,
        history: usize,
        old: &[S; CAPACITY],
        compare_orders: bool,
    ) -> Result<(), SciOxideError> {
        for (slot, order) in [q.saturating_sub(1), q, q + 1].into_iter().enumerate() {
            if !policy.automatic()
                || order == 0
                || order > policy.maximum()
                || (order != q && (!compare_orders || order > history))
            {
                self.errors[slot] = None;
            }
        }
        let count = prepared_order(q, policy, history, compare_orders);
        let ordinary = super::polynomial::ordinary_geometry(&old[..count]);
        let native = if ordinary {
            self.native::<false>(method, q, policy, history, old, compare_orders)
        } else {
            self.native::<true>(method, q, policy, history, old, compare_orders)
        };
        if native {
            Ok(())
        } else {
            self.fallback(method, q, policy, history, old, compare_orders)
        }
    }

    /// A single Newton-prefix recurrence keeps only the current divided
    /// difference and integral rows. Completed functionals are written directly;
    /// no matrix of every intermediate prefix is constructed or copied.
    #[inline(never)]
    fn native<const CHECK: bool>(
        &mut self,
        method: Method,
        q: usize,
        policy: MultistepOrder,
        history: usize,
        old: &[S; CAPACITY],
        compare_orders: bool,
    ) -> bool {
        let orders = [q.saturating_sub(1), q, q + 1];
        let admitted = |order| {
            policy.automatic()
                && order > 0
                && order <= policy.maximum()
                && (order == q || (compare_orders && order <= history))
        };
        let len = prepared_order(q, policy, history, compare_orders) + 1;
        // The capacity is fixed; explicit construction avoids a dynamic
        // short-copy call. Only the prefix selected by len participates below.
        let nodes = [
            S::ONE,
            old[0],
            old[1],
            old[2],
            old[3],
            old[4],
            old[5],
            old[6],
        ];
        // M_p = integral_0^1 x^p P(x) dx for the current old-node product P.
        // Appending root a updates M_p <- M_(p+1) + a*M_p. All terms are
        // nonnegative; M_0-M_1 loses at most log2(q+1) bits for degree <= q-1.
        let mut count = S::ONE;
        let mut moments: [S; CAPACITY] = core::array::from_fn(|_| {
            let value = S::ONE / count;
            count += S::ONE;
            value
        });
        let mut leading = [S::ZERO; CAPACITY];
        let mut integral = [S::ZERO; CAPACITY];
        let mut predictor = [S::ZERO; CAPACITY];
        let mut corrector = [S::ZERO; CAPACITY];
        let mut predictor_moment = S::ONE;
        for (k, &node) in nodes.iter().enumerate().take(len) {
            let mut old_moment = S::ONE;
            let needs_old_moment =
                k + 1 == q || (matches!(method, Method::Bashforth) && admitted(k));
            let moment = if k == 0 {
                leading[0] = S::ONE;
                S::ONE
            } else {
                let mut denominator = S::ONE;
                for (j, &previous) in nodes.iter().enumerate().take(k) {
                    let distance = previous - node;
                    if CHECK && (!distance.is_normal() || distance <= S::ZERO) {
                        return false;
                    }
                    leading[j] /= distance;
                    denominator *= -distance;
                }
                leading[k] = S::ONE / denominator;
                if CHECK
                    && (!denominator.is_normal() || leading[..=k].iter().any(|v| !v.is_normal()))
                {
                    return false;
                }
                let moment = moments[1] - moments[0];
                if CHECK && !moment.is_normal() {
                    return false;
                }
                if k + 1 < len || needs_old_moment {
                    let root = -old[k - 1];
                    if CHECK && root < S::ZERO {
                        return false;
                    }
                    for p in 0..len - k {
                        moments[p] = scioxide_core::fast_mul_add(root, moments[p], moments[p + 1]);
                        if CHECK && !moments[p].is_normal() {
                            return false;
                        }
                    }
                    if needs_old_moment {
                        old_moment = moments[0];
                    }
                }
                moment
            };
            if k + 1 == q {
                predictor_moment = old_moment;
            }
            if k < q {
                for (j, value) in integral.iter_mut().enumerate().take(k + 1) {
                    *value = scioxide_core::fast_mul_add(moment, leading[j], *value);
                    if (CHECK && !value.is_normal()) || (!CHECK && *value == S::ZERO) {
                        return false;
                    }
                }
            }
            if k + 1 == q {
                corrector = integral;
            }
            if k == q {
                // Replacing endpoint 1 by the last old node changes the integral
                // by (old_last-1)*integral(P_old_prefix). The degree-q integral
                // row and final polynomial product are therefore unnecessary.
                let replacement = (old[q - 1] - S::ONE) * predictor_moment;
                for j in 0..q {
                    predictor[j] =
                        scioxide_core::fast_mul_add(replacement, leading[j + 1], integral[j + 1]);
                }
                if predictor[..q].iter().any(|v| !v.is_normal()) {
                    return false;
                }
            }
            for (slot, &order) in orders.iter().enumerate() {
                if order != k || !admitted(order) {
                    continue;
                }
                let factor = if matches!(method, Method::Bashforth) {
                    old_moment
                } else {
                    moment
                };
                let mut weights = [S::ZERO; CAPACITY];
                for (out, &coefficient) in weights.iter_mut().zip(&leading).take(k + 1) {
                    *out = factor * coefficient;
                    if CHECK && !out.is_normal() {
                        return false;
                    }
                }
                self.errors[slot]
                    .get_or_insert_with(|| Weights::from_native([S::ZERO; CAPACITY], 0, S::ZERO))
                    .replace_native(weights, k + 1, S::ZERO);
            }
        }
        self.predictor.replace_native(predictor, q, S::ONE);
        self.corrector.replace_native(corrector, q, S::ONE);
        true
    }

    #[cold]
    fn fallback(
        &mut self,
        method: Method,
        q: usize,
        policy: MultistepOrder,
        history: usize,
        old: &[S; CAPACITY],
        compare_orders: bool,
    ) -> Result<(), SciOxideError> {
        let mut nodes = [S::ONE; CAPACITY];
        nodes[1..].copy_from_slice(&old[..CAPACITY - 1]);
        let predictor = Weights::new(&old[..q], Functional::Integral)?;
        let corrector = Weights::new(&nodes[..q], Functional::Integral)?;
        let mut errors = core::array::from_fn(|_| None);
        for (slot, order) in [q.saturating_sub(1), q, q + 1].into_iter().enumerate() {
            if !policy.automatic()
                || order == 0
                || order > policy.maximum()
                || (order != q && (!compare_orders || order > history))
            {
                continue;
            }
            let factor = integral_product(if matches!(method, Method::Bashforth) {
                &old[..order]
            } else {
                &nodes[..order]
            })?;
            errors[slot] = Some(
                Weights::new(&nodes[..=order], Functional::Coefficient(order))?.multiply(factor)?,
            );
        }
        *self = Self {
            predictor,
            corrector,
            errors,
        };
        Ok(())
    }
}

/// Borrow the current, optional endpoint, and accepted derivative rows once;
/// applying weights never recomputes history-ring positions per coordinate.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn startup<'a, S, T, System>(
    order: usize,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut MultistepWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    if order <= 4 {
        workspace
            .predictor
            .as_mut_slice()
            .copy_from_slice(y.as_slice());
        // RK4 startup builds no extension and uses no component policy of its own.
        crate::explicit_rk::perform_explicit_step(
            system,
            t,
            &mut workspace.predictor,
            dt,
            &workspace.startup_rk4,
            StepRequest {
                dense_output: false,
                error_tolerance: None,
                ..request
            },
            &mut workspace.startup,
            Some(workspace.current_derivative.as_slice()),
            cx,
        )?;
        evaluate_rhs_at_finite_state(
            system,
            t + dt,
            &workspace.predictor,
            &mut workspace.next,
            &mut workspace.rhs_candidate,
            request,
            cx,
        )?;
        workspace.next_derivative_valid = true;
        if request.requires_dense_output() {
            workspace.dense.prepare_endpoints(
                dt,
                y,
                &workspace.predictor,
                &workspace.current_derivative,
                &workspace.next,
            )?;
            workspace.dense.finish(
                crate::dense_output::DenseExtension::CubicHermite,
                t,
                dt,
                y,
                &workspace.predictor,
            )?;
        }
        workspace
            .proposed_state
            .as_mut_slice()
            .copy_from_slice(workspace.predictor.as_slice());
        workspace.push_current(t, y.as_slice());
        y.as_mut_slice()
            .copy_from_slice(workspace.proposed_state.as_slice());
        let dense = request.requires_dense_output().then_some(&workspace.dense);
        let mut result = StepResult::from_checked_parts(None, dense);

        result.endpoint_rhs = Some(workspace.next.as_slice());
        return Ok(result);
    }
    workspace
        .predictor
        .as_mut_slice()
        .copy_from_slice(y.as_slice());
    let result = crate::explicit_rk::perform_explicit_step(
        system,
        t,
        &mut workspace.predictor,
        dt,
        &workspace.startup_dp5,
        request,
        &mut workspace.startup,
        Some(workspace.current_derivative.as_slice()),
        cx,
    )?;
    if let Some(errors) = result.raw_error {
        workspace.errors.as_mut_slice().copy_from_slice(errors);
    }
    if let Some(dense) = result.dense_output {
        workspace.dense.copy_from(dense)?;
    }
    workspace
        .next
        .as_mut_slice()
        .copy_from_slice(workspace.startup.endpoint_rhs.as_slice());
    workspace.next_derivative_valid = true;
    workspace
        .proposed_state
        .as_mut_slice()
        .copy_from_slice(workspace.predictor.as_slice());
    workspace.push_current(t, y.as_slice());
    y.as_mut_slice()
        .copy_from_slice(workspace.proposed_state.as_slice());
    let dense = request.requires_dense_output().then_some(&workspace.dense);
    let mut result = StepResult::from_checked_parts(Some(workspace.errors.as_slice()), dense);
    result.error_order = Some(5);
    result.endpoint_rhs = Some(workspace.next.as_slice());

    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod geometry_tests {
    use super::*;
    use crate::stepper::testing::with_context;
    use crate::stepper::{StepTrace, Stepper};
    use scioxide_core::EvaluationRole;
    fn prepare(
        method: Method,
        q: usize,
        policy: MultistepOrder,
        history: usize,
        old: &[f64; CAPACITY],
        compare_orders: bool,
    ) -> Result<Prepared<f64>, SciOxideError> {
        let mut prepared = Prepared::default();
        prepared.prepare(method, q, policy, history, old, compare_orders)?;
        Ok(prepared)
    }

    #[test]
    fn settled_order_wait_retains_current_errors_and_compares_failed_trials() {
        struct Cosine;
        impl OdeSystem<f64> for Cosine {
            fn dimension(&self) -> usize {
                1
            }
            fn rhs(
                &self,
                time: f64,
                _: &[f64],
                out: &mut [f64],
                _: &mut SolveContext<'_, '_>,
            ) -> Result<(), SciOxideError> {
                out[0] = time.cos();
                Ok(())
            }
        }
        for (tol, compared) in [(1e-8, false), (1e-30, true)] {
            let method =
                crate::AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5)).unwrap();
            let mut workspace = MultistepWorkspace::default();
            Stepper::<f64, f64>::reserve(&method, 1, false, &mut workspace).unwrap();
            for time in [-0.06_f64, -0.05, -0.04, -0.03, -0.02, -0.01] {
                workspace.current_derivative.as_mut_slice()[0] = time.cos();
                workspace.push_current(time, &[time.sin()]);
                workspace.accept_step().unwrap();
            }
            workspace.history_method = Some((Stepper::<f64, f64>::info(&method).name, 5));
            workspace.automatic_order = true;
            workspace.history_step = Some(0.01);
            workspace.history_endpoint = Some(0.0);
            workspace.accepted_endpoint.as_mut_slice()[0] = 0.0;
            workspace.current_derivative.as_mut_slice()[0] = 1.0;
            workspace.current_derivative_valid = true;
            workspace.active_order = 5;
            workspace.accepted_at_order = 7;
            let tolerance =
                ErrorTolerance::scalar(tol, tol, scioxide_core::NormSelection::Infinity).unwrap();
            let mut state = Vector::from_vec(alloc::vec![0.0]);
            let trace = StepTrace::default();
            with_context(|cx| {
                let result = method.step(
                    &Cosine,
                    0.0,
                    &mut state,
                    0.01,
                    StepRequest::new(&trace).with_error_tolerance(&tolerance),
                    &mut workspace,
                    cx,
                )?;
                let error = tolerance.normalized_norm(
                    &[result.raw_error.unwrap()[0].abs()],
                    &[state.as_slice()[0].abs()],
                )?;
                assert_eq!(error > 1.0, compared);
                Ok(())
            })
            .unwrap();
            assert!((state.as_slice()[0] - 0.01_f64.sin()).abs() < 1e-12);
            assert_eq!(workspace.adams_coefficients.errors[0].is_some(), compared);
        }
    }

    #[test]
    fn geometry_domain_boundaries_preserve_quadratic_moments() {
        for (near, far) in [
            (-0.25_f64, -8.0_f64),
            (-0.25_f64.next_up(), -8.0),
            (-0.25, (-8.0_f64).next_down()),
        ] {
            let mut old = [0.0; CAPACITY];
            old[1] = near;
            old[2] = far;
            let prepared =
                prepare(Method::Pece, 3, MultistepOrder::Automatic(3), 2, &old, true).unwrap();
            let actual = prepared
                .predictor
                .affine(0.0, 1.0, |i| old[i] * old[i])
                .unwrap();
            assert!((actual - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn reused_geometry_tracks_order_changes_without_retaining_inactive_rows() {
        let old = [0.0_f64, -0.5, -1.5, -2.25, -3.0, -4.0, -5.0, -6.0];
        let mut prepared = Prepared::default();
        for q in [5, 2, 1, 4] {
            prepared
                .prepare(Method::Pece, q, MultistepOrder::Automatic(q), q, &old, true)
                .unwrap();
            let expected = 1.0 / q as f64;
            let predicted = prepared
                .predictor
                .affine(0.0, 1.0, |i| old[i].powi((q - 1) as i32))
                .unwrap();
            assert!((predicted - expected).abs() < 1e-10);
            assert!(prepared.errors[2].is_none());
            if q == 1 {
                assert!(prepared.errors[0].is_none());
            }
        }
    }

    #[test]
    fn moulton_retains_the_rhs_of_its_verified_endpoint() {
        struct Decay(core::cell::Cell<usize>);
        impl OdeSystem<f64> for Decay {
            fn dimension(&self) -> usize {
                1
            }
            fn rhs(
                &self,
                _: f64,
                state: &[f64],
                out: &mut [f64],
                _: &mut SolveContext<'_, '_>,
            ) -> Result<(), SciOxideError> {
                self.0.set(self.0.get() + 1);
                out[0] = -state[0];
                Ok(())
            }
        }
        let system = Decay(core::cell::Cell::new(0));
        let solver = crate::AdamsMoulton::with_order_policy(MultistepOrder::Automatic(1)).unwrap();
        let mut workspace = MultistepWorkspace::default();
        Stepper::<f64, f64>::reserve(&solver, 1, false, &mut workspace).unwrap();
        Stepper::<f64, f64>::begin(&solver, 1, false, &mut workspace).unwrap();
        let tolerance =
            ErrorTolerance::scalar(1e-8, 1e-8, scioxide_core::NormSelection::Infinity).unwrap();
        let mut state = Vector::from_vec(alloc::vec![1.0]);
        let trace = StepTrace::default();
        with_context(|cx| {
            for index in 0..2 {
                let before = system.0.get();
                let metered = cx.counts().own().provider(EvaluationRole::RightHandSide);
                let result = solver.step(
                    &system,
                    f64::from(index) * 1e-3,
                    &mut state,
                    1e-3,
                    StepRequest::new(&trace).with_error_tolerance(&tolerance),
                    &mut workspace,
                    cx,
                )?;
                assert!(
                    (result.endpoint_rhs.unwrap()[0] + state.as_slice()[0]).abs() <= f64::EPSILON
                );
                let evaluations = 2 + usize::from(index == 0);
                assert_eq!(system.0.get() - before, evaluations);
                assert_eq!(
                    cx.counts().own().provider(EvaluationRole::RightHandSide) - metered,
                    evaluations
                );
                solver.accept(&mut workspace)?;
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn one_geometry_reproduces_predictor_corrector_and_error_moments() {
        let old = [0.0_f64, -0.5, -1.5, -2.25, -3.0, -4.0, -5.0, -6.0];
        for q in 1..=5 {
            let prepared =
                prepare(Method::Pece, q, MultistepOrder::Automatic(5), q, &old, true).unwrap();
            for degree in 0..q {
                let expected = 1.0 / (degree + 1) as f64;
                let predictor = prepared
                    .predictor
                    .affine(0.0, 1.0, |i| old[i].powi(degree as i32))
                    .unwrap();
                let corrector = prepared
                    .corrector
                    .affine(0.0, 1.0, |i| {
                        if i == 0 {
                            1.0
                        } else {
                            old[i - 1].powi(degree as i32)
                        }
                    })
                    .unwrap();
                assert!((predictor - expected).abs() < 1e-10);
                assert!((corrector - expected).abs() < 1e-10);
                let error = prepared.errors[1]
                    .as_ref()
                    .unwrap()
                    .affine(0.0, 1.0, |i| {
                        if i == 0 {
                            1.0
                        } else {
                            old[i - 1].powi(degree as i32)
                        }
                    })
                    .unwrap();
                assert!(error.abs() < 1e-10);
            }
        }
        for (method, expected) in [(Method::Bashforth, 0.5), (Method::Pece, -0.5)] {
            let prepared = prepare(method, 1, MultistepOrder::Automatic(1), 0, &old, true).unwrap();
            let error = prepared.errors[1]
                .as_ref()
                .unwrap()
                .affine(0.0, 1.0, |i| if i == 0 { 1.0 } else { old[i - 1] })
                .unwrap();
            assert!((error - expected).abs() < 1e-14);
        }
    }
    #[test]
    fn exceptional_node_products_preserve_constant_history() {
        let mut old = [0.0_f64; CAPACITY];
        old[1] = -1e200;
        let prepared =
            prepare(Method::Pece, 2, MultistepOrder::Automatic(2), 1, &old, true).unwrap();
        assert_eq!(prepared.predictor.affine(0.0, 1.0, |_| 1.0).unwrap(), 1.0);
        assert_eq!(prepared.corrector.affine(0.0, 1.0, |_| 1.0).unwrap(), 1.0);
        assert_eq!(
            prepared.errors[1]
                .as_ref()
                .unwrap()
                .affine(0.0, 1.0, |_| 1.0)
                .unwrap(),
            0.0
        );
    }
}
