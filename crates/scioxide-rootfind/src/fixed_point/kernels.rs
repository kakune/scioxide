//! Kernels for fixed-point solves.
use super::*;

/// Apply the stored inverse-Jacobian history.
pub(crate) fn apply_inverse<T: NumericScalar>(
    first: &[T],
    second: &[T],
    used: usize,
    input: &[T],
    output: &mut [T],
    alpha: T::Real,
    adjoint: bool,
) -> Result<(), SciOxideError> {
    let n = input.len();
    for (out, &value) in output.iter_mut().zip(input) {
        *out = affine(T::ZERO, T::from_real(-alpha), value)?;
    }
    for k in 0..used {
        let (u, v) = (&first[k * n..(k + 1) * n], &second[k * n..(k + 1) * n]);
        let (u, v) = if adjoint { (v, u) } else { (u, v) };
        let coefficient = dot(v, input)?;
        if let Some(coefficient) = crate::nonlinear_math::native_coefficient(coefficient) {
            for (out, &value) in output.iter_mut().zip(u) {
                *out = affine(*out, coefficient, value)?;
            }
        } else {
            for (out, &value) in output.iter_mut().zip(u) {
                *out = affine_scaled(*out, coefficient, value)?;
            }
        }
    }
    Ok(())
}

/// Balanced factors for H + (s-Hy) v^H/(v^H y), guarding the secant denominator.
/// Scaling both factors avoids overflow in an otherwise representable update.
pub(crate) fn broyden_correction<T: NumericScalar>(
    step: &[T],
    product: &[T],
    row: &mut [T],
    delta: &[T],
    correction: &mut [T],
) -> Result<bool, SciOxideError> {
    let denominator = dot(row, delta)?;
    let magnitude = denominator.magnitude()?;
    let threshold = dot(row, row)?
        .real()
        .sqrt()?
        .mul(dot(delta, delta)?.real().sqrt()?)?
        .mul(Scaled::from_value(T::Real::SQRT_EPSILON)?)?;
    if magnitude.abs_le(threshold) {
        return Ok(false);
    }
    if let Some(changed) = balanced_native(step, product, row, correction, denominator)? {
        return Ok(changed);
    }
    let mut correction_scale = Scaled::ZERO;
    let mut row_scale = Scaled::ZERO;
    for ((&s, &hy), &v) in step.iter().zip(product).zip(row.iter()) {
        let c = scaled(s)?.sub(scaled(hy)?)?.magnitude()?;
        if c.abs_cmp(correction_scale).is_gt() {
            correction_scale = c;
        }
        let r = scaled(v)?.magnitude()?;
        if r.abs_cmp(row_scale).is_gt() {
            row_scale = r;
        }
    }
    if correction_scale.is_zero() {
        return Ok(false);
    }
    let amplitude = correction_scale.mul(row_scale)?.div(magnitude)?.sqrt()?;
    let real = |value| ScaledValue::from_components(value, Scaled::ZERO);
    let phase = denominator.div(real(magnitude))?;
    for (((out, &s), &hy), v) in correction
        .iter_mut()
        .zip(step)
        .zip(product)
        .zip(row.iter_mut())
    {
        *out = rounded(
            scaled(s)?
                .sub(scaled(hy)?)?
                .div(real(correction_scale))?
                .mul(real(amplitude))?
                .div(phase)?,
        )?;
        *v = rounded(scaled(*v)?.div(real(row_scale))?.mul(real(amplitude))?)?;
    }
    Ok(true)
}

/// Normalize native factors before forming their amplitude. Only the scalar
/// amplitude needs extended range; every coordinate uses ordinary arithmetic.
fn balanced_native<T: NumericScalar>(
    step: &[T],
    product: &[T],
    row: &mut [T],
    correction: &mut [T],
    denominator: ScaledValue<T>,
) -> Result<Option<bool>, SciOxideError> {
    let mut left = T::Real::ZERO;
    let mut right = T::Real::ZERO;
    for ((&s, &p), &v) in step.iter().zip(product).zip(row.iter()) {
        let c = s - p;
        let (a, b) = (c.abs(), v.abs());
        if !c.has_finite_components() || !a.is_finite() || !b.is_finite() {
            return Ok(None);
        }
        left = left.max(a);
        right = right.max(b);
    }
    if left == T::Real::ZERO {
        return Ok(Some(false));
    }
    let magnitude = denominator.magnitude()?;
    let amplitude = Scaled::from_value(left)?
        .mul(Scaled::from_value(right)?)?
        .div(magnitude)?
        .sqrt()?;
    let Ok(amplitude) = amplitude.try_value() else {
        return Ok(None);
    };
    if !amplitude.is_normal() {
        return Ok(None);
    }
    let phase = denominator
        .div(ScaledValue::from_components(magnitude, Scaled::ZERO))?
        .round()?;
    let factor = |s: T, p: T| ((s - p) / T::from_real(left)) * T::from_real(amplitude) / phase;
    let row_factor = |v: T| (v / T::from_real(right)) * T::from_real(amplitude);
    // A failed native attempt must leave the row available to the range replay.
    if !step
        .iter()
        .zip(product)
        .zip(row.iter())
        .all(|((&s, &p), &v)| {
            let c = s - p;
            let ratios_preserve_range = [
                (c.real_component(), left),
                (c.imaginary_component(), left),
                (v.real_component(), right),
                (v.imaginary_component(), right),
            ]
            .into_iter()
            .all(|(part, scale)| part == T::Real::ZERO || (part / scale).is_normal());
            ratios_preserve_range
                && factor(s, p).has_finite_components()
                && row_factor(v).has_finite_components()
        })
    {
        return Ok(None);
    }
    for (((out, &s), &p), v) in correction.iter_mut().zip(step).zip(product).zip(row) {
        *out = factor(s, p);
        *v = row_factor(*v);
    }
    Ok(Some(true))
}

/// Outcome of one evaluated fixed-point trial.
enum Advance<R> {
    /// The trial was accepted and the residual norm is its norm.
    Accepted(R),
    /// The solve terminates, after accepting the trial where applicable.
    Done(RootTermination),
}

impl<T: NumericScalar> FixedPointSolver<T> {
    /// Run one fixed-point solve on storage that the input checks have validated.
    ///
    /// A budget or execution stop after an accepted point writes that point with
    /// the stop's termination; other failures return the error.
    pub(crate) fn kernel<F>(
        &self,
        problem: &F,
        initial: &[T],
        options: &FixedPointOptions<T::Real>,
        result: &mut FixedPointResult<T>,
        w: &mut FixedPointWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: ResidualOperator<T> + ?Sized,
    {
        w.used = 0;
        w.diagnostics = FixedPointDiagnostics::default();
        w.x.as_mut_slice().copy_from_slice(initial);
        w.diagonal
            .as_mut_slice()
            .fill(T::from_real(options.damping));
        let session = crate::context::Session::new(context).with_policy(options.policy);
        let mut scale = options.scale.unwrap_or(T::Real::ONE);
        let outcome = self.iterate(problem, options, w, &session, &mut scale);
        w.accepted = session.accepted.get();
        let termination = match (outcome, session.stopped.get(), session.accepted.get()) {
            (Ok(reason), _, _) => reason,
            (Err(_), Some(reason), true) => reason,
            (Err(error), _, _) => return Err(error),
        };
        if session.stopped.get().is_none() {
            let _ = session.check_stop()?;
        }
        let termination = session.stopped.get().unwrap_or(termination);
        let raw = norm(w.f.as_slice())?;
        // The report pairs the stopping norm with its reference, in one unit.
        let report = crate::context::residual(residual_norm(w, options)?, scale)?;
        let counts = session.accounting()?;
        result.point.as_mut_slice().copy_from_slice(w.x.as_slice());
        result
            .residual
            .as_mut_slice()
            .copy_from_slice(w.f.as_slice());
        result.report = Some(report);
        result.raw_norm = raw;
        result.termination = termination;
        result.counts = counts;
        result.diagnostics = w.diagnostics;
        Ok(())
    }

    /// Evaluate the initial point, fix the residual reference `scale`, and iterate.
    fn iterate<F>(
        &self,
        problem: &F,
        options: &FixedPointOptions<T::Real>,
        w: &mut FixedPointWorkspace<T>,
        session: &crate::context::Session<'_, '_, '_>,
        scale: &mut T::Real,
    ) -> Result<RootTermination, SciOxideError>
    where
        F: ResidualOperator<T> + ?Sized,
    {
        evaluate(problem, self.input, &w.x, &mut w.f, &mut w.scratch, session)?;
        let mut current = residual_norm(w, options)?;
        *scale = options.scale.unwrap_or(current);
        session.assess()?;
        let tolerance = options.residual.tolerance();
        for _ in 0..options.iterations.get() {
            if crate::termination::residual_ok(&tolerance, current, *scale)? {
                return Ok(RootTermination::ResidualTolerance);
            }
            session.iteration()?;
            propose(w, options, session)?;
            if self.method == FixedPointMethod::Steffensen {
                self.steffensen(problem, options, w, session)?;
            }
            match self.advance(problem, options, w, session, current, *scale)? {
                Advance::Accepted(next) => current = next,
                Advance::Done(termination) => return Ok(termination),
            }
        }
        Ok(
            if crate::termination::residual_ok(&tolerance, current, *scale)? {
                RootTermination::ResidualTolerance
            } else {
                RootTermination::IterationLimit
            },
        )
    }

    /// Replace the proposed damped map step by Steffensen's coordinatewise
    /// Aitken extrapolation `x − (βf)²/(βf(x + βf) − βf)`; near-zero
    /// denominators keep the proposal and count a skipped update.
    fn steffensen<F>(
        &self,
        problem: &F,
        options: &FixedPointOptions<T::Real>,
        w: &mut FixedPointWorkspace<T>,
        session: &crate::context::Session<'_, '_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: ResidualOperator<T> + ?Sized,
    {
        evaluate(
            problem,
            self.input,
            &w.trial,
            &mut w.trial_f,
            &mut w.scratch,
            session,
        )?;
        for i in 0..w.dimension {
            let (x, f, next) = (
                w.x.as_slice()[i],
                w.f.as_slice()[i],
                w.trial_f.as_slice()[i],
            );
            match steffensen_native(x, options.damping, f, next)
                .map_or_else(|| steffensen_range(x, options.damping, f, next), Ok)?
            {
                Some(value) => w.trial.as_mut_slice()[i] = value,
                None => w.diagnostics.skipped_updates += 1,
            }
        }
        Ok(())
    }

    /// Evaluate the proposed trial, apply the termination tests, and otherwise
    /// update the method's history and accept the trial.
    fn advance<F>(
        &self,
        problem: &F,
        options: &FixedPointOptions<T::Real>,
        w: &mut FixedPointWorkspace<T>,
        session: &crate::context::Session<'_, '_, '_>,
        current: T::Real,
        scale: T::Real,
    ) -> Result<Advance<T::Real>, SciOxideError>
    where
        F: ResidualOperator<T> + ?Sized,
    {
        let secant = matches!(
            self.method,
            FixedPointMethod::Anderson
                | FixedPointMethod::BroydenFirst
                | FixedPointMethod::BroydenSecond
                | FixedPointMethod::DiagonalBroyden
        );
        let step_test = options.step.tolerance().absolute() != T::Real::ZERO
            || options.step.tolerance().relative() != T::Real::ZERO;
        if secant || step_test {
            for ((step, &next), &x) in w
                .step
                .as_mut_slice()
                .iter_mut()
                .zip(w.trial.as_slice())
                .zip(w.x.as_slice())
            {
                *step = difference(next, x)?;
            }
        }
        let small_step = step_test
            && options.step.is_satisfied(
                norm(w.step.as_slice())?,
                norm(w.x.as_slice())?.max(T::Real::ONE),
            )?;
        if w.trial.as_slice() == w.x.as_slice() {
            return Ok(Advance::Done(RootTermination::Stagnation));
        }
        evaluate(
            problem,
            self.input,
            &w.trial,
            &mut w.trial_f,
            &mut w.scratch,
            session,
        )?;
        if secant {
            for ((delta, &next), &f) in w
                .difference
                .as_mut_slice()
                .iter_mut()
                .zip(w.trial_f.as_slice())
                .zip(w.f.as_slice())
            {
                *delta = difference(next, f)?;
            }
        }
        let next = component_norm(w.trial_f.as_slice(), options)?;
        let termination =
            if crate::termination::residual_ok(&options.residual.tolerance(), next, scale)? {
                Some(RootTermination::ResidualTolerance)
            } else if small_step && next >= current {
                Some(RootTermination::Stagnation)
            } else {
                update(w, options)?;
                None
            };
        session.accept()?;
        core::mem::swap(&mut w.x, &mut w.trial);
        core::mem::swap(&mut w.f, &mut w.trial_f);
        Ok(termination.map_or(Advance::Accepted(next), Advance::Done))
    }
}

/// Steffensen correction in scaled arithmetic; `None` skips a near-zero denominator.
fn steffensen_range<T: NumericScalar>(
    x: T,
    damping: T::Real,
    f: T,
    next: T,
) -> Result<Option<T>, SciOxideError> {
    let first = scaled(T::from_real(damping))?.mul(scaled(f)?)?;
    let second = scaled(T::from_real(damping))?.mul(scaled(next)?)?;
    let denominator = second.sub(first)?;
    let threshold = first
        .magnitude()?
        .add(second.magnitude()?)?
        .mul(Scaled::from_value(T::Real::SQRT_EPSILON)?)?;
    if denominator.magnitude()?.abs_le(threshold) {
        return Ok(None);
    }
    Ok(Some(rounded(
        scaled(x)?.sub(first.mul(first)?.div(denominator)?)?,
    )?))
}
/// Ordinary Steffensen correction, or a skipped near-zero denominator.
/// An absent outer value requests the scaled-arithmetic calculation.
fn steffensen_native<T: NumericScalar>(x: T, damping: T::Real, f: T, next: T) -> Option<Option<T>> {
    use scioxide_core::backend::standard::product_needs_retry;
    let damping = T::from_real(damping);
    let first = damping * f;
    let second = damping * next;
    let denominator = second - first;
    let threshold = (first.abs() + second.abs()) * T::Real::SQRT_EPSILON;
    if product_needs_retry(damping, f, first)
        || product_needs_retry(damping, next, second)
        || !denominator.has_finite_components()
        || !threshold.is_finite()
    {
        return None;
    }
    if denominator.abs() <= threshold {
        return Some(None);
    }
    let square = first * first;
    if product_needs_retry(first, first, square) {
        return None;
    }
    let correction = square / denominator;
    let value = x - correction;
    (correction.has_finite_components()
        && value.has_finite_components()
        && (correction != T::ZERO || square == T::ZERO))
        .then_some(Some(value))
}

/// Diagonal secant update in ordinary arithmetic; `None` requests the scaled
/// calculation.
fn diagonal_native<T: NumericScalar>(
    diagonal: T,
    step: T,
    delta: T,
    denominator: T,
    damping: T::Real,
) -> Option<(T, bool)> {
    use scioxide_core::backend::standard::product_needs_retry;
    let old = -T::ONE / diagonal;
    let product = old * step;
    let correction = delta - product;
    let weight = step.conj() / denominator;
    let update = correction * weight;
    let next = old + update;
    if !old.has_finite_components()
        || product_needs_retry(old, step, product)
        || !correction.has_finite_components()
        || !weight.has_finite_components()
        || (weight == T::ZERO && step != T::ZERO)
        || product_needs_retry(correction, weight, update)
        || !next.has_finite_components()
    {
        return None;
    }
    let threshold = old.abs() * T::Real::SQRT_EPSILON;
    if next.abs() <= threshold {
        return Some((T::from_real(damping), true));
    }
    let value = -T::ONE / next;
    value.has_finite_components().then_some((value, false))
}

pub(crate) fn update<T: NumericScalar>(
    w: &mut FixedPointWorkspace<T>,
    o: &FixedPointOptions<T::Real>,
) -> Result<(), SciOxideError> {
    let n = w.dimension;
    if matches!(
        w.method,
        FixedPointMethod::Anderson
            | FixedPointMethod::BroydenFirst
            | FixedPointMethod::BroydenSecond
    ) && w.capacity > 0
    {
        if w.used == w.capacity {
            match o.restart {
                HistoryRestart::Oldest => discard(w, 0),
                HistoryRestart::Restart => {
                    w.used = 0;
                    w.diagnostics.restarts += 1;
                }
            }
        }
        if w.method == FixedPointMethod::Anderson {
            w.first[w.used * n..(w.used + 1) * n].copy_from_slice(w.step.as_slice());
            w.second[w.used * n..(w.used + 1) * n].copy_from_slice(w.difference.as_slice());
        } else {
            apply_inverse(
                &w.first,
                &w.second,
                w.used,
                w.difference.as_slice(),
                w.work.as_mut_slice(),
                o.damping,
                false,
            )?;
            if w.method == FixedPointMethod::BroydenFirst {
                apply_inverse(
                    &w.first,
                    &w.second,
                    w.used,
                    w.step.as_slice(),
                    w.row.as_mut_slice(),
                    o.damping,
                    true,
                )?;
            } else {
                w.row
                    .as_mut_slice()
                    .copy_from_slice(w.difference.as_slice());
            }
            if !broyden_correction(
                w.step.as_slice(),
                w.work.as_slice(),
                w.row.as_mut_slice(),
                w.difference.as_slice(),
                &mut w.first[w.used * n..(w.used + 1) * n],
            )? {
                w.diagnostics.skipped_updates += 1;
                return Ok(());
            }
            w.second[w.used * n..(w.used + 1) * n].copy_from_slice(w.row.as_slice());
        }
        w.used += 1;
        if w.method != FixedPointMethod::Anderson {
            w.diagnostics.history_rank = w.used;
        }
    } else if w.method == FixedPointMethod::DiagonalBroyden {
        let denominator = dot(w.step.as_slice(), w.step.as_slice())?;
        if denominator.is_zero() {
            return Ok(());
        }
        let native_denominator = crate::nonlinear_math::native_coefficient(denominator);
        for i in 0..n {
            if let Some((value, restart)) = native_denominator.and_then(|denominator| {
                diagonal_native(
                    w.diagonal.as_slice()[i],
                    w.step.as_slice()[i],
                    w.difference.as_slice()[i],
                    denominator,
                    o.damping,
                )
            }) {
                w.diagonal.as_mut_slice()[i] = value;
                w.diagnostics.restarts += usize::from(restart);
                continue;
            }
            // Store alpha=-1/J_ii; update the diagonal of J by the good Broyden rule.
            let old = scaled(T::ONE)?
                .neg()
                .div(scaled(w.diagonal.as_slice()[i])?)?;
            let correction =
                scaled(w.difference.as_slice()[i])?.sub(old.mul(scaled(w.step.as_slice()[i])?)?)?;
            let next = old.add(
                correction
                    .mul(scaled(w.step.as_slice()[i].conj())?)?
                    .div(denominator)?,
            )?;
            let threshold = old
                .magnitude()?
                .mul(Scaled::from_value(T::Real::SQRT_EPSILON)?)?;
            if next.magnitude()?.abs_le(threshold) {
                w.diagonal.as_mut_slice()[i] = T::from_real(o.damping);
                w.diagnostics.restarts += 1;
            } else {
                w.diagonal.as_mut_slice()[i] = rounded(scaled(T::ONE)?.neg().div(next)?)?;
            }
        }
    } else if w.method == FixedPointMethod::ExcitingMixing {
        for i in 0..n {
            let a = w.f.as_slice()[i].real_component();
            let b = w.trial_f.as_slice()[i].real_component();
            w.diagonal.as_mut_slice()[i] = T::from_real(
                if (a > T::Real::ZERO && b > T::Real::ZERO)
                    || (a < T::Real::ZERO && b < T::Real::ZERO)
                {
                    // Saturating addition avoids overflow before applying the upper bound.
                    let current = w.diagonal.as_slice()[i].real_component();
                    current + o.damping.min(o.maximum_mixing - current)
                } else {
                    w.diagnostics.restarts += 1;
                    o.damping
                },
            );
        }
    }
    Ok(())
}

/// Native common history scale, declining exceptional weighted magnitudes.
fn native_history_scale<T: NumericScalar>(
    history: &[T],
    residual: &[T],
    options: &FixedPointOptions<T::Real>,
) -> Option<T::Real> {
    let n = residual.len();
    let mut largest = T::Real::ZERO;
    for (i, &value) in history.iter().chain(residual).enumerate() {
        let magnitude = value.abs();
        let scaled = magnitude / options.divisor(i % n);
        if value != T::ZERO && (!magnitude.is_normal() || !scaled.is_normal()) {
            return None;
        }
        largest = largest.max(scaled);
    }
    Some(largest)
}

/// Replay the whole history contraction when an intermediate correction or
/// partial sum loses range; later terms may restore a representable candidate.
pub(super) fn subtract_history<T: NumericScalar>(
    initial: T,
    damping: T,
    terms: impl Iterator<Item = (T, T, T)> + Clone,
) -> Result<T, SciOxideError> {
    use scioxide_core::backend::standard::product_needs_retry;
    let mut value = initial;
    let mut ordinary = true;
    for (step, delta, coefficient) in terms.clone() {
        let product = damping * delta;
        let correction = step + product;
        let update = coefficient * correction;
        value = value - update;
        ordinary &= correction.has_finite_components()
            && value.has_finite_components()
            && !product_needs_retry(damping, delta, product)
            && !product_needs_retry(coefficient, correction, update);
    }
    if ordinary {
        return Ok(value);
    }
    let mut value = scaled(initial)?;
    let damping = scaled(damping)?;
    for (step, delta, coefficient) in terms {
        let correction = scaled(step)?.add(damping.mul(scaled(delta)?)?)?;
        value = value.sub(correction.mul(scaled(coefficient)?)?)?;
    }
    rounded(value)
}

pub(crate) fn anderson_propose<T: NumericScalar>(
    w: &mut FixedPointWorkspace<T>,
    o: &FixedPointOptions<T::Real>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    loop {
        let Some(a) = w.anderson.as_mut() else {
            return Ok(());
        };
        let m = w.used;
        if m == 0 {
            w.diagnostics.history_rank = 0;
            return Ok(());
        }
        let n = w.dimension;
        // One common scale preserves the weighted least-squares problem.
        let native_scale = native_history_scale(&w.second[..m * n], w.f.as_slice(), o);
        let scale = if let Some(scale) = native_scale {
            Scaled::from_value(scale)?
        } else {
            let mut scale = Scaled::ZERO;
            for k in 0..m {
                for i in 0..n {
                    let value = scaled(w.second[k * n + i])?
                        .magnitude()?
                        .div(Scaled::from_value(o.divisor(i))?)?;
                    if value.abs_cmp(scale).is_gt() {
                        scale = value;
                    }
                }
            }
            for (i, &f) in w.f.as_slice().iter().enumerate() {
                let value = scaled(f)?
                    .magnitude()?
                    .div(Scaled::from_value(o.divisor(i))?)?;
                if value.abs_cmp(scale).is_gt() {
                    scale = value;
                }
            }
            scale
        };
        if scale.is_zero() {
            w.used = 0;
            return Ok(());
        }
        let rows = n + m;
        a.matrix.try_resize_flat(rows, m)?;
        a.rhs.try_resize(rows, T::ZERO)?;
        a.coefficients.try_resize(m, T::ZERO)?;
        a.matrix.as_mut_slice().fill(T::ZERO);
        a.rhs.as_mut_slice().fill(T::ZERO);
        if let Some(scale) =
            native_scale.filter(|&scale| (0..n).all(|i| (o.divisor(i) * scale).is_normal()))
        {
            for i in 0..n {
                let divisor = o.divisor(i) * scale;
                let divide = |value: T| {
                    T::from_components(
                        value.real_component() / divisor,
                        value.imaginary_component() / divisor,
                    )
                };
                a.rhs.as_mut_slice()[i] = divide(w.f.as_slice()[i]);
                for k in 0..m {
                    a.matrix.as_mut_slice()[i * m + k] = divide(w.second[k * n + i]);
                }
            }
        } else {
            for i in 0..n {
                let divisor = ScaledValue::from_components(
                    Scaled::from_value(o.divisor(i))?.mul(scale)?,
                    Scaled::ZERO,
                );
                a.rhs.as_mut_slice()[i] = rounded(scaled(w.f.as_slice()[i])?.div(divisor)?)?;
                for k in 0..m {
                    a.matrix.as_mut_slice()[i * m + k] =
                        rounded(scaled(w.second[k * n + i])?.div(divisor)?)?;
                }
            }
        }
        session.admit(EvaluationRole::Factorization)?;
        HouseholderDecomposition::factor_into(
            &a.matrix,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
            &scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(o.policy),
            &mut a.factor,
            &mut a.scratch,
            session.execution()?,
        )?;
        let rank = a.factor.analyze_rank(o.rank)?.rank();
        w.diagnostics.history_rank = rank;
        if rank < m && o.regularization == T::Real::ZERO {
            let rejected = a.factor.permutation()[rank];
            discard(w, rejected);
            w.diagnostics.skipped_updates += 1;
            continue;
        }
        if o.regularization > T::Real::ZERO {
            for k in 0..m {
                a.matrix.as_mut_slice()[(n + k) * m + k] = T::from_real(o.regularization.sqrt());
            }
            session.admit(EvaluationRole::Factorization)?;
            HouseholderDecomposition::factor_into(
                &a.matrix,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(o.policy),
                &mut a.factor,
                &mut a.scratch,
                session.execution()?,
            )?;
            w.diagnostics.regularized_solves += 1;
        }
        session.admit(EvaluationRole::LinearSolve)?;
        let report = a.factor.solve_into(
            &a.rhs,
            &(scioxide_linalg::DirectSolveOptions::new(RankPolicy::Exact)).with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(o.policy),
            ),
            a.coefficients.view_mut(),
            &mut a.scratch,
            session.execution()?,
        )?;
        if !report.rank().is_full_rank() {
            return Err(invalid(
                scioxide_core::argument!("options.regularization"),
                InvalidInputReason::Inconsistent,
                "regularized history remains singular",
            ));
        }
        for i in 0..n {
            w.trial.as_mut_slice()[i] = subtract_history(
                w.trial.as_slice()[i],
                T::from_real(o.damping),
                (0..m).map(|k| {
                    (
                        w.first[k * n + i],
                        w.second[k * n + i],
                        a.coefficients.as_slice()[k],
                    )
                }),
            )?;
        }
        return Ok(());
    }
}

pub(crate) fn propose<T: NumericScalar>(
    w: &mut FixedPointWorkspace<T>,
    o: &FixedPointOptions<T::Real>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if matches!(
        w.method,
        FixedPointMethod::BroydenFirst | FixedPointMethod::BroydenSecond
    ) {
        apply_inverse(
            &w.first,
            &w.second,
            w.used,
            w.f.as_slice(),
            w.step.as_mut_slice(),
            o.damping,
            false,
        )?;
        for ((next, &x), &step) in w
            .trial
            .as_mut_slice()
            .iter_mut()
            .zip(w.x.as_slice())
            .zip(w.step.as_slice())
        {
            *next = difference(x, step)?;
        }
    } else if matches!(
        w.method,
        FixedPointMethod::DiagonalBroyden | FixedPointMethod::ExcitingMixing
    ) {
        for (i, ((next, &x), &f)) in w
            .trial
            .as_mut_slice()
            .iter_mut()
            .zip(w.x.as_slice())
            .zip(w.f.as_slice())
            .enumerate()
        {
            let alpha = w
                .diagonal
                .as_slice()
                .get(i)
                .copied()
                .unwrap_or(T::from_real(o.damping));
            *next = affine(x, alpha, f)?;
        }
    } else {
        crate::nonlinear_math::affine_into(
            w.trial.as_mut_slice(),
            w.x.as_slice(),
            T::from_real(o.damping),
            w.f.as_slice(),
            crate::nonlinear_math::StepRounding::Separate,
        )?;
    }
    if w.method == FixedPointMethod::Anderson && w.used > 0 {
        anderson_propose(w, o, session)?;
    }
    Ok(())
}

pub(crate) fn component_norm<T: NumericScalar>(
    values: &[T],
    o: &FixedPointOptions<T::Real>,
) -> Result<T::Real, SciOxideError> {
    if o.component_scales.is_empty() {
        return norm(values);
    }
    let ordinary = values.iter().enumerate().all(|(i, &value)| {
        let d = o.divisor(i);
        [value.real_component(), value.imaginary_component()]
            .into_iter()
            .all(|part| {
                let quotient = part / d;
                quotient.is_normal() || part == T::Real::ZERO
            })
    });
    if ordinary {
        return crate::nonlinear_math::norm_iter(values.iter().enumerate().map(|(i, &value)| {
            let d = o.divisor(i);
            T::from_components(value.real_component() / d, value.imaginary_component() / d)
        }));
    }
    let sum = values
        .iter()
        .enumerate()
        .try_fold(Scaled::ZERO, |sum, (i, &value)| {
            let magnitude = scaled(value)?
                .magnitude()?
                .div(Scaled::from_value(o.divisor(i))?)?;
            sum.add(magnitude.mul(magnitude)?)
        })?;
    sum.sqrt()?.try_value()
}

pub(crate) fn residual_norm<T: NumericScalar>(
    w: &mut FixedPointWorkspace<T>,
    o: &FixedPointOptions<T::Real>,
) -> Result<T::Real, SciOxideError> {
    component_norm(w.f.as_slice(), o)
}

pub(crate) fn evaluate<T: NumericScalar, F>(
    problem: &F,
    input: FixedPointInput,
    x: &Vector<T>,
    output: &mut Vector<T>,
    scratch: &mut [T],
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    F: ResidualOperator<T> + ?Sized,
{
    crate::residual::evaluate(
        problem,
        if input == FixedPointInput::Map {
            EvaluationRole::Function
        } else {
            EvaluationRole::Residual
        },
        x.as_slice(),
        output.as_mut_slice(),
        scratch,
        session,
    )?;
    if input == FixedPointInput::Map {
        for (out, &value) in output.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *out = difference(*out, value)?;
        }
    }
    Ok(())
}

pub(crate) fn discard<T: NumericScalar>(w: &mut FixedPointWorkspace<T>, index: usize) {
    let n = w.dimension;
    w.first.copy_within((index + 1) * n..w.used * n, index * n);
    w.second.copy_within((index + 1) * n..w.used * n, index * n);
    w.used -= 1;
}
