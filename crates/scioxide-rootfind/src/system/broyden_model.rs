//! Jacobian model of the first Broyden update, with a checked inverse.
//!
//! Steps are defined by the forward model `B`. The inverse only proposes a
//! direction, which must satisfy `B d = −F` within a bound for product rounding.
//! A rejected inverse is recomputed from a pivoted LU factorization.
use super::dense::{
    infinity_norm, product_bound_into, product_into, rank_one_into, transpose_product_into,
};
use crate::context::Session;
use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix, RankPolicy, Vector,
};

#[derive(Debug)]
pub(crate) struct GoodBroydenModel<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) forward: Matrix<S>,
    inverse: Matrix<S>,
    candidate_forward: Matrix<S>,
    candidate_inverse: Matrix<S>,
    change: Vector<S>,
    forward_step: Vector<S>,
    inverse_change: Vector<S>,
    inverse_row: Vector<S>,
    forward_left: Vector<S>,
    inverse_left: Vector<S>,
    pub(crate) product: Vector<S>,
    rounding_bound: Vector<S>,
    rhs: Vector<S>,
    solve_scratch: Vector<S>,
    next_direction: Vector<S>,
    next_direction_valid: bool,
    inverse_direction_used: bool,
    active: bool,
    inverse_valid: bool,
    factor_current: bool,
    full_rank: bool,
    forward_norm: S,
    inverse_norm: S,
    roundoff: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> GoodBroydenModel<S> {
    pub(crate) const fn new() -> Self {
        Self {
            forward: Matrix::empty(),
            inverse: Matrix::empty(),
            candidate_forward: Matrix::empty(),
            candidate_inverse: Matrix::empty(),
            change: Vector::empty(),
            forward_step: Vector::empty(),
            inverse_change: Vector::empty(),
            inverse_row: Vector::empty(),
            forward_left: Vector::empty(),
            inverse_left: Vector::empty(),
            product: Vector::empty(),
            rounding_bound: Vector::empty(),
            rhs: Vector::empty(),
            solve_scratch: Vector::empty(),
            next_direction: Vector::empty(),
            next_direction_valid: false,
            inverse_direction_used: false,
            active: false,
            inverse_valid: false,
            factor_current: false,
            full_rank: false,
            forward_norm: <S as AlgebraicScalar>::ZERO,
            inverse_norm: <S as AlgebraicScalar>::ZERO,
            roundoff: <S as AlgebraicScalar>::ZERO,
        }
    }
    pub(crate) fn reset(&mut self) {
        self.active = false;
        self.inverse_valid = false;
        self.factor_current = false;
        self.forget_direction();
    }
    pub(crate) fn forget_direction(&mut self) {
        self.next_direction_valid = false;
        self.inverse_direction_used = false;
    }
    pub(crate) fn is_active(&self) -> bool {
        self.active
    }
    pub(crate) fn storage_bytes(n: usize) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("broyden_model_bytes");
        let matrices = scioxide_core::backend::checked_product(&[4, n, n], q)?;
        let vectors = scioxide_core::backend::checked_mul(11, n, q)?;
        scioxide_core::backend::checked_mul(
            scioxide_core::backend::checked_add(matrices, vectors, q)?,
            core::mem::size_of::<S>(),
            q,
        )
    }
    pub(crate) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        crate::workspace::sum([
            self.forward.retained_bytes(),
            self.inverse.retained_bytes(),
            self.candidate_forward.retained_bytes(),
            self.candidate_inverse.retained_bytes(),
            self.change.retained_bytes(),
            self.forward_step.retained_bytes(),
            self.inverse_change.retained_bytes(),
            self.inverse_row.retained_bytes(),
            self.forward_left.retained_bytes(),
            self.inverse_left.retained_bytes(),
            self.product.retained_bytes(),
            self.rounding_bound.retained_bytes(),
            self.rhs.retained_bytes(),
            self.solve_scratch.retained_bytes(),
            self.next_direction.retained_bytes(),
        ])
    }
    pub(crate) fn try_reserve(&mut self, n: usize) -> Result<(), SciOxideError> {
        for matrix in [
            &mut self.forward,
            &mut self.inverse,
            &mut self.candidate_forward,
            &mut self.candidate_inverse,
        ] {
            matrix.try_resize_flat(n, n)?;
        }
        for vector in [
            &mut self.change,
            &mut self.forward_step,
            &mut self.inverse_change,
            &mut self.inverse_row,
            &mut self.forward_left,
            &mut self.inverse_left,
            &mut self.product,
            &mut self.rounding_bound,
            &mut self.rhs,
            &mut self.solve_scratch,
            &mut self.next_direction,
        ] {
            vector.try_resize(n, <S as AlgebraicScalar>::ZERO)?;
        }
        let count = S::try_from_usize_exact(n).unwrap_or(S::MAX);
        self.roundoff = (count * S::EPSILON) * (S::TWO * S::TWO * S::TWO);
        Ok(())
    }
    pub(crate) fn initialize(
        &mut self,
        jacobian: &Matrix<S>,
        lu: &mut LuDecomposition<S>,
        work: &mut LinalgWorkspace<'static, S>,
        session: &Session<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.forward
            .as_mut_slice()
            .copy_from_slice(jacobian.as_slice());
        self.forward_norm = infinity_norm(&self.forward);
        self.active = true;
        self.inverse_valid = false;
        self.factor_current = false;
        self.refactor(lu, work, session)
    }
    fn refactor(
        &mut self,
        lu: &mut LuDecomposition<S>,
        work: &mut LinalgWorkspace<'static, S>,
        session: &Session<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.inverse_valid = false;
        self.forget_direction();
        let options = LinalgExecutionOptions::new().with_execution_policy(session.policy());
        session.invoke_context(scioxide_core::EvaluationRole::Factorization, |context| {
            match LuDecomposition::factor_into(
                &self.forward,
                &options,
                lu,
                work,
                context.execution(),
            ) {
                Err(error) if error.kind() == SciOxideErrorKind::Singular => {
                    self.full_rank = false;
                    self.factor_current = true;
                    return Ok(());
                }
                outcome => outcome?,
            }
            self.full_rank = lu
                .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
                .is_full_rank();
            self.factor_current = true;
            if self.full_rank {
                match lu.inverse_into(&options, self.inverse.view_mut(), work, context.execution())
                {
                    Ok(()) => {
                        self.inverse_valid = true;
                        self.inverse_norm = infinity_norm(&self.inverse);
                    }
                    Err(error)
                        if matches!(
                            error.kind(),
                            SciOxideErrorKind::NonFinite
                                | SciOxideErrorKind::NumericalInstability
                                | SciOxideErrorKind::Singular
                        ) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(())
        })
    }
    pub(crate) fn direction(
        &mut self,
        residual: &Vector<S>,
        residual_norm: S,
        output: &mut Vector<S>,
        lu: &mut LuDecomposition<S>,
        work: &mut LinalgWorkspace<'static, S>,
        session: &Session<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        self.inverse_direction_used = false;
        let hinted = self.next_direction_valid;
        self.next_direction_valid = false;
        let amplification =
            self.forward_norm * self.inverse_norm * self.roundoff * (S::TWO * S::TWO);
        if !amplification.is_finite() || amplification > S::SQRT_EPSILON {
            self.inverse_valid = false;
        }
        if self.inverse_valid {
            for attempt in 0..if hinted { 2 } else { 1 } {
                let candidate = (|| {
                    session.admit(scioxide_core::EvaluationRole::LinearSolve)?;
                    if hinted && attempt == 0 {
                        output
                            .as_mut_slice()
                            .copy_from_slice(self.next_direction.as_slice());
                    } else {
                        product_into(
                            &self.inverse,
                            residual.as_slice(),
                            self.inverse_norm,
                            output.as_mut_slice(),
                        )?;
                        for x in output.as_mut_slice() {
                            *x = -*x;
                        }
                    }
                    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
                    product_bound_into(
                        &self.forward,
                        output.as_slice(),
                        self.product.as_mut_slice(),
                        self.rounding_bound.as_mut_slice(),
                    )?;
                    Ok(verified_direction(
                        residual.as_slice(),
                        residual_norm,
                        self.product.as_slice(),
                        self.rounding_bound.as_slice(),
                        self.roundoff,
                    ))
                })();
                match candidate {
                    Ok(true) => {
                        self.inverse_direction_used = true;
                        return Ok(true);
                    }
                    Err(error) if cache_failure(&error) => {}
                    outcome => {
                        let _ = outcome?;
                    }
                }
            }
            self.inverse_valid = false;
        }
        if !self.factor_current {
            self.refactor(lu, work, session)?;
        }
        if !self.full_rank {
            return Ok(false);
        }
        for (out, &f) in self.rhs.as_mut_slice().iter_mut().zip(residual.as_slice()) {
            *out = -f;
        }
        super::factored_solve_into(lu, &self.rhs, output, &mut self.solve_scratch, session)?;
        session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
        product_into(
            &self.forward,
            output.as_slice(),
            self.forward_norm,
            self.product.as_mut_slice(),
        )?;
        Ok(true)
    }
    pub(crate) fn update(
        &mut self,
        step: &Vector<S>,
        step_norm: S,
        direction: &Vector<S>,
        current: &Vector<S>,
        trial: &Vector<S>,
        session: &Session<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        self.next_direction_valid = false;
        if step_norm == <S as AlgebraicScalar>::ZERO {
            return Ok(false);
        }
        session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
        let denominator = step_norm * step_norm;
        let condition = self.forward_norm * self.inverse_norm;
        let allowance = condition * self.roundoff * (S::TWO * S::TWO);
        if !denominator.is_normal()
            || !self.inverse_valid
            || !allowance.is_finite()
            || allowance > S::SQRT_EPSILON
        {
            return self.update_scaled(
                step,
                current,
                trial,
                super::scaled_dot(step.as_slice(), step.as_slice())?,
            );
        }
        match self.update_candidate(step, direction, current, trial, denominator) {
            Err(error) if cache_failure(&error) => self.update_scaled(
                step,
                current,
                trial,
                super::scaled_dot(step.as_slice(), step.as_slice())?,
            ),
            outcome => outcome,
        }
    }
    fn update_candidate(
        &mut self,
        step: &Vector<S>,
        direction: &Vector<S>,
        current: &Vector<S>,
        trial: &Vector<S>,
        denominator: S,
    ) -> Result<bool, SciOxideError> {
        for ((out, &a), &b) in self
            .change
            .as_mut_slice()
            .iter_mut()
            .zip(trial.as_slice())
            .zip(current.as_slice())
        {
            *out = crate::nonlinear_math::difference(a, b)?;
        }
        product_into(
            &self.forward,
            step.as_slice(),
            self.forward_norm,
            self.forward_step.as_mut_slice(),
        )?;
        if !correction_factors(
            self.change.as_slice(),
            self.forward_step.as_slice(),
            self.forward_left.as_mut_slice(),
            denominator,
        )? {
            return Ok(false);
        }
        let mut inverse_candidate = false;
        if self.inverse_valid {
            let proposal = (|| {
                product_into(
                    &self.inverse,
                    self.change.as_slice(),
                    self.inverse_norm,
                    self.inverse_change.as_mut_slice(),
                )?;
                transpose_product_into(
                    &self.inverse,
                    step.as_slice(),
                    self.inverse_norm,
                    self.inverse_row.as_mut_slice(),
                )?;
                let row_norm = crate::nonlinear_math::norm(self.inverse_row.as_slice())?;
                let change_norm = crate::nonlinear_math::norm(self.change.as_slice())?;
                let native = self
                    .inverse_row
                    .as_slice()
                    .iter()
                    .zip(self.change.as_slice())
                    .fold(<S as AlgebraicScalar>::ZERO, |sum, (&a, &b)| sum + a * b);
                let native_threshold = S::SQRT_EPSILON * row_norm * change_norm;
                if !native.is_normal()
                    || !native_threshold.is_finite()
                    || native.abs() <= native_threshold
                {
                    return Ok(false);
                }
                let d = native;
                {
                    inverse_candidate = correction_factors(
                        step.as_slice(),
                        self.inverse_change.as_slice(),
                        self.inverse_left.as_mut_slice(),
                        d,
                    )?;
                }
                Ok(inverse_candidate)
            })();
            inverse_candidate = match proposal {
                Err(error) if cache_failure(&error) => false,
                outcome => outcome?,
            };
        }
        let (changed, forward_norm) = rank_one_into::<S, true>(
            &self.forward,
            &self.forward_left,
            step,
            self.forward_norm,
            S::ONE + self.roundoff,
            &mut self.candidate_forward,
        )?;
        let mut inverse_norm = self.inverse_norm;
        if inverse_candidate {
            match rank_one_into::<S, false>(
                &self.inverse,
                &self.inverse_left,
                &self.inverse_row,
                self.inverse_norm,
                S::ONE + self.roundoff,
                &mut self.candidate_inverse,
            ) {
                Ok((_, norm)) => inverse_norm = norm,
                Err(error)
                    if matches!(
                        error.kind(),
                        SciOxideErrorKind::NonFinite | SciOxideErrorKind::NumericalInstability
                    ) =>
                {
                    inverse_candidate = false
                }
                Err(error) => return Err(error),
            }
        }
        if inverse_candidate && self.inverse_direction_used {
            // Sherman–Morrison applied to F_new gives
            // d_new = d - s + (s-Hy)(sᵀd)/(sᵀHy). The next use still checks
            // this hint against the updated forward model before accepting it.
            let projection = step
                .as_slice()
                .iter()
                .zip(direction.as_slice())
                .fold(<S as AlgebraicScalar>::ZERO, |sum, (&s, &d)| sum + s * d);
            let mut finite = projection.is_finite();
            for (((out, &d), &s), &left) in self
                .next_direction
                .as_mut_slice()
                .iter_mut()
                .zip(direction.as_slice())
                .zip(step.as_slice())
                .zip(self.inverse_left.as_slice())
            {
                *out = (d - s) + left * projection;
                finite &= out.is_finite();
            }
            self.next_direction_valid = finite;
        }
        core::mem::swap(&mut self.forward, &mut self.candidate_forward);
        self.forward_norm = forward_norm;
        if inverse_candidate {
            core::mem::swap(&mut self.inverse, &mut self.candidate_inverse);
            self.inverse_norm = inverse_norm;
        }
        self.inverse_valid = inverse_candidate;
        self.factor_current = false;
        Ok(changed)
    }
    /// Preserve the original residuals and products until each complete updated
    /// entry is projected. This replay also handles an unrepresentable ΔF or Bs.
    #[cold]
    fn update_scaled(
        &mut self,
        step: &Vector<S>,
        current: &Vector<S>,
        trial: &Vector<S>,
        denominator: Scaled<S>,
    ) -> Result<bool, SciOxideError> {
        let n = step.len();
        let mut changed = false;
        for i in 0..n {
            let delta = Scaled::from_value(trial.as_slice()[i])?
                .sub(Scaled::from_value(current.as_slice()[i])?)?;
            let row = &self.forward.as_slice()[i * n..(i + 1) * n];
            let product = super::scaled_dot(row, step.as_slice())?;

            for (j, &old) in row.iter().enumerate() {
                let value = if n == 1 {
                    delta
                        .div(Scaled::from_value(step.as_slice()[0])?)?
                        .round()?
                } else {
                    {
                        let coefficient =
                            Scaled::from_value(step.as_slice()[j])?.div(denominator)?;
                        let mut sum = scioxide_core::ScaledAccumulator::ZERO;
                        sum.add_scaled(Scaled::from_value(old)?)?;
                        sum.add_scaled(delta.mul(coefficient)?)?;
                        sum.add_scaled(-product.mul(coefficient)?)?;
                        sum.total()?.round()?
                    }
                };
                self.candidate_forward.as_mut_slice()[i * n + j] = value;
                changed |= value != old;
            }
        }
        core::mem::swap(&mut self.forward, &mut self.candidate_forward);
        self.forward_norm = infinity_norm(&self.forward);
        self.inverse_valid = false;
        self.factor_current = false;
        Ok(changed)
    }
}
/// Fuse the coordinate-wise error enclosure and its L2 comparison. Normal
/// threshold squares and a factor-two margin keep ordinary square-sum rounding
/// away from the decision; exceptional ranges decline the inverse cache.
fn verified_direction<S: RealScalar>(
    residual: &[S],
    norm: S,
    product: &[S],
    absolute: &[S],
    gamma: S,
) -> bool {
    let threshold = S::SQRT_EPSILON * norm;
    let squared_threshold = threshold * threshold;
    if gamma >= S::HALF || !squared_threshold.is_normal() {
        return false;
    }
    let mut sums = [S::ZERO; 4];
    for (i, ((&f, &p), &a)) in residual.iter().zip(product).zip(absolute).enumerate() {
        let error = (f + p).abs() + gamma * a;
        sums[i % 4] += error * error;
    }
    let squared_error = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    squared_error.is_finite() && squared_error <= squared_threshold * (S::HALF * S::HALF)
}

fn cache_failure(error: &SciOxideError) -> bool {
    matches!(
        error.kind(),
        SciOxideErrorKind::NonFinite
            | SciOxideErrorKind::NumericalInstability
            | SciOxideErrorKind::Singular
    )
}
/// Native coefficients are consumed immediately by the staged outer product.
/// A lost component requests original-component replay of the forward update,
/// or rejection of the disposable inverse cache.
fn correction_factors<S: RealScalar + NumericScalar<Real = S>>(
    left: &[S],
    product: &[S],
    correction: &mut [S],
    denominator: S,
) -> Result<bool, SciOxideError> {
    let reciprocal = S::ONE / denominator;
    let multiply = reciprocal.is_normal();
    let mut changed = false;
    let mut ordinary = true;
    for ((out, &a), &b) in correction.iter_mut().zip(left).zip(product) {
        let difference = a - b;
        *out = if multiply {
            difference * reciprocal
        } else {
            difference / denominator
        };
        ordinary &= difference.is_finite()
            && (out.is_normal() || difference == <S as AlgebraicScalar>::ZERO);
        changed |= difference != <S as AlgebraicScalar>::ZERO;
    }
    if !ordinary {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "secant coefficient requires extended-range replay",
        ));
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Broyden, BroydenOptions, BroydenUpdate, ResidualFn, SystemRootSolver};
    fn prepared<S: RealScalar + NumericScalar<Real = S>>(
        n: usize,
        operation: impl FnOnce(
            &mut super::super::SystemWorkspace<S>,
            &Session<'_, '_, '_>,
        ) -> Result<(), SciOxideError>,
    ) {
        let function = ResidualFn::new(n, |_: &[S], _: &mut [S]| Ok(()));
        let solver = Broyden::new(BroydenUpdate::First, BroydenOptions::<S>::default()).unwrap();
        let mut workspace = solver.workspace(&function).unwrap();
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Iteration,
            scioxide_core::Budget::try_from(1000).unwrap(),
        );
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                scioxide_core::SolveIdentity::new(
                    scioxide_core::ProblemToken::new(0),
                    solver.algorithm_id(),
                    n as u128,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        operation(&mut workspace, &Session::new(&mut context)).unwrap();
    }
    fn general_models<S: RealScalar + NumericScalar<Real = S>>() {
        for n in [16, 64] {
            prepared::<S>(n, |w, session| {
                w.jacobian.as_mut_slice().fill(<S as AlgebraicScalar>::ZERO);
                for i in 0..n {
                    w.jacobian.as_mut_slice()[i * n + i] = S::ONE / (S::TWO * S::TWO * S::TWO);
                    w.jacobian.as_mut_slice()[i * n + (i + 1) % n] = S::TWO;
                }
                w.good_broyden.initialize(
                    &w.jacobian,
                    &mut w.lu,
                    &mut w.factor_workspace,
                    session,
                )?;
                assert!(w.good_broyden.inverse_valid);
                w.state.current_residual.as_mut_slice().fill(S::ONE);
                for iteration in 0..3 {
                    for (i, out) in w.step.as_mut_slice().iter_mut().enumerate() {
                        *out = if (i + iteration) % 2 == 0 {
                            S::HALF
                        } else {
                            -S::HALF
                        };
                    }
                    let count = S::try_from_usize_exact(n)?;
                    for i in 0..n {
                        let value = (0..n).fold(<S as AlgebraicScalar>::ZERO, |sum, j| {
                            sum + w.good_broyden.forward.as_slice()[i * n + j]
                                * w.step.as_slice()[j]
                        });
                        w.state.trial_residual.as_mut_slice()[i] =
                            S::ONE + value * (S::ONE + S::EPSILON.sqrt() * S::HALF);
                    }
                    assert!(w.good_broyden.update(
                        &w.step,
                        crate::nonlinear_math::norm(w.step.as_slice())?,
                        &w.direction,
                        &w.state.current_residual,
                        &w.state.trial_residual,
                        session
                    )?);
                    for i in 0..n {
                        let actual = (0..n).fold(<S as AlgebraicScalar>::ZERO, |sum, j| {
                            sum + w.good_broyden.forward.as_slice()[i * n + j]
                                * w.step.as_slice()[j]
                        });
                        let expected = w.state.trial_residual.as_slice()[i] - S::ONE;
                        assert!(
                            (actual - expected).abs()
                                <= count
                                    * S::EPSILON
                                    * S::try_from_usize_exact(32)?
                                    * expected.abs().max(S::ONE)
                        );
                    }
                    assert!(w.good_broyden.direction(
                        &w.state.current_residual,
                        crate::nonlinear_math::norm(w.state.current_residual.as_slice())?,
                        &mut w.direction,
                        &mut w.lu,
                        &mut w.factor_workspace,
                        session
                    )?);
                }
                // A numerically unusable inverse cannot authorize its own step.
                w.good_broyden
                    .inverse
                    .as_mut_slice()
                    .fill(<S as AlgebraicScalar>::ZERO);
                assert!(w.good_broyden.direction(
                    &w.state.current_residual,
                    crate::nonlinear_math::norm(w.state.current_residual.as_slice())?,
                    &mut w.direction,
                    &mut w.lu,
                    &mut w.factor_workspace,
                    session
                )?);
                for i in 0..n {
                    let residual = S::ONE
                        + (0..n).fold(<S as AlgebraicScalar>::ZERO, |sum, j| {
                            sum + w.good_broyden.forward.as_slice()[i * n + j]
                                * w.direction.as_slice()[j]
                        });
                    assert!(residual.abs() <= S::SQRT_EPSILON);
                }
                Ok(())
            });
        }
    }
    #[test]
    fn general_models_and_rejected_caches_f32() {
        general_models::<f32>();
    }
    #[test]
    fn general_models_and_rejected_caches_f64() {
        general_models::<f64>();
    }
    fn exceptional_models<S: RealScalar + NumericScalar<Real = S>>() {
        prepared::<S>(1, |w, session| {
            w.jacobian.as_mut_slice()[0] = S::MIN_SUBNORMAL;
            w.good_broyden
                .initialize(&w.jacobian, &mut w.lu, &mut w.factor_workspace, session)?;
            assert!(!w.good_broyden.inverse_valid);
            w.state.current_residual.as_mut_slice()[0] = S::MIN_SUBNORMAL;
            assert!(w.good_broyden.direction(
                &w.state.current_residual,
                crate::nonlinear_math::norm(w.state.current_residual.as_slice())?,
                &mut w.direction,
                &mut w.lu,
                &mut w.factor_workspace,
                session
            )?);
            assert!((w.direction.as_slice()[0] + S::ONE).abs() <= S::EPSILON);
            w.jacobian.as_mut_slice()[0] = S::ONE;
            w.good_broyden
                .initialize(&w.jacobian, &mut w.lu, &mut w.factor_workspace, session)?;
            w.step.as_mut_slice()[0] = S::TWO;
            w.state.current_residual.as_mut_slice()[0] = -S::MAX;
            w.state.trial_residual.as_mut_slice()[0] = S::MAX;
            assert!(w.good_broyden.update(
                &w.step,
                crate::nonlinear_math::norm(w.step.as_slice())?,
                &w.direction,
                &w.state.current_residual,
                &w.state.trial_residual,
                session
            )?);
            assert_eq!(w.good_broyden.forward.as_slice()[0], S::MAX);
            let before = w.good_broyden.forward.clone();
            w.step.as_mut_slice()[0] = S::HALF;
            assert!(
                w.good_broyden
                    .update(
                        &w.step,
                        crate::nonlinear_math::norm(w.step.as_slice())?,
                        &w.direction,
                        &w.state.current_residual,
                        &w.state.trial_residual,
                        session
                    )
                    .is_err()
            );
            assert_eq!(w.good_broyden.forward, before);
            Ok(())
        });
    }
    #[test]
    fn exceptional_updates_and_unrepresentable_inverses_f32() {
        exceptional_models::<f32>();
    }
    #[test]
    fn exceptional_updates_and_unrepresentable_inverses_f64() {
        exceptional_models::<f64>();
    }
}
