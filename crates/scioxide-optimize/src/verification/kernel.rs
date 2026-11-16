//! Pure residual calculations with counted operator applications and scratch.
use super::*;
use crate::{LinearConstraintSense, context::SolveSession};

use scioxide_core::checked_sum;
use scioxide_core::{EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};

/// Map an interrupted check to an `Incomplete` report carrying the stop reason.
pub(crate) fn interrupted<S: RealScalar>(
    result: Result<VerificationReport<S>, SciOxideError>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<VerificationReport<S>, SciOxideError> {
    match result {
        Err(_) if session.stopped().is_some() => {
            let mut report = VerificationReport::empty();
            report.status = VerificationStatus::Incomplete;
            report.termination = session.stopped();
            Ok(report)
        }
        other => other,
    }
}
fn invoke<T>(
    session: Option<&SolveSession<'_, '_, '_>>,
    role: EvaluationRole,
    operation: impl FnOnce() -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    match session {
        Some(session) => session.invoke(role, operation),
        None => operation(),
    }
}
/// Check claim shapes, finiteness and workspace sizes before any operator work.
/// Infeasibility claims never inspect the Hessian.
pub(crate) fn validate_claim<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    q: Option<&QpVerificationProblem<'_, S>>,
    certificate: ProgrammingCertificate<'_, S>,
    w: &VerificationWorkspace<S>,
) -> Result<(), SciOxideError> {
    let q = q.filter(|_| !matches!(certificate, ProgrammingCertificate::Infeasibility(_)));
    validate(p, q, certificate, w)
}
fn validate<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    q: Option<&QpVerificationProblem<'_, S>>,
    certificate: ProgrammingCertificate<'_, S>,
    w: &VerificationWorkspace<S>,
) -> Result<(), SciOxideError> {
    let n = p.dimension();
    let m = p.constraints_len();
    for array in [&w.gradient, &w.transpose, &w.direction, &w.lower, &w.upper] {
        shape(n, array.len())?;
    }
    shape(m, w.rows.len())?;
    shape(m, w.row_dual.len())?;
    for op in [p.operator, q.map(|q| q.hessian())].into_iter().flatten() {
        let need = op
            .apply_workspace_len()?
            .max(op.apply_adjoint_workspace_len()?);
        if w.scratch.len() < need {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "verification operator workspace is too small",
            ));
        }
    }
    if matches!(certificate, ProgrammingCertificate::Optimality { .. })
        && q.is_some_and(|q| q.inspectable())
    {
        let matrix = w.matrix.as_ref().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "reserve explicit QP convexity workspace",
            )
        })?;
        shape(n, matrix.rows())?;
    }
    let point = match certificate {
        ProgrammingCertificate::Optimality { point, .. }
        | ProgrammingCertificate::Unboundedness { point, .. } => Some(point),
        ProgrammingCertificate::Infeasibility(_) => None,
    };
    if let Some(point) = point {
        shape(n, point.len())?;
        finite(point)?;
    }
    match certificate {
        ProgrammingCertificate::Optimality { dual, .. }
        | ProgrammingCertificate::Infeasibility(dual) => {
            shape(m, dual.rows.len())?;
            shape(n, dual.lower.len())?;
            shape(n, dual.upper.len())?;
            finite(dual.rows)?;
            finite(dual.lower)?;
            finite(dual.upper)?;
        }
        ProgrammingCertificate::Unboundedness { direction, .. } => {
            shape(n, direction.len())?;
            finite(direction)?;
        }
    }
    Ok(())
}
pub(crate) fn lp_session<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    certificate: ProgrammingCertificate<'_, S>,
    t: crate::OptimalityTolerances<S>,
    w: &mut VerificationWorkspace<S>,
    session: Option<&SolveSession<'_, '_, '_>>,
) -> Result<VerificationReport<S>, SciOxideError> {
    validate(p, None, certificate, w)?;
    if let Some(session) = session {
        session.check()?;
    }
    run(p, None, certificate, t, w, session, Some(true))
}
pub(crate) fn qp_session<S: RealScalar + NumericScalar<Real = S>>(
    q: &QpVerificationProblem<'_, S>,
    certificate: ProgrammingCertificate<'_, S>,
    t: crate::OptimalityTolerances<S>,
    w: &mut VerificationWorkspace<S>,
    session: Option<&SolveSession<'_, '_, '_>>,
) -> Result<VerificationReport<S>, SciOxideError> {
    validate(
        &q.linear,
        if matches!(certificate, ProgrammingCertificate::Infeasibility(_)) {
            None
        } else {
            Some(q)
        },
        certificate,
        w,
    )?;
    if let Some(session) = session {
        session.check()?;
    }
    let convex = if matches!(certificate, ProgrammingCertificate::Optimality { .. }) {
        convexity(q, t, w, session)?
    } else {
        None
    };
    run(&q.linear, Some(q), certificate, t, w, session, convex)
}
fn convexity<S: RealScalar + NumericScalar<Real = S>>(
    q: &QpVerificationProblem<'_, S>,
    t: crate::OptimalityTolerances<S>,
    w: &mut VerificationWorkspace<S>,
    session: Option<&SolveSession<'_, '_, '_>>,
) -> Result<Option<bool>, SciOxideError> {
    if !q.inspectable() {
        return Ok(None);
    }
    let matrix = w.matrix.as_mut().ok_or_else(|| {
        invalid(
            scioxide_core::argument!("workspace"),
            "missing convexity workspace",
        )
    })?;
    let n = q.linear.dimension();
    for i in 0..n {
        matrix.view_mut().set(i, i, q.coefficient(i, i)?)?;
        for j in i + 1..n {
            let a = q.coefficient(i, j)?;
            let b = q.coefficient(j, i)?;
            let value = if a == b {
                a
            } else {
                Scaled::from_value(a)?
                    .mul(Scaled::from_value(S::HALF)?)?
                    .add(Scaled::from_value(b)?.mul(Scaled::from_value(S::HALF)?)?)?
                    .round()?
            };
            matrix.view_mut().set(i, j, value)?;
            matrix.view_mut().set(j, i, value)?;
        }
    }
    if invoke(session, EvaluationRole::Factorization, || {
        if crate::dense::positive_gershgorin(matrix)? {
            return Ok(true);
        }
        crate::dense::positive_definite(scioxide_linalg::CholeskyDecomposition::factor_into(
            &*matrix,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::CholeskyPivoting::None,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut w.cholesky,
            &mut w.cholesky_workspace,
            session.map_or(
                Ok(scioxide_core::ExecutionContext::sequential()),
                SolveSession::execution,
            )?,
        ))
    })? {
        return Ok(Some(true));
    }
    invoke(session, EvaluationRole::Factorization, || {
        scioxide_linalg::HermitianEigenvaluesBuffer::compute_into(
            &*matrix,
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::HermitianEigenOptions::default(),
            &mut w.eigenvalues,
            &mut w.eigen_workspace,
            session.map_or(
                Ok(scioxide_core::ExecutionContext::sequential()),
                SolveSession::execution,
            )?,
        )
    })?;
    let scale = max_abs(matrix.as_slice());
    // Negative curvature within the requested rank threshold is inconclusive,
    // not a license to certify a slightly indefinite original Hessian.
    let cutoff = t.rank().threshold(scale)?;
    if w.eigenvalues.eigenvalues().iter().any(|&e| e < -cutoff) {
        return Ok(Some(false));
    }
    if w.eigenvalues
        .eigenvalues()
        .iter()
        .any(|&e| e < <S as scioxide_core::AlgebraicScalar>::ZERO)
    {
        return Ok(None);
    }
    Ok(Some(true))
}
fn max_abs<S: RealScalar>(x: &[S]) -> S {
    x.iter().fold(S::ZERO, |m, x| m.max(x.abs()))
}
fn product<S: RealScalar>(a: S, b: S) -> Result<S, SciOxideError> {
    Scaled::from_value(a)?.mul(Scaled::from_value(b)?)?.round()
}
fn add_product<S: RealScalar>(
    sum: &mut ScaledAccumulator<S>,
    a: S,
    b: S,
) -> Result<(), SciOxideError> {
    sum.add_product(Scaled::from_value(a)?, Scaled::from_value(b)?)
}
fn rows<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    x: &[S],
    w: &mut VerificationWorkspace<S>,
    session: Option<&SolveSession<'_, '_, '_>>,
) -> Result<(), SciOxideError> {
    if let Some(op) = p.operator {
        invoke(session, EvaluationRole::LinearOperator, || match session {
            Some(session) => session.with_context(|context| {
                op.apply_into_scoped(
                    x,
                    &mut w.rows,
                    scratch(&mut w.scratch, op.apply_workspace_len()?)?,
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require)
            })?,
            None => op.apply_into(
                x,
                &mut w.rows,
                scratch(&mut w.scratch, op.apply_workspace_len()?)?,
            ),
        })?;
        products_finite(&w.rows)?;
    }
    Ok(())
}
fn gradient<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    q: Option<&QpVerificationProblem<'_, S>>,
    x: &[S],
    w: &mut VerificationWorkspace<S>,
    session: Option<&SolveSession<'_, '_, '_>>,
) -> Result<S, SciOxideError> {
    w.gradient.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    if let Some(q) = q {
        invoke(session, EvaluationRole::LinearOperator, || match session {
            Some(session) => session.with_context(|context| {
                q.hessian()
                    .apply_into_scoped(
                        x,
                        &mut w.gradient,
                        scratch(&mut w.scratch, q.hessian().apply_workspace_len()?)?,
                        &mut scioxide_core::OperatorContext::new(context),
                    )
                    .and_then(scioxide_core::Invocation::require)
            })?,
            None => q.hessian().apply_into(
                x,
                &mut w.gradient,
                scratch(&mut w.scratch, q.hessian().apply_workspace_len()?)?,
            ),
        })?;
        invoke(session, EvaluationRole::LinearOperator, || match session {
            Some(session) => session.with_context(|context| {
                q.hessian()
                    .apply_adjoint_into_scoped(
                        x,
                        &mut w.transpose,
                        scratch(&mut w.scratch, q.hessian().apply_adjoint_workspace_len()?)?,
                        &mut scioxide_core::OperatorContext::new(context),
                    )
                    .and_then(scioxide_core::Invocation::require)
            })?,
            None => q.hessian().apply_adjoint_into(
                x,
                &mut w.transpose,
                scratch(&mut w.scratch, q.hessian().apply_adjoint_workspace_len()?)?,
            ),
        })?;
        products_finite(&w.gradient)?;
        products_finite(&w.transpose)?;
        for (g, &gt) in w.gradient.iter_mut().zip(&w.transpose) {
            let a = Scaled::from_value(*g)?.mul(Scaled::from_value(S::HALF)?)?;
            let b = Scaled::from_value(gt)?.mul(Scaled::from_value(S::HALF)?)?;
            *g = a.add(b)?.round()?;
        }
    }
    w.direction.copy_from_slice(&w.gradient);
    let mut objective = ScaledAccumulator::ZERO;
    for ((&xi, g), &c) in x.iter().zip(&mut w.gradient).zip(p.objective) {
        objective.add_product(
            Scaled::from_value(xi)?.mul(Scaled::from_value(S::HALF)?)?,
            Scaled::from_value(*g)?,
        )?;
        add_product(&mut objective, xi, c)?;
        *g = checked_sum(&[*g, c])?;
    }
    objective.round()
}
fn row_violation<S: RealScalar>(value: S, sense: LinearConstraintSense) -> S {
    match sense {
        LinearConstraintSense::LessEqual => value.max(S::ZERO),
        LinearConstraintSense::GreaterEqual => (-value).max(S::ZERO),
        LinearConstraintSense::Equal => value.abs(),
    }
}
fn primal<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    x: &[S],
    t: crate::OptimalityTolerances<S>,
    w: &VerificationWorkspace<S>,
    report: &mut VerificationReport<S>,
    homogeneous: bool,
) -> Result<bool, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut pass = true;
    for ((&value, &rhs), &sense) in w.rows.iter().zip(p.rhs).zip(p.senses) {
        let rhs = if homogeneous { zero } else { rhs };
        let violation = row_violation(checked_sum(&[value, -rhs])?, sense);
        report.primal_violation = report.primal_violation.max(violation);
        pass &= violation
            <= t.feasibility()
                .threshold(NumericScalar::abs(value).max(NumericScalar::abs(rhs)))?;
        if homogeneous {
            pass &= violation == zero;
        }
    }
    for (i, &value) in x.iter().enumerate() {
        let (lower, upper) = p.bound(i);
        for (bound, sign) in [
            (lower, -<S as scioxide_core::AlgebraicScalar>::ONE),
            (upper, <S as scioxide_core::AlgebraicScalar>::ONE),
        ] {
            if let Some(bound) = bound {
                let bound = if homogeneous { zero } else { bound };
                let residual = product(sign, checked_sum(&[value, -bound])?)?.max(zero);
                report.bound_violation = report.bound_violation.max(residual);
                pass &= residual
                    <= t.feasibility()
                        .threshold(NumericScalar::abs(value).max(NumericScalar::abs(bound)))?;
                if homogeneous {
                    pass &= residual == zero;
                }
            }
        }
    }
    Ok(pass)
}
fn copy_dual<S: RealScalar + NumericScalar<Real = S>>(
    dual: DualCertificate<'_, S>,
    w: &mut VerificationWorkspace<S>,
    normalize: bool,
) -> Result<bool, SciOxideError> {
    let scale = if normalize {
        max_abs(dual.rows)
            .max(max_abs(dual.lower))
            .max(max_abs(dual.upper))
    } else {
        <S as scioxide_core::AlgebraicScalar>::ONE
    };
    if scale == <S as scioxide_core::AlgebraicScalar>::ZERO {
        return Ok(false);
    }
    for (target, source) in [
        (&mut w.row_dual, dual.rows),
        (&mut w.lower, dual.lower),
        (&mut w.upper, dual.upper),
    ] {
        for (out, &value) in target.iter_mut().zip(source) {
            *out = if normalize {
                normalized(value, scale)?
            } else {
                value
            };
        }
    }
    Ok(true)
}
fn dual_checks<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    t: crate::OptimalityTolerances<S>,
    w: &VerificationWorkspace<S>,
    report: &mut VerificationReport<S>,
) -> Result<(bool, bool), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut pass = true;
    let mut exact = true;
    for (&y, &sense) in w.row_dual.iter().zip(p.senses) {
        let violation = row_violation(-y, sense);
        let violation = if matches!(sense, LinearConstraintSense::Equal) {
            zero
        } else {
            violation
        };
        report.dual_violation = report.dual_violation.max(violation);
        exact &= violation == zero;
        pass &= violation <= t.dual_feasibility().threshold(NumericScalar::abs(y))?;
    }
    for (i, (&lower, &upper)) in w.lower.iter().zip(&w.upper).enumerate() {
        let bounds = p.bound(i);
        for (y, bound) in [(lower, bounds.0), (upper, bounds.1)] {
            let violation = if bound.is_some() {
                (-y).max(zero)
            } else {
                NumericScalar::abs(y)
            };
            report.dual_violation = report.dual_violation.max(violation);
            exact &= violation == zero;
            pass &= violation <= t.dual_feasibility().threshold(NumericScalar::abs(y))?;
            if bound.is_none() {
                pass &= y == zero;
            }
        }
    }
    Ok((pass, exact))
}
fn stationarity<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    t: crate::OptimalityTolerances<S>,
    w: &mut VerificationWorkspace<S>,
    report: &mut VerificationReport<S>,
    optimality: bool,
    session: Option<&SolveSession<'_, '_, '_>>,
) -> Result<bool, SciOxideError> {
    w.transpose
        .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    if let Some(op) = p.operator {
        invoke(session, EvaluationRole::LinearOperator, || match session {
            Some(session) => session.with_context(|context| {
                op.apply_adjoint_into_scoped(
                    &w.row_dual,
                    &mut w.transpose,
                    scratch(&mut w.scratch, op.apply_adjoint_workspace_len()?)?,
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require)
            })?,
            None => op.apply_adjoint_into(
                &w.row_dual,
                &mut w.transpose,
                scratch(&mut w.scratch, op.apply_adjoint_workspace_len()?)?,
            ),
        })?;
        products_finite(&w.transpose)?;
    }
    let mut pass = true;
    for (i, (((g, &aty), &lower), &upper)) in w
        .gradient
        .iter_mut()
        .zip(&w.transpose)
        .zip(&w.lower)
        .zip(&w.upper)
        .enumerate()
    {
        let objective_scale = if optimality {
            NumericScalar::abs(p.objective[i]).max(NumericScalar::abs(w.direction[i]))
        } else {
            <S as scioxide_core::AlgebraicScalar>::ZERO
        };
        let scale = NumericScalar::abs(*g)
            .max(objective_scale)
            .max(NumericScalar::abs(aty))
            .max(NumericScalar::abs(lower))
            .max(NumericScalar::abs(upper));
        *g = checked_sum(&[*g, aty, -lower, upper])?;
        report.stationarity = report.stationarity.max(NumericScalar::abs(*g));
        pass &= NumericScalar::abs(*g) <= t.gradient().threshold(scale)?;
    }
    Ok(pass)
}
fn dual_constant<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    w: &VerificationWorkspace<S>,
) -> Result<ScaledAccumulator<S>, SciOxideError> {
    let mut value = ScaledAccumulator::ZERO;
    for (&y, &rhs) in w.row_dual.iter().zip(p.rhs) {
        add_product(&mut value, -y, rhs)?;
    }
    for (i, (&lower, &upper)) in w.lower.iter().zip(&w.upper).enumerate() {
        let bounds = p.bound(i);
        if let Some(l) = bounds.0 {
            add_product(&mut value, lower, l)?;
        }
        if let Some(u) = bounds.1 {
            add_product(&mut value, -upper, u)?;
        }
    }
    Ok(value)
}
fn residual_support<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    w: &VerificationWorkspace<S>,
    mut value: ScaledAccumulator<S>,
) -> Result<Option<S>, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for (i, &r) in w.gradient.iter().enumerate() {
        if r == zero {
            continue;
        }
        let bounds = p.bound(i);
        let bound = if r > zero { bounds.0 } else { bounds.1 };
        let Some(bound) = bound else {
            return Ok(None);
        };
        add_product(&mut value, r, bound)?;
    }
    value.round().map(Some)
}
fn complementarity<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    x: &[S],
    t: crate::OptimalityTolerances<S>,
    w: &VerificationWorkspace<S>,
    report: &mut VerificationReport<S>,
) -> Result<bool, SciOxideError> {
    let mut pass = true;
    let mut check = |y: S, value: S, rhs: S| -> Result<(), SciOxideError> {
        let r = NumericScalar::abs(product(y, checked_sum(&[value, -rhs])?)?);
        let scale =
            NumericScalar::abs(product(y, value)?).max(NumericScalar::abs(product(y, rhs)?));
        report.complementarity = report.complementarity.max(r);
        pass &= r <= t.complementarity().threshold(scale)?;
        Ok(())
    };
    for (((&y, &value), &rhs), &sense) in w.row_dual.iter().zip(&w.rows).zip(p.rhs).zip(p.senses) {
        if !matches!(sense, LinearConstraintSense::Equal) {
            check(y, value, rhs)?;
        }
    }
    for (i, &xi) in x.iter().enumerate() {
        let (l, u) = p.bound(i);
        if let Some(l) = l {
            check(w.lower[i], xi, l)?;
        }
        if let Some(u) = u {
            check(w.upper[i], xi, u)?;
        }
    }
    Ok(pass)
}
#[allow(clippy::too_many_arguments)]
fn run<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    q: Option<&QpVerificationProblem<'_, S>>,
    certificate: ProgrammingCertificate<'_, S>,
    t: crate::OptimalityTolerances<S>,
    w: &mut VerificationWorkspace<S>,
    session: Option<&SolveSession<'_, '_, '_>>,
    convex: Option<bool>,
) -> Result<VerificationReport<S>, SciOxideError> {
    let mut report = VerificationReport::empty();
    report.convexity_checked = convex.is_some();
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    match certificate {
        ProgrammingCertificate::Optimality { point, dual } => {
            copy_dual(dual, w, false)?;
            rows(p, point, w, session)?;
            let mut pass = primal(p, point, t, w, &mut report, false)?;
            let objective = gradient(p, q, point, w, session)?;
            report.primal_objective = Some(objective);
            let mut dual_value = dual_constant(p, w)?;
            // -xᵀQx/2, with the linear part removed from the computed gradient.
            for (&xi, &qx) in point.iter().zip(&w.direction) {
                dual_value.add_product(
                    Scaled::from_value(-xi)?.mul(Scaled::from_value(S::HALF)?)?,
                    Scaled::from_value(qx)?,
                )?;
            }
            let raw_dual = dual_value.round()?;
            let gap = checked_sum(&[objective, -raw_dual])?;
            report.objective_gap = Some(gap);
            pass &= NumericScalar::abs(gap)
                <= t.gap()
                    .threshold(NumericScalar::abs(objective).max(NumericScalar::abs(raw_dual)))?;
            let (dual_pass, exact_signs) = dual_checks(p, t, w, &mut report)?;
            pass &= dual_pass;
            pass &= complementarity(p, point, t, w, &mut report)?;
            pass &= stationarity(p, t, w, &mut report, true, session)?;
            if exact_signs && convex == Some(true) {
                report.dual_bound = residual_support(p, w, dual_value)?;
            }
            report.status = if !pass || convex == Some(false) {
                VerificationStatus::Rejected
            } else if convex.is_none() {
                VerificationStatus::KktOnly
            } else {
                VerificationStatus::Accepted
            };
        }
        ProgrammingCertificate::Infeasibility(dual) => {
            if !copy_dual(dual, w, true)? {
                return Ok(report);
            }
            w.gradient.fill(zero);
            let (mut pass, exact_signs) = dual_checks(p, t, w, &mut report)?;
            pass &= stationarity(p, t, w, &mut report, false, session)?;
            let constant = dual_constant(p, w)?;
            let constant_scale = NumericScalar::abs(constant.round()?);
            report.witness_margin = residual_support(p, w, constant)?;
            pass &= exact_signs && report.witness_margin.is_some_and(|margin| margin > zero);
            if let Some(margin) = report.witness_margin {
                pass &= margin > t.gap().threshold(constant_scale)?;
            }
            report.status = if pass {
                VerificationStatus::Accepted
            } else {
                VerificationStatus::Rejected
            };
        }
        ProgrammingCertificate::Unboundedness { point, direction } => {
            rows(p, point, w, session)?;
            let mut pass = primal(p, point, t, w, &mut report, false)?;
            report.primal_objective = Some(gradient(p, q, point, w, session)?);
            let norm = max_abs(direction);
            if norm == zero {
                return Ok(report);
            }
            for (out, &d) in w.direction.iter_mut().zip(direction) {
                *out = normalized(d, norm)?;
            }
            // Keep the normalized direction in owned scratch while applying products.
            if let Some(op) = p.operator {
                invoke(session, EvaluationRole::LinearOperator, || match session {
                    Some(session) => session.with_context(|context| {
                        op.apply_into_scoped(
                            &w.direction,
                            &mut w.rows,
                            scratch(&mut w.scratch, op.apply_workspace_len()?)?,
                            &mut scioxide_core::OperatorContext::new(context),
                        )
                        .and_then(scioxide_core::Invocation::require)
                    })?,
                    None => op.apply_into(
                        &w.direction,
                        &mut w.rows,
                        scratch(&mut w.scratch, op.apply_workspace_len()?)?,
                    ),
                })?;
                products_finite(&w.rows)?;
            }
            pass &= primal(p, &w.direction, t, w, &mut report, true)?;
            if let Some(q) = q {
                invoke(session, EvaluationRole::LinearOperator, || match session {
                    Some(session) => session.with_context(|context| {
                        q.hessian()
                            .apply_into_scoped(
                                &w.direction,
                                &mut w.gradient,
                                scratch(&mut w.scratch, q.hessian().apply_workspace_len()?)?,
                                &mut scioxide_core::OperatorContext::new(context),
                            )
                            .and_then(scioxide_core::Invocation::require)
                    })?,
                    None => q.hessian().apply_into(
                        &w.direction,
                        &mut w.gradient,
                        scratch(&mut w.scratch, q.hessian().apply_workspace_len()?)?,
                    ),
                })?;
                invoke(session, EvaluationRole::LinearOperator, || match session {
                    Some(session) => session.with_context(|context| {
                        q.hessian()
                            .apply_adjoint_into_scoped(
                                &w.direction,
                                &mut w.transpose,
                                scratch(
                                    &mut w.scratch,
                                    q.hessian().apply_adjoint_workspace_len()?,
                                )?,
                                &mut scioxide_core::OperatorContext::new(context),
                            )
                            .and_then(scioxide_core::Invocation::require)
                    })?,
                    None => q.hessian().apply_adjoint_into(
                        &w.direction,
                        &mut w.transpose,
                        scratch(&mut w.scratch, q.hessian().apply_adjoint_workspace_len()?)?,
                    ),
                })?;
                products_finite(&w.gradient)?;
                products_finite(&w.transpose)?;
                for (&a, &b) in w.gradient.iter().zip(&w.transpose) {
                    let value = Scaled::from_value(a)?
                        .mul(Scaled::from_value(S::HALF)?)?
                        .add(Scaled::from_value(b)?.mul(Scaled::from_value(S::HALF)?)?)?
                        .round()?;
                    report.stationarity = report.stationarity.max(NumericScalar::abs(value));
                    pass &= value == zero;
                }
            }
            let descent = -crate::utils::dot(p.objective, &w.direction)?;
            report.witness_margin = Some(descent);
            pass &= descent > t.gap().threshold(NumericScalar::abs(descent))?;
            report.status = if pass {
                VerificationStatus::Accepted
            } else {
                VerificationStatus::Rejected
            };
        }
    }
    if let Some(session) = session {
        session.check()?;
    }
    Ok(report)
}

/// Reject a claimed result whose termination or reported objective disagrees
/// with the independently verified certificate.
pub(crate) fn reported_claim<S: RealScalar + NumericScalar<Real = S>>(
    certificate: ProgrammingCertificate<'_, S>,
    termination: crate::OptimizeTermination,
    objective: S,
    mut report: VerificationReport<S>,
    t: crate::OptimalityTolerances<S>,
) -> Result<VerificationReport<S>, SciOxideError> {
    let matches = matches!(
        (certificate, termination),
        (
            ProgrammingCertificate::Optimality { .. },
            crate::OptimizeTermination::Converged
        ) | (
            ProgrammingCertificate::Infeasibility(_),
            crate::OptimizeTermination::Infeasible
        ) | (
            ProgrammingCertificate::Unboundedness { .. },
            crate::OptimizeTermination::Unbounded
        )
    );
    if !matches {
        report.status = VerificationStatus::Rejected;
    }
    if let Some(reconstructed) = report.primal_objective {
        finite(&[objective])?;
        let error = NumericScalar::abs(checked_sum(&[objective, -reconstructed])?);
        if error > t.gap().threshold(NumericScalar::abs(reconstructed))? {
            report.status = VerificationStatus::Rejected;
        }
    }
    Ok(report)
}

fn normalized<S: RealScalar>(value: S, maximum: S) -> Result<S, SciOxideError> {
    let scioxide_core::FloatDecomposition::Finite {
        significand: mantissa,
        exponent,
        ..
    } = maximum.decompose()
    else {
        return Err(scioxide_core::SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "finite scalar required",
        ));
    };
    let shift = i64::from(exponent) + i64::from(63 - mantissa.leading_zeros());
    let result = Scaled::from_value(value)?.shifted(-shift)?.round()?;
    if value != S::ZERO && result == S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("certificate"),
            "witness normalization loses a nonzero component",
        ));
    }
    Ok(result)
}

fn scratch<S>(workspace: &mut [S], required: usize) -> Result<&mut [S], SciOxideError> {
    workspace.get_mut(..required).ok_or_else(|| {
        invalid(
            scioxide_core::argument!("workspace"),
            "operator workspace requirement changed",
        )
    })
}

fn products_finite<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Callback,
            "operator returned nonfinite values",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn result_corruption_is_independently_rejected() {
        let problem = crate::LpProblem::new(alloc::vec![1f64]).unwrap();
        let view = LpVerificationProblem::from_problem(&problem);
        let verifier = crate::ProgrammingVerifier::default();
        let result = crate::SimplexSolver::default().solve(&problem).unwrap();
        assert!(verifier.verify(&view, &result).unwrap().is_accepted());
        let mut corrupt = result.clone();
        corrupt.fun = 1.;
        assert!(!verifier.verify(&view, &corrupt).unwrap().is_accepted());
        let mut corrupt = result.clone();
        corrupt.termination = crate::OptimizeTermination::Unbounded;
        assert!(!verifier.verify(&view, &corrupt).unwrap().is_accepted());
        let mut corrupt = result;
        corrupt.evidence = None;
        assert!(!verifier.verify(&view, &corrupt).unwrap().is_accepted());
    }
}
