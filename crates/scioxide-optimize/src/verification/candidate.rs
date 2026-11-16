//! Solver-side certificate construction. No acceptance decision is made here.
use super::{LpVerificationProblem, OwnedDualCertificate};
use crate::{LinearConstraintSense, context::SolveSession};
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_product, try_filled, try_with_capacity};
use scioxide_core::checked_sum;
use scioxide_core::{EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, Vector};

#[derive(Clone, Copy)]
enum Source {
    Row(usize),
    Lower(usize),
    Upper(usize),
}
struct Row<S> {
    coefficients: Vec<S>,
    rhs: S,
    equality: bool,
    sign: S,
    source: Source,
}

/// Recover candidate multipliers from the active original rows, or construct a
/// Farkas candidate using all rows and a unit contradiction. Rank-revealing solves
/// are merely witness producers; the independent verifier must check the result.
pub(crate) fn dual<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    point: &[S],
    gradient: &[S],
    t: crate::OptimalityTolerances<S>,
    farkas: bool,
    session: &SolveSession<'_, '_, '_>,
) -> Result<OwnedDualCertificate<S>, SciOxideError> {
    let n = p.dimension();
    let m = p.constraints_len();
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let capacity = checked_add(
        m,
        checked_product(&[2, n], scioxide_core::quantity!("candidate_bound_rows"))?,
        scioxide_core::quantity!("candidate_rows"),
    )?;
    let mut rows = try_with_capacity(capacity, scioxide_core::quantity!("candidate_rows"))?;
    if let Some(op) = p.operator {
        let mut unit = try_filled(m, zero, scioxide_core::quantity!("candidate_row_selector"))?;
        let mut scratch = try_filled(
            op.apply_adjoint_workspace_len()?,
            zero,
            scioxide_core::quantity!("candidate_row_scratch"),
        )?;
        for i in 0..m {
            unit.fill(zero);
            unit[i] = one;
            let mut coefficients = try_filled(n, zero, scioxide_core::quantity!("candidate_row"))?;
            session.invoke(EvaluationRole::LinearOperator, || {
                session.with_context(|context| {
                    op.apply_adjoint_into_scoped(
                        &unit,
                        &mut coefficients,
                        &mut scratch,
                        &mut scioxide_core::OperatorContext::new(context),
                    )
                    .and_then(scioxide_core::Invocation::require)
                })?
            })?;
            super::finite(&coefficients)?;
            let sign = if matches!(p.senses[i], LinearConstraintSense::GreaterEqual) {
                -one
            } else {
                one
            };
            coefficients.iter_mut().for_each(|c| *c *= sign);
            rows.push(Row {
                coefficients,
                rhs: p.rhs[i] * sign,
                equality: matches!(p.senses[i], LinearConstraintSense::Equal),
                sign,
                source: Source::Row(i),
            });
        }
    }
    for i in 0..n {
        let (l, u) = p.bound(i);
        for (bound, sign, source) in [(l, -one, Source::Lower(i)), (u, one, Source::Upper(i))] {
            if let Some(bound) = bound {
                let mut coefficients =
                    try_filled(n, zero, scioxide_core::quantity!("candidate_bound_row"))?;
                coefficients[i] = sign;
                rows.push(Row {
                    coefficients,
                    rhs: bound * sign,
                    equality: false,
                    sign: one,
                    source,
                });
            }
        }
    }
    let mut active = try_with_capacity(
        rows.len(),
        scioxide_core::quantity!("candidate_active_rows"),
    )?;
    for (i, row) in rows.iter().enumerate() {
        let value = crate::utils::dot(&row.coefficients, point)?;
        let residual = checked_sum(&[value, -row.rhs])?;
        if farkas
            || row.equality
            || NumericScalar::abs(residual)
                <= t.feasibility()
                    .threshold(NumericScalar::abs(value).max(NumericScalar::abs(row.rhs)))?
        {
            active.push(i);
        }
    }
    let equations = checked_add(
        n,
        usize::from(farkas),
        scioxide_core::quantity!("candidate_equations"),
    )?;
    let mut rhs = try_filled(equations, zero, scioxide_core::quantity!("candidate_rhs"))?;
    if farkas {
        rhs[n] = one;
    } else {
        for (r, &g) in rhs.iter_mut().zip(gradient) {
            *r = -g;
        }
    }
    let rhs = Vector::from_vec(rhs);
    let mut multipliers = try_filled(
        rows.len(),
        zero,
        scioxide_core::quantity!("candidate_multipliers"),
    )?;
    while !active.is_empty() {
        let mut matrix = Matrix::zeros(equations, active.len())?;
        for (column, &index) in active.iter().enumerate() {
            for (i, &a) in rows[index].coefficients.iter().enumerate() {
                matrix.view_mut().set(i, column, a)?;
            }
            if farkas {
                matrix.view_mut().set(n, column, -rows[index].rhs)?;
            }
        }
        let solution = crate::dense::svd_solve_with_policy(
            &matrix,
            &rhs,
            scioxide_linalg::RankPolicy::Threshold(t.rank()),
            session,
        )?;
        let negative = active
            .iter()
            .zip(solution.as_slice())
            .filter(|(i, y)| !rows[**i].equality && **y < zero)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(&i, _)| i);
        if let Some(index) = negative {
            active.retain(|&i| i != index);
            continue;
        }
        for (&i, &y) in active.iter().zip(solution.as_slice()) {
            multipliers[i] = y;
        }
        break;
    }
    let mut result = OwnedDualCertificate {
        rows: try_filled(m, zero, scioxide_core::quantity!("original_dual_rows"))?,
        lower: try_filled(n, zero, scioxide_core::quantity!("original_lower_dual"))?,
        upper: try_filled(n, zero, scioxide_core::quantity!("original_upper_dual"))?,
    };
    for (row, &y) in rows.iter().zip(&multipliers) {
        match row.source {
            Source::Row(i) => result.rows[i] = y * row.sign,
            Source::Lower(i) => result.lower[i] = y,
            Source::Upper(i) => result.upper[i] = y,
        }
    }
    repair_bounds(p, point, gradient, t, farkas, session, &mut result)?;
    Ok(result)
}

pub(crate) struct Assessment<S: RealScalar> {
    pub(crate) diagnostics: crate::OptimalityDiagnostics<S>,
    pub(crate) accepted: bool,
    pub(crate) evidence: Option<super::OwnedProgrammingCertificate<S>>,
    pub(crate) report: Option<super::VerificationReport<S>>,
}
impl<S: RealScalar> Assessment<S> {
    pub(crate) fn unavailable() -> Self {
        Self {
            diagnostics: crate::OptimalityDiagnostics::unavailable(),
            accepted: false,
            evidence: None,
            report: None,
        }
    }
    pub(crate) fn new(
        evidence: super::OwnedProgrammingCertificate<S>,
        report: super::VerificationReport<S>,
    ) -> Self {
        let diagnostics = match &evidence {
            super::OwnedProgrammingCertificate::Optimality(_) => report.diagnostics(),
            super::OwnedProgrammingCertificate::Infeasibility(_) => crate::OptimalityDiagnostics {
                stationarity: Some(report.stationarity),
                stationarity_norm: scioxide_core::NormSelection::Infinity,
                dual_feasibility: Some(report.dual_violation),
                ..crate::OptimalityDiagnostics::unavailable()
            },
            super::OwnedProgrammingCertificate::Unboundedness(_) => crate::OptimalityDiagnostics {
                primal_feasibility: Some(report.primal_violation),
                bound_violation: Some(report.bound_violation),
                ..crate::OptimalityDiagnostics::unavailable()
            },
        };
        Self {
            diagnostics,
            accepted: report.is_accepted(),
            evidence: Some(evidence),
            report: Some(report),
        }
    }
}
pub(crate) fn assess_qp<S: RealScalar + NumericScalar<Real = S>>(
    problem: &crate::QpProblem<S>,
    point: &[S],
    gradient: &[S],
    t: crate::OptimalityTolerances<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<Assessment<S>, SciOxideError> {
    let p = super::QpVerificationProblem::from_problem(problem);
    let candidate = dual(&p.linear, point, gradient, t, false, session)?;
    let evidence = super::OwnedProgrammingCertificate::Optimality(candidate);
    let mut workspace = super::VerificationWorkspace::for_qp(&p)?;
    let report =
        super::kernel::qp_session(&p, evidence.borrow(point), t, &mut workspace, Some(session))?;
    Ok(Assessment::new(evidence, report))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn from_rows<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    point: &[S],
    gradient: &[S],
    rows: Vec<S>,
    t: crate::OptimalityTolerances<S>,
    farkas: bool,
    session: &SolveSession<'_, '_, '_>,
) -> Result<OwnedDualCertificate<S>, SciOxideError> {
    let n = p.dimension();
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut result = OwnedDualCertificate {
        rows,
        lower: try_filled(n, zero, scioxide_core::quantity!("candidate_lower_dual"))?,
        upper: try_filled(n, zero, scioxide_core::quantity!("candidate_upper_dual"))?,
    };
    repair_bounds(p, point, gradient, t, farkas, session, &mut result)?;
    Ok(result)
}
#[allow(clippy::too_many_arguments)]
fn repair_bounds<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpVerificationProblem<'_, S>,
    point: &[S],
    gradient: &[S],
    t: crate::OptimalityTolerances<S>,
    farkas: bool,
    session: &SolveSession<'_, '_, '_>,
    result: &mut OwnedDualCertificate<S>,
) -> Result<(), SciOxideError> {
    let n = p.dimension();
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut reduced = try_filled(
        n,
        zero,
        scioxide_core::quantity!("candidate_original_reduced_costs"),
    )?;
    if let Some(op) = p.operator {
        let mut scratch = try_filled(
            op.apply_adjoint_workspace_len()?,
            zero,
            scioxide_core::quantity!("candidate_reduced_cost_scratch"),
        )?;
        session.invoke(EvaluationRole::LinearOperator, || {
            session.with_context(|context| {
                op.apply_adjoint_into_scoped(
                    &result.rows,
                    &mut reduced,
                    &mut scratch,
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require)
            })?
        })?;
        super::finite(&reduced)?;
    }
    for (i, r) in reduced.iter_mut().enumerate() {
        if !farkas {
            *r = checked_sum(&[*r, gradient[i]])?;
        }
        let (lower, upper) = p.bound(i);
        let active = |bound: Option<S>| -> Result<bool, SciOxideError> {
            match bound {
                None => Ok(false),
                Some(b) => Ok(farkas
                    || NumericScalar::abs(checked_sum(&[point[i], -b])?)
                        <= t.feasibility()
                            .threshold(NumericScalar::abs(point[i]).max(NumericScalar::abs(b)))?),
            }
        };
        if *r >= zero && active(lower)? {
            result.lower[i] = *r;
            result.upper[i] = zero;
        } else if *r <= zero && active(upper)? {
            result.upper[i] = -*r;
            result.lower[i] = zero;
        }
    }
    Ok(())
}
