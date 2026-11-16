//! Original-model incumbent checks, shared by all primal heuristics.
use super::*;
use crate::LinearConstraintSense as Sense;

pub(super) struct Candidate<S> {
    pub x: Vec<S>,
    pub fun: S,
    pub primal: S,
    pub bounds: S,
    pub integrality: S,
}

pub(super) fn distance<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    Ok(checked_difference(value, value.round())?.abs())
}
pub(super) fn row_violation<S: RealScalar>(
    lhs: S,
    rhs: S,
    sense: Sense,
) -> Result<S, SciOxideError> {
    let d = checked_difference(lhs, rhs)?;
    Ok(match sense {
        Sense::LessEqual => d.max(S::ZERO),
        Sense::GreaterEqual => (-d).max(S::ZERO),
        Sense::Equal => d.abs(),
    })
}
pub(super) fn candidate<S: RealScalar + NumericScalar<Real = S>>(
    p: &MipProblem<S>,
    x: Vec<S>,
    round: bool,
    feasibility: scioxide_core::FeasibilityTolerance<S>,
) -> Result<Option<Candidate<S>>, SciOxideError> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut c = Candidate {
        x,
        fun: z,
        primal: z,
        bounds: z,
        integrality: z,
    };
    Ok(if check_candidate(p, &mut c, round, feasibility)? {
        Some(c)
    } else {
        None
    })
}
fn check_candidate<S: RealScalar + NumericScalar<Real = S>>(
    p: &MipProblem<S>,
    c: &mut Candidate<S>,
    round: bool,
    feasibility: scioxide_core::FeasibilityTolerance<S>,
) -> Result<bool, SciOxideError> {
    let x = &mut c.x;
    if x.len() != p.relaxation().dimension() || x.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            scioxide_core::argument!("lp"),
            "malformed incumbent point",
        ));
    }
    if round {
        for &j in p.integer_variables() {
            x[j] = RealScalar::round(x[j]);
        }
    }
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut primal = z;
    let mut bounds = z;
    let mut integrality = z;
    for &j in p.integer_variables() {
        integrality = integrality.max(distance(x[j])?);
    }
    if integrality != z {
        return Ok(false);
    }
    let lp = p.relaxation();
    if let Some(c) = lp.constraints() {
        for i in 0..c.len() {
            let mut sum = scioxide_core::ProductSum::new();
            for (j, a) in c.coefficients().row_entries(i)? {
                sum.add_product(a, x[j])?;
            }
            let lhs = sum.finish()?;
            let v = row_violation(lhs, c.rhs()[i], c.senses()[i])?;
            if v > feasibility
                .threshold(NumericScalar::abs(lhs).max(NumericScalar::abs(c.rhs()[i])))?
            {
                return Ok(false);
            }
            primal = primal.max(v);
        }
    }
    for (j, &value) in x.iter().enumerate() {
        let (lo, hi) = lp.bounds().map_or((Some(z), None), |b| b.as_slice()[j]);
        for (endpoint, sense) in [(lo, Sense::GreaterEqual), (hi, Sense::LessEqual)] {
            if let Some(endpoint) = endpoint {
                let v = row_violation(value, endpoint, sense)?;
                if v > feasibility
                    .threshold(NumericScalar::abs(value).max(NumericScalar::abs(endpoint)))?
                {
                    return Ok(false);
                }
                bounds = bounds.max(v);
            }
        }
    }
    let fun = crate::utils::dot(lp.objective(), x)?;
    c.fun = fun;
    c.primal = primal.max(bounds);
    c.bounds = bounds;
    c.integrality = integrality;
    Ok(true)
}

pub(super) fn node_candidate_into<'a, S: 'a + RealScalar + NumericScalar<Real = S>>(
    original: &MipProblem<S>,
    model: &model::Model<S>,
    rows: impl Iterator<Item = &'a model::Row<S>>,
    point: &[S],
    c: &mut Candidate<S>,
    feasibility: scioxide_core::FeasibilityTolerance<S>,
) -> Result<bool, SciOxideError> {
    model.map.reconstruct_into(point, &mut c.x)?;
    if !check_candidate(original, c, true, feasibility)? {
        return Ok(false);
    }
    for row in rows {
        let mut sum = scioxide_core::ProductSum::new();
        for &(j, a) in &row.entries {
            sum.add_product(a, c.x[model.map.active[j]])?;
        }
        let lhs = sum.finish()?;
        if row_violation(lhs, row.rhs, row.sense)?
            > feasibility.threshold(NumericScalar::abs(lhs).max(NumericScalar::abs(row.rhs)))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn result<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &MipSearchState<S, W>,
    termination: OptimizeTermination,
    message: &str,
) -> Result<MipResult<S>, SciOxideError> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let lower = global_bound(s);
    let candidate = s.incumbent.as_ref().or(s.partial.as_ref());
    let mut r = MipResult::empty();
    r.termination = termination;
    r.message = message.into();
    r.nodes = s.stats.nodes;
    r.iterations = s.stats.iterations;
    r.search.statistics = s.stats;
    r.search.has_incumbent = s.incumbent.is_some();
    r.search.lower_bound = lower;
    if let Some(c) = candidate {
        r.x = copy(&c.x, scioxide_core::quantity!("mip_result"))?;
        r.fun = c.fun;
        r.diagnostics = crate::OptimalityDiagnostics {
            primal_feasibility: Some(c.primal),
            bound_violation: Some(c.bounds),
            integrality_violation: Some(c.integrality),
            ..crate::OptimalityDiagnostics::unavailable()
        };
    }
    if let Some(c) = &s.incumbent {
        r.trust = if termination == OptimizeTermination::Converged {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::Admissible
        };
        if let Some(lb) = lower {
            let gap = checked_difference(c.fun, lb)?.max(z);
            r.search.absolute_gap = Some(gap);
            r.search.relative_gap = Some(
                gap / NumericScalar::abs(c.fun)
                    .max(NumericScalar::abs(lb))
                    .max(<S as scioxide_core::AlgebraicScalar>::ONE),
            );
        }
    }
    Ok(r)
}
pub(super) fn global_bound<S: RealScalar + NumericScalar<Real = S>, W>(
    s: &MipSearchState<S, W>,
) -> Option<S> {
    let mut lower = s.incumbent.as_ref().map(|c| c.fun);
    if let Some(b) = s.closed_bound {
        lower = Some(lower.map_or(b, |v| v.min(b)));
    }
    for node in s.queue.first().into_iter().chain(s.active.iter()) {
        let bound = node.lower_bound?;
        lower = Some(lower.map_or(bound, |v| v.min(bound)));
    }
    lower
}
