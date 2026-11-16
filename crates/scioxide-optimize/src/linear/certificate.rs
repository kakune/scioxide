//! LP certificate construction and independent verification.
use crate::verification::{
    OwnedProgrammingCertificate,
    candidate::{self, Assessment},
    kernel::lp_session,
};
use crate::{
    LpProblem, LpVerificationProblem, OptimalityTolerances, OptimizeTermination,
    VerificationWorkspace, context::SolveSession,
};
use alloc::vec::Vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

pub(super) fn assess<S: RealScalar + NumericScalar<Real = S>>(
    problem: &LpProblem<S>,
    point: &[S],
    t: OptimalityTolerances<S>,
    termination: OptimizeTermination,
    row_dual: Option<Vec<S>>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<Assessment<S>, SciOxideError> {
    if session.termination()?.is_some() {
        return Ok(Assessment::unavailable());
    }
    let outcome = (|| {
        let p = &LpVerificationProblem::from_problem(problem);
        let farkas = matches!(termination, OptimizeTermination::Infeasible);
        let dual = match row_dual {
            Some(rows) => {
                candidate::from_rows(p, point, problem.objective(), rows, t, farkas, session)?
            }
            None => candidate::dual(p, point, problem.objective(), t, farkas, session)?,
        };
        let evidence = if farkas {
            OwnedProgrammingCertificate::Infeasibility(dual)
        } else {
            OwnedProgrammingCertificate::Optimality(dual)
        };
        let mut workspace = VerificationWorkspace::for_lp(p)?;
        let report = lp_session(p, evidence.borrow(point), t, &mut workspace, Some(session))?;
        Ok(Assessment::new(evidence, report))
    })();
    match outcome {
        Err(_) if session.stopped().is_some() => Ok(Assessment::unavailable()),
        other => other,
    }
}

pub(super) fn ray<S: RealScalar + NumericScalar<Real = S>>(
    problem: &LpProblem<S>,
    point: &[S],
    direction: Vec<S>,
    t: OptimalityTolerances<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<Assessment<S>, SciOxideError> {
    let evidence = OwnedProgrammingCertificate::Unboundedness(direction);
    let p = &LpVerificationProblem::from_problem(problem);
    let mut workspace = VerificationWorkspace::for_lp(p)?;
    let report = lp_session(p, evidence.borrow(point), t, &mut workspace, Some(session))?;
    Ok(Assessment::new(evidence, report))
}
