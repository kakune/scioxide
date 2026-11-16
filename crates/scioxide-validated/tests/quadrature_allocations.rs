//! Prepared paths must allocate nothing; reported payload must match allocations.
#![cfg(test)]

use scioxide_core::{Complex, ErrorTolerance, ExecutionContext, NormSelection};
use scioxide_validated::IntervalBox;
use scioxide_validated::{
    DecoratedInterval as D, ProofRequest, QuadratureAssumption, QuadratureCertificate,
    QuadratureObligation, QuadraturePiece, QuadratureQuery, QuadratureRequest,
    QuadratureTermination, QuadratureVerificationStatus, QuadratureVerifier, ValidatedOptions,
};

#[test]
fn gauss4_generation_and_replay_execute_after_reservation() {
    let domain = D::new(0_f64, 1.).unwrap().set();
    let pieces = [QuadraturePiece {
        domain,
        assumption: QuadratureAssumption::EightTimesContinuouslyDifferentiable,
    }];
    let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| match q.query {
        QuadratureQuery::Value | QuadratureQuery::EighthDerivative => {
            let value = D::from(q.domain).exp()?.hull();
            out.fill(Complex::new(value.set(), value.neg()?.set()));
            Ok(QuadratureObligation::Established)
        }
        _ => Ok(QuadratureObligation::Unavailable),
    };
    let plan =
        QuadratureVerifier::<Complex<f64>>::new(&pieces, 2, &oracle, ValidatedOptions::new())
            .unwrap();
    let mut workspace = plan.workspace(ProofRequest::refinement(32)).unwrap();
    workspace
        .try_reserve(
            &plan
                .workspace_requirements(ProofRequest::proofs(3))
                .unwrap(),
        )
        .unwrap();
    let mut certificate = QuadratureCertificate::new(32, 2).unwrap();
    let tolerance = ErrorTolerance::scalar(1e-12, 0., NormSelection::Infinity).unwrap();
    for _ in 0..2 {
        let report = plan
            .generate_into(
                &tolerance,
                &mut certificate,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached
        );
        let view = certificate.view();
        let mut statuses = [QuadratureVerificationStatus::Verified; 3];
        plan.verify_batch_into(
            &[view; 3],
            &mut statuses,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(statuses, [QuadratureVerificationStatus::Verified; 3]);
    }
}

#[test]
fn bisecting_panels_without_finite_bounds_allocates_nothing_after_reservation() {
    // The natural extension of 1 / (x·x − x + 1) is unbounded on [0, 2], [0, 1] and [1, 2].
    let domain = D::new(0_f64, 2.).unwrap().set();
    let pieces = [QuadraturePiece {
        domain,
        assumption: QuadratureAssumption::IntegrableRange,
    }];
    let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        let x = D::from(q.domain);
        let one = D::point(1.)?;
        let value = x.mul(&x)?.sub(&x)?.add(&one)?.reciprocal()?;
        out.fill(Complex::new(value.set(), D::point(0.)?.set()));
        Ok(QuadratureObligation::Established)
    };
    let plan =
        QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, ValidatedOptions::new()).unwrap();
    let mut workspace = plan.workspace(ProofRequest::refinement(512)).unwrap();
    let mut certificate = QuadratureCertificate::new(512, 1).unwrap();
    let tolerance = ErrorTolerance::scalar(1e-2, 0., NormSelection::Infinity).unwrap();
    for _ in 0..2 {
        let report = plan
            .generate_into(
                &tolerance,
                &mut certificate,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            QuadratureTermination::ToleranceReached
        );
        let status = plan
            .verify_scoped(
                certificate.view(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(status, QuadratureVerificationStatus::Verified);
    }
}
