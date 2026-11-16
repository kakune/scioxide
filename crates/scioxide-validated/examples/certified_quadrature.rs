//! Independently certify the integral of exp(x) on [0,1].
use scioxide_core::{Complex, ErrorTolerance, ExecutionContext, NormSelection, SciOxideError};
use scioxide_validated::IntervalBox;
use scioxide_validated::{
    DecoratedInterval as D, ProofRequest, QuadratureAssumption, QuadratureCertificate,
    QuadratureObligation, QuadraturePiece, QuadratureQuery, QuadratureRequest,
    QuadratureTermination, QuadratureVerificationStatus, QuadratureVerifier, ValidatedOptions,
};

fn main() -> Result<(), SciOxideError> {
    let answer = scioxide_validated::integrate(
        |x| Ok(x.exp()?.hull()),
        |x| Ok(x.exp()?.hull()),
        0.0_f64,
        1.0,
    )?;
    assert!(answer.contains(core::f64::consts::E - 1.0));

    let domain = D::new(0.0_f64, 1.0)?.set();
    let pieces = [QuadraturePiece {
        domain,
        assumption: QuadratureAssumption::EightTimesContinuouslyDifferentiable,
    }];
    let oracle = |q: QuadratureRequest<f64>, out: &mut [IntervalBox<f64>]| {
        let value = match q.query {
            QuadratureQuery::Value | QuadratureQuery::EighthDerivative => {
                D::from(q.domain).exp()?.hull()
            }
            _ => return Ok(QuadratureObligation::Unavailable),
        };
        out[0] = Complex::new(value.set(), D::point(0.)?.set());
        Ok(QuadratureObligation::Established)
    };
    let plan = QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, ValidatedOptions::new())?;
    // The certificate's panel capacity bounds refinement; the workspace is sized for it.
    let mut certificate = QuadratureCertificate::new(16, 1)?;
    let mut work = plan.workspace(ProofRequest::refinement(certificate.capacity()))?;
    let tolerance = ErrorTolerance::scalar(1e-12, 0., NormSelection::Infinity)?;
    let report = plan.generate_into(
        &tolerance,
        &mut certificate,
        &mut work,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(
        report.termination(),
        QuadratureTermination::ToleranceReached
    );
    assert_eq!(
        plan.verify_scoped(
            certificate.view(),
            &mut work,
            ExecutionContext::sequential()
        )?,
        QuadratureVerificationStatus::Verified
    );
    assert!(
        certificate.view().total[0]
            .re
            .contains(core::f64::consts::E - 1.0)
    );
    Ok(())
}
