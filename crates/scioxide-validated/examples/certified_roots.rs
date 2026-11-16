//! Ordinary root candidates are inputs to a separate rigorous count verifier.
use scioxide_core::SciOxideError;
use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions};
use scioxide_validated::{
    DecoratedInterval, PolynomialRootPlan, RootClaim, RootDisk, ScalarRootCertificate,
    ScalarRootVerifier, ValidatedOptions,
};

#[expect(
    clippy::approx_constant,
    reason = "illustrate certification from deliberately rounded root candidates"
)]
fn main() -> Result<(), SciOxideError> {
    let approximations = [
        scioxide_core::Complex::new(-1.4142, 0.0),
        scioxide_core::Complex::new(1.4142, 0.0),
    ];
    let answer = scioxide_validated::polynomial_roots(&[-2.0_f64, 0.0, 1.0], &approximations)?;
    assert_eq!(answer.view().iter().map(|c| c.count).sum::<usize>(), 2);

    let coefficients = [-2.0_f64, 0., 1.];
    let ordinary = scioxide_rootfind::polynomial_roots_with_options(
        &coefficients,
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth),
    )?;
    let options = ValidatedOptions::new();
    let plan = PolynomialRootPlan::new(&coefficients, options)?;
    let seeds: Vec<_> = ordinary
        .roots()
        .iter()
        .map(|&center| RootDisk {
            center,
            radius: 0.125,
        })
        .collect();
    let (report, cover) = plan.isolate(&seeds)?;
    assert!(report.is_complete());
    assert!(plan.verify_cover(cover.view())?.is_proved());

    let value = |x: DecoratedInterval<f64>| x.square()?.sub(&DecoratedInterval::point(2.)?);
    let derivative = |x: DecoratedInterval<f64>| x.mul(&DecoratedInterval::point(2.)?);
    let scalar = ScalarRootVerifier::new(&value, &derivative, options);
    let proposal = ScalarRootCertificate {
        region: scioxide_core::ExtendedInterval::new(Some(1.), Some(2.))?,
        center: 1.5,
        claim: RootClaim::Unique,
    };
    assert!(scalar.verify(proposal)?.is_proved());
    println!(
        "Certified {} roots in {} disjoint disks; scalar uniqueness independently proved.",
        report.roots(),
        report.certificates()
    );
    Ok(())
}
