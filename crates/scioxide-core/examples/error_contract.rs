//! External use of the canonical classification, borrowed views and sources.
use core::error::Error;
use scioxide_core::{
    DifferentiationLocation, DomainReason, NonFiniteOrigin, RealScalar, SciOxideError,
    SciOxideErrorKind,
};

fn square_root(x: f64) -> Result<f64, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "argument must be finite",
        ));
    }
    if x < 0.0 {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            DomainReason::OutsideDomain,
            "argument must be nonnegative",
        ));
    }
    Ok(x.sqrt())
}
fn print_causes(error: &SciOxideError) {
    let mut current: Option<&dyn Error> = Some(error);
    while let Some(cause) = current {
        println!("{cause}");
        current = cause.source();
    }
}
fn main() -> Result<(), SciOxideError> {
    if let Err(cause) = square_root(-1.0) {
        let error =
            SciOxideError::differentiation_failure(DifferentiationLocation::variable(0), 1, cause);
        if let scioxide_core::SciOxideErrorView::DifferentiationFailure { frames, cause } =
            error.view()
        {
            assert_eq!(cause.kind(), SciOxideErrorKind::DomainError);
            assert_eq!(frames[0].evaluations, 1);
        }
        print_causes(&error);
    }
    if let Err(error) = f32::try_from_f64_lossy(f64::MAX)
        && let scioxide_core::SciOxideErrorView::ScalarConversion {
            from_type,
            to_type,
            reason,
        } = error.view()
    {
        println!("conversion {} -> {}: {}", from_type, to_type, reason);
    }
    assert_eq!(square_root(4.0)?, 2.0);
    Ok(())
}
