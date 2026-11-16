//! C¹ cubic splines on the sphere: the basis, the spline model, the
//! least-squares plan with fixed breakpoints and the adaptive fit.

mod adaptive;
mod basis;
mod model;
mod plan;

pub use adaptive::{
    fit_spherical_spline, fit_spherical_spline_scoped, fit_spherical_spline_with_options,
};
pub use model::{SphericalSpline, SphericalSplineWorkspace};
pub use plan::{
    SphericalSamples, SphericalSplineFit, SphericalSplineFitRequirements,
    SphericalSplineFitWorkspace, SphericalSplinePlan,
};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::SplineFitLimits;
    use crate::protocol::{ExtrapolationPolicy, Interpolant};
    use alloc::vec::Vec;
    use core::f64::consts::PI;
    use scioxide_core::Complex;

    fn shared<T: Send + Sync + Clone>(_: &T) {}

    #[test]
    fn plans_fits_and_models_are_shareable() {
        let samples: Vec<[f64; 2]> = (0..=4)
            .flat_map(|i| (0..8).map(move |j| [PI * f64::from(i) / 4., PI * f64::from(j) / 4.]))
            .collect();
        let plan = SphericalSplinePlan::new(
            &[0., PI],
            &[0., PI / 2., PI, 1.5 * PI, 2. * PI],
            SphericalSamples::Scattered(&samples),
            None,
            0.,
            SplineFitLimits::new(),
        )
        .unwrap();
        shared(&plan);
        let fit = plan
            .fit(&vec![Complex::new(1.0, -2.0); samples.len()])
            .unwrap();
        shared(&fit);
        shared(fit.spline());
        shared(&fit.spline().workspace().unwrap());
        let value = fit
            .spline()
            .evaluate([0.3, -4.0], ExtrapolationPolicy::Reject)
            .unwrap();
        assert!((value.re - 1.0).abs() < 1e-13 && (value.im + 2.0).abs() < 1e-13);
    }
}
