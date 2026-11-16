//! Options for the special-function families.
//!
//! Most options hold only a work budget: the accuracy of those functions is
//! fixed and is reported with each result. Only [`crate::EllipsoidalOptions`]
//! (adaptive quadrature) and [`crate::SpectralOptions`] (spectral truncation)
//! also take a tolerance.

use scioxide_core::{Budget, RealScalar, RelativeTolerance};

macro_rules! budget_options {
    ($name:ident, $budget:expr, $description:literal) => {
        #[doc = $description]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[must_use]
        pub struct $name {
            iteration_budget: Budget,
        }

        impl $name {
            /// Creates options with the given work budget.
            pub const fn new(iteration_budget: Budget) -> Self {
                Self { iteration_budget }
            }

            /// Returns the work budget, which also covers nested evaluations.
            pub const fn iteration_budget(self) -> Budget {
                self.iteration_budget
            }

            /// Returns a copy with a different work budget.
            pub const fn with_iteration_budget(mut self, iteration_budget: Budget) -> Self {
                self.iteration_budget = iteration_budget;
                self
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new(Budget::ONE.saturating_add($budget - 1))
            }
        }
    };
}

budget_options!(
    GammaOptions,
    65536,
    "Holds the work budget for gamma-family functions, including incomplete gamma and its inverses; the default is 65536 work units."
);
budget_options!(
    CombinatorialOptions,
    65536,
    "Holds the work budget for combinatorial functions; the default is 65536 work units."
);
budget_options!(
    BetaOptions,
    65536,
    "Holds the work budget for beta-family functions, including incomplete beta and its inverses; the default is 65536 work units."
);
budget_options!(
    AiryOptions,
    65536,
    "Holds the work budget for Airy functions; the default is 65536 work units."
);
budget_options!(
    BesselOptions,
    65536,
    "Holds the work budget for cylindrical Bessel and Hankel functions; the default is 65536 work units."
);
budget_options!(
    EllipticOptions,
    65536,
    "Holds the work budget for elliptic integrals, Carlson integrals and Jacobi elliptic functions; the default is 65536 work units."
);
budget_options!(
    ErrorFunctionOptions,
    65536,
    "Holds the work budget for error functions, their inverses and Dawson's integral; the default is 65536 work units."
);
budget_options!(
    HypergeometricOptions,
    65536,
    "Holds the work budget for hypergeometric functions and their parameter derivatives; the default is 65536 work units."
);
budget_options!(
    LegendreOptions,
    65536,
    "Holds the work budget for Legendre functions and spherical harmonics (one unit per recurrence step); the default is 65536 work units."
);
budget_options!(
    ZetaOptions,
    65536,
    "Holds the work budget for Riemann and Hurwitz zeta and Dirichlet eta functions; the default is 65536 work units."
);
budget_options!(
    SynchrotronOptions,
    128,
    "Holds the work budget for synchrotron functions; the default is 128 work units."
);
budget_options!(
    TransportOptions,
    128,
    "Holds the work budget for transport integrals; the default is 128 work units."
);
budget_options!(
    ClausenOptions,
    128,
    "Holds the work budget for the Clausen function; the default is 128 work units."
);
budget_options!(
    ProbabilityOptions,
    65536,
    "Holds the work budget for normal-distribution functions, their inverses and Owen's T function; the default is 65536 work units."
);
budget_options!(
    IntegralOptions,
    512,
    "Holds the work budget for exponential, trigonometric, hyperbolic and logarithmic integrals and Fresnel integrals; the default is 512 work units."
);
budget_options!(
    SphericalBesselOptions,
    65536,
    "Holds the work budget for spherical Bessel functions; the default is 65536 work units."
);
budget_options!(
    KelvinOptions,
    65536,
    "Holds the work budget for Kelvin functions; the default is 65536 work units."
);
budget_options!(
    CoulombOptions,
    65536,
    "Holds the work budget for Coulomb wave functions and hydrogenic radial functions; the default is 65536 work units."
);
budget_options!(
    StruveOptions,
    262144,
    "Holds the work budget for Struve functions; the default is 262144 work units."
);
budget_options!(
    ParabolicCylinderOptions,
    262144,
    "Holds the work budget for parabolic cylinder functions; the default is 262144 work units."
);
budget_options!(
    BesselIntegralOptions,
    65536,
    "Holds the work budget for integrals of Bessel and Struve functions; the default is 65536 work units."
);
budget_options!(
    LambertOptions,
    65536,
    "Holds the work budget for Lambert W and Wright omega functions; the default is 65536 work units."
);
budget_options!(
    PolylogOptions,
    65536,
    "Holds the work budget for polylogarithms, the dilogarithm and Spence's function; the default is 65536 work units."
);
budget_options!(
    LerchOptions,
    65536,
    "Holds the work budget for the Lerch transcendent; the default is 65536 work units."
);
budget_options!(
    DebyeOptions,
    65536,
    "Holds the work budget for Debye functions; the default is 65536 work units."
);
budget_options!(
    FermiDiracOptions,
    65536,
    "Holds the work budget for Fermi–Dirac integrals; the default is 65536 work units."
);

/// Sets the quadrature tolerance and the work budget for ellipsoidal harmonics.
///
/// The relative tolerance controls the adaptive quadrature for second-kind values and
/// normalization integrals: a looser tolerance accepts coarser panels and does less work.
/// The coefficients and the first-kind Lamé–Chebyshev sums have fixed accuracy and use
/// only the budget. The budget applies separately to plan preparation and to each
/// evaluation. Defaults are 262144 work units and a relative tolerance of `1e-12` for
/// `f64` or `1e-5` for `f32`. A [`crate::EllipsoidalHarmonicPlan`] keeps the options it
/// was built with.
///
/// # Examples
///
/// ```
/// use scioxide_core::RelativeTolerance;
/// let options = scioxide_special::EllipsoidalOptions::default()
///     .with_relative_tolerance(RelativeTolerance::new(1e-8_f64)?);
/// assert_eq!(options.relative_tolerance().relative(), 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct EllipsoidalOptions<S: RealScalar> {
    relative_tolerance: RelativeTolerance<S>,
    iteration_budget: Budget,
}

impl<S: RealScalar> EllipsoidalOptions<S> {
    /// Creates options with the given relative tolerance and work budget.
    pub const fn new(relative_tolerance: RelativeTolerance<S>, iteration_budget: Budget) -> Self {
        Self {
            relative_tolerance,
            iteration_budget,
        }
    }

    /// Returns the relative tolerance of the adaptive quadrature.
    pub const fn relative_tolerance(self) -> RelativeTolerance<S> {
        self.relative_tolerance
    }

    /// Returns the work budget of plan preparation or of one evaluation.
    pub const fn iteration_budget(self) -> Budget {
        self.iteration_budget
    }

    /// Returns a copy with a different relative quadrature tolerance.
    pub const fn with_relative_tolerance(
        mut self,
        relative_tolerance: RelativeTolerance<S>,
    ) -> Self {
        self.relative_tolerance = relative_tolerance;
        self
    }

    /// Returns a copy with a different work budget.
    pub const fn with_iteration_budget(mut self, iteration_budget: Budget) -> Self {
        self.iteration_budget = iteration_budget;
        self
    }
}

impl<S: RealScalar> Default for EllipsoidalOptions<S> {
    fn default() -> Self {
        // RealScalar is sealed to f32/f64, where these constants are finite.
        let tolerance = if S::MANTISSA_DIGITS <= f32::MANTISSA_DIGITS {
            1e-5
        } else {
            1e-12
        };
        Self {
            relative_tolerance: S::try_from_f64_lossy(tolerance)
                .and_then(RelativeTolerance::new)
                .unwrap_or_default(),
            iteration_budget: Budget::ONE.saturating_add(262_143),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_positive_budgets() {
        assert_eq!(GammaOptions::default().iteration_budget().get(), 65536);
        assert_eq!(IntegralOptions::default().iteration_budget().get(), 512);
        assert_eq!(
            GammaOptions::default()
                .with_iteration_budget(Budget::ONE)
                .iteration_budget(),
            Budget::ONE
        );
    }

    #[test]
    fn ellipsoidal_tolerances_are_validated_and_precision_specific() {
        assert_eq!(
            EllipsoidalOptions::<f64>::default()
                .relative_tolerance()
                .relative(),
            1e-12
        );
        assert_eq!(
            EllipsoidalOptions::<f32>::default()
                .relative_tolerance()
                .relative(),
            1e-5
        );
        assert!(RelativeTolerance::new(-1.0_f64).is_err());
        assert!(RelativeTolerance::new(f64::NAN).is_err());
        assert_eq!(
            EllipsoidalOptions::<f64>::default()
                .with_relative_tolerance(RelativeTolerance::new(1e-6).unwrap())
                .relative_tolerance()
                .relative(),
            1e-6
        );
    }
}
