//! Shared formula-law capabilities: laws supply mathematical formulas, and
//! the shared driver owns tail selection, inversion and work accounting.
use crate::law::charge;
use crate::{LogCdf, LogSurvival, Rng, Support};
use scioxide_core::{Budget, RealScalar, SciOxideError};

/// How a formula law inverts its tails.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Inverse {
    /// Closed-form `inverse` and `inverse_upper`, charged one work unit.
    Closed,
    /// Bounded root solve of the smooth logarithmic tail, stopping at the
    /// law's probability tolerance.
    Smooth,
    /// Bounded root solve that resolves the scalar coordinate lattice.
    Lattice,
}

/// The one capability source of continuous scalar laws; `formula_law!` derives
/// every evaluation, inversion and sampling trait from it.
pub(super) trait Formula<S: RealScalar>: LogCdf<S> + LogSurvival<S> + Support<S> {
    /// Inversion method. `Closed` laws override `inverse` and `inverse_upper`.
    const INVERSE: Inverse = Inverse::Closed;
    fn bounds(&self) -> (Option<S>, Option<S>);
    fn density_log(&self, x: S) -> Result<S, SciOxideError>;
    fn density_with_work(&self, x: S, _left: &mut usize) -> Result<S, SciOxideError> {
        self.density_log(x)
    }
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        Ok(if upper {
            self.log_sf(x)?
        } else {
            self.log_cdf(x)?
        }
        .exp())
    }
    /// Approximate median: tails first evaluate the side that excludes it.
    fn center(&self) -> Option<S> {
        None
    }
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        scioxide_core::Tolerance::from_machine_epsilon()
    }
    fn work_budget(&self) -> Budget {
        Budget::ONE
    }
    fn byte_limit(&self) -> usize {
        usize::MAX
    }
    /// Evaluate a logarithmic tail, charging numerical work to the shared budget.
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError>;
    fn log_probability_with_work(
        &self,
        x: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        let first = self.center().map_or(upper, |center| x >= center);
        Ok(crate::law::smaller_tail_first(first, upper, |upper| {
            self.tail_with_work(x, upper, left)
        })?
        .0)
    }
    fn log_probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        self.log_probability_with_work(x, upper, &mut self.work_budget().get())
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        self.inverse_with_remaining(p, false, &mut self.work_budget().get())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        self.inverse_with_remaining(p, true, &mut self.work_budget().get())
    }
    /// The one inversion entry point: closed forms, or the shared bounded
    /// inversion of `log_probability_with_work` within the remaining work.
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        let tolerance = match Self::INVERSE {
            Inverse::Closed => {
                charge(left)?;
                return if upper {
                    self.inverse_upper(p)
                } else {
                    self.inverse(p)
                };
            }
            Inverse::Smooth => Some(self.tolerance()),
            Inverse::Lattice => None,
        };
        numerical_inverse(self, p, upper, tolerance, left)
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        self.inverse_with_remaining(crate::util::validated_open_float(rng)?, false, left)
    }
    fn draw<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.draw_with_remaining(rng, &mut self.work_budget().get())
    }
}

/// The one bounded inversion of a formula law's logarithmic tails within the
/// remaining work. A tolerance stops at the smooth tail's probability accuracy;
/// without one, the solve resolves the coordinate lattice.
pub(super) fn numerical_inverse<S: RealScalar, F: Formula<S> + ?Sized>(
    law: &F,
    p: S,
    upper: bool,
    tolerance: Option<scioxide_core::Tolerance<S>>,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    crate::inversion::inverse_remaining(
        law.support()?,
        p,
        upper,
        &crate::inversion::law_options(law.work_budget())?,
        tolerance,
        left,
        |x, upper, left| law.log_probability_with_work(x, upper, left),
    )
}

/// Validate the common public inverse contract independently of its work driver.
pub(super) fn checked_inverse<S: RealScalar>(
    p: S,
    bounds: (Option<S>, Option<S>),
    evaluate: impl FnOnce() -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    crate::law::probability(p, scioxide_core::argument!("p"))?;
    let value = evaluate()?;
    if p > S::ZERO && p < S::ONE {
        crate::law::interior_result(value, bounds.0, bounds.1)
    } else {
        Ok(value)
    }
}

macro_rules! formula_law {
    ($name:ident) => {
        impl<S: RealScalar> crate::Support<S> for $name<S> {
            fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
                let (low, high) = self.bounds();
                scioxide_core::ExtendedInterval::new(low, high)
            }
        }
        impl<S: RealScalar> crate::LogDensity<S> for $name<S> {
            fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
                crate::law::argument(x, "x")?;
                crate::law::checked_log_density(self.density_log(x)?)
            }

            fn log_pdf_scoped(
                &self,
                x: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.work_budget(), |left| {
                    crate::law::argument(x, "x")?;
                    crate::law::charge(left)?;
                    crate::law::checked_log_density(self.density_with_work(x, left)?)
                })
            }
        }
        impl<S: RealScalar> crate::Density<S> for $name<S> {
            fn pdf(&self, x: S) -> Result<S, SciOxideError> {
                let log = crate::LogDensity::log_pdf(self, x)?;
                crate::law::density_from_log(log)
            }
        }
        impl<S: RealScalar> crate::LogCdf<S> for $name<S> {
            fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
                crate::law::argument(x, "x")?;
                self.log_probability(x, false)
            }

            fn log_cdf_scoped(
                &self,
                x: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.work_budget(), |left| {
                    crate::law::argument(x, "x")?;
                    crate::law::charge(left)?;
                    self.log_probability_with_work(x, false, left)
                })
            }
        }
        impl<S: RealScalar> crate::Cdf<S> for $name<S> {
            fn cdf(&self, x: S) -> Result<S, SciOxideError> {
                self.probability(x, false)
            }
        }
        impl<S: RealScalar> crate::LogSurvival<S> for $name<S> {
            fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
                crate::law::argument(x, "x")?;
                self.log_probability(x, true)
            }

            fn log_sf_scoped(
                &self,
                x: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.work_budget(), |left| {
                    crate::law::argument(x, "x")?;
                    crate::law::charge(left)?;
                    self.log_probability_with_work(x, true, left)
                })
            }
        }
        impl<S: RealScalar> crate::Survival<S> for $name<S> {
            fn sf(&self, x: S) -> Result<S, SciOxideError> {
                self.probability(x, true)
            }
        }
        impl<S: RealScalar> crate::Quantile<S> for $name<S> {
            fn quantile(&self, p: S) -> Result<S, SciOxideError> {
                crate::distribution::formula::checked_inverse(p, self.bounds(), || self.inverse(p))
            }

            fn quantile_scoped(
                &self,
                p: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.work_budget(), |left| {
                    crate::distribution::formula::checked_inverse(p, self.bounds(), || {
                        self.inverse_with_remaining(p, false, left)
                    })
                })
            }
        }
        impl<S: RealScalar> crate::InverseSurvival<S> for $name<S> {
            fn isf(&self, p: S) -> Result<S, SciOxideError> {
                crate::distribution::formula::checked_inverse(p, self.bounds(), || {
                    self.inverse_upper(p)
                })
            }

            fn isf_scoped(
                &self,
                p: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.work_budget(), |left| {
                    crate::distribution::formula::checked_inverse(p, self.bounds(), || {
                        self.inverse_with_remaining(p, true, left)
                    })
                })
            }
        }
        impl<S: RealScalar> crate::Distribution<S> for $name<S> {
            fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
                Formula::byte_limit(self)
            }
            fn sample<R: crate::Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
                crate::law::finite(self.draw(rng)?)
            }

            fn sample_scoped<R: crate::Rng<S>>(
                &self,
                rng: &mut R,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.work_budget(), |left| {
                    crate::law::finite(self.draw_with_remaining(rng, left)?)
                })
            }
        }
    };
}
pub(super) use formula_law;

/// Closed hull of strictly positive scalar laws.
pub(super) fn positive_support<S: RealScalar>() -> (Option<S>, Option<S>) {
    (Some(S::ZERO), None)
}
