//! Central and noncentral test-statistic laws.
use crate::distribution::LawOptions;
use crate::distribution::formula::{Formula, formula_law};
use crate::law::{argument, finite, invalid, log1mexp, logadd};
use crate::{InverseSurvival, LogCdf, LogDensity, LogSurvival, Mean, Quantile, Rng, Variance};
use scioxide_core::{Budget, RealScalar, SciOxideError};
mod chi_f_t;
mod ks;
pub(super) mod math;
mod mixture;
mod range;
mod rice;
pub use chi_f_t::{FisherF, NoncentralChiSquared, NoncentralF, NoncentralStudentT};
pub use ks::{
    Kolmogorov, KolmogorovRequirements, KolmogorovSmirnovOneSided, KolmogorovSmirnovTwoSided,
    KolmogorovWorkspace,
};
use math::{Query, Work, exhausted, number, positive};
pub use range::StudentizedRange;
pub use rice::Rice;

pub(super) use chi_f_t::{NoncentralChi, inverse_result, mixture};

/// Tail probability of a test statistic with an error estimate and a work count.
///
/// Returned by `evaluate_tail` on each test-statistic law, for example
/// [`FisherF::evaluate_tail`] and [`KolmogorovSmirnovTwoSided::evaluate_tail`].
/// The error is an estimate, not a bound.
///
/// ```
/// use scioxide_random::FisherF;
/// let law = FisherF::new(2.0_f64, 2.0)?;
/// let tail = law.evaluate_tail(3.0, true)?; // P(X > 3) = 1/4
/// assert!((tail.log_probability - 0.25_f64.ln()).abs() < 1e-12);
/// assert!(tail.absolute_error < 1e-12 && tail.evaluations > 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TestStatisticResult<S: RealScalar> {
    /// Natural logarithm of the requested probability.
    pub log_probability: S,
    /// Estimated absolute error in the ordinary probability.
    pub absolute_error: S,
    /// Number of recurrence updates and integrand evaluations used.
    pub evaluations: usize,
}

/// Binary64 kernels of one test-statistic law. The blanket adapter below
/// derives the shared [`Formula`] capabilities from them, so `formula_law!`
/// wires every evaluation, inversion and sampling trait.
pub(in crate::distribution) trait Statistic<S: RealScalar>:
    Formula<S>
{
    /// Accuracy, work and sampling policy of every query.
    fn options(&self) -> LawOptions<S>;
    /// Closed support hull in the scalar coordinate.
    fn hull(&self) -> (Option<S>, Option<S>);
    /// Approximate median; it only orders tail evaluations.
    fn median(&self) -> f64;
    /// Logarithmic density or tail at `x`, charged to `work`.
    fn kernel(&self, x: f64, query: Query, work: &mut Work) -> Result<f64, SciOxideError>;
    /// Ordinary probability; specialized laws avoid the logarithmic route.
    fn ordinary(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        Ok(if upper {
            self.log_sf(x)?
        } else {
            self.log_cdf(x)?
        }
        .exp())
    }
    /// Inverse within the remaining work. The default resolves the coordinate
    /// lattice of the logarithmic tails; specialized laws keep binary64 inverses.
    fn solve(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        crate::distribution::formula::numerical_inverse(self, p, upper, None, left)
    }
    /// One variate, charging the law's per-draw word allowance.
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError>;
    /// Requested logarithmic tail and the probability scale of the tail that
    /// was computed, which bounds its absolute error.
    fn tail_parts(
        &self,
        x: f64,
        upper: bool,
        work: &mut Work,
    ) -> Result<(f64, f64), SciOxideError> {
        let (value, computed) =
            crate::law::smaller_tail_first(x >= self.median(), upper, |upper| {
                self.kernel(x, if upper { Query::Upper } else { Query::Lower }, work)
            })?;
        Ok((value, computed.exp()))
    }
}

// The one binary64 adapter: tails and densities evaluate in binary64 within
// the caller's remaining work and are rounded once to the law's scalar.
fn density<S: RealScalar, T: Statistic<S>>(
    law: &T,
    x: S,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let value = Work::within(law.options(), left, |work| {
        law.kernel(x.into_f64(), Query::Density, work)
    })?;
    // Singular endpoint densities are an explicit mathematical limit.
    if value == f64::INFINITY {
        Ok(S::INFINITY)
    } else {
        number(value)
    }
}
fn tail<S: RealScalar, T: Statistic<S>>(
    law: &T,
    x: S,
    upper: bool,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let query = if upper { Query::Upper } else { Query::Lower };
    number(Work::within(law.options(), left, |work| {
        law.kernel(x.into_f64(), query, work)
    })?)
}
fn log_tail<S: RealScalar, T: Statistic<S>>(
    law: &T,
    x: S,
    upper: bool,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    number(
        Work::within(law.options(), left, |work| {
            law.tail_parts(x.into_f64(), upper, work)
        })?
        .0,
    )
}

/// One tail with work accounting and an estimated error.
fn evaluate_tail<S: RealScalar, T: Statistic<S>>(
    law: &T,
    x: S,
    upper: bool,
) -> Result<TestStatisticResult<S>, SciOxideError> {
    argument(x, "x")?;
    let mut work = Work::new(law.options());
    let (value, scale) = law.tail_parts(x.into_f64(), upper, &mut work)?;
    math::tail_result(value, scale, &work)
}

// Derive the shared formula capabilities of a test-statistic law from its
// binary64 kernels; `report` adds the public `evaluate_tail` form.
macro_rules! statistic_law {
    ($law:ident, report) => {
        impl<S: RealScalar> $law<S> {
            /// Computes `ln P(X > x)` if `upper` is true, or `ln P(X ≤ x)`
            /// otherwise, with an absolute error estimate and a work count.
            ///
            /// A tail above 15/16 is computed as `ln(1 − q)` from the
            /// complementary tail `q`, so logarithms close to zero keep their
            /// relative accuracy. The error estimate includes the final
            /// rounding to `S`.
            ///
            /// # Errors
            ///
            /// - `NonFinite` if `x` is NaN or infinite.
            /// - `ConvergenceFailure` if the evaluation exhausts the work limit
            ///   in [`LawOptions`](crate::LawOptions).
            /// - `NumericalInstability` if a sum or integral breaks down.
            pub fn evaluate_tail(
                &self,
                x: S,
                upper: bool,
            ) -> Result<TestStatisticResult<S>, SciOxideError> {
                super::evaluate_tail(self, x, upper)
            }
        }
        statistic_law!($law);
    };
    ($law:ident) => {
        impl<S: RealScalar> Formula<S> for $law<S> {
            fn bounds(&self) -> (Option<S>, Option<S>) {
                Statistic::hull(self)
            }
            fn density_log(&self, x: S) -> Result<S, SciOxideError> {
                super::density(self, x, &mut Statistic::options(self).work_budget().get())
            }
            fn density_with_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
                super::density(self, x, left)
            }
            fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
                Statistic::ordinary(self, x, upper)
            }
            fn tolerance(&self) -> scioxide_core::Tolerance<S> {
                Statistic::options(self).tolerance()
            }
            fn work_budget(&self) -> scioxide_core::Budget {
                Statistic::options(self).work_budget()
            }
            fn byte_limit(&self) -> usize {
                Statistic::options(self).max_bytes()
            }
            fn tail_with_work(
                &self,
                x: S,
                upper: bool,
                left: &mut usize,
            ) -> Result<S, SciOxideError> {
                super::tail(self, x, upper, left)
            }
            fn log_probability_with_work(
                &self,
                x: S,
                upper: bool,
                left: &mut usize,
            ) -> Result<S, SciOxideError> {
                super::log_tail(self, x, upper, left)
            }
            fn inverse_with_remaining(
                &self,
                p: S,
                upper: bool,
                left: &mut usize,
            ) -> Result<S, SciOxideError> {
                Statistic::solve(self, p, upper, left)
            }
            fn draw_with_remaining<R: Rng<S>>(
                &self,
                rng: &mut R,
                _left: &mut usize,
            ) -> Result<S, SciOxideError> {
                Statistic::variate(self, rng)
            }
        }
        formula_law!($law);
    };
}
use statistic_law;
