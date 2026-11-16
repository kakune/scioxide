//! Shared bounded integer inversion and count-law arithmetic.
pub(super) use crate::float::open_midpoint as open;
use crate::law::logadd;
pub(super) use crate::law::{exhausted, log_probability};

use crate::distribution::LawOptions;
use crate::{BitGenerator, CountSupport};
use scioxide_core::{Budget, RealScalar, SciOxideError};

pub(super) fn parameter<S: RealScalar>(
    x: S,
    positive: bool,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    Ok(if positive {
        crate::law::positive(x, argument)?
    } else {
        crate::law::nonnegative(x, argument)?
    }
    .into_f64())
}
pub(super) fn probability<S: RealScalar>(
    x: S,
    zero: bool,
    one: bool,
    argument: scioxide_core::Argument,
) -> Result<f64, SciOxideError> {
    crate::law::probability(x, argument)?;
    let p = x.into_f64();
    if (!zero && p == 0.0) || (!one && p == 1.0) {
        return Err(crate::law::domain(
            argument,
            "count-law probability outside its domain",
        ));
    }
    Ok(p)
}
pub(super) fn moment(v: f64) -> Result<f64, SciOxideError> {
    crate::law::finite(v)
}
/// The one documented count range: exact count arithmetic covers integers of
/// magnitude at most `2^53`, the consecutive binary64 integers. Larger counts
/// are `NotImplemented` in every conversion below and in integer inversion.
pub(super) const MAX_EXACT_COUNT: u64 = 1 << 53;
/// Direct CDF summations stop at this many terms, so their work never grows
/// with the support.
pub(super) const DIRECT_TERMS: usize = 32;
/// Binary64 count result as `usize`.
pub(super) fn count(v: f64) -> Result<usize, SciOxideError> {
    // A float comparison bounds the target width without a 128-bit
    // conversion call; it folds away where usize exceeds 2^53.
    if !((0.0..=MAX_EXACT_COUNT as f64).contains(&v) && v <= usize::MAX as f64) {
        return Err(SciOxideError::not_implemented(
            "sample or quantile exceeds exact count arithmetic",
        ));
    }
    Ok(v as usize)
}
/// Exact conversion of a count argument to binary64.
pub(super) fn analytic_count(k: usize) -> Result<f64, SciOxideError> {
    if k as u128 > u128::from(MAX_EXACT_COUNT) {
        return Err(SciOxideError::not_implemented("analytic counts above 2^53"));
    }
    Ok(k as f64)
}
/// Exact conversion of a signed count argument to binary64.
pub(super) fn signed_count_value(k: i64) -> Result<f64, SciOxideError> {
    if k.unsigned_abs() > MAX_EXACT_COUNT {
        return Err(SciOxideError::not_implemented(
            "signed analytic counts above 2^53 in magnitude",
        ));
    }
    Ok(k as f64)
}
/// Unsigned support conversion for the single integer search.
pub(super) fn inverse<S: RealScalar>(
    p: S,
    upper: bool,
    lower: usize,
    maximum: Option<usize>,
    guess: Option<usize>,
    left: &mut usize,
    mut tail: impl FnMut(usize, bool, &mut usize) -> Result<f64, SciOxideError>,
) -> Result<usize, SciOxideError> {
    let value = integer_inverse(
        p,
        upper,
        Some(lower as i128),
        maximum.map(|v| v as i128),
        (
            0,
            maximum.map_or(i128::from(MAX_EXACT_COUNT), |v| v as i128),
        ),
        guess.map(|v| v as i128),
        left,
        |k, upper, left| tail(k as usize, upper, left),
    )?;
    usize::try_from(value)
        .map_err(|_| SciOxideError::not_implemented("integer quantile exceeds count range"))
}

/// Signed support conversion; i128 arithmetic protects full-width i64 midpoints.
pub(super) fn inverse_signed<S: RealScalar>(
    p: S,
    upper: bool,
    minimum: Option<i64>,
    maximum: Option<i64>,
    left: &mut usize,
    mut tail: impl FnMut(i64, bool, &mut usize) -> Result<f64, SciOxideError>,
) -> Result<i64, SciOxideError> {
    let value = integer_inverse(
        p,
        upper,
        minimum.map(i128::from),
        maximum.map(i128::from),
        (i128::from(i64::MIN), i128::from(i64::MAX)),
        None,
        left,
        |k, upper, left| tail(k as i64, upper, left),
    )?;
    i64::try_from(value)
        .map_err(|_| SciOxideError::not_implemented("integer quantile exceeds signed count range"))
}

/// Smallest integer reaching the requested tail threshold. Discovery and bisection
/// share the caller's complete work budget, including nested tail summations.
#[allow(clippy::too_many_arguments)]
fn integer_inverse<S: RealScalar>(
    p: S,
    upper: bool,
    minimum: Option<i128>,
    maximum: Option<i128>,
    range: (i128, i128),
    guess: Option<i128>,
    left: &mut usize,
    mut tail: impl FnMut(i128, bool, &mut usize) -> Result<f64, SciOxideError>,
) -> Result<i128, SciOxideError> {
    crate::law::probability(p, scioxide_core::argument!("p"))?;
    let p = p.into_f64();
    let endpoint = if p == if upper { 1. } else { 0. } {
        Some(minimum)
    } else if p == if upper { 0. } else { 1. } {
        Some(maximum)
    } else {
        None
    };
    if let Some(value) = endpoint {
        return value
            .ok_or_else(|| SciOxideError::not_implemented("infinite integer quantile endpoint"));
    }
    let (target, upper) = if p > 0.5 {
        ((-p).ln_1p(), !upper)
    } else {
        (p.ln(), upper)
    };
    let mut below = |k| {
        crate::law::charge(left)?;
        let value = log_probability(tail(k, upper, left)?)?;
        Ok::<_, SciOxideError>(if upper {
            value > target
        } else {
            value < target
        })
    };
    let (mut lo, mut hi) = if let Some(guess) = guess {
        let low = minimum.unwrap_or(range.0).max(range.0);
        let high = maximum.unwrap_or(range.1).min(range.1);
        let mut lo = guess.clamp(low, high);
        let mut hi = lo;
        let mut step = 1;
        if below(lo)? {
            loop {
                hi = (lo + step).min(high);
                if !below(hi)? {
                    break;
                }
                if hi == high {
                    return Err(SciOxideError::not_implemented(
                        "integer quantile exceeds count range",
                    ));
                }
                lo = hi;
                step *= 2;
            }
        } else {
            loop {
                if hi == low {
                    return Ok(low);
                }
                lo = (hi - step).max(low);
                if below(lo)? {
                    break;
                }
                hi = lo;
                step *= 2;
            }
        }
        (lo, hi)
    } else {
        if let Some(low) = minimum
            && !below(low)?
        {
            return Ok(low);
        }
        let mut lo = minimum
            .unwrap_or_else(|| maximum.map_or(-1, |high| high.min(0) - 1))
            .max(range.0);
        let mut hi = maximum.unwrap_or(lo.max(0) + 1).min(range.1);
        while minimum.is_none() && !below(lo)? {
            let next = (lo - (hi - lo).max(1)).max(range.0);
            if next == lo {
                return Err(SciOxideError::not_implemented(
                    "integer quantile exceeds count range",
                ));
            }
            lo = next;
        }
        while maximum.is_none() && below(hi)? {
            let next = (hi + (hi - lo).max(1)).min(range.1);
            if next == hi {
                return Err(SciOxideError::not_implemented(
                    "integer quantile exceeds count range",
                ));
            }
            hi = next;
        }
        (lo, hi)
    };
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if below(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(hi)
}
/// Positive log sum with an explicit term budget.
pub(super) fn sum(
    start: usize,
    end: usize,
    left: &mut usize,
    mut term: impl FnMut(usize) -> Result<f64, SciOxideError>,
) -> Result<f64, SciOxideError> {
    if end < start {
        return Ok(f64::NEG_INFINITY);
    }
    if end - start >= *left {
        return Err(exhausted());
    }
    (start..=end).try_fold(f64::NEG_INFINITY, |s, k| {
        crate::law::charge(left)?;
        logadd(s, term(k)?)
    })
}
/// Per-draw uniform-word allowance and batch byte limit of a configured sampler.
#[derive(Debug, Clone, Copy)]
pub(super) struct DrawLimits {
    pub words: usize,
    pub max_bytes: usize,
}
impl DrawLimits {
    pub(super) fn of<S: RealScalar>(options: super::LawOptions<S>) -> Self {
        Self {
            words: options.sample_word_budget().get(),
            max_bytes: options.max_bytes(),
        }
    }
}

// The configured count contract for a type with `limits()` and an uncharged
// `draw(&mut Words)`. A public draw wraps the provider exactly once, so nested
// kernels share its allowance; batches apply the configured byte limit.
macro_rules! count_distribution {
    ([$($generics:tt)*] $sampler:ty, $count:ty) => {
        impl $($generics)* crate::CountDistribution for $sampler {
            type Count = $count;
            fn sample<R: crate::BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
            ) -> Result<$count, scioxide_core::SciOxideError> {
                self.draw(&mut crate::distribution::words::Words::new(rng, self.limits().words))
            }
            fn sample_n_into<R: crate::BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
                policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
                output: &mut [$count],
                workspace: &mut crate::SamplingWorkspace<$count>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::workspace::sample_scalars(
                    rng,
                    policy,
                    output,
                    0,
                    workspace,
                    self.limits().max_bytes,
                    |rng| <Self as crate::CountDistribution>::sample(self, rng),
                )
            }
            fn sample_n<R: crate::BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
                n: usize,
            ) -> Result<alloc::vec::Vec<$count>, scioxide_core::SciOxideError> {
                crate::workspace::admit_bytes(
                    crate::workspace::storage_bytes::<$count>(n)?,
                    self.limits().max_bytes,
                )?;
                crate::workspace::sample_owned(rng, n, 0, |rng| {
                    <Self as crate::CountDistribution>::sample(self, rng)
                })
            }
        }
    };
}
pub(super) use count_distribution;

// The count contract of a law that draws through its prepared sampler: each
// call prepares the sampler once, before any draw, and forwards to it, so the
// draws and errors equal those of the sampler.
macro_rules! prepared_count_distribution {
    ($law:ident, $count:ty) => {
        impl<S: scioxide_core::RealScalar> crate::CountDistribution for $law<S> {
            type Count = $count;
            fn sample<R: crate::BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
            ) -> Result<$count, scioxide_core::SciOxideError> {
                crate::CountDistribution::sample(&self.prepare_sampler()?, rng)
            }
            fn sample_n_into<R: crate::BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
                policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
                output: &mut [$count],
                workspace: &mut crate::SamplingWorkspace<$count>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::CountDistribution::sample_n_into(
                    &self.prepare_sampler()?,
                    rng,
                    policy,
                    output,
                    workspace,
                )
            }
            fn sample_n<R: crate::BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
                n: usize,
            ) -> Result<alloc::vec::Vec<$count>, scioxide_core::SciOxideError> {
                crate::CountDistribution::sample_n(&self.prepare_sampler()?, rng, n)
            }
        }
    };
}
pub(super) use prepared_count_distribution;

/// Charge a nested special-function call to the remaining operation work.
pub(super) fn remaining_budget(left: &usize) -> Result<Budget, SciOxideError> {
    Budget::try_from(*left).map_err(|_| exhausted())
}
pub(super) fn special_value(
    report: scioxide_special::SpecialFunctionResult<f64>,
    left: &mut usize,
) -> Result<f64, SciOxideError> {
    *left = left
        .checked_sub(report.counts().iterations())
        .ok_or_else(exhausted)?;
    report.try_value()
}

pub(super) fn log_result<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    let value = log_probability(value)?;
    if value == f64::NEG_INFINITY {
        Ok(S::NEG_INFINITY)
    } else {
        S::try_from_f64_lossy(value)
    }
}

/// Exact count kernels of one law; `count_capabilities!` derives every
/// mass, tail, inverse, moment and sampling capability from them.
pub(super) trait Kernel<S: RealScalar>: CountSupport {
    fn options(&self) -> LawOptions<S>;
    fn limits(&self) -> crate::distribution::count_math::DrawLimits {
        crate::distribution::count_math::DrawLimits::of(self.options())
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError>;
    fn tail_with_work(&self, k: usize, upper: bool, left: &mut usize)
    -> Result<f64, SciOxideError>;
    fn tail(&self, k: usize, upper: bool) -> Result<f64, SciOxideError> {
        self.tail_with_work(k, upper, &mut self.options().work_budget().get())
    }
    fn probability(&self, k: usize, upper: bool) -> Result<f64, SciOxideError> {
        Ok(crate::distribution::count_math::log_probability(self.tail(k, upper)?)?.exp())
    }
    fn average(&self) -> Result<f64, SciOxideError>;
    fn spread(&self) -> Result<f64, SciOxideError>;
    fn average_with_work(&self, left: &mut usize) -> Result<f64, SciOxideError> {
        crate::law::charge(left)?;
        self.average()
    }
    fn spread_with_work(&self, left: &mut usize) -> Result<f64, SciOxideError> {
        crate::law::charge(left)?;
        self.spread()
    }
    fn inverse_with_work(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<usize, SciOxideError> {
        let (lo, hi) = self.count_support();
        inverse(p, upper, lo, hi, None, left, |k, upper, left| {
            self.tail_with_work(k, upper, left)
        })
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        // Default sampler: exact inversion of the tail functions with a
        // uniform that never hits 0 or 1, so the search never needs the
        // (possibly infinite) support endpoints. Laws with a cheaper exact
        // construction override this.
        let (lo, hi) = self.count_support();
        if hi == Some(lo) {
            return Ok(lo);
        }
        crate::distribution::count_math::inverse(
            crate::distribution::count_math::open(rng)?,
            false,
            lo,
            hi,
            None,
            &mut self.options().work_budget().get(),
            |k, u, left| self.tail_with_work(k, u, left),
        )
    }
}
macro_rules! count_capabilities {
    ($law:ident) => {
        crate::distribution::count_math::count_distribution!([<S: RealScalar>] $law<S>, usize);
        impl<S: RealScalar> crate::Mass<S> for $law<S> {
            fn pmf(&self, k: usize) -> Result<S, SciOxideError> {
                S::try_from_f64_lossy(crate::distribution::count_math::log_probability(self.mass(k)?)?.exp())
            }
        }
        impl<S: RealScalar> crate::LogMass<S> for $law<S> {
            fn log_pmf(&self, k: usize) -> Result<S, SciOxideError> {
                crate::distribution::count_math::log_result(self.mass(k)?)
            }
        }
        impl<S: RealScalar> crate::Cdf<S, usize> for $law<S> {
            fn cdf(&self, k: usize) -> Result<S, SciOxideError> {
                S::try_from_f64_lossy(self.probability(k, false)?)
            }
        }
        impl<S: RealScalar> crate::Survival<S, usize> for $law<S> {
            fn sf(&self, k: usize) -> Result<S, SciOxideError> {
                S::try_from_f64_lossy(self.probability(k, true)?)
            }
        }
        impl<S: RealScalar> crate::LogCdf<S, usize> for $law<S> {
            fn log_cdf(&self, k: usize) -> Result<S, SciOxideError> {
                crate::distribution::count_math::log_result(self.tail(k, false)?)
            }

            fn log_cdf_scoped(&self,k:usize,context:&mut scioxide_core::SolveContext<'_, '_>) -> Result<S,SciOxideError> { crate::law::with_work(context,self.options.work_budget(),|left| crate::distribution::count_math::log_result(self.tail_with_work(k,false,left)?)) }
}
        impl<S: RealScalar> crate::LogSurvival<S, usize> for $law<S> {
            fn log_sf(&self, k: usize) -> Result<S, SciOxideError> {
                crate::distribution::count_math::log_result(self.tail(k, true)?)
            }

            fn log_sf_scoped(&self,k:usize,context:&mut scioxide_core::SolveContext<'_, '_>) -> Result<S,SciOxideError> { crate::law::with_work(context,self.options.work_budget(),|left| crate::distribution::count_math::log_result(self.tail_with_work(k,true,left)?)) }
}
        impl<S: RealScalar> crate::Quantile<S, usize> for $law<S> {
            fn quantile(&self, p: S) -> Result<usize, SciOxideError> {
                self.inverse_with_work(p, false, &mut self.options().work_budget().get())
            }
            fn quantile_scoped(&self, p: S, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<usize, SciOxideError> {
                crate::law::with_work(context, self.options().work_budget(), |left| self.inverse_with_work(p, false, left))
            }
        }
        impl<S: RealScalar> crate::InverseSurvival<S, usize> for $law<S> {
            fn isf(&self, p: S) -> Result<usize, SciOxideError> {
                self.inverse_with_work(p, true, &mut self.options().work_budget().get())
            }
            fn isf_scoped(&self, p: S, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<usize, SciOxideError> {
                crate::law::with_work(context, self.options().work_budget(), |left| self.inverse_with_work(p, true, left))
            }
        }
        impl<S: RealScalar> crate::Mean<S> for $law<S> {
            fn mean(&self) -> Result<S, SciOxideError> {
                crate::law::finite(S::try_from_f64_lossy(crate::distribution::count_math::moment(self.average()?)?)?)
            }
            fn mean_scoped(&self, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.options().work_budget(), |left| {
                    crate::law::finite(S::try_from_f64_lossy(self.average_with_work(left)?)?)
                })
            }
        }
        impl<S: RealScalar> crate::Variance<S> for $law<S> {
            fn variance(&self) -> Result<S, SciOxideError> {
                crate::law::finite(S::try_from_f64_lossy(crate::distribution::count_math::moment(self.spread()?)?)?)
            }
            fn variance_scoped(&self, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.options().work_budget(), |left| {
                    crate::law::finite(S::try_from_f64_lossy(self.spread_with_work(left)?)?)
                })
            }
        }
    };
}
pub(super) use count_capabilities;

/// Signed count laws supply `mass`, `tail`, `inverse`, `average` and `spread`;
/// `sampled` laws also derive `CountDistribution` from their `draw`.
macro_rules! signed_capabilities {
    ($law:ident, sampled) => {
        impl<S: RealScalar> $law<S> {
            fn limits(&self) -> crate::distribution::count_math::DrawLimits {
                crate::distribution::count_math::DrawLimits::of(self.options)
            }
        }
        crate::distribution::count_math::count_distribution!([<S: RealScalar>] $law<S>, i64);
        signed_capabilities!($law);
    };
    ($law:ident) => {
        impl<S: RealScalar> crate::Mass<S, i64> for $law<S> {
            fn pmf(&self, k: i64) -> Result<S, SciOxideError> {
                S::try_from_f64_lossy(self.mass(k)?.exp())
            }
        }
        impl<S: RealScalar> crate::LogMass<S, i64> for $law<S> {
            fn log_pmf(&self, k: i64) -> Result<S, SciOxideError> {
                crate::distribution::count_math::log_result(self.mass(k)?)
            }
        }
        impl<S: RealScalar> crate::Cdf<S, i64> for $law<S> {
            fn cdf(&self, k: i64) -> Result<S, SciOxideError> {
                S::try_from_f64_lossy(self.tail(k, false)?.exp())
            }
        }
        impl<S: RealScalar> crate::Survival<S, i64> for $law<S> {
            fn sf(&self, k: i64) -> Result<S, SciOxideError> {
                S::try_from_f64_lossy(self.tail(k, true)?.exp())
            }
        }
        impl<S: RealScalar> crate::LogCdf<S, i64> for $law<S> {
            fn log_cdf(&self, k: i64) -> Result<S, SciOxideError> {
                crate::distribution::count_math::log_result(self.tail(k, false)?)
            }
            fn log_cdf_scoped(&self, k: i64, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.options.work_budget(), |left| {
                    crate::distribution::count_math::log_result(self.tail_with_work(k, false, left)?)
                })
            }
        }
        impl<S: RealScalar> crate::LogSurvival<S, i64> for $law<S> {
            fn log_sf(&self, k: i64) -> Result<S, SciOxideError> {
                crate::distribution::count_math::log_result(self.tail(k, true)?)
            }
            fn log_sf_scoped(&self, k: i64, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<S, SciOxideError> {
                crate::law::with_work(context, self.options.work_budget(), |left| {
                    crate::distribution::count_math::log_result(self.tail_with_work(k, true, left)?)
                })
            }
        }
        impl<S: RealScalar> crate::Quantile<S, i64> for $law<S> {
            fn quantile(&self, p: S) -> Result<i64, SciOxideError> {
                self.inverse(p, false, &mut self.options.work_budget().get())
            }
            fn quantile_scoped(&self, p: S, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<i64, SciOxideError> {
                crate::law::with_work(context, self.options.work_budget(), |left| self.inverse(p, false, left))
            }
        }
        impl<S: RealScalar> crate::InverseSurvival<S, i64> for $law<S> {
            fn isf(&self, p: S) -> Result<i64, SciOxideError> {
                self.inverse(p, true, &mut self.options.work_budget().get())
            }
            fn isf_scoped(&self, p: S, context: &mut scioxide_core::SolveContext<'_, '_>) -> Result<i64, SciOxideError> {
                crate::law::with_work(context, self.options.work_budget(), |left| self.inverse(p, true, left))
            }
        }
        impl<S: RealScalar> crate::Mean<S> for $law<S> {
            fn mean(&self) -> Result<S, SciOxideError> {
                crate::law::finite(S::try_from_f64_lossy(self.average()?)?)
            }
        }
        impl<S: RealScalar> crate::Variance<S> for $law<S> {
            fn variance(&self) -> Result<S, SciOxideError> {
                crate::law::finite(S::try_from_f64_lossy(self.spread()?)?)
            }
        }
    };
}
pub(super) use signed_capabilities;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod inversion_tests {
    use super::*;
    #[test]
    fn signed_and_unsigned_thresholds_and_endpoints() {
        let tail = |k: i128, upper: bool, _: &mut usize| {
            let p: f64 = if k < 7 { 0.25 } else { 1. };
            Ok(if upper { (-p).ln_1p() } else { p.ln() })
        };
        for upper in [false, true] {
            let p = if upper { 0.6 } else { 0.4 };
            assert_eq!(
                inverse(p, upper, 0, Some(10), None, &mut 100, |k, u, w| tail(
                    k as i128, u, w
                ))
                .unwrap(),
                7
            );
            assert_eq!(
                inverse_signed(p, upper, Some(-10), Some(10), &mut 100, |k, u, w| tail(
                    i128::from(k),
                    u,
                    w
                ))
                .unwrap(),
                7
            );
        }
        let pivot = i64::MIN + 8;
        let result = inverse_signed(
            0.5,
            false,
            Some(i64::MIN),
            Some(i64::MAX),
            &mut 100,
            |k, _, _| Ok(if k < pivot { 0.25_f64.ln() } else { 0. }),
        )
        .unwrap();
        assert_eq!(result, pivot);
        assert_eq!(
            inverse(0., false, 2, Some(2), None, &mut 1, |_, _, _| Err(
                exhausted()
            ))
            .unwrap(),
            2
        );
        assert!(inverse(1., false, 0, None, None, &mut 1, |_, _, _| Err(exhausted())).is_err());
    }
}
