//! Hypergeometric laws: central, Fisher noncentral and Wallenius noncentral.
use super::central_urn::CentralUrn;
use super::count_math::analytic_count;
use super::count_math::{self as math};
use super::count_math::{Kernel, count_capabilities};
use super::finite_table::{FiniteTable, storage};
use crate::distribution::LawOptions;
use crate::law::{log1mexp, logadd};
use crate::{BitGenerator, CountSupport};
use scioxide_core::{RealScalar, SciOxideError};

pub(super) fn urn_support(
    m: usize,
    n: usize,
    draws: usize,
) -> Result<(usize, usize), SciOxideError> {
    if n > m || draws > m {
        // Callers are the urn constructors `(population, successes, draws, ..)`.
        return Err(crate::law::domain(
            if n > m {
                scioxide_core::argument!("successes")
            } else {
                scioxide_core::argument!("draws")
            },
            "urn successes and draws must not exceed population",
        ));
    }
    Ok((draws.saturating_sub(m - n), draws.min(n)))
}

fn point(k: usize) -> Result<FiniteTable, SciOxideError> {
    let mut masses = storage(1)?;
    masses.push(0.0);
    FiniteTable::new(k, masses)
}

/// `ln P(k + 1) - ln P(k)` of Fisher's urn with `log_odds`, for `lo <= k < hi`.
pub(super) fn log_ratio(m: usize, n: usize, draws: usize, log_odds: f64, k: usize) -> f64 {
    // Positive factors are formed as integers before conversion.
    // log1p near unity avoids cancellation between four large logarithms.
    let a = (n - k) as f64 * (draws - k) as f64;
    let b = (k + 1) as f64 * (m - n - (draws - k) + 1) as f64;
    let relative = (a - b) / b;
    (if relative.abs() < 0.5 {
        relative.ln_1p()
    } else {
        a.ln() - b.ln()
    }) + log_odds
}

/// The adjacent mass ratio decreases monotonically; locate the mode without scanning.
fn mode(lo: usize, hi: usize, ratio: impl Fn(usize) -> f64) -> usize {
    let (mut left, mut right) = (lo, hi);
    while left < right {
        let mid = left + (right - left) / 2;
        if ratio(mid) > 0.0 {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// Draw through the univariate law's bounded HRUA kernel. Nested categories
/// share the work allowance and the word counter of the enclosing draw.
pub(super) fn central_draw<R: BitGenerator + ?Sized>(
    population: usize,
    successes: usize,
    draws: usize,
    rng: &mut R,
    work: &mut usize,
) -> Result<usize, SciOxideError> {
    CentralUrn::new(population, successes, draws)?.draw(rng, work)
}

fn fisher<S: RealScalar>(
    m: usize,
    n: usize,
    draws: usize,
    odds: f64,
    options: LawOptions<S>,
) -> Result<FiniteTable, SciOxideError> {
    let (lo, hi) = urn_support(m, n, draws)?;
    if lo == hi {
        options.admit::<f64>(1)?;
        return point(lo);
    }
    analytic_count(m)?;
    let len = hi - lo + 1;
    options.admit::<f64>(len)?;
    if len > options.work_budget().get() {
        return Err(math::exhausted());
    }
    let log_odds = odds.ln();
    let ratio = |k| log_ratio(m, n, draws, log_odds, k);
    let mode = mode(lo, hi, ratio);
    // Unnormalized log masses built outward from the mode (fixed at 0) with the
    // adjacent-ratio recurrence, so every step moves toward smaller masses and
    // no binomial coefficient is ever formed. `FiniteTable::new` normalizes.
    let mut masses = storage(len)?;
    masses.resize(len, 0.0);
    for k in (lo..mode).rev() {
        masses[k - lo] = masses[k + 1 - lo] - ratio(k);
    }
    if odds == 1.0 && (n == m - n || draws == m - draws) {
        // Exact color/draw-complement symmetry keeps paired masses identical,
        // including the two modal atoms of an odd-width central lattice.
        for k in mode..hi {
            masses[k + 1 - lo] = masses[hi - k - 1];
        }
    } else {
        for k in mode..hi {
            masses[k + 1 - lo] = masses[k - lo] + ratio(k);
        }
    }
    FiniteTable::new(lo, masses)
}

fn wallenius<S: RealScalar>(
    m: usize,
    mut n: usize,
    draws: usize,
    odds: f64,
    options: LawOptions<S>,
) -> Result<FiniteTable, SciOxideError> {
    let (lo, hi) = urn_support(m, n, draws)?;
    if lo == hi {
        options.admit::<f64>(1)?;
        return point(lo);
    }
    if odds == 1.0 {
        return fisher(m, n, draws, odds, options);
    }
    analytic_count(m)?;
    options.admit::<f64>(hi - lo + 1)?;
    // Swap colors, never complement draws (the latter changes the Wallenius model).
    let reflect = n > m - n;
    let log_odds = if reflect {
        n = m - n;
        -odds.ln()
    } else {
        odds.ln()
    };
    let failures = m - n;
    let mut updates = 0usize;
    for t in 1..=draws {
        let width = t.min(n) - t.saturating_sub(failures) + 1;
        updates = updates.checked_add(width).ok_or_else(math::exhausted)?;
        if updates > options.work_budget().get() {
            return Err(math::exhausted());
        }
    }
    // Wallenius has no closed-form pmf that is cheap to evaluate, so its law
    // is built by forward dynamic programming over the draw sequence:
    // row[k] = log P(k successes after t draws). Iterating k downward lets the
    // single row be updated in place.
    options.admit::<f64>(draws.min(n) + 1)?;
    let mut row = storage(draws.min(n) + 1)?;
    row.resize(draws.min(n) + 1, f64::NEG_INFINITY);
    row[0] = 0.0;
    // Return logarithmic probabilities of selecting each color without forming odds*count.
    let transition = |t: usize, k: usize| -> Result<(f64, f64), SciOxideError> {
        let a = n - k;
        let b = failures - (t - k);
        if a == 0 {
            return Ok((f64::NEG_INFINITY, 0.0));
        }
        if b == 0 {
            return Ok((0.0, f64::NEG_INFINITY));
        }
        let delta = log_odds + (a as f64).ln() - (b as f64).ln();
        Ok(if delta >= 0.0 {
            let q = -logadd(0.0, delta)?;
            (log1mexp(q)?, q)
        } else {
            let p = -logadd(0.0, -delta)?;
            (p, log1mexp(p)?)
        })
    };
    for t in 1..=draws {
        let start = t.saturating_sub(failures);
        let end = t.min(n);
        for k in (start..=end).rev() {
            let from_red =
                if k > 0 && k - 1 <= (t - 1).min(n) && k > (t - 1).saturating_sub(failures) {
                    row[k - 1] + transition(t - 1, k - 1)?.0
                } else {
                    f64::NEG_INFINITY
                };
            let from_white = if k <= (t - 1).min(n) && k >= (t - 1).saturating_sub(failures) {
                row[k] + transition(t - 1, k)?.1
            } else {
                f64::NEG_INFINITY
            };
            row[k] = logadd(from_red, from_white)?;
        }
    }
    let start = draws.saturating_sub(failures);
    let end = draws.min(n);
    let mut masses = storage(end - start + 1)?;
    if reflect {
        masses.extend(row[start..=end].iter().rev().copied());
    } else {
        masses.extend_from_slice(&row[start..=end]);
    }
    FiniteTable::new(lo, masses)
}

macro_rules! urn {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $name<S: RealScalar> {
            table: FiniteTable,
            options: LawOptions<S>,
            moments: (f64, f64),
        }
        impl<S: RealScalar> CountSupport for $name<S> {
            fn count_support(&self) -> (usize, Option<usize>) {
                (self.table.offset, Some(self.table.maximum()))
            }
        }
        impl<S: RealScalar> Kernel<S> for $name<S> {
            fn options(&self) -> LawOptions<S> {
                self.options
            }
            fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
                Ok(self.table.mass(k))
            }
            fn tail_with_work(
                &self,
                k: usize,
                u: bool,
                _left: &mut usize,
            ) -> Result<f64, SciOxideError> {
                Ok(self.table.tail(k, u))
            }
            fn average(&self) -> Result<f64, SciOxideError> {
                Ok(self.moments.0)
            }
            fn spread(&self) -> Result<f64, SciOxideError> {
                Ok(self.moments.1)
            }
        }
        count_capabilities!($name);
    };
}
/// Hypergeometric distribution: the number of successes in `n` draws without replacement.
///
/// An urn holds `N` items, `K` of them successes. The mass function is
///
/// ```text
/// P(k) = C(K, k) C(N − K, n − k) / C(N, n),   max(0, n − (N − K)) ≤ k ≤ min(n, K)
/// ```
///
/// with mean `n K / N`. The arguments of [`Hypergeometric::new`] are
/// `population = N`, `successes = K` and `draws = n`. Tails sum masses with
/// the adjacent-mass ratio recurrence, within the
/// [work limit](crate#work-limits). Draws use Stadlober's ratio-of-uniforms
/// algorithm HRUA, which has bounded expected time.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Hypergeometric, Mass, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival};
/// // 3 draws from 10 items, 4 of them successes: P(k) = C(4,k) C(6,3-k) / C(10,3).
/// let law = Hypergeometric::<f64>::new(10, 4, 3)?;
/// assert!((law.pmf(1)? - 0.5).abs() < 1e-14 && (law.pmf(2)? - 0.3).abs() < 1e-14);
/// assert!((law.cdf(1)? - 2.0 / 3.0).abs() < 1e-14 && (law.sf(1)? - 1.0 / 3.0).abs() < 1e-14);
/// assert!((law.mean()? - 1.2).abs() < 1e-14 && law.quantile(0.5)? == 1);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? <= 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// E. Stadlober, "The ratio of uniforms approach for generating discrete
/// random variates", *J. Comput. Appl. Math.* 31(1), 1990, 181–189.
#[derive(Debug, Clone, Copy)]
pub struct Hypergeometric<S: RealScalar> {
    urn: CentralUrn,
    options: LawOptions<S>,
}
impl<S: RealScalar> CountSupport for Hypergeometric<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (self.urn.lo, Some(self.urn.hi))
    }
}
impl<S: RealScalar> Kernel<S> for Hypergeometric<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        Ok(self.urn.mass(k))
    }
    fn tail_with_work(
        &self,
        k: usize,
        upper: bool,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        self.urn.tail(k, upper, left)
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(self.urn.mean())
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        Ok(self.urn.variance())
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        self.urn.draw(rng, &mut self.options.work_budget().get())
    }
}
count_capabilities!(Hypergeometric);

urn!(
    /// Fisher's noncentral hypergeometric distribution with odds ratio `ω > 0`.
    ///
    /// With the urn of [`Hypergeometric`], the mass function is
    ///
    /// ```text
    /// P(k) ∝ C(K, k) C(N − K, n − k) ω^k
    /// ```
    ///
    /// on the hypergeometric support. It is the law of independent binomial
    /// selections of each color, conditioned on their total being `n`. The
    /// constructor builds a table of masses and both tails, so mass and tail
    /// queries take O(1) time. `ω = 1` is the central law.
    ///
    /// ```
    /// use scioxide_random::{Cdf, CountDistribution, FisherNoncentralHypergeometric, Mass};
    /// use scioxide_random::{Quantile, SeedableRng, SplitMix64, Survival};
    /// // Odds 2: P(k) is proportional to C(4,k) C(6,3-k) 2^k = 20, 120, 144, 32.
    /// let law = FisherNoncentralHypergeometric::new(10, 4, 3, 2.0_f64)?;
    /// assert!((law.pmf(1)? - 120.0 / 316.0).abs() < 1e-14);
    /// assert!((law.cdf(1)? - 140.0 / 316.0).abs() < 1e-14);
    /// assert!((law.sf(1)? - 176.0 / 316.0).abs() < 1e-14);
    /// assert_eq!(law.quantile(0.5)?, 2);
    /// assert!(law.sample(&mut SplitMix64::from_seed(42))? <= 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    FisherNoncentralHypergeometric
);
urn!(
    /// Wallenius' noncentral hypergeometric distribution with odds ratio `ω > 0`.
    ///
    /// Items are drawn one at a time without replacement. With the urn of
    /// [`Hypergeometric`], each remaining success item has weight `ω` and each
    /// remaining failure item has weight 1. The law of the success count has no
    /// cheap closed form. The constructor computes it by dynamic programming
    /// over the `n` draws and builds a table of masses and both tails. `ω = 1`
    /// is the central law.
    ///
    /// ```
    /// use scioxide_random::{Cdf, CountDistribution, Mass, Quantile, SeedableRng, SplitMix64};
    /// use scioxide_random::{Survival, WalleniusNoncentralHypergeometric};
    /// // Successes weigh 2: no success in 3 draws has probability 6/14 * 5/13 * 4/12.
    /// let law = WalleniusNoncentralHypergeometric::new(10, 4, 3, 2.0_f64)?;
    /// assert!((law.pmf(0)? - 5.0 / 91.0).abs() < 1e-14);
    /// assert!((law.sf(2)? - 4.0 / 35.0).abs() < 1e-14); // 8/14 * 6/12 * 4/10
    /// assert!((law.cdf(2)? + law.sf(2)? - 1.0).abs() < 1e-14);
    /// assert_eq!(law.quantile(0.5)?, 2);
    /// assert!(law.sample(&mut SplitMix64::from_seed(42))? <= 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    WalleniusNoncentralHypergeometric
);

impl<S: RealScalar> Hypergeometric<S> {
    /// Creates a hypergeometric law with `draws` draws from an urn of
    /// `population` items, `successes` of them successes.
    ///
    /// Construction takes O(1) time and storage.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `successes > population` or `draws > population`.
    /// - `NotImplemented` if the support has more than one point and
    ///   `population > 2⁵³`.
    pub fn new(population: usize, successes: usize, draws: usize) -> Result<Self, SciOxideError> {
        Self::with_options(population, successes, draws, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        population: usize,
        successes: usize,
        draws: usize,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            urn: CentralUrn::new(population, successes, draws)?,
            options,
        })
    }
}
macro_rules! biased_constructor {
    ($name:ident,$kernel:ident) => {
        impl<S: RealScalar> $name<S> {
            /// Creates the law with `draws` draws from an urn of `population`
            /// items, `successes` of them successes, and odds ratio `odds`.
            ///
            /// The law stores three tables over the support. Building them
            /// counts toward the work limit.
            ///
            /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
            /// tolerance and work limits.
            ///
            /// # Errors
            ///
            /// - `NonFinite` if `odds` is NaN or infinite.
            /// - `DomainError` if `odds ≤ 0`, `successes > population` or
            ///   `draws > population`.
            /// - `NotImplemented` if the support has more than one point and
            ///   `population > 2⁵³`.
            /// - `ConvergenceFailure` if building the tables exceeds the work
            ///   limit.
            /// - `AllocationFailure` if the tables cannot be allocated.
            /// - `ResourceLimit` if a table exceeds the configured byte limit.
            pub fn new(
                population: usize,
                successes: usize,
                draws: usize,
                odds: S,
            ) -> Result<Self, SciOxideError> {
                Self::with_options(population, successes, draws, odds, LawOptions::default())
            }

            /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::new`].
            /// - `InvalidInput` if the tolerance in `options` is outside the
            ///   range accepted by [`LawOptions`].
            pub fn with_options(
                population: usize,
                successes: usize,
                draws: usize,
                odds: S,
                options: LawOptions<S>,
            ) -> Result<Self, SciOxideError> {
                options.validate()?;
                let odds = math::parameter(odds, true, scioxide_core::argument!("odds"))?;
                let table = $kernel(population, successes, draws, odds, options)?;
                let moments = table.moments();
                Ok(Self {
                    table,
                    options,
                    moments,
                })
            }
        }
    };
}
biased_constructor!(FisherNoncentralHypergeometric, fisher);
biased_constructor!(WalleniusNoncentralHypergeometric, wallenius);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Cdf, LogCdf, LogMass, LogSurvival, Quantile, Survival};

    struct Word {
        word: Option<u64>,
        calls: usize,
    }
    impl BitGenerator for Word {
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u64().map(|w| w as u32)
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.calls += 1;
            self.word
                .ok_or_else(|| SciOxideError::not_implemented("injected provider failure"))
        }
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u32()
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.next_u64()
        }
        fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
            self.next_u64().map(|_| ())
        }
    }

    #[test]
    fn central_analytics_match_independently_normalized_fisher_tables() {
        for (population, successes, draws) in [
            (20, 8, 6),
            (40, 20, 20),
            (60, 25, 50),
            (100, 80, 90),
            (20, 1, 10),
            (20, 19, 10),
            (20, 10, 1),
            (20, 10, 19),
            (2000, 1000, 1000),
        ] {
            let law = Hypergeometric::<f64>::new(population, successes, draws).unwrap();
            let table = fisher(
                population,
                successes,
                draws,
                1.0,
                LawOptions::<f64>::default(),
            )
            .unwrap();
            for k in 0..=population {
                for (actual, expected) in [
                    (law.log_pmf(k).unwrap(), table.mass(k)),
                    (law.log_cdf(k).unwrap(), table.tail(k, false)),
                    (law.log_sf(k).unwrap(), table.tail(k, true)),
                ] {
                    assert!(
                        actual == expected
                            || (actual - expected).abs() <= 5e-12 * expected.abs().max(1.0),
                        "({population}, {successes}, {draws}), k={k}: {actual} vs {expected}"
                    );
                }
            }
            for p in [0.01, 0.25, 0.5, 0.75, 0.99] {
                let k = law.quantile(p).unwrap();
                assert!(law.cdf(k).unwrap() >= p);
                assert!(k == 0 || law.cdf(k - 1).unwrap() < p);
                assert!((law.cdf(k).unwrap() + law.sf(k).unwrap() - 1.0).abs() < 2e-15);
            }
        }
    }

    #[test]
    fn central_draw_charges_candidates_before_words_and_propagates_errors() {
        let mut rng = Word {
            word: Some(1_u64 << 63),
            calls: 0,
        };
        assert_eq!(central_draw(10, 10, 3, &mut rng, &mut 0).unwrap(), 3);
        assert_eq!(rng.calls, 0);
        let error = central_draw(200_000, 100_000, 100_000, &mut rng, &mut 0).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(rng.calls, 0);
        let mut work = 1;
        assert_eq!(
            central_draw(200_000, 100_000, 100_000, &mut rng, &mut work).unwrap(),
            50_000
        );
        assert_eq!((rng.calls, work), (2, 0));
        let mut failing = Word {
            word: None,
            calls: 0,
        };
        let error = central_draw(20, 8, 6, &mut failing, &mut 7).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
        assert_eq!(failing.calls, 1);
        let mut rejecting = Word {
            word: Some(0),
            calls: 0,
        };
        let mut work = 2048;
        let error = central_draw(200_000, 100_000, 100_000, &mut rejecting, &mut work).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!((rejecting.calls, work), (2048, 1024));
    }
}
