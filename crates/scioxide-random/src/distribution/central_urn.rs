//! Central hypergeometric evaluation in O(1) storage and HRUA sampling.
use super::count_math::analytic_count;
use super::count_math::{self as math};
use super::discrete_math::binomial_log_mass;
use super::finite_urn::{log_ratio, urn_support};
use crate::BitGenerator;
use crate::law::log1mexp;
use crate::util::neumaier;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(unused_imports)]
use scioxide_core::RealScalar as _;
use scioxide_core::SciOxideError;

/// Colors and draws reduce to at most N/2 before evaluating masses or sampling.
#[derive(Debug, Clone, Copy)]
pub(super) struct CentralUrn {
    population: usize,
    successes: usize,
    draws: usize,
    pub(super) lo: usize,
    pub(super) hi: usize,
    marked: usize,
    sampled: usize,
    probability: f64,
    log_denominator: f64,
    log_mode_mass: f64,
    center: f64,
    hat: f64,
}
impl CentralUrn {
    pub(super) fn new(
        population: usize,
        successes: usize,
        draws: usize,
    ) -> Result<Self, SciOxideError> {
        let (lo, hi) = urn_support(population, successes, draws)?;
        let mut urn = Self {
            population,
            successes,
            draws,
            lo,
            hi,
            marked: successes.min(population - successes),
            sampled: draws.min(population - draws),
            probability: 0.0,
            log_denominator: 0.0,
            log_mode_mass: 0.0,
            center: 0.0,
            hat: 0.0,
        };
        if lo != hi {
            let total = analytic_count(population)?;
            urn.probability = urn.sampled as f64 / total;
            urn.log_denominator = binomial_log_mass(total, urn.sampled as f64, urn.probability);
            let mean = urn.sampled as f64 * (urn.marked as f64 / total);
            let variance = mean
                * ((population - urn.marked) as f64 / total)
                * ((population - urn.sampled) as f64 / (population - 1) as f64);
            urn.center = mean + 0.5;
            // Stadlober (1989), Algorithm HRUA, Eq. 5.17. The table-mountain
            // envelope dominates the full support; no tail cutoff is applied.
            urn.hat = 1.715_527_769_921_413_5 * (variance + 0.5).sqrt() + 0.898_916_162_058_898_8;
            let mode = (((urn.sampled as u128 + 1) * (urn.marked as u128 + 1))
                / (population as u128 + 2)) as usize;
            urn.log_mode_mass = urn.reduced_mass(mode);
        }
        Ok(urn)
    }

    fn reduced_mass(&self, k: usize) -> f64 {
        let k = if self.symmetric() {
            k.min(self.hi - self.lo - k)
        } else {
            k
        };
        // Independent Bin(marked,p), Bin(N-marked,p), conditioned on their
        // sum being sampled. Loader's kernels avoid log-factorial cancellation.
        binomial_log_mass(self.marked as f64, k as f64, self.probability)
            + binomial_log_mass(
                (self.population - self.marked) as f64,
                (self.sampled - k) as f64,
                self.probability,
            )
            - self.log_denominator
    }

    fn symmetric(&self) -> bool {
        self.marked == self.population - self.marked
            || self.sampled == self.population - self.sampled
    }

    fn reduced_count(&self, mut k: usize) -> usize {
        if self.successes > self.population - self.successes {
            k = self.draws - k;
        }
        if self.draws > self.population - self.draws {
            k = self.marked - k;
        }
        k
    }

    fn original_count(&self, mut k: usize) -> usize {
        if self.draws > self.population - self.draws {
            k = self.marked - k;
        }
        if self.successes > self.population - self.successes {
            k = self.draws - k;
        }
        k
    }

    pub(super) fn mass(&self, k: usize) -> f64 {
        if k < self.lo || k > self.hi {
            f64::NEG_INFINITY
        } else if self.lo == self.hi {
            0.0
        } else {
            self.reduced_mass(self.reduced_count(k))
        }
    }

    pub(super) fn mean(&self) -> f64 {
        if self.lo == self.hi {
            return self.lo as f64;
        }
        self.draws as f64 * (self.successes as f64 / self.population as f64)
    }

    pub(super) fn variance(&self) -> f64 {
        if self.lo == self.hi {
            return 0.0;
        }
        self.mean()
            * ((self.population - self.successes) as f64 / self.population as f64)
            * ((self.population - self.draws) as f64 / (self.population - 1) as f64)
    }

    fn tail_sum(&self, k: usize, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        crate::law::charge(left)?;
        let mut atom = if upper { k + 1 } else { k };
        let initial = self.mass(atom);
        let mut weight = 1.0;
        let mut sum = (1.0, 0.0);
        while atom != if upper { self.hi } else { self.lo } {
            let ratio = if upper {
                log_ratio(self.population, self.successes, self.draws, 0.0, atom)
            } else {
                -log_ratio(self.population, self.successes, self.draws, 0.0, atom - 1)
            };
            // Adjacent ratios decrease away from the mode. The geometric
            // series bounds the unvisited tail. Retain near-f64 precision even
            // when the configured tolerance is looser.
            if ratio < 0.0 && weight / (-ratio).exp_m1() <= 4.0 * f64::EPSILON * (sum.0 + sum.1) {
                break;
            }
            crate::law::charge(left)?;
            weight *= ratio.exp();
            sum = neumaier(sum, weight);
            if upper {
                atom += 1;
            } else {
                atom -= 1;
            }
        }
        math::log_probability(initial + (sum.0 + sum.1).ln())
    }

    pub(super) fn tail(
        &self,
        k: usize,
        upper: bool,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        if k < self.lo {
            return Ok(if upper { 0.0 } else { f64::NEG_INFINITY });
        }
        if k >= self.hi {
            return Ok(if upper { f64::NEG_INFINITY } else { 0.0 });
        }
        if self.symmetric()
            && (self.hi - self.lo) % 2 == 1
            && k == self.lo + (self.hi - self.lo) / 2
        {
            return Ok(-core::f64::consts::LN_2);
        }
        let side = k as f64 >= self.mean();
        let tail = self.tail_sum(k, side, left)?;
        // A skewed mode can put the mean on the larger side. Evaluate the
        // smaller side positively rather than subtracting a near-one value.
        if tail > -core::f64::consts::LN_2 {
            let other = self.tail_sum(k, !side, left)?;
            return if upper == side {
                log1mexp(other)
            } else {
                Ok(other)
            };
        }
        if upper == side {
            Ok(tail)
        } else {
            log1mexp(tail)
        }
    }

    pub(super) fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        work: &mut usize,
    ) -> Result<usize, SciOxideError> {
        if self.lo == self.hi {
            return Ok(self.lo);
        }
        if self.sampled < 10 {
            let mut marked = self.marked;
            let mut total = self.population;
            let mut selected = 0;
            for _ in 0..self.sampled {
                crate::law::charge(work)?;
                if crate::sampling::uniform_below(total as u64, rng)? < marked as u64 {
                    selected += 1;
                    marked -= 1;
                }
                total -= 1;
            }
            return Ok(self.original_count(selected));
        }
        crate::util::rejection(|| {
            crate::law::charge(work)?;
            let u = math::open(rng)?;
            let v = math::open(rng)?;
            let proposal = self.center + self.hat * (v - 0.5) / u;
            if !(0.0..(self.marked.min(self.sampled) + 1) as f64).contains(&proposal) {
                return Ok(None);
            }
            let candidate = proposal.floor() as usize;
            let ratio = self.reduced_mass(candidate) - self.log_mode_mass;
            let accepted =
                u * (4.0 - u) - 3.0 <= ratio || (u * (u - ratio) < 1.0 && 2.0 * u.ln() <= ratio);
            Ok(accepted.then(|| self.original_count(candidate)))
        })
    }
}
