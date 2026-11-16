//! Tables of log masses and of both tails for laws with finite support.
use crate::law::{log1mexp, logadd};
use alloc::vec::Vec;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(unused_imports)] // Needed when float methods resolve through RealScalar under no_std.
use scioxide_core::RealScalar as _;
use scioxide_core::SciOxideError;

pub(super) fn storage(n: usize) -> Result<Vec<f64>, SciOxideError> {
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("finite_law_preparation"), n)
    })?;
    Ok(values)
}

#[derive(Debug, Clone)]
pub(super) struct FiniteTable {
    pub(super) offset: usize,
    masses: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl FiniteTable {
    pub(super) fn new(offset: usize, mut masses: Vec<f64>) -> Result<Self, SciOxideError> {
        let norm = masses.iter().copied().try_fold(f64::NEG_INFINITY, logadd)?;
        crate::law::finite(norm)?;
        masses.iter_mut().for_each(|m| *m -= norm);
        // Both cumulative tails are accumulated independently from their own
        // end, so each starts from its smallest masses and keeps relative
        // precision far into the tail.
        let mut lower = storage(masses.len())?;
        let mut cumulative = f64::NEG_INFINITY;
        for &mass in &masses {
            cumulative = logadd(cumulative, mass)?;
            lower.push(crate::law::log_probability(cumulative)?);
        }
        let mut upper = storage(masses.len())?;
        upper.resize(masses.len(), f64::NEG_INFINITY);
        cumulative = f64::NEG_INFINITY;
        for (sf, &mass) in upper.iter_mut().zip(&masses).rev() {
            *sf = crate::law::log_probability(cumulative)?;
            cumulative = logadd(cumulative, mass)?;
        }
        // Equal independently accumulated tails identify the half-probability
        // crossing despite rounding in their shared normalizer. Otherwise retain
        // the small tail, without another normalization/transcendental per atom.
        for (lo, hi) in lower.iter_mut().zip(&mut upper) {
            if *lo == *hi {
                *lo = -core::f64::consts::LN_2;
                *hi = *lo;
            } else if *lo < *hi {
                *hi = log1mexp(*lo)?;
            } else {
                *lo = log1mexp(*hi)?;
            }
        }
        Ok(Self {
            offset,
            masses,
            lower,
            upper,
        })
    }
    pub(super) fn maximum(&self) -> usize {
        self.offset + (self.masses.len() - 1)
    }
    pub(super) fn mass(&self, k: usize) -> f64 {
        k.checked_sub(self.offset)
            .and_then(|j| self.masses.get(j))
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }
    pub(super) fn tail(&self, k: usize, upper: bool) -> f64 {
        if k < self.offset {
            return if upper { 0.0 } else { f64::NEG_INFINITY };
        }
        if upper {
            self.upper
                .get(k - self.offset)
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
        } else {
            self.lower.get(k - self.offset).copied().unwrap_or(0.0)
        }
    }
    pub(super) fn moments(&self) -> (f64, f64) {
        let mean = self
            .masses
            .iter()
            .enumerate()
            .map(|(j, m)| j as f64 * m.exp())
            .sum::<f64>();
        let variance = self
            .masses
            .iter()
            .enumerate()
            .map(|(j, m)| (j as f64 - mean).powi(2) * m.exp())
            .sum();
        (self.offset as f64 + mean, variance)
    }
}
