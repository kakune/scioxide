//! Piecewise-uniform law on given bins.
use crate::law::{argument, finite, invalid, logadd, logdiff, probability};
use crate::{
    Cdf, Density, Distribution, Entropy, InverseSurvival, LogCdf, LogDensity, LogSurvival, Mean,
    Quantile, Rng, Support, Survival, Variance,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{ExtendedInterval, RealScalar, Scaled, ScaledAccumulator, SciOxideError};
/// Piecewise-uniform law on bins `[e₀, e₁), [e₁, e₂), …, [eₙ₋₁, eₙ]` with given masses.
///
/// With bin masses `w₁, …, wₙ` normalized to `pᵢ = wᵢ / Σⱼ wⱼ`, the density is `pᵢ / (eᵢ − eᵢ₋₁)`
/// on bin `i`, and the distribution function is linear within each bin. The bins are half-open
/// for the density, and the support is the closure of the bins with positive mass. The law does
/// not fit data: you supply the edges and the masses ([`Self::from_masses`]) or the density
/// heights ([`Self::from_densities`]). Masses are stored in log form, so tails stay accurate
/// for very unequal masses.
///
/// Construction costs O(n) time and storage. The density, tails, quantiles and samples cost
/// O(log n) by binary search over the cumulative masses, and sampling inverts one uniform. The
/// mean, variance and entropy cost O(n).
/// Means and variances combine logarithmic weights with scaled moments before
/// final rounding. Zero-mass bins do not contribute to these moments; a nonzero
/// result outside the scalar range returns numerical `Underflow` or `Overflow`.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Density, Distribution, Histogram, Mean, SeedableRng, SplitMix64};
/// let law = Histogram::from_densities(&[0.0_f64, 1.0, 3.0], &[0.5, 0.25])?;
/// assert!((law.pdf(2.0)? - 0.25).abs() < 1e-15);
/// assert!((law.mean()? - 1.25).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(86);
/// assert!((0.0..=3.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Histogram<S> {
    edges: Vec<S>,
    log_mass: Vec<S>,
    log_prefix: Vec<S>,
    log_suffix: Vec<S>,
    /// Linear cumulative masses for sampling, from `log_prefix`.
    prefix: Vec<S>,
    /// Linear bin masses for sampling, from `log_mass`.
    mass: Vec<S>,
    first: usize,
    last: usize,
}
impl<S: RealScalar> Histogram<S> {
    /// Creates the law from `n + 1` increasing edges and `n` nonnegative bin masses.
    ///
    /// The masses need not sum to one. Bins of zero mass are allowed.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `masses` is empty or `edges.len() ≠ masses.len() + 1`.
    /// - `NonFinite` if an edge or mass is NaN or infinite.
    /// - `DomainError` if a mass is negative, the edges are not strictly increasing, or all
    ///   masses are zero.
    /// - `NumericalInstability` if a bin width overflows.
    /// - `AllocationFailure` if storage cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{Histogram, Quantile};
    /// let law = Histogram::from_masses(&[0.0_f64, 1.0, 3.0], &[1.0, 1.0])?;
    /// assert_eq!(law.quantile(0.5)?, 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_masses(edges: &[S], masses: &[S]) -> Result<Self, SciOxideError> {
        Self::prepare(edges, masses, false, scioxide_core::argument!("masses"))
    }
    /// Creates the law from `n + 1` increasing edges and `n` nonnegative density heights.
    ///
    /// Bin `i` gets mass `hᵢ (eᵢ − eᵢ₋₁)` before normalization.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::from_masses`], with `densities` in place of `masses`.
    pub fn from_densities(edges: &[S], densities: &[S]) -> Result<Self, SciOxideError> {
        Self::prepare(
            edges,
            densities,
            true,
            scioxide_core::argument!("densities"),
        )
    }
    /// Builds the law from masses or, with `density`, from density heights.
    /// `weights_argument` names the weight parameter in errors.
    fn prepare(
        edges: &[S],
        weights: &[S],
        density: bool,
        weights_argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        if weights.is_empty() || edges.len().checked_sub(1) != Some(weights.len()) {
            return Err(invalid(
                weights_argument,
                "histogram requires one more edge than weights",
            ));
        }
        let mut log_mass =
            try_with_capacity(weights.len(), scioxide_core::quantity!("histogram_masses"))?;
        let mut log_prefix =
            try_with_capacity(edges.len(), scioxide_core::quantity!("histogram_prefix"))?;
        log_prefix.push(S::NEG_INFINITY);
        let mut total = S::NEG_INFINITY;
        for (edge, &w) in edges.windows(2).zip(weights) {
            argument(edge[0], "edges")?;
            argument(edge[1], "edges")?;
            crate::law::nonnegative(w, weights_argument)?;
            let width = crate::law::finite(edge[1] - edge[0])?;
            if width <= S::ZERO {
                return Err(crate::law::domain(
                    scioxide_core::argument!("edges"),
                    "histogram edges must be strictly increasing",
                ));
            }
            // Density input is converted to bin mass (density x width); both
            // forms are then normalized to log masses summing to one.
            let log = w.ln() + if density { width.ln() } else { S::ZERO };
            total = logadd(total, log)?;
            log_mass.push(log);
            log_prefix.push(total);
        }
        if !total.is_finite() {
            return Err(crate::law::domain(
                weights_argument,
                "histogram has no positive mass",
            ));
        }
        let first = weights
            .iter()
            .position(|&w| w > S::ZERO)
            .ok_or_else(|| invalid(weights_argument, "empty histogram"))?;
        let last = weights
            .iter()
            .rposition(|&w| w > S::ZERO)
            .ok_or_else(|| invalid(weights_argument, "empty histogram"))?;
        let mut log_suffix =
            try_with_capacity(edges.len(), scioxide_core::quantity!("histogram_suffix"))?;
        log_suffix.push(S::NEG_INFINITY);
        let mut sum = S::NEG_INFINITY;
        for &mass in log_mass.iter().rev() {
            sum = logadd(sum, mass)?;
            log_suffix.push((sum - total).min(S::ZERO));
        }
        log_suffix.reverse();
        log_suffix[..=first].fill(S::ZERO);
        log_prefix
            .iter_mut()
            .for_each(|x| *x = (*x - total).min(S::ZERO));
        log_mass.iter_mut().for_each(|x| *x -= total);
        let mut owned =
            try_with_capacity(edges.len(), scioxide_core::quantity!("histogram_edges"))?;
        owned.extend_from_slice(edges);
        let mut prefix =
            try_with_capacity(edges.len(), scioxide_core::quantity!("histogram_prefix"))?;
        prefix.extend(log_prefix.iter().map(|log| log.exp()));
        let mut mass =
            try_with_capacity(weights.len(), scioxide_core::quantity!("histogram_masses"))?;
        mass.extend(log_mass.iter().map(|log| log.exp()));
        Ok(Self {
            edges: owned,
            log_mass,
            log_prefix,
            log_suffix,
            prefix,
            mass,
            first,
            last,
        })
    }
    fn tail(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let right = self.edges.partition_point(|&edge| edge <= x);
        if right == 0 {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if right == self.edges.len() {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        let i = right - 1;
        let width = self.edges[i + 1] - self.edges[i];
        let lower = logadd(
            self.log_prefix[i],
            self.log_mass[i] + (x - self.edges[i]).ln() - width.ln(),
        )?;
        let higher = logadd(
            self.log_suffix[i + 1],
            self.log_mass[i] + (self.edges[i + 1] - x).ln() - width.ln(),
        )?;
        // Form the near-one tail from its small complement.
        Ok(crate::law::smaller_tail_first(upper, upper, |upper| {
            Ok(if upper { higher } else { lower })
        })?
        .0)
    }
    fn tail_at_log(&self, y: S, upper: bool) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let log_edge = |e: S| if e > S::ZERO { e.ln() } else { S::NEG_INFINITY };
        let right = self.edges.partition_point(|&e| log_edge(e) <= y);
        if right == 0 {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if right == self.edges.len() {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        let i = right - 1;
        let log_width = (self.edges[i + 1] - self.edges[i]).ln();
        let lower = logadd(
            self.log_prefix[i],
            self.log_mass[i]
                + logdiff(y, log_edge(self.edges[i]), scioxide_core::argument!("self"))?
                - log_width,
        )?;
        let higher = logadd(
            self.log_suffix[i + 1],
            self.log_mass[i]
                + logdiff(
                    log_edge(self.edges[i + 1]),
                    y,
                    scioxide_core::argument!("self"),
                )?
                - log_width,
        )?;
        Ok(crate::law::smaller_tail_first(upper, upper, |upper| {
            Ok(if upper { higher } else { lower })
        })?
        .0)
    }
    fn invert(&self, p: S, upper: bool) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                self.edges[self.first]
            } else {
                self.edges[self.last + 1]
            });
        }
        // Search from the requested tail to preserve tiny probabilities and
        // the lower/upper generalized-inverse choices at zero-mass gaps.
        let target = p.ln();
        let i = if upper {
            self.log_suffix.partition_point(|&sum| sum >= target) - 1
        } else {
            self.log_prefix.partition_point(|&sum| sum < target) - 1
        };
        let before = if upper {
            self.log_suffix[i + 1]
        } else {
            self.log_prefix[i]
        };
        let fraction = (logdiff(target, before, scioxide_core::argument!("self"))?
            - self.log_mass[i])
            .exp()
            .min(S::ONE);
        let width = self.edges[i + 1] - self.edges[i];
        finite(if upper {
            self.edges[i + 1] - fraction * width
        } else {
            self.edges[i] + fraction * width
        })
    }
}
impl<S: RealScalar> Support<S> for Histogram<S> {
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.edges
            .windows(2)
            .zip(&self.log_mass)
            .filter(|(_, m)| m.is_finite())
            .try_fold(ExtendedInterval::empty(), |h, (e, _)| {
                let clipped =
                    ExtendedInterval::new(Some(e[0]), Some(e[1]))?.intersection(&interval);
                Ok(if clipped.bounds().is_some_and(|(a, b)| a != b) {
                    h.hull(&clipped)
                } else {
                    h
                })
            })
    }

    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        ExtendedInterval::new(
            Some(self.edges[self.first]),
            Some(self.edges[self.last + 1]),
        )
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        if !x.is_finite() {
            return Ok(false);
        }
        let right = self.edges.partition_point(|&edge| edge < x);
        Ok((right > 0
            && right <= self.log_mass.len()
            && self.log_mass[right - 1].is_finite()
            && x <= self.edges[right])
            || (right < self.log_mass.len()
                && x == self.edges[right]
                && self.log_mass[right].is_finite()))
    }
}
impl<S: RealScalar> LogDensity<S> for Histogram<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let right = self.edges.partition_point(|&edge| edge <= x);
        Ok(if right == 0 || right == self.edges.len() {
            S::NEG_INFINITY
        } else {
            let i = right - 1;
            self.log_mass[i] - (self.edges[i + 1] - self.edges[i]).ln()
        })
    }
}
impl<S: RealScalar> Density<S> for Histogram<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        finite(self.log_pdf(x)?.exp())
    }
}
impl<S: RealScalar> LogCdf<S> for Histogram<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.tail(x, false)
    }
}
impl<S: RealScalar> LogSurvival<S> for Histogram<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.tail(x, true)
    }
}
impl<S: RealScalar> Cdf<S> for Histogram<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf(x)?.exp())
    }
}
impl<S: RealScalar> Survival<S> for Histogram<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf(x)?.exp())
    }
}
impl<S: RealScalar> Quantile<S> for Histogram<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        self.invert(p, false)
    }
}
impl<S: RealScalar> InverseSurvival<S> for Histogram<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        self.invert(p, true)
    }
}
impl<S: RealScalar> Distribution<S> for Histogram<S> {
    /// Draws by inversion of a uniform on `(0, 1)`: one binary search over the bins with
    /// positive mass and a linear interpolation within the bin.
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let u = crate::util::validated_open_float(rng)?;
        let i = (self.prefix.partition_point(|&cumulative| cumulative <= u) - 1)
            .clamp(self.first, self.last);
        let fraction = ((u - self.prefix[i]) / self.mass[i]).clamp(S::ZERO, S::ONE);
        finite(self.edges[i] + fraction * (self.edges[i + 1] - self.edges[i]))
    }
}
impl<S: RealScalar> Mean<S> for Histogram<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.mean_scaled()?.try_value_to()
    }
}
fn bin_midpoint<S: RealScalar>(edges: &[S]) -> Result<Scaled<f64>, SciOxideError> {
    Scaled::from_value(edges[0].into_f64())?
        .add(Scaled::from_value(edges[1].into_f64())?)?
        .mul(Scaled::HALF)
}
impl<S: RealScalar> Histogram<S> {
    fn mean_scaled(&self) -> Result<Scaled<f64>, SciOxideError> {
        let mut sum = ScaledAccumulator::ZERO;
        for (edges, mass) in self
            .edges
            .windows(2)
            .zip(&self.log_mass)
            .filter(|(_, m)| m.is_finite())
        {
            sum.add_product(
                Scaled::from_log(1.0, mass.into_f64())?,
                bin_midpoint(edges)?,
            )?;
        }
        sum.total()
    }
}
impl<S: RealScalar> Variance<S> for Histogram<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let mean = self.mean_scaled()?;
        let mut sum = ScaledAccumulator::ZERO;
        for (edges, mass) in self
            .edges
            .windows(2)
            .zip(&self.log_mass)
            .filter(|(_, m)| m.is_finite())
        {
            let deviation = bin_midpoint(edges)?.sub(mean)?;
            let width = Scaled::from_value(edges[1].into_f64())?
                .sub(Scaled::from_value(edges[0].into_f64())?)?;
            let within = width.square()?.div(Scaled::from_value(12.0)?)?;
            sum.add_product(
                Scaled::from_log(1.0, mass.into_f64())?,
                within.add(deviation.square()?)?,
            )?;
        }
        sum.total()?.try_value_to()
    }
}
impl<S: RealScalar> Entropy<S> for Histogram<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        finite(
            self.edges
                .windows(2)
                .zip(&self.log_mass)
                .filter(|(_, m)| m.is_finite())
                .fold(S::ZERO, |v, (e, m)| v - m.exp() * (*m - (e[1] - e[0]).ln())),
        )
    }
}

impl<S: RealScalar> crate::LogCoordinateLaw<S> for Histogram<S> {
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.tail_at_log(y, false)
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.tail_at_log(y, true)
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let right = self.edges.partition_point(|&e| e <= S::ZERO || e.ln() <= y);
        if right == 0 || right == self.edges.len() {
            return Ok(S::NEG_INFINITY);
        }
        let i = right - 1;
        Ok(self.log_mass[i] + y - (self.edges[i + 1] - self.edges[i]).ln())
    }
    fn log_support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        let _ = crate::nonlinear::logarithmic_support(self.support()?)?;
        self.edges
            .windows(2)
            .zip(&self.log_mass)
            .filter(|(_, m)| m.is_finite())
            .try_fold(ExtendedInterval::empty(), |h, (e, _)| {
                let bin = crate::nonlinear::logarithmic_support(ExtendedInterval::new(
                    Some(e[0]),
                    Some(e[1]),
                )?)?
                .intersection(&interval);
                Ok(
                    if bin.bounds().is_some_and(|(a, b)| a.is_some() && a == b) {
                        h
                    } else {
                        h.hull(&bin)
                    },
                )
            })
    }
    fn contains_at_log(&self, y: S) -> Result<bool, SciOxideError> {
        if !y.is_finite() {
            return Ok(false);
        }
        Ok(self.edges.windows(2).zip(&self.log_mass).any(|(e, m)| {
            m.is_finite() && e[1] > S::ZERO && y <= e[1].ln() && (e[0] <= S::ZERO || y >= e[0].ln())
        }))
    }
}
