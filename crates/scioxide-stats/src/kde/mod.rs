//! Gaussian kernel density estimation, optionally truncated to a box.
mod evaluate;
mod kernel;
mod mass;
mod workspace;

pub(crate) use workspace::{KdeWorkspaceRequirements, RawKdeWorkspace, kde_into};

use crate::arithmetic::Compensated;
use crate::central::Central;
use crate::central::merge_cross;
use crate::error_helpers::invalid;
use crate::{KernelWeights, MomentOrder, ObservationData};
use alloc::vec::Vec;
use kernel::{Kernel, KernelWeight, OrdinaryInterval};
use mass::{NormalMass, normal_interval_mass};
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{ExecutionContext, Interval, RealScalar, SciOxideError};

/// Kernel covariance (bandwidth matrix) of a [`GaussianKde`](crate::GaussianKde).
///
/// The estimate is `f̂(x) = Σ wᵢ φ_H(x − xᵢ) / Σ wᵢ`, where `xᵢ` are the
/// observations, `wᵢ` their weights and `φ_H` the Gaussian density with mean
/// zero and covariance `H`. The rules `Scott`, `Silverman` and `Factor` scale the sample covariance:
///
/// ```text
/// H = f² Σ̂
/// ```
///
/// where `Σ̂` is the unbiased weighted sample covariance of the observations
/// (only its diagonal with [`KdeCovariance::Diagonal`]) and `f` is the bandwidth
/// factor. The factor multiplies standard deviations, so the covariance scales
/// by `f²`. The rules use the effective sample size `n_eff` (the replicated count
/// for unweighted data and frequency weights, and Kish's `(Σwᵢ)² / Σwᵢ²` for
/// mass weights) and the dimension `d`. They fail if `Σ̂` is singular, for
/// example for one observation or a constant sample. The absolute bandwidths
/// accept such samples.
///
/// # References
///
/// - D. W. Scott, *Multivariate Density Estimation*, Wiley, 1992.
/// - B. W. Silverman, *Density Estimation for Statistics and Data Analysis*,
///   Chapman & Hall, 1986.
#[derive(Clone, Copy, Debug)]
pub enum KdeBandwidth<'a, S: RealScalar> {
    /// Scott's rule, `f = n_eff^(−1/(d+4))`.
    Scott,
    /// Silverman's rule, `f = (n_eff (d + 2) / 4)^(−1/(d+4))`.
    Silverman,
    /// A given factor `f`, positive and finite.
    Factor(S),
    /// Diagonal `H` with these `d` positive standard deviations.
    StandardDeviations(&'a [S]),
    /// `H` itself: a symmetric positive definite `d × d` matrix in row-major order.
    Covariance(&'a [S]),
}
/// Structure of the sample covariance used by the bandwidth rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdeCovariance {
    /// Full covariance, keeping correlations between coordinates.
    Full,
    /// Variances only, so each coordinate has an independent kernel.
    Diagonal,
}
/// Options for [`GaussianKde::build`](crate::GaussianKde::build).
///
/// The default is Scott's rule with a full covariance, no support bounds, and
/// the default [`StatisticsOptions`](crate::StatisticsOptions).
#[derive(Clone, Copy)]
pub struct KdeOptions<'a, S: RealScalar> {
    /// Bandwidth rule or explicit bandwidth.
    pub(crate) bandwidth: KdeBandwidth<'a, S>,
    /// Structure of the fitted covariance.
    pub(crate) covariance: KdeCovariance,
    /// Optional box that every observation must lie in.
    pub(crate) support: Option<&'a [Interval<S>]>,
    /// Missing-value and non-finite policies for the observations.
    pub(crate) admission: crate::StatisticsOptions,
}
impl<S: RealScalar> Default for KdeOptions<'_, S> {
    fn default() -> Self {
        Self {
            bandwidth: KdeBandwidth::Scott,
            covariance: KdeCovariance::Full,
            support: None,
            admission: crate::StatisticsOptions::new(),
        }
    }
}

impl<'a, S: RealScalar> KdeOptions<'a, S> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the bandwidth. [`GaussianKde::build`](crate::GaussianKde::build) checks it.
    pub const fn with_bandwidth(mut self, value: KdeBandwidth<'a, S>) -> Self {
        self.bandwidth = value;
        self
    }
    /// Returns the bandwidth.
    pub const fn bandwidth(self) -> KdeBandwidth<'a, S> {
        self.bandwidth
    }
    /// Sets the structure of the fitted covariance.
    pub const fn with_covariance(mut self, value: KdeCovariance) -> Self {
        self.covariance = value;
        self
    }
    /// Returns the structure of the fitted covariance.
    pub const fn covariance(self) -> KdeCovariance {
        self.covariance
    }
    /// Sets the support: one interval per coordinate, or `None` for all of `ℝᵈ`.
    ///
    /// With a support, each kernel is a Gaussian truncated to the box and
    /// renormalized, so the density integrates to 1 over the box. Every
    /// observation must lie in the box, and the kernel covariance must be
    /// diagonal. [`GaussianKde::build`](crate::GaussianKde::build) checks this.
    pub const fn with_support(mut self, value: Option<&'a [Interval<S>]>) -> Self {
        self.support = value;
        self
    }
    /// Returns the support.
    pub const fn support(self) -> Option<&'a [Interval<S>]> {
        self.support
    }
    /// Sets the missing-value and non-finite policies for the observations.
    pub const fn with_admission(mut self, value: crate::StatisticsOptions) -> Self {
        self.admission = value;
        self
    }
    /// Returns the missing-value and non-finite policies for the observations.
    pub const fn admission(self) -> crate::StatisticsOptions {
        self.admission
    }
}

/// Weighted Gaussian KDE with its own copy of the kernel centers.
/// Evaluation uses compensated `f64` sums, and switches to scaled or logarithmic
/// arithmetic for extreme ranges.
pub(crate) struct RawGaussianKde<S: RealScalar> {
    centers: Vec<S>,
    weights: Vec<KernelWeight>,
    ordinary_interval: Option<OrdinaryInterval>,
    kernel: Kernel,
    support: Option<Vec<Interval<S>>>,
    interval_masses: Vec<NormalMass>,
    effective_size: f64,
    factor: Option<S>,
    dimensions: usize,
}
impl<S: RealScalar> RawGaussianKde<S> {
    /// Builds the estimator from one batch. A fitted bandwidth needs positive
    /// residual degrees of freedom. An absolute bandwidth accepts one observation.
    ///
    /// # Errors
    /// Fails on invalid observations, support, weights or bandwidth, a covariance
    /// that is not positive definite, no residual degrees of freedom, or
    /// allocation failure.
    pub(crate) fn prepare(
        view: ObservationData<'_, S>,
        batch: usize,
        options: KdeOptions<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if batch >= view.batches() {
            return Err(invalid(
                scioxide_core::argument!("batch"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "KDE batch index out of range",
            ));
        }
        let d = view.dimensions();
        let square = d.checked_mul(d).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("kde_covariance"))
        })?;
        let size = view
            .samples()
            .checked_mul(d)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("kde_centers")))?;
        if options.support.is_some_and(|support| {
            support.len() != d || support.iter().any(|s| s.lower() >= s.upper())
        }) {
            return Err(invalid(
                scioxide_core::argument!("options.support"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid KDE support",
            ));
        }
        let mut centers = try_with_capacity(size, scioxide_core::quantity!("kde_centers"))?;
        let mut masses = try_with_capacity(view.samples(), scioxide_core::quantity!("kde_masses"))?;
        let mut total = Compensated::<Scaled<f64>>::zero();
        let mut squares = Compensated::<Scaled<f64>>::zero();
        let mut pairs = Compensated::<Scaled<f64>>::zero();
        let mut count = 0usize;
        let mut central = try_filled(d, Central::empty(), scioxide_core::quantity!("kde_central"))?;
        let mut covariance = try_filled(
            square,
            Scaled::<f64>::ZERO,
            scioxide_core::quantity!("kde_covariance"),
        )?;
        for i in 0..view.samples() {
            context.check()?;
            let Some((mass, n)) = view.admit(
                batch,
                i,
                options.admission,
                scioxide_core::argument!("view"),
            )?
            else {
                continue;
            };
            if options
                .support
                .is_some_and(|bounds| (0..d).any(|j| !bounds[j].contains(view.value(batch, i, j))))
            {
                return Err(invalid(
                    scioxide_core::argument!("options.support"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "KDE center outside support",
                ));
            }
            let old = total.total()?;
            total = total.added(mass)?;
            let next = total.total()?;
            pairs = pairs.added(old.mul(mass)?.mul(Scaled::<f64>::from_value(2.)?)?)?;
            squares = squares.added(mass.square()?)?;
            count = count.checked_add(n).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("kde_replication_count"))
            })?;
            if !matches!(
                options.bandwidth,
                KdeBandwidth::StandardDeviations(_) | KdeBandwidth::Covariance(_)
            ) {
                let p = old.div(next)?;
                let q = mass.div(next)?;
                for a in 0..d {
                    for b in 0..=a {
                        let cross = merge_cross(
                            covariance[a * d + b],
                            Scaled::<f64>::ZERO,
                            (central[a].mean()?, central[b].mean()?),
                            (
                                scioxide_core::ScaledAccumulator::from_value(
                                    scioxide_core::RealScalar::into_f64(view.value(batch, i, a)),
                                )?,
                                scioxide_core::ScaledAccumulator::from_value(
                                    scioxide_core::RealScalar::into_f64(view.value(batch, i, b)),
                                )?,
                            ),
                            p,
                            q,
                        )?;
                        covariance[a * d + b] = cross;
                        covariance[b * d + a] = cross;
                    }
                }
                for (j, c) in central.iter_mut().enumerate() {
                    *c = if masses.is_empty() {
                        Central::singleton(view.value(batch, i, j))?
                    } else {
                        c.merged(
                            Central::singleton(view.value(batch, i, j))?,
                            crate::arithmetic::Shares::rounded(p, q),
                            MomentOrder::Second,
                        )?
                    };
                }
            }
            centers.extend((0..d).map(|j| view.value(batch, i, j)));
            masses.push(mass);
        }
        if masses.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty retained KDE sample",
            ));
        }
        let total = total.total()?;
        let frequency = !matches!(view.weights, Some(KernelWeights::Mass(_)));
        let effective_size = if frequency {
            count as f64
        } else {
            total.square()?.div(squares.total()?)?.round_to()?
        };
        let mut factor = None;
        match options.bandwidth {
            KdeBandwidth::StandardDeviations(sd) => {
                if sd.len() != d || sd.iter().any(|x| !x.is_finite() || *x <= S::ZERO) {
                    return Err(invalid(
                        scioxide_core::argument!("options.bandwidth"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid KDE standard deviations",
                    ));
                }
                covariance.fill(Scaled::<f64>::ZERO);
                for (j, value) in sd.iter().enumerate() {
                    covariance[j * d + j] =
                        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(*value))?
                            .square()?;
                }
            }
            KdeBandwidth::Covariance(values) => {
                if values.len() != square || values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid(
                        scioxide_core::argument!("options.bandwidth"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid explicit KDE covariance",
                    ));
                }
                for (out, value) in covariance.iter_mut().zip(values) {
                    *out = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(*value))?;
                }
            }
            bandwidth => {
                let correction = if frequency {
                    crate::arithmetic::count_ratio(count.saturating_sub(1), count)?
                } else {
                    pairs.total()?.div(total.square()?)?
                };
                if correction.is_zero() {
                    return Err(invalid(
                        scioxide_core::argument!("view"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "KDE sample has no residual degrees of freedom",
                    ));
                }
                let f = match bandwidth {
                    KdeBandwidth::Scott => effective_size.powf(-1. / (d as f64 + 4.)),
                    KdeBandwidth::Silverman => {
                        (effective_size * (d as f64 + 2.) / 4.).powf(-1. / (d as f64 + 4.))
                    }
                    KdeBandwidth::Factor(value) => value.into_f64(),
                    _ => {
                        return Err(invalid(
                            scioxide_core::argument!("options.bandwidth"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "invalid bandwidth state",
                        ));
                    }
                };
                if !f.is_finite() || f <= 0. {
                    return Err(invalid(
                        scioxide_core::argument!("options.bandwidth"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "bandwidth factor must be positive and finite",
                    ));
                }
                factor = Some(S::try_from_f64_lossy(f)?);
                for (i, value) in covariance.iter_mut().enumerate() {
                    *value = if options.covariance == KdeCovariance::Diagonal && i / d != i % d {
                        Scaled::<f64>::ZERO
                    } else {
                        value
                            .div(correction)?
                            .mul(Scaled::<f64>::from_value(f)?.square()?)?
                    };
                }
            }
        }
        let kernel = Kernel::new(covariance, d, scioxide_core::argument!("options.bandwidth"))?;
        if options.support.is_some() && !kernel.diagonal {
            return Err(invalid(
                scioxide_core::argument!("options.covariance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "bounded KDE requires diagonal covariance",
            ));
        }
        let support = if let Some(bounds) = options.support {
            let mut owned = try_with_capacity(d, scioxide_core::quantity!("kde_support"))?;
            owned.extend_from_slice(bounds);
            Some(owned)
        } else {
            None
        };
        let mut weights = try_with_capacity(masses.len(), scioxide_core::quantity!("kde_weights"))?;
        let mut interval_masses = if support.is_some() {
            try_filled(
                masses.len() * d,
                NormalMass {
                    width: Scaled::<f64>::ONE,
                    log_average: 0.,
                },
                scioxide_core::quantity!("kde_interval_masses"),
            )?
        } else {
            Vec::new()
        };
        for (i, mass) in masses.iter().enumerate() {
            let log_mass = mass.div(total)?.ln_abs()?;
            let mut log_truncation = 0.;
            if let Some(bounds) = &support {
                for (j, bound) in bounds.iter().enumerate() {
                    let interval_mass = normal_interval_mass(
                        bound.lower(),
                        bound.upper(),
                        centers[i * d + j],
                        kernel.scales[j],
                    )?;
                    let z = interval_mass.log_value()?;
                    interval_masses[i * d + j] = interval_mass;
                    if !z.is_finite() {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Underflow,
                            "unrepresentable truncated kernel mass",
                        ));
                    }
                    log_truncation += z;
                }
            }
            let log_peak = log_mass + kernel.log_normalization - log_truncation;
            weights.push(KernelWeight {
                log_mass,
                log_truncation,
                log_peak,
                peak: log_peak.exp(),
            });
        }
        let ordinary_interval = OrdinaryInterval::new(&centers, &kernel, &weights);
        Ok(Self {
            centers,
            weights,
            ordinary_interval,
            kernel,
            support,
            interval_masses,
            effective_size,
            factor,
            dimensions: d,
        })
    }
    /// Returns the dimension.
    pub(crate) const fn dimensions(&self) -> usize {
        self.dimensions
    }
    /// Returns the number of centers with positive weight, not counting replications.
    pub(crate) fn observations(&self) -> usize {
        self.weights.len()
    }
    /// Returns the replicated sample size, or Kish's effective sample size for mass weights.
    pub(crate) const fn effective_size(&self) -> f64 {
        self.effective_size
    }
    /// Returns the bandwidth factor, or `None` for an absolute bandwidth.
    pub(crate) const fn bandwidth_factor(&self) -> Option<S> {
        self.factor
    }
}
