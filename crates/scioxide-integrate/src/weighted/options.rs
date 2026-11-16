//! Options of weighted quadrature and Fourier tail plans.
use crate::SubdivisionOptions;
use scioxide_core::{Budget, RealScalar};

/// Options for [`WeightedQuadraturePlan`](crate::WeightedQuadraturePlan).
///
/// The [`SubdivisionOptions`] set the tolerance, the subdivision and evaluation
/// limits and the execution policy. The moment levels, 16 by default, bound the
/// bisection depth at which a Fourier plan uses Chebyshev moments (see
/// [`WeightedQuadraturePlan::fourier`](crate::WeightedQuadraturePlan::fourier)).
/// Only depths that need moments are prepared, so a large value costs no
/// storage. Other weights ignore it.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::{SubdivisionOptions, WeightedQuadratureOptions};
///
/// let options = WeightedQuadratureOptions::<f64>::new(SubdivisionOptions::default())
///     .with_moment_levels(Budget::try_from(12)?);
/// assert_eq!(options.moment_levels().get(), 12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct WeightedQuadratureOptions<S: RealScalar> {
    pub(super) subdivision: SubdivisionOptions<S>,
    moments: Budget,
}
impl<S: RealScalar> Default for WeightedQuadratureOptions<S> {
    fn default() -> Self {
        Self::new(SubdivisionOptions::default())
    }
}
impl<S: RealScalar> WeightedQuadratureOptions<S> {
    /// Creates options with the given subdivision options and 16 moment levels.
    pub fn new(subdivision: SubdivisionOptions<S>) -> Self {
        Self {
            subdivision,
            moments: Budget::ONE.saturating_add(15),
        }
    }
    /// Replaces the subdivision options.
    pub fn with_subdivision_options(mut self, value: SubdivisionOptions<S>) -> Self {
        self.subdivision = value;
        self
    }
    /// Sets the number of bisection depths with Fourier moments.
    pub const fn with_moment_levels(mut self, value: Budget) -> Self {
        self.moments = value;
        self
    }
    /// Returns the subdivision options.
    pub const fn subdivision_options(&self) -> &SubdivisionOptions<S> {
        &self.subdivision
    }
    /// Returns the number of bisection depths with Fourier moments.
    pub const fn moment_levels(&self) -> Budget {
        self.moments
    }
}

/// Options for [`FourierTailPlan`](crate::FourierTailPlan).
///
/// The [`SubdivisionOptions`] set the tolerance of the whole integral, the
/// subdivision and evaluation limits of the whole call and the execution
/// policy. The [`WeightedQuadratureOptions`] set the moment levels of each
/// cycle. At most [`max_cycles`](Self::max_cycles) cycles, 128 by default, are
/// integrated on each half-line.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::FourierTailOptions;
///
/// let options = FourierTailOptions::<f64>::default().with_max_cycles(Budget::try_from(64)?);
/// assert_eq!(options.max_cycles().get(), 64);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct FourierTailOptions<S: RealScalar> {
    pub(super) weighted: WeightedQuadratureOptions<S>,
    cycles: Budget,
}
impl<S: RealScalar> Default for FourierTailOptions<S> {
    fn default() -> Self {
        Self::new(SubdivisionOptions::default())
    }
}
impl<S: RealScalar> FourierTailOptions<S> {
    /// Creates options with the given subdivision options, 16 moment levels and
    /// 128 cycles.
    pub fn new(subdivision: SubdivisionOptions<S>) -> Self {
        Self {
            weighted: WeightedQuadratureOptions::new(subdivision),
            cycles: Budget::ONE.saturating_add(127),
        }
    }
    /// Replaces the options of the cycle quadrature, including the subdivision
    /// options.
    pub fn with_quadrature_options(mut self, value: WeightedQuadratureOptions<S>) -> Self {
        self.weighted = value;
        self
    }
    /// Sets the maximum number of cycles per half-line.
    ///
    /// [`FourierTailPlan::new`](crate::FourierTailPlan::new) requires at least six.
    pub const fn with_max_cycles(mut self, value: Budget) -> Self {
        self.cycles = value;
        self
    }
    /// Sets the number of bisection depths with Fourier moments in each cycle.
    pub fn with_moment_levels(mut self, value: Budget) -> Self {
        self.weighted = self.weighted.with_moment_levels(value);
        self
    }
    /// Returns the options of the cycle quadrature.
    pub const fn quadrature_options(&self) -> &WeightedQuadratureOptions<S> {
        &self.weighted
    }
    /// Returns the subdivision options.
    pub const fn subdivision_options(&self) -> &SubdivisionOptions<S> {
        self.weighted.subdivision_options()
    }
    /// Returns the maximum number of cycles per half-line.
    pub const fn max_cycles(&self) -> Budget {
        self.cycles
    }
    /// Returns the number of bisection depths with Fourier moments in each cycle.
    pub const fn moment_levels(&self) -> Budget {
        self.weighted.moment_levels()
    }
}
