//! Options of oscillatory quadrature plans.
use crate::SubdivisionOptions;
use scioxide_core::{Budget, RealScalar};

/// Options for [`OscillatoryQuadraturePlan`](crate::OscillatoryQuadraturePlan).
///
/// The [`SubdivisionOptions`] set the tolerance, the subdivision and evaluation
/// limits and the execution policy of amplitude samples. The ordinary-panel
/// limit, 4096 by default, bounds the number of Gauss–Kronrod panels in one
/// stationary or weakly oscillating region. A region that needs more panels is
/// not evaluated.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::{OscillatoryOptions, SubdivisionOptions};
///
/// let options = OscillatoryOptions::<f64>::new(SubdivisionOptions::default())
///     .with_ordinary_panels(Budget::try_from(256)?);
/// assert_eq!(options.ordinary_panels().get(), 256);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct OscillatoryOptions<S: RealScalar> {
    subdivision: SubdivisionOptions<S>,
    ordinary: Budget,
}
impl<S: RealScalar> Default for OscillatoryOptions<S> {
    fn default() -> Self {
        Self::new(SubdivisionOptions::default())
    }
}
impl<S: RealScalar> OscillatoryOptions<S> {
    /// Creates options with the given subdivision options and a limit of 4096
    /// ordinary panels.
    pub fn new(subdivision: SubdivisionOptions<S>) -> Self {
        Self {
            subdivision,
            ordinary: Budget::ONE.saturating_add(4095),
        }
    }
    /// Replaces the subdivision options.
    pub fn with_subdivision_options(mut self, value: SubdivisionOptions<S>) -> Self {
        self.subdivision = value;
        self
    }
    /// Sets the maximum number of Gauss–Kronrod panels in one region.
    pub const fn with_ordinary_panels(mut self, value: Budget) -> Self {
        self.ordinary = value;
        self
    }
    /// Returns the subdivision options.
    pub const fn subdivision_options(&self) -> &SubdivisionOptions<S> {
        &self.subdivision
    }
    /// Returns the maximum number of Gauss–Kronrod panels in one region.
    pub const fn ordinary_panels(&self) -> Budget {
        self.ordinary
    }
}
