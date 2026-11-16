//! Globally adaptive one-dimensional quadrature with Gauss–Kronrod or Simpson
//! regions, and the region scheduler that the weighted, oscillatory,
//! sensitivity and iterated rules reuse.

pub(crate) mod extrapolation;
pub(crate) mod kernel;
pub(crate) mod kronrod;
mod options;
mod plan;
pub(crate) mod regional;
pub(crate) mod simpson;
mod simpson_regional;

use alloc::vec::Vec;

use scioxide_core::RealScalar;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};

use crate::report::{IntegrationErrorComponents, IntegrationReport};
use crate::workspace::{Attempts, IntegrationRequirements, workspace_protocol};

pub use options::SubdivisionOptions;
pub use plan::{AdaptiveIntegrator, integrate};
/// Selects the rule that [`AdaptiveIntegrator`] applies on each region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveRule {
    /// 15-point Gauss–Kronrod rule with its embedded 7-point Gauss rule (G7K15).
    ///
    /// The nodes lie strictly inside each region, so the integrand is never
    /// evaluated at a region endpoint. The first region costs 15 callbacks and
    /// each bisection 30.
    GaussKronrod,
    /// Composite Boole rule on nine equally spaced points, with an error
    /// estimate from the Boole rule on every second point.
    ///
    /// Boole's rule is Simpson's rule with one Richardson extrapolation step.
    /// The endpoints of each region are sampled. The first region costs 10
    /// callbacks: nine grid points and one off-grid check point. Each bisection costs 8, because the children reuse the five
    /// samples they share with the parent. Every region is bisected to depth 4
    /// before convergence is accepted, so a successful call takes at least 15
    /// subdivisions and 130 callbacks.
    Simpson,
}

/// Reusable storage for [`AdaptiveIntegrator`].
///
/// Holds up to `max_subdivisions + 1` regions with the estimates of every value
/// component, and for [`AdaptiveRule::Simpson`] the nine samples of each region.
/// Memory is `O(c · m)` for `c` value components and `m` subdivisions. Storage for
/// [`AdaptiveIntegrator::with_extrapolation`] is always included.
///
/// Create it with [`AdaptiveIntegrator::workspace`]. A workspace created for one
/// rule serves the other after [`Self::try_reserve`] with that integrator's
/// requirements. See [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_integrate::{AdaptiveIntegrator, AdaptiveRule, SubdivisionOptions};
///
/// let options = SubdivisionOptions::<f64>::default();
/// let kronrod = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone())?;
/// let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options)?;
/// let mut workspace = kronrod.workspace(1)?;
/// workspace.try_reserve(&simpson.workspace_requirements(1)?)?;
/// let mut value = 0.0;
/// let unit = Interval::new(0.0, 1.0)?;
/// let cx = ExecutionContext::sequential_native();
/// let f = |x: f64| Ok(x.exp());
/// let report = simpson.integrate_into(&f, unit, &mut value, &mut workspace, cx)?;
/// // ∫₀¹ eˣ dx = e − 1
/// assert!(report.is_success() && (value - (1.0_f64.exp() - 1.0)).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct AdaptiveWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    regional: regional::RegionalWorkspace<S>,
    preparation: kernel::PreparedRules<S>,
    retained: Vec<S>,
    errors: Vec<IntegrationErrorComponents<S>>,
    published_errors: bool,
}

impl<S: RealScalar> AdaptiveWorkspace<S> {
    fn allocate(
        requirements: &IntegrationRequirements,
    ) -> Result<Self, scioxide_core::SciOxideError> {
        let components = requirements.components();
        let [regions, stencil] = requirements.units();
        let regional = regional::RegionalWorkspace::new(components, regions, 15)?;
        // One retained stencil per region slot, including both staging slots,
        // and one witness row for stencil rules.
        let retained = checked_mul(
            checked_add(
                checked_mul(
                    regional.regions.len(),
                    stencil,
                    scioxide_core::quantity!("retained_samples"),
                )?,
                stencil.min(1),
                scioxide_core::quantity!("retained_samples"),
            )?,
            components,
            scioxide_core::quantity!("retained_samples"),
        )?;
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            regional,
            preparation: kernel::PreparedRules::new()?,
            retained: try_filled(
                retained,
                S::ZERO,
                scioxide_core::quantity!("retained_samples"),
            )?,
            errors: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("adaptive_errors"),
            )?,
            published_errors: false,
        })
    }

    /// Returns the error estimate of each value component from the last call.
    ///
    /// Returns `None` before the first call, after a call that returned an
    /// error, and after a zero-width interval, which has no error estimate.
    #[must_use]
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        crate::execution::component_errors(self.published_errors, self.errors.as_slice())
    }
}

workspace_protocol!(AdaptiveWorkspace, IntegrationReport<S>);
