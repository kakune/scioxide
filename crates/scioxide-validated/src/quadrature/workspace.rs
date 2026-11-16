//! Workspace of certified quadrature.
use super::*;
use crate::shell::{Requirements, Storage, Workspace, add, grow, grown_bytes, mul};
use scioxide_core::{AttemptRecord, Capacity};

/// One proof's fresh bounds and value/derivative scratch.
#[derive(Debug)]
pub(super) struct Lane<R: RealScalar> {
    total: Vec<IntervalBox<R>>,
    scratch: Vec<IntervalBox<R>>,
}

/// Borrowed lane cut to a component count.
pub(super) struct LaneScratch<'a, R: RealScalar> {
    pub(super) total: &'a mut [IntervalBox<R>],
    pub(super) scratch: &'a mut [IntervalBox<R>],
}

impl<R: RealScalar> Lane<R> {
    const fn new() -> Self {
        Self {
            total: Vec::new(),
            scratch: Vec::new(),
        }
    }
    fn projected_bytes(&self, components: usize) -> Result<usize> {
        add(
            grown_bytes(&self.total, components)?,
            grown_bytes(&self.scratch, mul(3, components)?)?,
        )
    }
    fn fits(&self, components: usize) -> Result<bool> {
        Ok(self.total.len() >= components && self.scratch.len() >= mul(3, components)?)
    }
    fn try_reserve(&mut self, components: usize) -> Result<()> {
        grow(
            &mut self.total,
            components,
            sentinel(),
            scioxide_core::quantity!("workspace.total"),
        )?;
        grow(
            &mut self.scratch,
            mul(3, components)?,
            sentinel(),
            scioxide_core::quantity!("workspace.scratch"),
        )
    }
    /// This lane for `components` components; the lane must fit them.
    pub(super) fn view(&mut self, components: usize) -> LaneScratch<'_, R> {
        LaneScratch {
            total: &mut self.total[..components],
            scratch: &mut self.scratch[..3 * components],
        }
    }
}

/// Borrowed generation staging: the live cover plus two child slots.
pub(super) struct CoverScratch<'a, R: RealScalar> {
    pub(super) panels: &'a mut [QuadraturePanel<R>],
    pub(super) refinement: scheduler::Scratch<'a, R>,
    pub(super) bounds: &'a mut [IntervalBox<R>],
}

/// Workspace size that the calls of a [`QuadratureVerifier`] need, from
/// [`QuadratureVerifier::workspace_requirements`].
///
/// [`Self::required_bytes`] is what a new [`QuadratureWorkspace`] allocates. A
/// reused workspace is checked against the byte limit with all the storage it
/// holds, including storage reserved for other plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadratureRequirements {
    components: usize,
    panels: usize,
    lanes: usize,
    bytes: usize,
    limit: usize,
    staged: usize,
    bounds: usize,
}

impl QuadratureRequirements {
    /// Requirements of real scalar `R`; the required bytes are those of a new
    /// workspace.
    pub(super) fn new<R: RealScalar>(
        components: usize,
        panels: usize,
        lanes: usize,
        limit: usize,
    ) -> Result<Self> {
        let staged = if panels == 0 { 0 } else { add(panels, 2)? };
        let mut requirements = Self {
            components,
            panels,
            lanes,
            bytes: 0,
            limit,
            staged,
            bounds: mul(staged, components)?,
        };
        requirements.bytes = QuadratureWorkspace::<R>::new()
            .core
            .projected_bytes(&requirements)?;
        Ok(requirements)
    }
    /// Returns the number of integrand components.
    pub const fn components(&self) -> usize {
        self.components
    }
    /// Returns the panel capacity of a generation.
    pub const fn panels(&self) -> usize {
        self.panels
    }
    /// Returns the number of independent verifications.
    pub const fn proofs(&self) -> usize {
        self.lanes
    }
    /// Returns the bytes that a new workspace allocates for these requirements.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the byte limit of the plan.
    pub const fn byte_limit(&self) -> usize {
        self.limit
    }
}

/// Reusable storage for the calls of [`QuadratureVerifier`].
///
/// [`Self::new`] creates an empty workspace without allocating. A call grows a
/// workspace that is too small before it queries the oracle, and does not
/// allocate when the workspace is large enough. One workspace can serve
/// several plans and keeps the largest size it has reserved. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::{Capacity};
/// use scioxide_validated::{
///     QuadratureVerifier, DecoratedInterval as D, ProofRequest, QuadratureAssumption,
///     QuadratureObligation, QuadraturePiece, QuadratureWorkspace, ValidatedOptions,
/// };
///
/// let domain = D::new(0_f64, 1.)?.set();
/// let pieces = [QuadraturePiece {
///     domain,
///     assumption: QuadratureAssumption::IntegrableRange,
/// }];
/// let oracle = |_, _: &mut [_]| Ok(QuadratureObligation::Unavailable);
/// let plan =
///     QuadratureVerifier::<f64>::new(&pieces, 1, &oracle, ValidatedOptions::new())?;
/// let requirements = plan.workspace_requirements(ProofRequest::proofs(8))?;
/// let mut workspace = QuadratureWorkspace::<f64>::new();
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct QuadratureWorkspace<R: RealScalar> {
    pub(super) core: Workspace<Lane<R>, Cover<R>, QuadratureVerificationStatus>,
}

impl<R: RealScalar> Default for QuadratureWorkspace<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RealScalar> core::fmt::Debug for QuadratureWorkspace<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuadratureWorkspace")
            .field("panels", &self.core.shared.panels.len())
            .field("lanes", &self.core.lanes.len())
            .field("last_attempt", &self.core.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<R: RealScalar> QuadratureWorkspace<R> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            core: Workspace::new(Cover::new()),
        }
    }

    /// Returns whether the workspace meets `requirements` without growing.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace would exceed the byte limit after
    ///   growing.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(&self, requirements: &QuadratureRequirements) -> Result<Capacity> {
        self.core.check(requirements)
    }

    /// Grows the workspace to meet `requirements`. If a reservation fails, the
    /// workspace may keep the part it has already grown.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace would exceed the byte limit; it is
    ///   checked before anything grows.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn try_reserve(&mut self, requirements: &QuadratureRequirements) -> Result<()> {
        self.core.reserve(requirements)
    }

    /// Returns the record of the last call: its outcome, solve epoch and work
    /// counts. A call rejected before it started has no epoch and zero counts.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.core.shell.last_attempt()
    }

    pub(super) fn staged(
        &self,
        count: usize,
        components: usize,
    ) -> QuadratureCertificateView<'_, R> {
        QuadratureCertificateView {
            panels: &self.core.shared.panels[..count],
            local_bounds: &self.core.shared.bounds[..count * components],
            total: &self.core.lanes[0].total[..components],
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CoverShape {
    panels: usize,
    components: usize,
    staged: usize,
    bounds: usize,
}

pub(super) struct Cover<R: RealScalar> {
    panels: Vec<QuadraturePanel<R>>,
    bounds: Vec<IntervalBox<R>>,
    refinement: scheduler::Refinement<R>,
}
impl<R: RealScalar> Cover<R> {
    const fn new() -> Self {
        Self {
            panels: Vec::new(),
            bounds: Vec::new(),
            refinement: scheduler::Refinement::new(),
        }
    }
    pub(super) fn view(&mut self, requirements: &QuadratureRequirements) -> CoverScratch<'_, R> {
        CoverScratch {
            refinement: self
                .refinement
                .view(requirements.panels, requirements.components),
            panels: &mut self.panels[..requirements.staged],
            bounds: &mut self.bounds[..requirements.bounds],
        }
    }
}
impl<R: RealScalar> Storage for Cover<R> {
    type Shape = CoverShape;
    fn empty() -> Self {
        Self::new()
    }
    fn projected_bytes(&self, shape: CoverShape) -> Result<usize> {
        [
            grown_bytes(&self.panels, shape.staged)?,
            grown_bytes(&self.bounds, shape.bounds)?,
            self.refinement.bytes(shape.panels, shape.components)?,
        ]
        .into_iter()
        .try_fold(0, add)
    }
    fn fits(&self, shape: CoverShape) -> Result<bool> {
        Ok(self.panels.len() >= shape.staged
            && self.bounds.len() >= shape.bounds
            && self.refinement.fits(shape.panels, shape.components)?)
    }
    fn reserve(&mut self, shape: CoverShape) -> Result<()> {
        grow(
            &mut self.panels,
            shape.staged,
            placeholder(),
            scioxide_core::quantity!("workspace.panels"),
        )?;
        grow(
            &mut self.bounds,
            shape.bounds,
            sentinel(),
            scioxide_core::quantity!("workspace.bounds"),
        )?;
        self.refinement.reserve(shape.panels, shape.components)
    }
}
impl<R: RealScalar> Storage for Lane<R> {
    type Shape = usize;
    fn empty() -> Self {
        Self::new()
    }
    fn projected_bytes(&self, components: usize) -> Result<usize> {
        self.projected_bytes(components)
    }
    fn fits(&self, components: usize) -> Result<bool> {
        self.fits(components)
    }
    fn reserve(&mut self, components: usize) -> Result<()> {
        self.try_reserve(components)
    }
}
impl<R: RealScalar> Requirements<Lane<R>, Cover<R>> for QuadratureRequirements {
    fn lanes(&self) -> usize {
        self.lanes
    }
    fn lane_shape(&self) -> usize {
        self.components
    }
    fn shared_shape(&self) -> CoverShape {
        CoverShape {
            panels: self.panels,
            components: self.components,
            staged: self.staged,
            bounds: self.bounds,
        }
    }
    fn limit(&self) -> usize {
        self.limit
    }
}
