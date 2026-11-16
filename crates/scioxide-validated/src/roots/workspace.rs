//! Workspace shared by every root proof.
use super::*;
use crate::shell::{Requirements, Storage, Workspace, add, grow, grown_bytes, mul};
use scioxide_core::{AlgebraicScalar, AttemptRecord, Capacity};

/// Buffer sizes one plan needs for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Shape {
    /// Unknowns of a system proof lane.
    pub(super) order: usize,
    /// Provider scratch of a system proof lane.
    pub(super) scratch: usize,
    /// Polynomial coefficients of a count lane.
    pub(super) terms: usize,
    /// Polynomial disk projections in each proof lane.
    pub(super) projections: usize,
    /// Region order of isolation trees.
    pub(super) tree_order: usize,
    /// Tree node capacity.
    pub(super) nodes: usize,
    /// Polynomial isolation disk capacity.
    pub(super) disks: usize,
}

/// One proof's system and polynomial scratch.
#[derive(Debug)]
pub(super) struct Lane<S: NumericScalar> {
    boxes: [Vec<IntervalBox<S::Real>>; 5],
    rows: Vec<S::Real>,
    provider: Vec<IntervalBox<S::Real>>,
    coefficients: Vec<Box2<S::Real>>,
    magnitudes: Vec<D<S::Real>>,
    projections: Vec<polynomial::Projection<S::Real>>,
}

/// Borrowed Krawczyk scratch cut to a plan's order and provider requirement.
pub(super) struct SystemScratch<'a, R: RealScalar> {
    pub(super) point: &'a mut [IntervalBox<R>],
    pub(super) direction: &'a mut [IntervalBox<R>],
    pub(super) a: &'a mut [IntervalBox<R>],
    pub(super) r: &'a mut [IntervalBox<R>],
    pub(super) image: &'a mut [IntervalBox<R>],
    pub(super) rows: &'a mut [R],
    pub(super) provider: &'a mut [IntervalBox<R>],
}

/// Borrowed outward Taylor coefficients and magnitudes of a polynomial.
pub(super) struct PolynomialScratch<'a, R: RealScalar> {
    pub(super) coefficients: &'a mut [Box2<R>],
    pub(super) magnitudes: &'a mut [D<R>],
}

impl<S: NumericScalar> Lane<S> {
    const fn new() -> Self {
        Self {
            boxes: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            rows: Vec::new(),
            provider: Vec::new(),
            coefficients: Vec::new(),
            magnitudes: Vec::new(),
            projections: Vec::new(),
        }
    }
    fn projected_bytes(&self, shape: Shape) -> Result<usize> {
        let mut bytes = add(
            grown_bytes(&self.rows, shape.order)?,
            grown_bytes(&self.provider, shape.scratch)?,
        )?;
        for values in &self.boxes {
            bytes = add(bytes, grown_bytes(values, shape.order)?)?;
        }
        bytes = add(bytes, grown_bytes(&self.projections, shape.projections)?)?;
        add(
            bytes,
            add(
                grown_bytes(&self.coefficients, shape.terms)?,
                grown_bytes(&self.magnitudes, shape.terms)?,
            )?,
        )
    }
    fn fits(&self, shape: Shape) -> bool {
        self.boxes.iter().all(|values| values.len() >= shape.order)
            && self.rows.len() >= shape.order
            && self.provider.len() >= shape.scratch
            && self.coefficients.len() >= shape.terms
            && self.magnitudes.len() >= shape.terms
            && self.projections.len() >= shape.projections
    }
    fn try_reserve(&mut self, shape: Shape) -> Result<()> {
        let z = zero()?;
        for values in &mut self.boxes {
            grow(
                values,
                shape.order,
                z,
                scioxide_core::quantity!("workspace.boxes"),
            )?;
        }
        grow(
            &mut self.rows,
            shape.order,
            S::Real::ZERO,
            scioxide_core::quantity!("workspace.rows"),
        )?;
        grow(
            &mut self.provider,
            shape.scratch,
            z,
            scioxide_core::quantity!("workspace.provider"),
        )?;
        grow(
            &mut self.coefficients,
            shape.terms,
            Box2::zero()?,
            scioxide_core::quantity!("workspace.coefficients"),
        )?;
        grow(
            &mut self.projections,
            shape.projections,
            polynomial::Projection::zero(),
            scioxide_core::quantity!("workspace.projections"),
        )?;
        grow(
            &mut self.magnitudes,
            shape.terms,
            D::empty(),
            scioxide_core::quantity!("workspace.magnitudes"),
        )
    }
    /// Krawczyk scratch for `order` unknowns and `scratch` provider slots.
    pub(super) fn system(&mut self, order: usize, scratch: usize) -> SystemScratch<'_, S::Real> {
        let [point, direction, a, r, image] = &mut self.boxes;
        SystemScratch {
            point: &mut point[..order],
            direction: &mut direction[..order],
            a: &mut a[..order],
            r: &mut r[..order],
            image: &mut image[..order],
            rows: &mut self.rows[..order],
            provider: &mut self.provider[..scratch],
        }
    }
    /// Taylor scratch for `terms` coefficients.
    pub(super) fn polynomial(&mut self, terms: usize) -> PolynomialScratch<'_, S::Real> {
        PolynomialScratch {
            coefficients: &mut self.coefficients[..terms],
            magnitudes: &mut self.magnitudes[..terms],
        }
    }
    /// Sweep storage for `count` disks.
    pub(super) fn projections(&mut self, count: usize) -> &mut [polynomial::Projection<S::Real>] {
        &mut self.projections[..count]
    }
}

/// Staging and replay storage of subdivision trees and counted disks.
#[derive(Debug)]
pub(super) struct Refinement<S: NumericScalar> {
    nodes: Vec<RootIsolationNode<S::Real>>,
    regions: Vec<IntervalBox<S::Real>>,
    parents: Vec<usize>,
    region: Vec<IntervalBox<S::Real>>,
    center: Vec<S>,
    disks: Vec<RootDisk<S::Real>>,
    counts: Vec<Option<usize>>,
    staged: Vec<PolynomialRootCertificate<S::Real>>,
}

/// Borrowed tree staging cut to an order and a node capacity.
pub(super) struct TreeScratch<'a, S: NumericScalar> {
    pub(super) nodes: &'a mut [RootIsolationNode<S::Real>],
    pub(super) regions: &'a mut [IntervalBox<S::Real>],
    pub(super) parents: &'a mut [usize],
    pub(super) region: &'a mut [IntervalBox<S::Real>],
    pub(super) center: &'a mut [S],
}

/// Borrowed disk staging cut to a capacity.
pub(super) struct DiskScratch<'a, R: RealScalar> {
    pub(super) disks: &'a mut [RootDisk<R>],
    pub(super) counts: &'a mut [Option<usize>],
    pub(super) staged: &'a mut [PolynomialRootCertificate<R>],
}

fn unit_disk<R: RealScalar>() -> RootDisk<R> {
    RootDisk {
        center: Complex::new(R::ZERO, R::ZERO),
        radius: R::ONE,
    }
}

/// Unwritten certificate storage: the unit disk with no roots.
pub(super) fn blank_certificate<R: RealScalar>() -> PolynomialRootCertificate<R> {
    PolynomialRootCertificate {
        disk: unit_disk(),
        count: 0,
    }
}

impl<S: NumericScalar> Refinement<S> {
    const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            regions: Vec::new(),
            parents: Vec::new(),
            region: Vec::new(),
            center: Vec::new(),
            disks: Vec::new(),
            counts: Vec::new(),
            staged: Vec::new(),
        }
    }
    fn region_len(shape: Shape) -> Result<usize> {
        mul(shape.tree_order, shape.nodes)
    }
    fn projected_bytes(&self, shape: Shape) -> Result<usize> {
        let parts = [
            grown_bytes(&self.nodes, shape.nodes)?,
            grown_bytes(&self.regions, Self::region_len(shape)?)?,
            grown_bytes(&self.parents, shape.nodes)?,
            grown_bytes(&self.region, shape.tree_order)?,
            grown_bytes(&self.center, shape.tree_order)?,
            grown_bytes(&self.disks, shape.disks)?,
            grown_bytes(&self.counts, shape.disks)?,
            grown_bytes(&self.staged, shape.disks)?,
        ];
        parts.into_iter().try_fold(0, add)
    }
    fn fits(&self, shape: Shape) -> Result<bool> {
        Ok(self.nodes.len() >= shape.nodes
            && self.regions.len() >= Self::region_len(shape)?
            && self.parents.len() >= shape.nodes
            && self.region.len() >= shape.tree_order
            && self.center.len() >= shape.tree_order
            && self.disks.len() >= shape.disks
            && self.counts.len() >= shape.disks
            && self.staged.len() >= shape.disks)
    }
    fn try_reserve(&mut self, shape: Shape) -> Result<()> {
        let z = zero()?;
        grow(
            &mut self.nodes,
            shape.nodes,
            RootIsolationNode::Leaf(None),
            scioxide_core::quantity!("workspace.nodes"),
        )?;
        grow(
            &mut self.regions,
            Self::region_len(shape)?,
            z,
            scioxide_core::quantity!("workspace.regions"),
        )?;
        grow(
            &mut self.parents,
            shape.nodes,
            0,
            scioxide_core::quantity!("workspace.parents"),
        )?;
        grow(
            &mut self.region,
            shape.tree_order,
            z,
            scioxide_core::quantity!("workspace.region"),
        )?;
        grow(
            &mut self.center,
            shape.tree_order,
            S::ZERO,
            scioxide_core::quantity!("workspace.center"),
        )?;
        grow(
            &mut self.disks,
            shape.disks,
            unit_disk(),
            scioxide_core::quantity!("workspace.disks"),
        )?;
        grow(
            &mut self.counts,
            shape.disks,
            None,
            scioxide_core::quantity!("workspace.counts"),
        )?;
        grow(
            &mut self.staged,
            shape.disks,
            blank_certificate(),
            scioxide_core::quantity!("workspace.staged"),
        )
    }
    /// Tree staging for regions of `order` components and `nodes` nodes.
    pub(super) fn tree(&mut self, order: usize, nodes: usize) -> TreeScratch<'_, S> {
        TreeScratch {
            nodes: &mut self.nodes[..nodes],
            regions: &mut self.regions[..order * nodes],
            parents: &mut self.parents[..nodes],
            region: &mut self.region[..order],
            center: &mut self.center[..order],
        }
    }
    /// The first `nodes` staged tree nodes and their `order`-component regions.
    pub(super) fn staged_tree(&self, order: usize, nodes: usize) -> RootIsolationView<'_, S::Real> {
        RootIsolationView {
            nodes: &self.nodes[..nodes],
            regions: &self.regions[..order * nodes],
        }
    }
    /// Disk staging for `capacity` disks.
    pub(super) fn disks(&mut self, capacity: usize) -> DiskScratch<'_, S::Real> {
        DiskScratch {
            disks: &mut self.disks[..capacity],
            counts: &mut self.counts[..capacity],
            staged: &mut self.staged[..capacity],
        }
    }
    /// The first `count` staged certificates.
    pub(super) fn staged_disks(&self, count: usize) -> &[PolynomialRootCertificate<S::Real>] {
        &self.staged[..count]
    }
}

/// Workspace size that the calls of a root plan need, from its
/// `workspace_requirements`.
///
/// [`Self::required_bytes`] is what a new [`RootWorkspace`] allocates. A reused
/// workspace is checked against the byte limit with all the storage it holds,
/// including storage reserved for other plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootRequirements {
    shape: Shape,
    lanes: usize,
    bytes: usize,
    limit: usize,
}

impl RootRequirements {
    /// Requirements of scalar `S`; the required bytes are those of a fresh workspace.
    pub(super) fn new<S: NumericScalar>(shape: Shape, lanes: usize, limit: usize) -> Result<Self> {
        let mut requirements = Self {
            shape,
            lanes,
            bytes: 0,
            limit,
        };
        requirements.bytes = RootWorkspace::<S>::new()
            .core
            .projected_bytes(&requirements)?;
        Ok(requirements)
    }
    /// Returns the number of independent proofs.
    pub const fn proofs(&self) -> usize {
        self.lanes
    }
    /// Returns the tree-node or disk capacity of an isolation.
    pub const fn capacity(&self) -> usize {
        if self.shape.nodes > self.shape.disks {
            self.shape.nodes
        } else {
            self.shape.disks
        }
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

/// Reusable storage for the calls of [`ScalarRootVerifier`], [`SystemRootVerifier`]
/// and [`PolynomialRootPlan`].
///
/// [`Self::new`] creates an empty workspace without allocating. A call grows a
/// workspace that is too small before it runs any callback, and does not
/// allocate when the workspace is large enough. One workspace can serve
/// several plans and keeps the largest size it has reserved. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_validated::{PolynomialRootPlan, ProofRequest, RootWorkspace, ValidatedOptions};
///
/// let coefficients = [-2.0_f64, 0.0, 1.0];
/// let plan = PolynomialRootPlan::new(&coefficients, ValidatedOptions::new())?;
/// let requirements = plan.workspace_requirements(ProofRequest::proofs(4))?;
/// let mut workspace = RootWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct RootWorkspace<S: NumericScalar> {
    pub(super) core: Workspace<Lane<S>, Refinement<S>, Staged<S::Real>>,
}

impl<S: NumericScalar> Default for RootWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: NumericScalar> core::fmt::Debug for RootWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RootWorkspace")
            .field("lanes", &self.core.lanes.len())
            .field("last_attempt", &self.core.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<S: NumericScalar> RootWorkspace<S> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            core: Workspace::new(Refinement::new()),
        }
    }

    /// Returns whether the workspace meets `requirements` without growing.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace would exceed the byte limit after
    ///   growing.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(&self, requirements: &RootRequirements) -> Result<Capacity> {
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
    pub fn try_reserve(&mut self, requirements: &RootRequirements) -> Result<()> {
        self.core.reserve(requirements)
    }

    /// Returns the record of the last call: its outcome, solve epoch and work
    /// counts. A call rejected before it started has no epoch and zero counts.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.core.shell.last_attempt()
    }
}

impl<S: NumericScalar> Storage for Lane<S> {
    type Shape = Shape;
    fn empty() -> Self {
        Self::new()
    }
    fn projected_bytes(&self, shape: Shape) -> Result<usize> {
        self.projected_bytes(shape)
    }
    fn fits(&self, shape: Shape) -> Result<bool> {
        Ok(self.fits(shape))
    }
    fn reserve(&mut self, shape: Shape) -> Result<()> {
        self.try_reserve(shape)
    }
}

impl<S: NumericScalar> Storage for Refinement<S> {
    type Shape = Shape;
    fn empty() -> Self {
        Self::new()
    }
    fn projected_bytes(&self, shape: Shape) -> Result<usize> {
        self.projected_bytes(shape)
    }
    fn fits(&self, shape: Shape) -> Result<bool> {
        self.fits(shape)
    }
    fn reserve(&mut self, shape: Shape) -> Result<()> {
        self.try_reserve(shape)
    }
}

impl<S: NumericScalar> Requirements<Lane<S>, Refinement<S>> for RootRequirements {
    fn lanes(&self) -> usize {
        self.lanes
    }
    fn lane_shape(&self) -> Shape {
        self.shape
    }
    fn shared_shape(&self) -> Shape {
        self.shape
    }
    fn limit(&self) -> usize {
        self.limit
    }
}
