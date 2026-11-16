//! Workspace of verified linear-system proofs.
use super::*;
use crate::shell::{Requirements, Storage, Workspace, add, grow, grown_bytes};
use scioxide_core::{AttemptRecord, Capacity};

/// Sizes of one proof lane, fixed by a plan's order and provider requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Shape {
    pub(super) order: usize,
    pub(super) provider: usize,
    pub(super) scalar: usize,
}

/// One proof's interval scratch and its staged, unverified certificate.
#[derive(Debug)]
pub(super) struct Lane<S: NumericScalar> {
    boxes: [Vec<IntervalBox<S::Real>>; 3],
    provider: Vec<IntervalBox<S::Real>>,
    radii: Vec<Complex<S::Real>>,
    center: Vec<S>,
    midpoint: Vec<S>,
    scalar: Vec<S>,
}

/// Borrowed proof scratch of one lane, cut to a plan's shape.
pub(super) struct ProofScratch<'a, R: RealScalar> {
    pub(super) input: &'a mut [IntervalBox<R>],
    pub(super) residual: &'a mut [IntervalBox<R>],
    pub(super) z: &'a mut [IntervalBox<R>],
    pub(super) provider: &'a mut [IntervalBox<R>],
}

/// Borrowed lane: proof scratch plus the staged center, radii and solve buffers.
pub(super) struct LaneView<'a, S: NumericScalar> {
    pub(super) proof: ProofScratch<'a, S::Real>,
    pub(super) center: &'a mut [S],
    pub(super) radii: &'a mut [Complex<S::Real>],
    pub(super) midpoint: &'a mut [S],
    pub(super) scalar: &'a mut [S],
}

impl<S: NumericScalar> Lane<S> {
    const fn new() -> Self {
        Self {
            boxes: [Vec::new(), Vec::new(), Vec::new()],
            provider: Vec::new(),
            radii: Vec::new(),
            center: Vec::new(),
            midpoint: Vec::new(),
            scalar: Vec::new(),
        }
    }
    fn projected_bytes(&self, shape: Shape) -> Result<usize> {
        let n = shape.order;
        let mut bytes = add(
            grown_bytes(&self.provider, shape.provider)?,
            grown_bytes(&self.scalar, shape.scalar)?,
        )?;
        for values in &self.boxes {
            bytes = add(bytes, grown_bytes(values, n)?)?;
        }
        for part in [
            grown_bytes(&self.radii, n)?,
            grown_bytes(&self.center, n)?,
            grown_bytes(&self.midpoint, n)?,
        ] {
            bytes = add(bytes, part)?;
        }
        Ok(bytes)
    }
    fn fits(&self, shape: Shape) -> bool {
        let n = shape.order;
        self.boxes.iter().all(|values| values.len() >= n)
            && self.provider.len() >= shape.provider
            && self.scalar.len() >= shape.scalar
            && [self.radii.len(), self.center.len(), self.midpoint.len()]
                .iter()
                .all(|&len| len >= n)
    }
    fn try_reserve(&mut self, shape: Shape) -> Result<()> {
        let n = shape.order;
        let z = zero()?;
        let c = Complex::new(S::Real::ZERO, S::Real::ZERO);
        for values in &mut self.boxes {
            grow(values, n, z, scioxide_core::quantity!("workspace.boxes"))?;
        }
        grow(
            &mut self.provider,
            shape.provider,
            z,
            scioxide_core::quantity!("workspace.provider"),
        )?;
        grow(
            &mut self.radii,
            n,
            c,
            scioxide_core::quantity!("workspace.radii"),
        )?;
        grow(
            &mut self.center,
            n,
            S::ZERO,
            scioxide_core::quantity!("workspace.center"),
        )?;
        grow(
            &mut self.midpoint,
            n,
            S::ZERO,
            scioxide_core::quantity!("workspace.midpoint"),
        )?;
        grow(
            &mut self.scalar,
            shape.scalar,
            S::ZERO,
            scioxide_core::quantity!("workspace.scalar"),
        )
    }
    /// Borrow this lane cut to `shape`; the lane must fit it.
    pub(super) fn view(&mut self, shape: Shape) -> LaneView<'_, S> {
        let n = shape.order;
        let [input, residual, z] = &mut self.boxes;
        LaneView {
            proof: ProofScratch {
                input: &mut input[..n],
                residual: &mut residual[..n],
                z: &mut z[..n],
                provider: &mut self.provider[..shape.provider],
            },
            center: &mut self.center[..n],
            radii: &mut self.radii[..n],
            midpoint: &mut self.midpoint[..n],
            scalar: &mut self.scalar[..shape.scalar],
        }
    }
    /// The staged center and radii of the last proof of `order` components.
    pub(super) fn staged(&self, order: usize) -> (&[S], &[Complex<S::Real>]) {
        (&self.center[..order], &self.radii[..order])
    }
}

/// Call-local matrix sweep storage, shared by every right-hand side. Its contents
/// are scratch, never a cached proof for a later call.
#[derive(Debug)]
pub(super) struct Sweep<R: RealScalar> {
    boxes: [Vec<IntervalBox<R>>; 3],
    provider: Vec<IntervalBox<R>>,
    rows: [Vec<R>; 3],
    image: Vec<Complex<R>>,
}

/// Borrowed streamed matrix proof, including an external certificate's weighted
/// defect image when requested. No full defect matrix is stored.
pub(super) struct SweepScratch<'a, R: RealScalar> {
    pub(super) input: &'a mut [IntervalBox<R>],
    pub(super) a: &'a mut [IntervalBox<R>],
    pub(super) r: &'a mut [IntervalBox<R>],
    pub(super) provider: &'a mut [IntervalBox<R>],
    pub(super) a_rows: &'a mut [R],
    pub(super) r_rows: &'a mut [R],
    pub(super) e_rows: &'a mut [R],
    pub(super) image: &'a mut [Complex<R>],
}

impl<R: RealScalar> Sweep<R> {
    const fn new() -> Self {
        Self {
            boxes: [Vec::new(), Vec::new(), Vec::new()],
            provider: Vec::new(),
            rows: [Vec::new(), Vec::new(), Vec::new()],
            image: Vec::new(),
        }
    }
    fn projected_bytes(&self, shape: Shape) -> Result<usize> {
        let mut bytes = add(
            grown_bytes(&self.provider, shape.provider)?,
            grown_bytes(&self.image, shape.order)?,
        )?;
        for values in &self.boxes {
            bytes = add(bytes, grown_bytes(values, shape.order)?)?;
        }
        for values in &self.rows {
            bytes = add(bytes, grown_bytes(values, shape.order)?)?;
        }
        Ok(bytes)
    }
    fn fits(&self, shape: Shape) -> bool {
        self.boxes.iter().all(|v| v.len() >= shape.order)
            && self.rows.iter().all(|v| v.len() >= shape.order)
            && self.provider.len() >= shape.provider
            && self.image.len() >= shape.order
    }
    fn try_reserve(&mut self, shape: Shape) -> Result<()> {
        let z = zero()?;
        for values in &mut self.boxes {
            grow(
                values,
                shape.order,
                z,
                scioxide_core::quantity!("workspace.sweep"),
            )?;
        }
        grow(
            &mut self.provider,
            shape.provider,
            z,
            scioxide_core::quantity!("workspace.provider"),
        )?;
        for values in &mut self.rows {
            grow(
                values,
                shape.order,
                R::ZERO,
                scioxide_core::quantity!("workspace.rows"),
            )?;
        }
        grow(
            &mut self.image,
            shape.order,
            Complex::new(R::ZERO, R::ZERO),
            scioxide_core::quantity!("workspace.image"),
        )
    }
    pub(super) fn defect_rows(&self, order: usize) -> &[R] {
        &self.rows[2][..order]
    }
    pub(super) fn view(&mut self, shape: Shape) -> SweepScratch<'_, R> {
        let n = shape.order;
        let [input, a, r] = &mut self.boxes;
        let [a_rows, r_rows, e_rows] = &mut self.rows;
        SweepScratch {
            input: &mut input[..n],
            a: &mut a[..n],
            r: &mut r[..n],
            provider: &mut self.provider[..shape.provider],
            a_rows: &mut a_rows[..n],
            r_rows: &mut r_rows[..n],
            e_rows: &mut e_rows[..n],
            image: &mut self.image[..n],
        }
    }
}

/// Workspace size that the calls of a [`LinearVerifier`] need, from
/// [`LinearVerifier::workspace_requirements`].
///
/// Requirements for `k > 0` right-hand sides cover every single-proof method
/// and [`LinearVerifier::solve_block_into`] with up to `k` columns; `k = 0`
/// covers only an empty block. The storage is O(n + kn) for order `n`, plus
/// the operators' scratch. [`Self::required_bytes`] is what a new workspace
/// allocates. A reused workspace is checked against the byte limit with all
/// the storage it holds, including storage reserved for other plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearRequirements {
    shape: Shape,
    right_hand_sides: usize,
    bytes: usize,
    limit: usize,
}

impl LinearRequirements {
    /// Requirements of scalar `S`; the required bytes are those of a new workspace.
    pub(super) fn new<S: NumericScalar>(
        shape: Shape,
        right_hand_sides: usize,
        limit: usize,
    ) -> Result<Self> {
        let mut requirements = Self {
            shape,
            right_hand_sides,
            bytes: 0,
            limit,
        };
        requirements.bytes = LinearWorkspace::<S>::new()
            .core
            .projected_bytes(&requirements)?;
        Ok(requirements)
    }
    /// Returns the order of the system.
    pub const fn order(&self) -> usize {
        self.shape.order
    }
    /// Returns the number of right-hand sides.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
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

/// Reusable storage for the calls of [`LinearVerifier`].
///
/// [`Self::new`] creates an empty workspace without allocating;
/// [`Self::try_reserve`] or [`LinearVerifier::workspace`] sizes it. A call
/// grows a workspace that is too small before it evaluates any operator, and
/// does not allocate when the workspace is large enough. One workspace can
/// serve several plans and keeps the largest size it has reserved. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_validated::{
///     DenseIntervalOperator, ValidatedOptions, LinearVerifier, LinearWorkspace,
/// };
///
/// let a = [2.0_f64];
/// let r = [0.5_f64];
/// let system = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a)?)?;
/// let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r)?)?;
/// let plan = LinearVerifier::new(&system, &inverse, ValidatedOptions::new())?;
/// let requirements = plan.workspace_requirements(1)?;
/// let mut workspace = LinearWorkspace::<f64>::new();
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(
///     workspace.check(&requirements)?,
///     scioxide_core::Capacity::Sufficient
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct LinearWorkspace<S: NumericScalar> {
    pub(super) core: Workspace<Lane<S>, Sweep<S::Real>, LinearVerificationReport<S::Real>>,
}

impl<S: NumericScalar> Default for LinearWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: NumericScalar> core::fmt::Debug for LinearWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LinearWorkspace")
            .field("lanes", &self.core.lanes.len())
            .field("last_attempt", &self.core.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<S: NumericScalar> LinearWorkspace<S> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            core: Workspace::new(Sweep::new()),
        }
    }

    /// Returns whether the workspace meets `requirements` without growing.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace would exceed the byte limit after
    ///   growing.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(&self, requirements: &LinearRequirements) -> Result<Capacity> {
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
    pub fn try_reserve(&mut self, requirements: &LinearRequirements) -> Result<()> {
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

impl<S: RealScalar> Storage for Sweep<S> {
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

impl<S: NumericScalar> Requirements<Lane<S>, Sweep<S::Real>> for LinearRequirements {
    fn lanes(&self) -> usize {
        self.right_hand_sides
    }
    fn lane_shape(&self) -> Shape {
        self.shape
    }
    fn shared_shape(&self) -> Shape {
        if self.right_hand_sides == 0 {
            Shape::default()
        } else {
            self.shape
        }
    }
    fn limit(&self) -> usize {
        self.limit
    }
}
