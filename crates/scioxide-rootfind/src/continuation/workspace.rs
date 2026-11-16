//! Scratch storage for continuation.
use super::*;
use crate::NewtonKrylovWorkspaceRequirements;
use scioxide_core::{AttemptRecord, Capacity};

/// Holds the scratch storage of a [`Continuation`]: the corrector, the
/// tangent solve and the optional dense diagnostic.
///
/// Create it with [`Continuation::workspace`], or create an empty one with
/// [`Self::new`] and size it with [`Self::try_reserve`]. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct ContinuationWorkspace<R: RealScalar> {
    pub(crate) shape: ProviderShape,
    pub(crate) shell: crate::attempt::Shell<ContinuationObservation<R>>,
    pub(crate) accepted: Vec<R>,
    pub(crate) residual_reference: R,
    pub(crate) guess: Vec<R>,
    pub(crate) row: Vec<R>,
    pub(crate) origin: Vec<R>,
    pub(crate) candidate: Vec<R>,
    pub(crate) tangent: Vector<R>,
    pub(crate) rhs: Vector<R>,
    pub(crate) units: Vec<R>,
    pub(crate) linear_residual: Vec<R>,
    pub(crate) product: Vec<R>,
    pub(crate) residual: Vec<R>,
    pub(crate) scratch: Vec<R>,
    pub(crate) newton: NewtonKrylovWorkspace<R>,
    pub(crate) result: NewtonKrylovResult<R>,
    pub(crate) linear: Option<IterativeWorkspace<R>>,
    pub(crate) child: SolveState,
    pub(super) dense: Option<diagnostics::DenseDiagnostic<R>>,
}

impl<R: RealScalar> ContinuationWorkspaceRequirements<R> {
    /// Returns the required workspace size in bytes.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
}

/// Describes the workspace a [`Continuation`] needs for one problem shape.
#[derive(Clone, Debug)]
pub struct ContinuationWorkspaceRequirements<R: RealScalar> {
    pub(crate) shape: ProviderShape,
    pub(crate) corrector: NewtonKrylovWorkspaceRequirements<R>,
    pub(crate) corrector_options: NewtonKrylovOptions<R>,
    pub(crate) tangent: Option<scioxide_linalg::IterativeRequirements<R>>,
    pub(crate) dense: bool,
    pub(crate) bytes: usize,
    pub(crate) limit: usize,
}

impl<R: RealScalar> ContinuationWorkspace<R> {
    pub(crate) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        use crate::workspace::{buffer, sum};
        sum([
            buffer(&self.accepted),
            buffer(&self.guess),
            buffer(&self.row),
            buffer(&self.origin),
            buffer(&self.candidate),
            self.tangent.retained_bytes(),
            self.rhs.retained_bytes(),
            buffer(&self.product),
            buffer(&self.units),
            buffer(&self.linear_residual),
            buffer(&self.residual),
            buffer(&self.scratch),
            self.newton.retained_bytes(),
            self.result.retained_bytes(),
            self.linear.as_ref().map_or(Ok(0), |w| w.retained_bytes()),
            self.dense.as_ref().map_or(Ok(0), |w| w.retained_bytes()),
        ])
    }

    /// Creates an empty workspace.
    ///
    /// Reserve the requirements of [`Continuation::workspace_requirements`] with
    /// [`Self::try_reserve`] before use.
    ///
    /// ```
    /// use scioxide_rootfind::ContinuationWorkspace;
    /// let workspace = ContinuationWorkspace::<f64>::new();
    /// assert!(workspace.last_attempt().is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            shape: ProviderShape {
                dimension: 0,
                residual: 0,
                jvp: 0,
                linear: false,
                diagnostic: false,
            },
            shell: crate::attempt::Shell::new(),
            accepted: Vec::new(),
            residual_reference: R::ZERO,
            guess: Vec::new(),
            row: Vec::new(),
            origin: Vec::new(),
            candidate: Vec::new(),
            tangent: Vector::empty(),
            rhs: Vector::empty(),
            units: Vec::new(),
            linear_residual: Vec::new(),
            product: Vec::new(),
            residual: Vec::new(),
            scratch: Vec::new(),
            newton: NewtonKrylovWorkspace::new(),
            result: NewtonKrylovResult::empty(),
            linear: None,
            child: SolveState::new(),
            dense: None,
        }
    }
    /// Checks whether this workspace is large enough for `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage held or required exceeds the byte limit.
    /// - Errors from checking the nested corrector and tangent workspaces.
    pub fn check(
        &self,
        r: &ContinuationWorkspaceRequirements<R>,
    ) -> Result<Capacity, SciOxideError> {
        if self.retained_bytes()?.max(r.bytes) > r.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                self.retained_bytes()?.max(r.bytes),
                r.limit,
            ));
        }
        let linear = match &r.tangent {
            Some(req) => self
                .linear
                .as_ref()
                .map(|w| w.check(req))
                .transpose()?
                .is_some_and(|c| c == Capacity::Sufficient),
            None => true,
        };
        Ok(
            if self.shape == r.shape
                && self.newton.check(&r.corrector)? == Capacity::Sufficient
                && linear
                && (!r.dense || self.dense.is_some())
                && self
                    .result
                    .has_capacity(r.shape.dimension + 1, &r.corrector_options)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows this workspace to satisfy `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the result would exceed the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn try_reserve(
        &mut self,
        r: &ContinuationWorkspaceRequirements<R>,
    ) -> Result<(), SciOxideError> {
        if self.check(r)? == Capacity::NeedsGrowth {
            let mut prepared = Self::allocate(r)?;
            prepared.shell = self.shell.retained();
            *self = prepared;
        }
        Ok(())
    }
    pub(crate) fn allocate(
        r: &ContinuationWorkspaceRequirements<R>,
    ) -> Result<Self, SciOxideError> {
        let m = r.shape.dimension + 1;
        let q = scioxide_core::quantity!("continuation_workspace");
        let mut newton = NewtonKrylovWorkspace::new();
        newton.try_reserve(&r.corrector)?;
        let mut linear = r.tangent.as_ref().map(|_| IterativeWorkspace::new());
        if let (Some(workspace), Some(requirements)) = (&mut linear, &r.tangent) {
            workspace.try_reserve(requirements)?;
        }
        Ok(Self {
            shape: r.shape,
            shell: crate::attempt::Shell::new(),
            accepted: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            residual_reference: R::ZERO,
            guess: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            row: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            origin: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            candidate: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            tangent: Vector::zeros(m)?,
            rhs: Vector::zeros(m)?,
            units: scioxide_core::backend::try_filled(m, R::ONE, q)?,
            linear_residual: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            product: scioxide_core::backend::try_filled(m, R::ZERO, q)?,
            residual: scioxide_core::backend::try_filled(m - 1, R::ZERO, q)?,
            scratch: scioxide_core::backend::try_filled(
                r.shape.residual.max(r.shape.jvp),
                R::ZERO,
                q,
            )?,
            newton,
            result: NewtonKrylovResult::new(m, &r.corrector_options)?,
            linear,
            child: SolveState::new(),
            dense: if r.dense {
                Some(diagnostics::DenseDiagnostic::new(m)?)
            } else {
                None
            },
        })
    }
    /// Returns the record of the most recent `advance` or `resume` call.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<ContinuationObservation<R>>> {
        self.shell.last_attempt()
    }
    /// Returns the point accepted by the most recent call, or `None` if that call
    /// accepted no point.
    pub fn last_accepted(&self) -> Option<&[R]> {
        self.shell.accepted(self.accepted.as_slice())
    }
}
crate::attempt::attempted!(
    ContinuationWorkspace<R>,
    [R: RealScalar],
    ContinuationObservation<R>
);

impl<R: RealScalar> Default for ContinuationWorkspace<R> {
    fn default() -> Self {
        Self::new()
    }
}
