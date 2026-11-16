use super::*;
use crate::SparseCholesky;
use crate::factor_solve::{Application, Oriented};
use scioxide_core::AlgebraicScalar;
use scioxide_core::Preconditioner;
/// Selects how often a multigrid cycle visits each coarser level: once (V
/// cycle) or twice (W cycle).
///
/// # Examples
///
/// ```
/// use scioxide_core::{BlockVectorView, ExecutionContext};
/// use scioxide_sparse::{
///     AmgCycleKind, AmgCycleOptions, AmgHierarchy, AmgLimits, AmgSetupOptions, CsrMatrix,
///     SparseSolveOptions, sparse_diagonals,
/// };
/// let a = sparse_diagonals(9, 9, &[-1, 0, 1], &[&[-1.0; 8], &[2.0; 9], &[-1.0; 8]])?;
/// let (options, context) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let h = AmgHierarchy::build(
///     a.view(),
///     BlockVectorView::new(&[1.0; 9], 9, 1)?,
///     AmgSetupOptions::new().with_coarse_size(1),
///     AmgLimits::new(4, 1000, 1 << 20),
///     options,
///     context,
/// )?;
/// assert_eq!(h.diagnostics().levels(), 3); // 9, 3 and 1 rows
/// let coarse = h.coarse_matrix().values()[0];
/// let inverse = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![1.0 / coarse])?;
/// let mut x = [[0.0; 9]; 2];
/// for (kind, x) in [AmgCycleKind::V, AmgCycleKind::W].into_iter().zip(&mut x) {
///     let cycle = h.cycle(&inverse, AmgCycleOptions::new().with_kind(kind))?;
///     cycle.apply_into(&[1.0; 9], options, x, &mut cycle.workspace(1)?, context)?;
/// }
/// // W corrects twice on the middle level, so its correction differs from V.
/// assert_ne!(x[0], x[1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmgCycleKind {
    /// V cycle: one coarse-grid correction per level.
    V,
    /// W cycle: two coarse-grid corrections per level. The second starts from
    /// the result of the first.
    W,
}
/// Holds the shape of a multigrid cycle: V or W, and the number of pre- and
/// post-smoothing sweeps.
///
/// A cycle applied from a zero initial guess is a fixed linear map. For use
/// with conjugate gradients it must be Hermitian positive definite. That needs
/// pre- and post-smoothers that are adjoint to each other, equal sweep counts,
/// `R = Pᴴ` on every level and a Hermitian positive definite coarse solver.
/// The [`Accuracy`](scioxide_core::Accuracy) comes from the [`SparseSolveOptions`]
/// of each call.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::{AmgCycleKind, AmgCycleOptions};
/// let options = AmgCycleOptions::new()
///     .with_kind(AmgCycleKind::W)
///     .with_post_sweeps(2);
/// assert_eq!(
///     (options.kind(), options.pre_sweeps(), options.post_sweeps()),
///     (AmgCycleKind::W, 1, 2)
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmgCycleOptions {
    kind: AmgCycleKind,
    pre_sweeps: usize,
    post_sweeps: usize,
}
impl Default for AmgCycleOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl AmgCycleOptions {
    /// Returns a V cycle with one pre-smoothing and one post-smoothing sweep.
    pub const fn new() -> Self {
        Self {
            kind: AmgCycleKind::V,
            pre_sweeps: 1,
            post_sweeps: 1,
        }
    }
    /// Sets the cycle kind.
    pub const fn with_kind(mut self, value: AmgCycleKind) -> Self {
        self.kind = value;
        self
    }
    /// Returns the cycle kind.
    pub const fn kind(self) -> AmgCycleKind {
        self.kind
    }
    /// Sets the number of pre-smoothing sweeps.
    pub const fn with_pre_sweeps(mut self, value: usize) -> Self {
        self.pre_sweeps = value;
        self
    }
    /// Returns the number of pre-smoothing sweeps.
    pub const fn pre_sweeps(self) -> usize {
        self.pre_sweeps
    }
    /// Sets the number of post-smoothing sweeps.
    pub const fn with_post_sweeps(mut self, value: usize) -> Self {
        self.post_sweeps = value;
        self
    }
    /// Returns the number of post-smoothing sweeps.
    pub const fn post_sweeps(self) -> usize {
        self.post_sweeps
    }
}
/// Selects the smoother of one multigrid level.
///
/// A smoother sweep computes the residual `r = b − A x`, a correction `δ = M r`,
/// and updates `x ← x + δ`. For use with conjugate gradients, an injected
/// smoother must be a fixed linear map.
///
/// # Examples
///
/// A two-level cycle with symmetric Gauss–Seidel smoothing, chosen so that every
/// step is exact in binary floating point:
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_sparse::{
///     AmgCycle, AmgCycleOptions, AmgLevelView, AmgSmootherView, CsrMatrix, SparseSolveOptions,
/// };
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![2.0_f64, -1., -1., 2.],
/// )?;
/// let p = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![1.0_f64, 1.])?;
/// let r = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 1.])?;
/// let coarse = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?; // R A P
/// let inverse = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![0.5_f64])?;
/// let unused = AmgSmootherView::Jacobi {
///     inverse_diagonal: &[],
///     weight: 1.0,
/// };
/// let levels = [
///     AmgLevelView {
///         operator: &a,
///         prolongation: Some(&p),
///         restriction: Some(&r),
///         smoother: AmgSmootherView::GaussSeidel { matrix: &a },
///     },
///     AmgLevelView {
///         operator: &coarse,
///         prolongation: None,
///         restriction: None,
///         smoother: unused,
///     },
/// ];
/// let cycle = AmgCycle::new(&levels, &inverse, AmgCycleOptions::new())?;
/// let mut x = [0.0; 2];
/// cycle.apply_into(
///     &[1.0, 1.0],
///     SparseSolveOptions::new(),
///     &mut x,
///     &mut cycle.workspace(1)?,
///     ExecutionContext::sequential(),
/// )?;
/// // Forward sweep [0.5, 0.75], coarse correction 0.375, then a backward sweep.
/// assert_eq!(x, [0.96875, 0.9375]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub enum AmgSmootherView<'a, S: NumericScalar> {
    /// Damped Jacobi: `δ = ω D⁻¹ r`.
    Jacobi {
        /// The inverse diagonal `D⁻¹`, one finite nonzero value per row.
        inverse_diagonal: &'a [S],
        /// The damping factor `ω`, finite and positive. For levels you build
        /// yourself, choosing `ω` so that the smoother converges is up to you.
        weight: S::Real,
    },
    /// Gauss–Seidel: before the coarse correction, `δ` solves `(D + L) δ = r`
    /// (forward sweep); after it, `(D + U) δ = r` (backward sweep). `L` and `U`
    /// are the strict lower and upper triangles of `matrix`.
    GaussSeidel {
        /// The matrix whose triangles define the sweeps. Its diagonal must be
        /// stored and nonzero.
        matrix: &'a CsrMatrix<S>,
    },
    /// Preconditioners of your own that map a residual to a correction.
    Injected {
        /// The pre-smoother.
        pre: &'a (dyn Preconditioner<S> + Sync),
        /// The post-smoother. For a symmetric cycle, use the adjoint of the
        /// pre-smoother.
        post: &'a (dyn Preconditioner<S> + Sync),
    },
}
/// Describes one level of a multigrid hierarchy for [`AmgCycle::new`].
///
/// The coarsest level has no prolongation and no restriction; every other level
/// has both. The operators can be sparse matrices, dense matrix views or
/// matrix-free [`LinearOperator`]s.
///
/// # Examples
///
/// ```
/// use scioxide_core::{BlockVectorView, ExecutionContext};
/// use scioxide_sparse::{
///     AmgHierarchy, AmgLimits, AmgSetupOptions, AmgSmootherView, CsrMatrix, SparseSolveOptions,
/// };
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![2.0_f64, -1., -1., 2., -1., -1., 2.],
/// )?;
/// let h = AmgHierarchy::build(
///     a.view(),
///     BlockVectorView::new(&[1.0; 3], 3, 1)?,
///     AmgSetupOptions::new().with_coarse_size(1),
///     AmgLimits::new(4, 1000, 1 << 20),
///     SparseSolveOptions::new(),
///     ExecutionContext::sequential(),
/// )?;
/// let fine = h.level(0).unwrap();
/// assert_eq!(
///     (fine.operator.rows(), fine.prolongation.map(|p| p.cols())),
///     (3, Some(1))
/// );
/// assert!(matches!(fine.smoother, AmgSmootherView::Jacobi { .. }));
/// // The coarsest level has no transfers; out-of-range levels are absent.
/// assert!(h.level(1).unwrap().restriction.is_none() && h.level(2).is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct AmgLevelView<'a, S: NumericScalar> {
    /// The square matrix `A_ℓ` of this level.
    pub operator: &'a (dyn LinearOperator<S> + Sync),
    /// The prolongation `P_ℓ` from the next coarser level to this one.
    pub prolongation: Option<&'a (dyn LinearOperator<S> + Sync)>,
    /// The restriction `R_ℓ` from this level to the next coarser one. It is
    /// used as given, not formed from `P_ℓ`.
    pub restriction: Option<&'a (dyn LinearOperator<S> + Sync)>,
    /// The smoother; ignored on the coarsest level.
    pub smoother: AmgSmootherView<'a, S>,
}
#[derive(Clone, Copy)]
pub(super) enum Source<'a, S: NumericScalar> {
    Owned(&'a AmgHierarchy<S>),
    Borrowed(&'a [AmgLevelView<'a, S>]),
}
impl<'a, S: NumericScalar> Source<'a, S> {
    fn levels(self) -> usize {
        match self {
            Self::Owned(h) => h.levels.len(),
            Self::Borrowed(l) => l.len(),
        }
    }
    fn level(self, i: usize) -> Option<AmgLevelView<'a, S>> {
        match self {
            Self::Owned(h) => h.level(i),
            Self::Borrowed(l) => l.get(i).copied(),
        }
    }
}
/// Coarse inverse: a caller operator, or the owner's Cholesky factor solved at
/// the cycle's arithmetic.
#[derive(Clone, Copy)]
pub(super) enum Coarse<'a, S: NumericScalar> {
    Operator(&'a (dyn LinearOperator<S> + Sync)),
    Cholesky(&'a SparseCholesky<S>),
}
impl<S: NumericScalar> Coarse<'_, S> {
    fn shape(self) -> (usize, usize) {
        match self {
            Self::Operator(op) => (op.rows(), op.cols()),
            Self::Cholesky(factor) => (factor.dimension(), factor.dimension()),
        }
    }
    fn work_len(self) -> Result<usize, SciOxideError> {
        match self {
            Self::Operator(op) => op.apply_workspace_len(),
            Self::Cholesky(factor) => Ok(factor.dimension()),
        }
    }
}
/// Checked shapes and exact provider workspaces of every level, kept so
/// applications never recheck or re-query providers. Fixed arrays keep
/// borrowed cycles free of allocation.
#[derive(Clone, Copy, Debug)]
pub(super) struct Layout {
    levels: usize,
    rows: [usize; 64],
    work: [usize; 64],
    provider: [usize; 64],
    operator: [usize; 64],
    prolongation: [usize; 64],
    restriction: [usize; 64],
    smoothers: [[usize; 2]; 64],
}
// Validate levels, the coarse inverse and options once; `argument` names the
// caller's parameter that supplied the levels.
pub(super) fn prepare<S: NumericScalar>(
    source: Source<'_, S>,
    coarse: Coarse<'_, S>,
    options: AmgCycleOptions,
    argument: Argument,
) -> Result<Layout, SciOxideError> {
    let scratch = scioxide_core::quantity!("cycle_scratch");
    let levels = source.levels();
    if levels == 0 || levels > 64 {
        return Err(invalid(argument, "AMG needs 1..=64 levels"));
    }
    add(
        options.pre_sweeps,
        options.post_sweeps,
        scioxide_core::quantity!("smoothing_sweeps"),
    )?;
    let mut layout = Layout {
        levels,
        rows: [0; 64],
        work: [0; 64],
        provider: [0; 64],
        operator: [0; 64],
        prolongation: [0; 64],
        restriction: [0; 64],
        smoothers: [[0; 2]; 64],
    };
    for i in (0..levels).rev() {
        let level = source
            .level(i)
            .ok_or_else(|| invalid(argument, "AMG level index"))?;
        let n = level.operator.rows();
        crate::helpers::exact(level.operator.cols(), n)?;
        layout.rows[i] = n;
        if i + 1 == levels {
            if level.prolongation.is_some() || level.restriction.is_some() {
                return Err(invalid(argument, "coarse level cannot have transfers"));
            }
            crate::helpers::shape(coarse.shape(), (n, n))?;
            layout.work[i] = coarse.work_len()?;
            continue;
        }
        let p = level
            .prolongation
            .ok_or_else(|| invalid(argument, "missing AMG prolongation"))?;
        let r = level
            .restriction
            .ok_or_else(|| invalid(argument, "missing AMG restriction"))?;
        let nc = layout.rows[i + 1];
        crate::helpers::shape((p.rows(), p.cols()), (n, nc))?;
        crate::helpers::shape((r.rows(), r.cols()), (nc, n))?;
        if nc >= n {
            return Err(invalid(
                argument,
                "AMG transfers must connect strictly decreasing levels",
            ));
        }
        layout.smoothers[i] = level.smoother.validate(n, argument)?;
        layout.operator[i] = level.operator.apply_workspace_len()?;
        layout.prolongation[i] = p.apply_workspace_len()?;
        layout.restriction[i] = r.apply_workspace_len()?;
        layout.provider[i] = layout.operator[i]
            .max(layout.prolongation[i])
            .max(layout.restriction[i])
            .max(layout.smoothers[i][0])
            .max(layout.smoothers[i][1]);
        layout.work[i] = add(
            add(mul(2, n, scratch)?, layout.provider[i], scratch)?,
            add(mul(2, nc, scratch)?, layout.work[i + 1], scratch)?,
            scratch,
        )?;
    }
    add(layout.rows[0], layout.work[0], scratch)?;
    Ok(layout)
}
/// A prepared cycle borrowed by every application; constructing it validates nothing.
pub(super) struct Engine<'a, S: NumericScalar> {
    pub(super) source: Source<'a, S>,
    pub(super) coarse: Coarse<'a, S>,
    pub(super) options: AmgCycleOptions,
    pub(super) layout: &'a Layout,
}
impl<S: NumericScalar> Engine<'_, S> {
    pub(super) const fn dimension(&self) -> usize {
        self.layout.rows[0]
    }
    pub(super) const fn work_len(&self) -> usize {
        self.layout.work[0]
    }
    // Recursive cycle at level `i`.
    fn run(
        &self,
        i: usize,
        (rhs, x): (&[S], &mut [S]),
        scratch: &mut [S],
        accuracy: scioxide_core::Accuracy,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        const SELF: Argument = scioxide_core::argument!("self");
        execution.check()?;
        let layout = self.layout;
        if i + 1 == layout.levels {
            let work = &mut scratch[..layout.work[i]];
            return match self.coarse {
                Coarse::Operator(op) => {
                    op.apply_into(rhs, x, work)?;
                    crate::helpers::finite_with_origin(x, scioxide_core::NonFiniteOrigin::Callback)
                }
                Coarse::Cholesky(factor) => Oriented {
                    factor,
                    operation: crate::SparseOperation::Normal,
                }
                .apply_column(
                    rhs,
                    x,
                    work,
                    SparseSolveOptions::new().with_accuracy(accuracy),
                    execution,
                ),
            };
        }
        let mut level = self
            .source
            .level(i)
            .ok_or_else(|| invalid(SELF, "AMG level index"))?;
        // Owned sparse levels apply at the selected arithmetic; supplied
        // operators keep their own.
        let (a, p, r) = if let Source::Owned(h) = self.source {
            let level = &h.levels[i];
            (
                Some(level.a.view().with_accuracy(accuracy)),
                level.p.as_ref().map(|m| m.view().with_accuracy(accuracy)),
                level.r.as_ref().map(|m| m.view().with_accuracy(accuracy)),
            )
        } else {
            (None, None, None)
        };
        if let Some(a) = a.as_ref() {
            level.operator = a;
        }
        if let Some(p) = p.as_ref() {
            level.prolongation = Some(p);
        }
        if let Some(r) = r.as_ref() {
            level.restriction = Some(r);
        }
        let n = layout.rows[i];
        let nc = layout.rows[i + 1];
        let (residual, rest) = scratch.split_at_mut(n);
        let (delta, rest) = rest.split_at_mut(n);
        let (provider, rest) = rest.split_at_mut(layout.provider[i]);
        let (bc, rest) = rest.split_at_mut(nc);
        let (xc, child) = rest.split_at_mut(nc);
        let operator = (level.operator, layout.operator[i]);
        for _ in 0..self.options.pre_sweeps {
            execution.check()?;
            residual_into(operator, (rhs, x), residual, provider, accuracy)?;
            level.smoother.apply(
                false,
                residual,
                delta,
                &mut provider[..layout.smoothers[i][0]],
                accuracy,
            )?;
            accumulate(x, delta, accuracy)?;
        }
        execution.check()?;
        residual_into(operator, (rhs, x), residual, provider, accuracy)?;
        let r = level
            .restriction
            .ok_or_else(|| invalid(SELF, "missing restriction"))?;
        r.apply_into(residual, bc, &mut provider[..layout.restriction[i]])?;
        crate::helpers::finite_with_origin(bc, scioxide_core::NonFiniteOrigin::Callback)?;
        xc.fill(S::ZERO);
        let repeats = if self.options.kind == AmgCycleKind::V {
            1
        } else {
            2
        };
        for _ in 0..repeats {
            self.run(i + 1, (bc, xc), child, accuracy, execution)?;
        }
        let p = level
            .prolongation
            .ok_or_else(|| invalid(SELF, "missing prolongation"))?;
        p.apply_into(xc, delta, &mut provider[..layout.prolongation[i]])?;
        accumulate(x, delta, accuracy)?;
        for _ in 0..self.options.post_sweeps {
            execution.check()?;
            residual_into(operator, (rhs, x), residual, provider, accuracy)?;
            level.smoother.apply(
                true,
                residual,
                delta,
                &mut provider[..layout.smoothers[i][1]],
                accuracy,
            )?;
            accumulate(x, delta, accuracy)?;
        }
        Ok(())
    }
    // One cycle from a zero initial iterate into `candidate`.
    pub(super) fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        work: &mut [S],
        options: SparseSolveOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        candidate.fill(S::ZERO);
        self.run(0, (rhs, candidate), work, options.accuracy(), execution)?;
        execution.check()?;
        crate::helpers::finite_with_origin(candidate, scioxide_core::NonFiniteOrigin::Intermediate)
    }
}
/// Applies one multigrid cycle as a preconditioner, on borrowed levels and a
/// coarse solver.
///
/// On level `ℓ` with right-hand side `b`, starting from `x = 0`, the cycle
///
/// 1. applies the pre-smoothing sweeps;
/// 2. restricts the residual, `b_c = R_ℓ (b − A_ℓ x)`;
/// 3. solves `A_(ℓ+1) x_c = b_c` approximately by the cycle on the next level,
///    once for a V cycle and twice for a W cycle, or by the coarse solver on the
///    coarsest level;
/// 4. corrects `x ← x + P_ℓ x_c`;
/// 5. applies the post-smoothing sweeps.
///
/// `apply_into` writes `x` for each right-hand side, and the cycle implements
/// the core [`Preconditioner`] trait. Shapes, transfers and smoothers are
/// checked once, when the cycle is created. The
/// [`Accuracy`](scioxide_core::Accuracy) of the call applies to the cycle updates,
/// the built-in smoothers and the matrices of an [`AmgHierarchy`]; operators you
/// supply use their own arithmetic. Each right-hand side counts as one
/// iteration.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::{AmgCycle, AmgCycleOptions, AmgLevelView, AmgSmootherView, CsrMatrix};
/// let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
/// let inverse = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![0.5_f64])?;
/// let levels = [AmgLevelView {
///     operator: &a,
///     prolongation: None,
///     restriction: None,
///     smoother: AmgSmootherView::Jacobi {
///         inverse_diagonal: &[],
///         weight: 1.0,
///     },
/// }];
/// let cycle = AmgCycle::new(&levels, &inverse, AmgCycleOptions::new())?;
/// assert_eq!(cycle.apply(&[4.0])?, [2.0]);
/// // A coarse inverse of the wrong shape is rejected at construction.
/// let wrong = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0])?;
/// assert!(AmgCycle::new(&levels, &wrong, AmgCycleOptions::new()).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct AmgCycle<'a, S: NumericScalar> {
    source: Source<'a, S>,
    coarse: &'a (dyn LinearOperator<S> + Sync),
    options: AmgCycleOptions,
    layout: Layout,
}
impl<'a, S: NumericScalar> AmgCycle<'a, S> {
    /// Creates a cycle on `levels`, finest first, with `coarse` applying the
    /// inverse of the coarsest matrix.
    ///
    /// Only shapes and smoother data are checked; whether `A_(ℓ+1) = R_ℓ A_ℓ P_ℓ`,
    /// whether the cycle is symmetric and whether it converges are not.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are no levels or more than 64, a level other
    ///   than the coarsest lacks a transfer, the coarsest level has one, a level
    ///   is not smaller than the one above it, or a Jacobi smoother has a zero
    ///   inverse diagonal entry or a weight that is not finite and positive.
    /// - `DimensionMismatch` if an operator, transfer, smoother or `coarse` has
    ///   the wrong shape.
    /// - `Singular` if a Gauss–Seidel matrix has a zero diagonal entry.
    /// - `NonFinite` if a Jacobi inverse diagonal contains NaN or infinity.
    /// - `SizeOverflow` if the sweep count or scratch length overflows `usize`.
    pub fn new(
        levels: &'a [AmgLevelView<'a, S>],
        coarse: &'a (dyn LinearOperator<S> + Sync),
        options: AmgCycleOptions,
    ) -> Result<Self, SciOxideError> {
        Self::prepared(
            Source::Borrowed(levels),
            coarse,
            options,
            scioxide_core::argument!("levels"),
        )
    }
    // Borrows an owned hierarchy.
    pub(super) fn from_owned(
        hierarchy: &'a AmgHierarchy<S>,
        coarse: &'a (dyn LinearOperator<S> + Sync),
        options: AmgCycleOptions,
    ) -> Result<Self, SciOxideError> {
        Self::prepared(
            Source::Owned(hierarchy),
            coarse,
            options,
            scioxide_core::argument!("self"),
        )
    }
    fn prepared(
        source: Source<'a, S>,
        coarse: &'a (dyn LinearOperator<S> + Sync),
        options: AmgCycleOptions,
        argument: Argument,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            layout: prepare(source, Coarse::Operator(coarse), options, argument)?,
            source,
            coarse,
            options,
        })
    }
    fn engine(&self) -> Engine<'_, S> {
        Engine {
            source: self.source,
            coarse: Coarse::Operator(self.coarse),
            options: self.options,
            layout: &self.layout,
        }
    }
}
impl<S: NumericScalar> Application<S> for AmgCycle<'_, S> {
    fn dimension(&self) -> usize {
        self.layout.rows[0]
    }
    fn work_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.layout.work[0])
    }
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        work: &mut [S],
        options: SparseSolveOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.engine()
            .apply_column(rhs, candidate, work, options, execution)
    }
}
crate::factor_solve::impl_applications!(AmgCycle<'_, S>);
// `x += delta`.
fn accumulate<S: NumericScalar>(
    x: &mut [S],
    delta: &[S],
    accuracy: scioxide_core::Accuracy,
) -> Result<(), SciOxideError> {
    for (x, &d) in x.iter_mut().zip(delta) {
        *x = scioxide_core::backend::standard::sum_products(
            [(*x, S::ONE), (d, S::ONE)].into_iter(),
            accuracy,
        )?;
    }
    Ok(())
}
// `result = rhs - A x`, with A's exact prepared workspace length.
fn residual_into<S: NumericScalar>(
    (a, workspace): (&(dyn LinearOperator<S> + Sync), usize),
    (rhs, x): (&[S], &[S]),
    result: &mut [S],
    scratch: &mut [S],
    accuracy: scioxide_core::Accuracy,
) -> Result<(), SciOxideError> {
    a.apply_into(x, result, &mut scratch[..workspace])?;
    for (&b, ax) in rhs.iter().zip(result) {
        *ax = scioxide_core::backend::standard::sum_products(
            [(b, S::ONE), (-*ax, S::ONE)].into_iter(),
            accuracy,
        )?;
    }
    Ok(())
}
impl<S: NumericScalar> AmgSmootherView<'_, S> {
    // Validates one smoother and returns its pre/post provider workspaces;
    // `source` names the caller's argument that supplied the levels.
    fn validate(self, n: usize, source: Argument) -> Result<[usize; 2], SciOxideError> {
        match self {
            Self::Jacobi {
                inverse_diagonal,
                weight,
            } => {
                crate::helpers::exact(inverse_diagonal.len(), n)?;
                if !weight.is_finite()
                    || weight <= S::Real::ZERO
                    || inverse_diagonal.contains(&S::ZERO)
                {
                    return Err(invalid(source, "invalid AMG Jacobi data"));
                }
                finite(inverse_diagonal)?;
                Ok([0; 2])
            }
            Self::GaussSeidel { matrix } => {
                crate::helpers::shape((matrix.rows(), matrix.cols()), (n, n))?;
                for row in 0..n {
                    if matrix.get(row, row)? == S::ZERO {
                        return Err(SciOxideError::singular(Some(row)));
                    }
                }
                Ok([0; 2])
            }
            Self::Injected { pre, post } => {
                crate::helpers::exact(pre.dimension(), n)?;
                crate::helpers::exact(post.dimension(), n)?;
                Ok([
                    pre.precondition_workspace_len()?,
                    post.precondition_workspace_len()?,
                ])
            }
        }
    }
    // One smoothing correction; `scratch` has the provider's exact length.
    fn apply(
        self,
        post_sweep: bool,
        rhs: &[S],
        result: &mut [S],
        scratch: &mut [S],
        accuracy: scioxide_core::Accuracy,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Jacobi {
                inverse_diagonal,
                weight,
            } => {
                for ((&b, &d), x) in rhs.iter().zip(inverse_diagonal).zip(result.iter_mut()) {
                    let value = scioxide_core::backend::standard::product(b, d, accuracy)?;
                    *x = scioxide_core::backend::standard::product(
                        value,
                        S::from_real(weight),
                        accuracy,
                    )?;
                }
            }
            Self::GaussSeidel { matrix } => {
                result.fill(S::ZERO);
                for k in 0..rhs.len() {
                    let i = if post_sweep { rhs.len() - 1 - k } else { k };
                    let pivot = matrix.get(i, i)?;
                    result[i] = match accuracy {
                        scioxide_core::Accuracy::Exact => {
                            let mut sum = ProductSum::new();
                            sum.add_product(rhs[i], S::ONE)?;
                            for pos in matrix.row_ptr()[i]..matrix.row_ptr()[i + 1] {
                                let j = matrix.col_indices()[pos];
                                if (post_sweep && j > i) || (!post_sweep && j < i) {
                                    sum.add_product(-matrix.values()[pos], result[j])?;
                                }
                            }
                            sum.finish()? / pivot
                        }
                        scioxide_core::Accuracy::Standard => {
                            scioxide_core::backend::standard::substitute_row(
                                rhs[i],
                                pivot,
                                (matrix.row_ptr()[i]..matrix.row_ptr()[i + 1]).map(|p| {
                                    let j = matrix.col_indices()[p];
                                    if (post_sweep && j > i) || (!post_sweep && j < i) {
                                        (matrix.values()[p], result[j])
                                    } else {
                                        (S::ZERO, S::ZERO)
                                    }
                                }),
                            )?
                        }
                    };
                    crate::helpers::finite_with_origin(
                        &result[i..=i],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
            Self::Injected { pre, post } => {
                let smoother = if post_sweep { post } else { pre };
                smoother.precondition_into(rhs, result, scratch)?;
            }
        }
        crate::helpers::finite_with_origin(
            result,
            if matches!(self, Self::Injected { .. }) {
                scioxide_core::NonFiniteOrigin::Callback
            } else {
                scioxide_core::NonFiniteOrigin::Intermediate
            },
        )
    }
}
