//! Deflation: finding further roots of a system after some are known.
use super::*;
use scioxide_core::{ResidualReport, Scaled, ScaledAccumulator};
/// Sets the deflation operator, the acceptance tests and the limits of a [`DeflationSet`].
///
/// The defaults are power `p = 2`, shift `σ = 1`, a residual tolerance `√ε`
/// relative with no absolute part, and a separation tolerance `√ε` absolute and
/// relative, where `ε` is the machine epsilon of `R`.
///
/// The relative residual test needs a reference. A deflated solve uses
/// `‖F(x₀)‖₂` at its initial point unless [`Self::with_residual_scale`] sets one.
/// [`DeflationSet::insert`] has no initial point, so with the defaults it accepts
/// only exact roots (`F(r) = 0`). To insert an approximate root, set a residual
/// scale or an absolute residual tolerance.
#[derive(Debug, Clone)]
pub struct DeflationOptions<R: RealScalar> {
    power: u32,
    shift: R,
    residual: ResidualTolerance<R>,
    residual_scale: Option<R>,
    separation: StepTolerance<R>,
    budgets: SolveBudgets,
    byte_limit: usize,
}
impl<R: RealScalar> Default for DeflationOptions<R> {
    fn default() -> Self {
        Self {
            budgets: SolveBudgets::unbounded(),
            byte_limit: usize::MAX,
            power: 2,
            shift: R::ONE,
            residual: crate::termination::relative_tolerance(),
            residual_scale: None,
            separation: StepTolerance::from_tolerance(Tolerance::from_sqrt_epsilon()),
        }
    }
}
impl<R: RealScalar> DeflationOptions<R> {
    /// Sets the work limits of each [`DeflationSet::insert_in_place`] call.
    ///
    /// Limits of a parent solve context still apply.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the work limits of each insertion.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }
    /// Limits the size in bytes of the root set and, separately, of each
    /// insertion workspace; a larger requirement is a `ResourceLimit` error.
    pub const fn with_workspace_byte_limit(mut self, bytes: usize) -> Self {
        self.byte_limit = bytes;
        self
    }
    /// Returns the size limit in bytes.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.byte_limit
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        if self.power == 0
            || !self.shift.is_finite()
            || self.shift <= R::ZERO
            || self
                .residual_scale
                .is_some_and(|v| !v.is_finite() || v <= R::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                InvalidInputReason::Inconsistent,
                "deflation needs positive finite power, shift and residual scale",
            ));
        }
        Ok(())
    }

    /// Sets the power `p` and shift `σ` of the deflation factor
    /// `‖x − r‖₂^(−p) + σ`.
    ///
    /// A larger `p` repels iterates from known roots more strongly. The shift keeps
    /// the deflated residual from vanishing far from the known roots.
    /// [`DeflationSet::new`] rejects `p = 0` and a shift that is not positive and finite.
    pub fn with_shifted_power(mut self, power: u32, shift: R) -> Self {
        self.power = power;
        self.shift = shift;
        self
    }
    /// Sets the residual tolerance for accepting a root of the original equation
    /// `F(x) = 0`, both on insertion and after a deflated solve.
    pub fn with_residual_tolerance(mut self, value: ResidualTolerance<R>) -> Self {
        self.residual = value;
        self
    }
    /// Sets a fixed reference `scale` for the relative residual test.
    ///
    /// [`DeflationSet::new`] rejects a scale that is not positive and finite.
    pub fn with_residual_scale(mut self, scale: R) -> Self {
        self.residual_scale = Some(scale);
        self
    }
    /// Sets the separation tolerance.
    ///
    /// A point `x` counts as the known root `r` when `‖x − r‖₂` passes this
    /// tolerance at scale `max(1, ‖x‖₂, ‖r‖₂)`. Such a point is rejected as a
    /// duplicate on insertion, makes a deflated solve `Unverified`, and is outside
    /// the domain of the deflated residual.
    pub fn with_separation_tolerance(mut self, value: StepTolerance<R>) -> Self {
        self.separation = value;
        self
    }
}
/// Stores known roots of a square system `F(x) = 0` for deflation.
///
/// The set has a fixed capacity and holds only roots you insert, after
/// [`Self::insert`] has checked their residual. It belongs to one equation: if
/// `F` changes, for example through a parameter, create a new set. Use it with
/// [`DeflatedProblem`] to search for roots not yet in the set. Deflation does not
/// show that all roots have been found.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_rootfind::{DeflationOptions, DeflationSet, ResidualFn};
///
/// // F(x) = x² − 1 has the exact roots ±1.
/// let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] * x[0] - 1.0;
///     Ok(())
/// });
/// let mut set = DeflationSet::new(1, Budget::try_from(2)?, DeflationOptions::default())?;
/// set.insert(&f, &[1.0])?;
/// assert_eq!(set.len(), 1);
/// // The same root again is rejected as a duplicate.
/// assert!(set.insert(&f, &[1.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. E. Farrell, Á. Birkisson and S. W. Funke, "Deflation techniques for
/// finding distinct solutions of nonlinear partial differential equations",
/// *SIAM J. Sci. Comput.* 37(4), A2026–A2045, 2015.
pub struct DeflationSet<R: RealScalar> {
    n: usize,
    limit: usize,
    options: DeflationOptions<R>,
    roots: Vec<R>,
}
impl<R: RealScalar> DeflationSet<R> {
    /// Creates an empty set for roots in `ℝ^dimension`, with room for `capacity` roots.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero or the options are invalid (see
    ///   [`DeflationOptions`]).
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the storage
    ///   cannot be reserved within the byte limit.
    pub fn new(
        dimension: usize,
        capacity: Budget,
        options: DeflationOptions<R>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                InvalidInputReason::Inconsistent,
                "empty deflation dimension",
            ));
        }
        let size = dimension
            .checked_mul(capacity.get())
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("roots")))?;
        let bytes = scioxide_core::backend::checked_mul(
            size,
            core::mem::size_of::<R>(),
            scioxide_core::quantity!("roots"),
        )?;
        if bytes > options.byte_limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("root_set_bytes"),
                bytes,
                options.byte_limit,
            ));
        }
        Ok(Self {
            n: dimension,
            limit: capacity.get(),
            options,
            roots: scioxide_core::backend::try_with_capacity(
                size,
                scioxide_core::quantity!("roots"),
            )?,
        })
    }
    /// Returns the number of roots in the set.
    pub fn len(&self) -> usize {
        self.roots.len() / self.n
    }
    /// Returns whether the set holds no roots.
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
    /// Returns root `index`, or `None` if there is no such root.
    pub fn root(&self, index: usize) -> Option<&[R]> {
        index
            .checked_mul(self.n)
            .and_then(|i| self.roots.get(i..i.checked_add(self.n)?))
    }
    /// Returns the scratch length, in scalars, needed to check a root of `p`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `p` is not square with the set's dimension.
    /// - `SizeOverflow`, or an error from the provider's workspace query.
    pub fn verification_workspace_len(
        &self,
        p: &dyn ResidualOperator<R>,
    ) -> Result<usize, SciOxideError> {
        self.check_provider(p, scioxide_core::argument!("p"))?;
        checked_size(
            self.n,
            p.residual_workspace_len()?,
            scioxide_core::quantity!("verification_workspace_len"),
        )
    }
    /// Checks that the provider `argument` is square with this set's dimension.
    fn check_provider(
        &self,
        p: &dyn ResidualOperator<R>,
        argument: Argument,
    ) -> Result<(), SciOxideError> {
        if p.input_dimension() != self.n || p.output_dimension() != self.n {
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "deflation requires a matching square problem",
            ));
        }
        Ok(())
    }
    /// Returns whether `x` is separated from every root in the set.
    fn separated(&self, x: &[R], difference_buffer: &mut [R]) -> Result<bool, SciOxideError> {
        let scale = norm(x)?.max(R::ONE);
        for root in self.roots.chunks_exact(self.n) {
            for ((d, &a), &b) in difference_buffer.iter_mut().zip(x).zip(root) {
                *d = difference(a, b)?;
            }
            if self
                .options
                .separation
                .is_satisfied(norm(difference_buffer)?, scale.max(norm(root)?))?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Checks the residual of `root` and appends it to the set.
    ///
    /// On error the set is unchanged; callback calls already made stay counted.
    fn insert_kernel(
        &mut self,
        p: &dyn ResidualOperator<R>,
        root: &[R],
        scratch: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<ResidualReport<R>, SciOxideError> {
        let (out, s) = scratch.split_at_mut(self.n);
        adapters::invoke_residual(p, root, out, s, &mut Some(c))?;
        let raw = norm(out)?;
        let scale = self.options.residual_scale.unwrap_or(R::ZERO);
        if !crate::termination::residual_ok(&self.options.residual.tolerance(), raw, scale)? {
            return Err(invalid(
                scioxide_core::argument!("root"),
                InvalidInputReason::Inconsistent,
                "inserted point fails original residual tolerance",
            ));
        }
        let report = crate::context::residual(raw, scale)?;
        c.assess_accepted(PartialResultTrust::ResidualChecked)?;
        self.roots.extend_from_slice(root);
        Ok(report)
    }
    /// Returns the workspace size needed to insert roots of `problem`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verification_workspace_len`].
    /// - `ResourceLimit` if the size exceeds the byte limit.
    pub fn workspace_requirements(
        &self,
        problem: &dyn ResidualOperator<R>,
    ) -> Result<DeflationWorkspaceRequirements<R>, SciOxideError> {
        DeflationWorkspaceRequirements::new(
            self.n,
            self.verification_workspace_len(problem)?,
            None,
            NewtonKrylovOptions::default(),
            self.options.byte_limit,
        )
    }
    /// Allocates a workspace for inserting roots of `problem`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace(
        &self,
        problem: &dyn ResidualOperator<R>,
    ) -> Result<DeflationWorkspace<R>, SciOxideError> {
        DeflationWorkspace::allocate(&self.workspace_requirements(problem)?)
    }
    /// Inserts `root` after checking it, using a workspace you provide.
    ///
    /// The call evaluates `F(root)` once and appends `root` only if its residual
    /// passes the tolerance of [`DeflationOptions`]. On error the set is unchanged.
    /// The workspace records the call in `last_attempt()`, with the residual
    /// report when the root was inserted.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `root.len()` differs from the set's dimension.
    /// - `NonFinite` if `root` or `F(root)` contains NaN or infinity.
    /// - `InvalidInput` if `problem` is not square with the set's dimension,
    ///   `workspace` is too small, `root` is within the separation tolerance of a
    ///   root already in the set, or the residual fails the tolerance.
    /// - `ResourceLimit` if the set is full.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the call.
    /// - Errors returned by `problem`, unchanged.
    pub fn insert_in_place<'scope, 'capability>(
        &mut self,
        problem: &dyn ResidualOperator<R>,
        root: &[R],
        workspace: &mut DeflationWorkspace<R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let verification = core::cell::Cell::new(0);
        crate::attempt::attempt(
            workspace,
            self,
            scope.into(),
            scioxide_core::AttemptKind::Call,
            |workspace, set| {
                let requirements = set.workspace_requirements(problem)?;
                verification.set(requirements.verification);
                if root.len() != set.n {
                    return Err(SciOxideError::dimension_mismatch([set.n], [root.len()]));
                }
                finite(root, NonFiniteOrigin::Input)?;
                if workspace.check(&requirements)? != scioxide_core::Capacity::Sufficient {
                    return Err(invalid(
                        scioxide_core::argument!("workspace"),
                        InvalidInputReason::Inconsistent,
                        "reserve insertion verification storage",
                    ));
                }
                if set.len() == set.limit {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("excluded_roots"),
                        set.len() + 1,
                        set.limit,
                    ));
                }
                if !set.separated(root, &mut workspace.verification[..set.n])? {
                    return Err(invalid(
                        scioxide_core::argument!("root"),
                        InvalidInputReason::Inconsistent,
                        "duplicate or excluded root",
                    ));
                }
                Ok(crate::attempt::Entry::call(
                    SolveIdentity::new(
                        ProblemToken::new(0),
                        scioxide_core::algorithm_id!("rootfind", "deflation-insertion", 1),
                        set.n as u128,
                        0,
                    ),
                    set.options.budgets,
                ))
            },
            |workspace, set, context| {
                let report = set.insert_kernel(
                    problem,
                    root,
                    &mut workspace.verification[..verification.get()],
                    context,
                )?;
                workspace.accepted.copy_from_slice(root);
                Ok(report)
            },
            |_, _, result| crate::attempt::Conclusion {
                succeeded: result.is_ok(),
                progress: result.as_ref().ok().copied(),
            },
        )
        .map(|_| ())
    }
    /// Inserts `root` after checking it.
    ///
    /// This allocates a workspace and runs [`Self::insert_in_place`] sequentially.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::insert_in_place`] and [`Self::workspace`].
    pub fn insert(
        &mut self,
        problem: &dyn ResidualOperator<R>,
        root: &[R],
    ) -> Result<(), SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        self.insert_in_place(
            problem,
            root,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}
/// Finds a root of `F(x) = 0` that is not in a [`DeflationSet`], by Newton–Krylov
/// iteration on the deflated system.
///
/// With known roots `r₁, …, r_k`, power `p` and shift `σ` from
/// [`DeflationOptions`], the deflated residual is
///
/// ```text
/// G(x) = M(x) F(x),   M(x) = ∏ᵢ (‖x − rᵢ‖₂^(−p) + σ).
/// ```
///
/// `G` has the same roots as `F` except the known ones, near which `M` grows
/// without bound, so Newton's method on `G` is repelled from them. The
/// Jacobian–vector product of `G` is formed analytically from that of `F`.
/// Points within the separation tolerance of a known root are outside the
/// domain of `G`.
///
/// [`Self::solve`] and [`Self::solve_into`] run the Newton–Krylov solve on `G`,
/// then evaluate `F` at the result. The result is successful only if `F` passes
/// the residual tolerance of [`DeflationOptions`] and the point is separated
/// from every known root. `DeflatedProblem` also implements
/// [`JacobianOperator`], so other solvers can use `G` directly, but they then
/// check only `G`, not `F`.
///
/// # References
///
/// P. E. Farrell, Á. Birkisson and S. W. Funke, "Deflation techniques for
/// finding distinct solutions of nonlinear partial differential equations",
/// *SIAM J. Sci. Comput.* 37(4), A2026–A2045, 2015.
pub struct DeflatedProblem<'a, R: RealScalar> {
    provider: &'a dyn JacobianOperator<R>,
    set: &'a DeflationSet<R>,
    linear: Option<&'a dyn NonlinearLinearSolver<R>>,
    solver: NewtonKrylovSolver<R>,
}
/// Holds the result of a [`DeflatedProblem`] solve.
///
/// [`Self::termination`] decides success for the original equation `F(x) = 0`.
/// It equals the termination of the inner Newton–Krylov solve on the deflated
/// residual `G`, except that an inner success becomes
/// [`RootTermination::Unverified`] when `F` fails its residual test or the point
/// is not separated from the known roots. [`Self::inner_termination`] and
/// [`Self::inner_residual_report`] describe the inner solve on `G`.
///
/// A newly created result reports [`RootTermination::NotStarted`].
///
/// ```
/// use scioxide_rootfind::{DeflatedRootResult, NewtonKrylovOptions, RootTermination};
/// let output = DeflatedRootResult::<f64>::new(1, &NewtonKrylovOptions::default())?;
/// assert_eq!(output.termination(), RootTermination::NotStarted);
/// assert!(output.residual_report().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct DeflatedRootResult<R: RealScalar> {
    inner: NewtonKrylovResult<R>,
    residual: Vec<R>,
    report: Option<ResidualReport<R>>,
    termination: RootTermination,
    counts: scioxide_core::SolveCounts,
}
impl<R: RealScalar> DeflatedRootResult<R> {
    /// Creates an empty result for `dimension` unknowns and the corrector `options`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(dimension: usize, options: &NewtonKrylovOptions<R>) -> Result<Self, SciOxideError> {
        Ok(Self {
            inner: NewtonKrylovResult::new(dimension, options)?,
            residual: scioxide_core::backend::try_filled(
                dimension,
                R::ZERO,
                scioxide_core::quantity!("original_residual"),
            )?,
            report: None,
            termination: RootTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        })
    }
    fn empty() -> Self {
        Self {
            inner: NewtonKrylovResult::empty(),
            residual: Vec::new(),
            report: None,
            termination: RootTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        }
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        crate::workspace::sum([
            self.inner.retained_bytes(),
            crate::workspace::buffer(&self.residual),
        ])
    }
    fn has_capacity(&self, dimension: usize, options: &NewtonKrylovOptions<R>) -> bool {
        self.residual.len() == dimension && self.inner.has_capacity(dimension, options)
    }
    /// Returns the final point, meaningful once [`Self::residual_report`] is `Some`.
    pub fn point(&self) -> &[R] {
        self.inner.point()
    }
    /// Returns `F(x)` at [`Self::point`], from a fresh evaluation.
    pub fn residual(&self) -> &[R] {
        &self.residual
    }
    /// Returns the norm of `F(x)` and the reference of its relative test, or `None`
    /// before the first solve.
    pub fn residual_report(&self) -> Option<ResidualReport<R>> {
        self.report
    }
    /// Returns `‖F(x)‖₂`, or `None` before the first solve.
    pub fn residual_norm(&self) -> Option<R> {
        self.report.map(|r| r.raw_norm())
    }
    /// Returns the termination status for the original equation `F(x) = 0`.
    pub const fn termination(&self) -> RootTermination {
        self.termination
    }
    /// Returns whether the solve found a new root of `F` that passed both checks.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the termination status of the inner Newton–Krylov solve on `G`.
    pub fn inner_termination(&self) -> RootTermination {
        self.inner.termination()
    }
    /// Returns the residual of `G` at the end of the inner solve.
    pub fn inner_residual_report(&self) -> Option<ResidualReport<R>> {
        self.inner.residual_report()
    }
    /// Returns the number of Newton iterations of the inner solve.
    pub fn iterations(&self) -> usize {
        self.inner.iterations()
    }
    /// Returns the work done outside the inner solve: the evaluations of `F`.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of the deflated solve, with the inner solve counted as nested work.
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
}
/// Holds the scratch storage for [`DeflationSet::insert_in_place`] and
/// [`DeflatedProblem::solve_into`].
///
/// See [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct DeflationWorkspace<R: RealScalar> {
    newton: NewtonKrylovWorkspace<R>,
    staged: DeflatedRootResult<R>,
    verification: Vec<R>,
    accepted: Vec<R>,
    shell: crate::attempt::Shell<ResidualReport<R>>,
}
/// Describes the workspace needed for a root insertion or a deflated solve.
#[derive(Clone, Debug)]
pub struct DeflationWorkspaceRequirements<R: RealScalar> {
    dimension: usize,
    verification: usize,
    newton: Option<crate::NewtonKrylovWorkspaceRequirements<R>>,
    options: NewtonKrylovOptions<R>,
    bytes: usize,
    limit: usize,
}
impl<R: RealScalar> DeflationWorkspaceRequirements<R> {
    /// Returns the required workspace size in bytes.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    fn new(
        dimension: usize,
        verification: usize,
        newton: Option<crate::NewtonKrylovWorkspaceRequirements<R>>,
        options: NewtonKrylovOptions<R>,
        limit: usize,
    ) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("deflation_workspace");
        let scalars = scioxide_core::backend::checked_add(
            verification,
            scioxide_core::backend::checked_mul(dimension, 2, q)?,
            q,
        )?;
        let bytes = crate::workspace::sum([
            Ok(scioxide_core::backend::checked_mul(
                scalars,
                core::mem::size_of::<R>(),
                q,
            )?),
            Ok(newton.as_ref().map_or(0, |r| r.required_bytes())),
            NewtonKrylovResult::<R>::required_bytes(dimension, &options),
        ])?;
        if bytes > limit {
            return Err(SciOxideError::resource_limit(q, bytes, limit));
        }
        Ok(Self {
            dimension,
            verification,
            newton,
            options,
            bytes,
            limit,
        })
    }
}
impl<R: RealScalar> DeflationWorkspace<R> {
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        crate::workspace::sum([
            self.newton.retained_bytes(),
            self.staged.retained_bytes(),
            crate::workspace::buffer(&self.verification),
            crate::workspace::buffer(&self.accepted),
        ])
    }

    /// Creates an empty workspace.
    ///
    /// Reserve the requirements with [`Self::try_reserve`] before use.
    ///
    /// ```
    /// use scioxide_rootfind::DeflationWorkspace;
    /// let workspace = DeflationWorkspace::<f64>::new();
    /// assert!(workspace.last_attempt().is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            newton: NewtonKrylovWorkspace::new(),
            staged: DeflatedRootResult::empty(),
            verification: Vec::new(),
            accepted: Vec::new(),
            shell: crate::attempt::Shell::new(),
        }
    }
    /// Checks whether this workspace is large enough for `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage held or required exceeds the byte limit.
    /// - Errors from checking the nested Newton–Krylov workspace.
    pub fn check(
        &self,
        r: &DeflationWorkspaceRequirements<R>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        if self.retained_bytes()?.max(r.bytes) > r.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                self.retained_bytes()?.max(r.bytes),
                r.limit,
            ));
        }
        let nonlinear = match &r.newton {
            Some(req) => {
                self.newton.check(req)? == scioxide_core::Capacity::Sufficient
                    && self.staged.has_capacity(r.dimension, &r.options)
            }
            None => true,
        };
        Ok(
            if self.verification.len() >= r.verification
                && self.accepted.len() == r.dimension
                && nonlinear
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
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
        r: &DeflationWorkspaceRequirements<R>,
    ) -> Result<(), SciOxideError> {
        if self.check(r)? == scioxide_core::Capacity::NeedsGrowth {
            let mut prepared = Self::allocate(r)?;
            prepared.shell = self.shell.retained();
            *self = prepared;
        }
        Ok(())
    }
    fn allocate(r: &DeflationWorkspaceRequirements<R>) -> Result<Self, SciOxideError> {
        let mut newton = NewtonKrylovWorkspace::new();
        if let Some(req) = &r.newton {
            newton.try_reserve(req)?;
        }
        Ok(Self {
            newton,
            staged: DeflatedRootResult::new(r.dimension, &r.options)?,
            verification: scioxide_core::backend::try_filled(
                r.verification,
                R::ZERO,
                scioxide_core::quantity!("verification"),
            )?,
            accepted: scioxide_core::backend::try_filled(
                r.dimension,
                R::ZERO,
                scioxide_core::quantity!("accepted"),
            )?,
            shell: crate::attempt::Shell::new(),
        })
    }
    /// Returns the record of the most recent insertion or solve.
    pub const fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<ResidualReport<R>>> {
        self.shell.last_attempt()
    }
    /// Returns the point checked by the most recent call, or `None` if it checked none.
    pub fn last_accepted(&self) -> Option<&[R]> {
        self.shell.accepted(self.accepted.as_slice())
    }
}
crate::attempt::attempted!(DeflationWorkspace<R>, [R: RealScalar], ResidualReport<R>);
impl<'a, R: RealScalar> DeflatedProblem<'a, R> {
    /// Creates the deflated problem for `provider` and the known roots in `set`,
    /// solved with a Newton–Krylov method configured by `options`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `provider` is not square with the set's dimension, or
    ///   `options` are invalid.
    /// - `SizeOverflow`, or an error from the provider's workspace queries.
    pub fn new(
        provider: &'a dyn JacobianOperator<R>,
        set: &'a DeflationSet<R>,
        options: NewtonKrylovOptions<R>,
    ) -> Result<Self, SciOxideError> {
        set.check_provider(provider, scioxide_core::argument!("provider"))?;
        let p = Self {
            provider,
            set,
            linear: None,
            solver: NewtonKrylovSolver::new(options)?,
        };
        p.residual_workspace_len()?;
        Ok(p)
    }
    /// Uses `solver` for the Newton correction equations instead of the built-in GMRES.
    pub fn with_linear_solver(mut self, solver: &'a dyn NonlinearLinearSolver<R>) -> Self {
        self.linear = Some(solver);
        self
    }
    fn problem(&self) -> NewtonKrylovProblem<'_, R> {
        let p = NewtonKrylovProblem::with_jacobian(self);
        self.linear.map_or(p, |s| p.with_linear_solver(s))
    }

    /// Returns the Newton–Krylov options.
    pub fn options(&self) -> &NewtonKrylovOptions<R> {
        self.solver.options()
    }
    /// Returns the workspace size needed by [`Self::solve_into`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow`, or an error from the provider's workspace queries.
    /// - `ResourceLimit` if the size exceeds the byte limit of the Newton–Krylov options.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<DeflationWorkspaceRequirements<R>, SciOxideError> {
        let newton = self.solver.workspace_requirements(&self.problem())?;
        DeflationWorkspaceRequirements::new(
            self.set.n,
            self.set.verification_workspace_len(self.provider)?,
            Some(newton),
            self.solver.options().clone(),
            self.solver.options().workspace_byte_limit(),
        )
    }
    /// Allocates a workspace for [`Self::solve_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace(&self, (): ()) -> Result<DeflationWorkspace<R>, SciOxideError> {
        DeflationWorkspace::allocate(&self.workspace_requirements(())?)
    }
    /// Searches for a new root of `F` starting from `initial`.
    ///
    /// This allocates the workspace and result and runs [`Self::solve_into`]
    /// sequentially. Check [`DeflatedRootResult::is_success`] before using the point.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::solve_into`] and [`Self::workspace`].
    pub fn solve(&self, initial: &[R]) -> Result<DeflatedRootResult<R>, SciOxideError> {
        let mut workspace = self.workspace(())?;
        let mut result = DeflatedRootResult::new(self.set.n, self.solver.options())?;
        self.solve_into(
            initial,
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(result)
    }
    /// Searches for a new root of `F` starting from `initial`, writing to `result`.
    ///
    /// The call evaluates `F(initial)`, runs the Newton–Krylov solve on `G`, and
    /// evaluates `F` at its final point. A limit, cancellation or deadline that
    /// stops the inner solve gives an unsuccessful termination, not an error. On
    /// error, `result` is unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `initial` or `result` does not match the dimension and
    ///   options, `workspace` is too small, or an iterate comes within the
    ///   separation tolerance of a known root.
    /// - `NonFinite` if `initial` contains NaN or infinity, or `F` returns NaN or
    ///   infinity.
    /// - Errors of the inner Newton–Krylov solve, and errors returned by
    ///   `provider`, unchanged.
    pub fn solve_into<'scope, 'capability>(
        &self,
        initial: &[R],
        result: &mut DeflatedRootResult<R>,
        workspace: &mut DeflationWorkspace<R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::attempt(
            workspace,
            result,
            scope.into(),
            scioxide_core::AttemptKind::Call,
            |workspace, result| {
                let requirements = self.workspace_requirements(())?;
                if initial.len() != self.set.n
                    || !result.has_capacity(self.set.n, self.solver.options())
                {
                    return Err(invalid(
                        scioxide_core::argument!("initial"),
                        InvalidInputReason::Inconsistent,
                        "prepare matching initial point and result",
                    ));
                }
                finite(initial, NonFiniteOrigin::Input)?;
                if workspace.check(&requirements)? != scioxide_core::Capacity::Sufficient {
                    return Err(invalid(
                        scioxide_core::argument!("workspace"),
                        InvalidInputReason::Inconsistent,
                        "reserve matching deflation storage",
                    ));
                }
                Ok(crate::attempt::Entry::call(
                    SolveIdentity::new(
                        ProblemToken::new(0),
                        scioxide_core::algorithm_id!("rootfind", "deflation", 1),
                        self.set.n as u128,
                        0,
                    ),
                    self.solver.options().budgets(),
                ))
            },
            |workspace, result, context| self.kernel(initial, result, workspace, context),
            |_, result, outcome| crate::attempt::Conclusion {
                succeeded: result.is_success(),
                progress: outcome.as_ref().ok().copied(),
            },
        )
        .map(|_| ())
    }
    /// Runs the deflated solve and the residual check on validated input and storage.
    fn kernel(
        &self,
        initial: &[R],
        result: &mut DeflatedRootResult<R>,
        w: &mut DeflationWorkspace<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ResidualReport<R>, SciOxideError> {
        // Original and transformed residuals have independent units and references.
        let (initial_out, initial_scratch) = w.verification.split_at_mut(self.set.n);
        adapters::invoke_residual(
            self.provider,
            initial,
            initial_out,
            initial_scratch,
            &mut Some(&mut *context),
        )?;
        let reference = self
            .set
            .options
            .residual_scale
            .unwrap_or(norm(initial_out)?);
        let p = self.problem();
        self.solver.solve_into(
            &p,
            initial,
            &mut w.staged.inner,
            &mut w.newton,
            &mut *context,
        )?;
        let (out, s) = w.verification.split_at_mut(self.set.n);
        let separated = self.set.separated(w.staged.point(), out)?;
        adapters::invoke_residual(self.provider, w.staged.point(), out, s, &mut Some(context))?;
        let raw = norm(out)?;
        let report = crate::context::residual(raw, reference)?;
        w.staged.termination = if w.staged.inner.is_success()
            && (!separated
                || !crate::termination::residual_ok(
                    &self.set.options.residual.tolerance(),
                    raw,
                    reference,
                )?) {
            RootTermination::Unverified
        } else {
            w.staged.inner.termination()
        };
        context.assess_accepted(PartialResultTrust::ResidualChecked)?;
        w.staged.residual.copy_from_slice(out);
        w.staged.report = Some(report);
        w.staged.counts = context.counts();
        w.accepted.copy_from_slice(w.staged.point());
        core::mem::swap(result, &mut w.staged);
        Ok(report)
    }
    /// Evaluates `G`, or its Jacobian–vector product when `v` is given.
    fn evaluate(
        &self,
        x: &[R],
        v: Option<&[R]>,
        y: &mut [R],
        s: &mut [R],
        mut c: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        let n = self.set.n;
        let argument = if x.len() != n {
            Some(scioxide_core::argument!("x"))
        } else if y.len() != n {
            Some(scioxide_core::argument!("y"))
        } else if s.len() != self.residual_workspace_len()? {
            Some(scioxide_core::argument!("s"))
        } else if v.is_some_and(|v| v.len() != n) {
            Some(scioxide_core::argument!("v"))
        } else {
            None
        };
        if let Some(argument) = argument {
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "deflation callback shape mismatch",
            ));
        }
        finite(x, NonFiniteOrigin::Input)?;
        if let Some(v) = v {
            finite(v, NonFiniteOrigin::Input)?;
        }
        let (delta, s) = s.split_at_mut(n);
        let (f, s) = s.split_at_mut(n);
        let (out, s) = s.split_at_mut(n);
        if !self.set.separated(x, delta)? {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                InvalidInputReason::OutOfRange,
                "point lies in an explicitly excluded root neighborhood",
            ));
        }
        let mut multiplier = Scaled::from_value(R::ONE)?;
        let mut logarithmic_derivative = ScaledAccumulator::ZERO;
        for root in self.set.roots.chunks_exact(n) {
            for ((d, &a), &b) in delta.iter_mut().zip(x).zip(root) {
                *d = difference(a, b)?;
            }
            let distance = Scaled::from_value(norm(delta)?)?;
            let mut exponent = self.set.options.power;
            let mut base = Scaled::from_value(R::ONE)?.div(distance)?;
            let mut inverse_power = Scaled::from_value(R::ONE)?;
            while exponent > 0 {
                if exponent & 1 != 0 {
                    inverse_power = inverse_power.mul(base)?;
                }
                exponent >>= 1;
                if exponent > 0 {
                    base = base.mul(base)?;
                }
            }
            let factor = inverse_power.add(Scaled::from_value(self.set.options.shift)?)?;
            multiplier = multiplier.mul(factor)?;
            if let Some(v) = v {
                let coefficient = inverse_power
                    .div(factor)?
                    .mul(Scaled::from_value(-R::try_from_f64_lossy(f64::from(
                        self.set.options.power,
                    ))?)?)?
                    .div(distance)?
                    .div(distance)?;
                logarithmic_derivative.add_scaled(coefficient.mul(dot(delta, v)?.real())?)?;
            }
        }
        adapters::invoke_residual(self.provider, x, f, s, &mut c)?;
        if let Some(v) = v {
            adapters::invoke_jvp(self.provider, x, v, out, s, &mut c)?;
            let derivative = logarithmic_derivative.total()?;
            for i in 0..n {
                out[i] = multiplier
                    .mul(
                        Scaled::from_value(out[i])?
                            .add(Scaled::from_value(f[i])?.mul(derivative)?)?,
                    )?
                    .round()?;
            }
        } else {
            for (out, &f) in out.iter_mut().zip(f.iter()) {
                *out = multiplier.mul(Scaled::from_value(f)?)?.round()?;
            }
        }
        finite(out, NonFiniteOrigin::Intermediate)?;
        y.copy_from_slice(out);
        Ok(())
    }
}
impl<R: RealScalar> ResidualOperator<R> for DeflatedProblem<'_, R> {
    fn input_dimension(&self) -> usize {
        self.set.n
    }
    fn output_dimension(&self) -> usize {
        self.set.n
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_size(
            self.set.n.checked_mul(3).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("state_vectors"))
            })?,
            self.provider
                .residual_workspace_len()?
                .max(self.provider.jvp_workspace_len()?),
            scioxide_core::quantity!("residual_workspace_len"),
        )
    }
    fn residual_into(&self, x: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(x, None, y, s, None)
    }
    fn residual_into_scoped(
        &self,
        x: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, None, y, s, Some(c))
    }
}
impl<R: RealScalar> JacobianOperator<R> for DeflatedProblem<'_, R> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.residual_workspace_len()
    }
    fn jvp_into(&self, x: &[R], v: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(v), y, s, None)
    }
    fn jvp_into_scoped(
        &self,
        x: &[R],
        v: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(v), y, s, Some(c))
    }
}

impl<R: RealScalar> Default for DeflationWorkspace<R> {
    fn default() -> Self {
        Self::new()
    }
}
