//! Workspace of the dense nonlinear-system solvers and its size requirements.
use super::options::{Settings, SystemRootOptions};
use crate::DenseJacobian;
use alloc::vec::Vec;
use scioxide_core::backend::checked_mul;
use scioxide_core::{Capacity, NumericScalar, RealScalar, ResidualReport, SciOxideError};
use scioxide_differentiate::{
    DerivativeRequest, DerivativeShape, DerivativeStrategy, FiniteDifference,
    FiniteDifferenceWorkspace,
};
use scioxide_linalg::{
    IterativeRequirements, IterativeWorkspace, LinalgWorkspace, LuDecomposition, Matrix,
    QrUpdateDecomposition, Vector,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseMethod {
    Newton,
    BroydenFirst,
    BroydenSecond,
    TrustRegion,
}
fn vector_used(index: usize, method: DenseMethod, line_search: bool) -> bool {
    match method {
        DenseMethod::Newton => matches!(index, 0 | 1 | 2 | 3 | 5) || (index == 4 && line_search),
        DenseMethod::BroydenFirst => {
            matches!(index, 0 | 1 | 2 | 3 | 5) || (index == 4 && line_search)
        }
        DenseMethod::BroydenSecond => {
            matches!(index, 0 | 1 | 2 | 3 | 5 | 6 | 7 | 8 | 9) || (index == 4 && line_search)
        }
        DenseMethod::TrustRegion => matches!(index, 1 | 3 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13),
    }
}
pub(crate) fn uses_finite_difference(strategy: DerivativeStrategy) -> bool {
    !matches!(
        strategy,
        DerivativeStrategy::Analytic | DerivativeStrategy::Automatic
    )
}

/// Describes the storage a dense-system solve needs.
///
/// Obtain it from `workspace_requirements` of [`crate::SystemRootSolver`]. It
/// depends on the problem dimension, the method, the Jacobian strategy, the line
/// search and the linear solver, but not on the tolerances. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
#[derive(Debug, Clone, Copy)]
pub struct SystemWorkspaceRequirements<S: RealScalar + NumericScalar<Real = S>> {
    dimension: usize,
    scratch: usize,
    finite_difference: bool,
    line_search: bool,
    kind: DenseMethod,
    linear: Option<IterativeRequirements<S>>,
    limit: usize,
    bytes: usize,
    policy: scioxide_core::ExecutionPolicy,
}
impl<S: RealScalar + NumericScalar<Real = S>> SystemWorkspaceRequirements<S> {
    /// Returns the number of unknowns `n`.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns `true` if the Jacobian strategy needs finite-difference storage.
    pub const fn finite_difference(&self) -> bool {
        self.finite_difference
    }
    /// Returns the required size in bytes, including the storage of an iterative
    /// linear solver.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn new<F: DenseJacobian<S> + ?Sized>(
        function: &F,
        kind: DenseMethod,
        settings: &Settings<S>,
        options: &SystemRootOptions<'_, S>,
    ) -> Result<Self, SciOxideError> {
        let mut requirements = Self::for_solve(function, kind, settings, options)?;
        // Nested factors have checked owner-specific requirements; reserve once to retain them.
        requirements.bytes = requirements.minimum_bytes()?;
        if requirements.bytes > requirements.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                requirements.bytes,
                requirements.limit,
            ));
        }
        Ok(requirements)
    }
    pub(crate) fn for_solve<F: DenseJacobian<S> + ?Sized>(
        function: &F,
        kind: DenseMethod,
        settings: &Settings<S>,
        options: &SystemRootOptions<'_, S>,
    ) -> Result<Self, SciOxideError> {
        let dimension =
            crate::residual::square_dimension(function, scioxide_core::argument!("function"))?;
        let scratch = crate::residual::scratch_len(function)?;
        let linear = if let Some(solver) = options.linear_solver {
            Some(solver.workspace_requirements(
                &scioxide_linalg::IterativeWorkspaceRequest::operator(
                    &crate::residual::LinearShape::<S>::new(dimension, dimension),
                )?,
            )?)
        } else {
            None
        };
        let finite_difference = uses_finite_difference(settings.strategy);
        let requirements = Self {
            policy: settings.policy,
            dimension,
            scratch,
            finite_difference,
            line_search: options.line_search.is_some(),
            kind,
            linear,
            limit: settings.byte_limit,
            bytes: 0,
        };
        Ok(requirements)
    }
    fn covers(&self, request: &Self) -> bool {
        self.dimension == request.dimension
            && self.scratch >= request.scratch
            && (!request.finite_difference || self.finite_difference)
            && (!request.line_search || self.line_search)
            && self.kind == request.kind
            && self.policy == request.policy
            && self.linear.is_some() == request.linear.is_some()
    }
    fn minimum_bytes(&self) -> Result<usize, SciOxideError> {
        SystemWorkspace::<S>::new().projected_bytes(self)
    }
}
#[derive(Debug)]
pub(crate) struct SystemState<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) current_point: Vector<S>,
    pub(crate) current_residual: Vector<S>,
    pub(crate) trial_point: Vector<S>,
    pub(crate) trial_residual: Vector<S>,
    pub(crate) scratch: Vec<S>,
    pub(crate) accepted: Option<ResidualReport<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> SystemState<S> {
    const fn new() -> Self {
        Self {
            current_point: Vector::empty(),
            current_residual: Vector::empty(),
            trial_point: Vector::empty(),
            trial_residual: Vector::empty(),
            scratch: Vec::new(),
            accepted: None,
        }
    }
}
/// Holds the scratch storage of [`crate::NewtonSystem`], [`crate::Broyden`] and
/// [`crate::TrustRegionSystem`].
///
/// The storage is O(n²) for `n` unknowns: the Jacobian or its model and its
/// factors, plus O(n) work vectors. One workspace can serve all three methods;
/// [`Self::try_reserve`] grows it as needed. After a solve,
/// [`Self::last_attempt`] and [`Self::last_accepted`] describe what happened.
/// See [Reusing workspaces](crate#reusing-workspaces).
#[derive(Debug)]
pub struct SystemWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) state: SystemState<S>,
    pub(crate) direction: Vector<S>,
    pub(crate) step: Vector<S>,
    pub(crate) work: Vector<S>,
    pub(crate) finite_difference_residual: Vector<S>,
    pub(crate) line_search_point: Vector<S>,
    pub(crate) line_search_residual: Vector<S>,
    pub(crate) qtf: Vector<S>,
    pub(crate) product: Vector<S>,
    pub(crate) difference: Vector<S>,
    pub(crate) gradient: Vector<S>,
    pub(crate) gauss_newton: Vector<S>,
    pub(crate) model: Vector<S>,
    pub(crate) scaled: Vector<S>,
    pub(crate) diag: Vector<S>,
    pub(crate) jacobian: Matrix<S>,
    pub(crate) candidate_jacobian: Matrix<S>,
    pub(crate) finite_difference: Option<FiniteDifferenceWorkspace<S>>,
    pub(crate) lu: LuDecomposition<S>,
    pub(crate) factor_workspace: LinalgWorkspace<'static, S>,
    pub(crate) secant: QrUpdateDecomposition<S>,
    pub(crate) secant_workspace: LinalgWorkspace<'static, S>,
    pub(crate) model_factor: LuDecomposition<S>,
    pub(crate) model_workspace: LinalgWorkspace<'static, S>,
    pub(crate) linear: IterativeWorkspace<S>,
    pub(crate) good_broyden: super::broyden_model::GoodBroydenModel<S>,
    pub(crate) shell: crate::attempt::Shell<ResidualReport<S>>,
    prepared: Option<SystemWorkspaceRequirements<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for SystemWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SystemWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            state: SystemState::new(),
            direction: Vector::empty(),
            step: Vector::empty(),
            work: Vector::empty(),
            finite_difference_residual: Vector::empty(),
            line_search_point: Vector::empty(),
            line_search_residual: Vector::empty(),
            qtf: Vector::empty(),
            product: Vector::empty(),
            difference: Vector::empty(),
            gradient: Vector::empty(),
            gauss_newton: Vector::empty(),
            model: Vector::empty(),
            scaled: Vector::empty(),
            diag: Vector::empty(),
            jacobian: Matrix::empty(),
            candidate_jacobian: Matrix::empty(),
            finite_difference: None,
            lu: LuDecomposition::empty(),
            factor_workspace: LinalgWorkspace::new(),
            secant: QrUpdateDecomposition::empty(),
            secant_workspace: LinalgWorkspace::new(),
            model_factor: LuDecomposition::empty(),
            model_workspace: LinalgWorkspace::new(),
            linear: IterativeWorkspace::new(),
            good_broyden: super::broyden_model::GoodBroydenModel::new(),
            shell: crate::attempt::Shell::new(),
            prepared: None,
        }
    }
    /// Returns the dimension the workspace is sized for, or zero before
    /// [`Self::try_reserve`].
    pub fn dimension(&self) -> usize {
        self.state.current_point.len()
    }
    /// Returns whether the workspace already holds the storage that `r`
    /// describes, or needs to grow.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage after growth would exceed the byte limit
    ///   of the options.
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn check(&self, r: &SystemWorkspaceRequirements<S>) -> Result<Capacity, SciOxideError> {
        let bytes = self.projected_bytes(r)?;
        crate::workspace::limit(bytes, r.limit)?;
        let sufficient = self.prepared.as_ref().is_some_and(|p| p.covers(r));
        let linear = r
            .linear
            .as_ref()
            .map(|r| self.linear.check(r))
            .transpose()?
            .is_none_or(|c| c == Capacity::Sufficient);
        Ok(if sufficient && linear {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to hold the storage that `r` describes. A workspace
    /// that is already large enough is unchanged.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage would exceed the byte limit of the
    ///   options.
    /// - `SizeOverflow` if the byte count overflows `usize`.
    /// - `AllocationFailure` if memory cannot be reserved. Storage reserved
    ///   before the failure stays in the workspace.
    pub fn try_reserve(&mut self, r: &SystemWorkspaceRequirements<S>) -> Result<(), SciOxideError> {
        if self.check(r)? == Capacity::Sufficient {
            return Ok(());
        }
        self.shell.clear();
        self.state.accepted = None;
        self.prepared = None;
        let n = r.dimension;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        for vector in [
            &mut self.state.current_point,
            &mut self.state.current_residual,
            &mut self.state.trial_point,
            &mut self.state.trial_residual,
        ] {
            vector.try_resize(n, zero)?;
        }
        for (index, vector) in [
            &mut self.direction,
            &mut self.step,
            &mut self.work,
            &mut self.finite_difference_residual,
            &mut self.line_search_point,
            &mut self.line_search_residual,
            &mut self.qtf,
            &mut self.product,
            &mut self.difference,
            &mut self.gradient,
            &mut self.gauss_newton,
            &mut self.model,
            &mut self.scaled,
            &mut self.diag,
        ]
        .into_iter()
        .enumerate()
        {
            vector.try_resize(
                if vector_used(index, r.kind, r.line_search) {
                    n
                } else {
                    0
                },
                zero,
            )?;
        }
        let additional = r.scratch.saturating_sub(self.state.scratch.len());
        scioxide_core::backend::try_reserve(
            &mut self.state.scratch,
            additional,
            scioxide_core::quantity!("system_scratch"),
        )?;
        self.state.scratch.resize(r.scratch, zero);
        self.jacobian.try_resize_flat(n, n)?;
        self.candidate_jacobian.try_resize_flat(n, n)?;
        if r.finite_difference {
            self.finite_difference = Some(FiniteDifferenceWorkspace::workspace(
                DerivativeRequest::new(
                    &FiniteDifference::<S>::default(),
                    DerivativeShape::jacobian(n, n),
                ),
            )?);
        }
        if (r.kind == DenseMethod::Newton && r.linear.is_none())
            || (r.kind == DenseMethod::BroydenFirst)
        {
            self.lu.try_reserve(n)?;
            self.factor_workspace.try_reserve(&lu_requirements::<S>(
                n,
                r.policy,
                r.kind == DenseMethod::BroydenFirst,
            )?)?;
        }
        if r.kind == DenseMethod::BroydenFirst {
            self.good_broyden.try_reserve(n)?;
        }
        if matches!(
            r.kind,
            DenseMethod::BroydenSecond | DenseMethod::TrustRegion
        ) {
            self.jacobian.as_mut_slice().fill(zero);
            for i in 0..n {
                self.jacobian.as_mut_slice()[i * n + i] =
                    <S as scioxide_core::AlgebraicScalar>::ONE;
            }
            super::secant::factor_secant_into(
                &self.jacobian,
                &mut self.secant,
                &mut self.secant_workspace,
                r.policy,
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            super::secant::secant_update_into(
                &mut self.secant,
                &mut self.secant_workspace,
                &self.difference,
                &self.gradient,
                r.policy,
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
        }
        if matches!(
            r.kind,
            DenseMethod::BroydenFirst | DenseMethod::BroydenSecond
        ) && r.line_search
        {
            self.model_factor.try_reserve(n)?;
            self.model_workspace
                .try_reserve(&lu_requirements::<S>(n, r.policy, false)?)?;
        }
        if let Some(linear) = &r.linear {
            self.linear.try_reserve(linear)?;
        }
        let mut prepared = *r;
        prepared.bytes = self.projected_bytes(r)?;
        crate::workspace::limit(prepared.bytes, r.limit)?;
        self.prepared = Some(prepared);
        let _ = self.check(r)?;
        Ok(())
    }
    fn projected_bytes(&self, r: &SystemWorkspaceRequirements<S>) -> Result<usize, SciOxideError> {
        // An admitted compatible request needs only actual retained capacity;
        // rebuilding identical nested factor and derivative plans is unnecessary.
        let prepared = self.prepared.as_ref().is_some_and(|p| p.covers(r));
        use crate::workspace::{buffer, sum};
        let n = r.dimension;
        let q = scioxide_core::quantity!("workspace_bytes");
        let matrix_bytes = checked_mul(checked_mul(n, n, q)?, core::mem::size_of::<S>(), q)?;
        let vectors = self.projected_vector_bytes(r)?;
        let fd = if r.finite_difference && !prepared {
            FiniteDifferenceWorkspace::workspace_requirements(DerivativeRequest::new(
                &FiniteDifference::<S>::default(),
                DerivativeShape::jacobian(n, n),
            ))?
            .bytes()
        } else {
            0
        };
        let fd = self
            .finite_difference
            .as_ref()
            .map_or(Ok(0), |w| w.retained_bytes())?
            .max(fd);
        let newton = !prepared
            && ((r.kind == DenseMethod::Newton && r.linear.is_none())
                || r.kind == DenseMethod::BroydenFirst);
        let qr = !prepared
            && matches!(
                r.kind,
                DenseMethod::BroydenSecond | DenseMethod::TrustRegion
            );
        let model = !prepared
            && matches!(
                r.kind,
                DenseMethod::BroydenFirst | DenseMethod::BroydenSecond
            )
            && r.line_search;
        let factor_workspace = if newton {
            self.factor_workspace
                .reservation_bytes(&lu_requirements::<S>(
                    n,
                    r.policy,
                    r.kind == DenseMethod::BroydenFirst,
                )?)?
        } else {
            self.factor_workspace.retained_bytes()?
        };
        let secant_workspace = if qr {
            self.secant_workspace.reservation_bytes(
                &QrUpdateDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::QrUpdateWorkspaceRequest::new(
                        n,
                        n,
                        &scioxide_linalg::QrUpdateOptions::new(scioxide_linalg::RankPolicy::Exact),
                    ),
                )?,
            )?
        } else {
            self.secant_workspace.retained_bytes()?
        };
        let model_workspace = if model {
            self.model_workspace
                .reservation_bytes(&lu_requirements::<S>(n, r.policy, false)?)?
        } else {
            self.model_workspace.retained_bytes()?
        };
        sum([
            Ok(vectors),
            Ok(fd),
            self.jacobian.retained_bytes().map(|b| b.max(matrix_bytes)),
            self.candidate_jacobian
                .retained_bytes()
                .map(|b| b.max(matrix_bytes)),
            buffer(&self.state.scratch)
                .and_then(|b| Ok(b.max(checked_mul(r.scratch, core::mem::size_of::<S>(), q)?))),
            self.lu.retained_bytes().and_then(|b| {
                Ok(b.max(if newton {
                    LuDecomposition::<S>::factor_storage_bytes(n)?
                } else {
                    0
                }))
            }),
            self.secant.retained_bytes().and_then(|b| {
                Ok(b.max(if qr {
                    QrUpdateDecomposition::<S>::factor_storage_bytes(n, n)?
                } else {
                    0
                }))
            }),
            self.model_factor.retained_bytes().and_then(|b| {
                Ok(b.max(if model {
                    LuDecomposition::<S>::factor_storage_bytes(n)?
                } else {
                    0
                }))
            }),
            Ok(factor_workspace),
            Ok(secant_workspace),
            Ok(model_workspace),
            self.good_broyden.retained_bytes().and_then(|bytes| {
                Ok(
                    bytes.max(if r.kind == DenseMethod::BroydenFirst && !prepared {
                        super::broyden_model::GoodBroydenModel::<S>::storage_bytes(n)?
                    } else {
                        0
                    }),
                )
            }),
            self.linear
                .retained_bytes()
                .map(|b| b.max(r.linear.as_ref().map_or(0, |r| r.required_bytes()))),
        ])
    }
    /// Bytes held by the state and work vectors after growing them to the
    /// vectors `r` uses.
    fn projected_vector_bytes(
        &self,
        r: &SystemWorkspaceRequirements<S>,
    ) -> Result<usize, SciOxideError> {
        use crate::workspace::sum;
        let vector_bytes = checked_mul(
            r.dimension,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let base = sum([
            self.state.current_point.retained_bytes(),
            self.state.current_residual.retained_bytes(),
            self.state.trial_point.retained_bytes(),
            self.state.trial_residual.retained_bytes(),
        ]
        .into_iter()
        .map(|b| b.map(|b| b.max(vector_bytes))))?;
        let vectors = [
            &self.direction,
            &self.step,
            &self.work,
            &self.finite_difference_residual,
            &self.line_search_point,
            &self.line_search_residual,
            &self.qtf,
            &self.product,
            &self.difference,
            &self.gradient,
            &self.gauss_newton,
            &self.model,
            &self.scaled,
            &self.diag,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.retained_bytes().map(|b| {
                b.max(if vector_used(i, r.kind, r.line_search) {
                    vector_bytes
                } else {
                    0
                })
            })
        });
        sum([Ok(base), sum(vectors)])
    }
    /// Returns the record of the most recent call, including a call rejected by
    /// its input checks.
    pub const fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<ResidualReport<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the last accepted point of the most recent call, or `None` if it
    /// accepted no point.
    pub fn last_accepted(&self) -> Option<&Vector<S>> {
        self.shell.accepted(&self.state.current_point)
    }
}
crate::attempt::attempted!(
    SystemWorkspace<S>,
    [S: RealScalar + NumericScalar<Real = S>],
    ResidualReport<S>
);

fn lu_requirements<S: RealScalar + NumericScalar<Real = S>>(
    n: usize,
    policy: scioxide_core::ExecutionPolicy,
    inverse: bool,
) -> Result<scioxide_linalg::LinalgRequirements, SciOxideError> {
    let options = scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(policy);
    let factor = LuDecomposition::<S>::workspace_requirements(
        &scioxide_linalg::LuWorkspaceRequest::factor(n, &options),
    )?;
    if inverse {
        Ok(factor.max(&LuDecomposition::<S>::workspace_requirements(
            &scioxide_linalg::LuWorkspaceRequest::solve_matrix(n, n, &options),
        )?))
    } else {
        Ok(factor)
    }
}
