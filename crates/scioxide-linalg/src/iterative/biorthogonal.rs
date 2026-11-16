//! BiCG, CGS, QMR and TFQMR: short recurrences based on the two-sided Lanczos process.
//!

use super::safety::{
    Step, StepFault, certified_shadow_product_is_small, checked_divide, shadow_product_is_small,
};
use super::support::validate_square;
use super::workspace::{Family, IterativeProgress, Job, Shape, StorageKind};
use super::{TrueResidual, evaluation_limit_error, solve_summary};
use crate::Preconditioning;
use crate::kernel::finite;
use crate::{
    IterativeRequirements, IterativeSolveReport, IterativeSolveResult, IterativeWorkspace,
    IterativeWorkspaceRequest, KrylovDiagnostics, LinalgExecutionContext, LinearSolveBreakdown,
    LinearSolveTermination, Vector, VectorView, VectorViewMut,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    AdjointOperator, Budget, ExecutionContext, LinearOperator, NumericScalar, RealScalar,
    ResidualTolerance, SciOxideError,
};

type Result<T> = core::result::Result<T, SciOxideError>;

fn denominator<S: NumericScalar>(
    value: S,
    a: &Vector<S>,
    b: &Vector<S>,
    c: &scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Step<S> {
    if shadow_product_is_small(
        scioxide_core::ScaledValue::from_value(value)?,
        finite::checked_norm(a.as_slice(), c.execution())?,
        finite::checked_norm(b.as_slice(), c.execution())?,
    )? {
        Err(StepFault::Breakdown)
    } else {
        Ok(value)
    }
}
/// CGS shadow product `shadowᴴ value` behind the epsilon-squared guard. The
/// Dot2 candidate decides the guard when its error bound excludes the
/// threshold; otherwise, and outside the ordinary frame, exact accumulation
/// decides and supplies the product.
fn compensated_denominator<S: NumericScalar>(
    value: &Vector<S>,
    shadow: &Vector<S>,
    kernels: crate::KernelContext<'_>,
) -> Step<S> {
    let products = crate::kernel::recurrence::compensated_products(
        value.as_slice(),
        shadow.as_slice(),
        kernels.dispatch(),
    )?;
    let value_norm = products.left.norm(value.as_slice())?;
    let shadow_norm = products.right.norm(shadow.as_slice())?;
    let certified = match products.compensated_error(value_norm, shadow_norm, value.len())? {
        Some(error) => certified_shadow_product_is_small(
            scioxide_core::ScaledValue::from_value(products.value)?,
            error,
            shadow_norm,
            value_norm,
        )?
        .map(|small| (small, products.value)),
        None => None,
    };
    let (small, product) = match certified {
        Some(decision) => decision,
        None => {
            let exact = crate::reduction::canonical_dot(
                crate::FiniteVectorView::from_workspace(shadow.view()),
                crate::FiniteVectorView::from_workspace(value.view()),
                true,
            )?;
            let small = shadow_product_is_small(
                scioxide_core::ScaledValue::from_value(exact)?,
                shadow_norm,
                value_norm,
            )?;
            (small, exact)
        }
    };
    if small {
        Err(StepFault::Breakdown)
    } else {
        Ok(product)
    }
}
fn divide<S: NumericScalar>(a: S, b: S) -> Step<S> {
    checked_divide(a, b)
}

pub(super) enum Operator<'a, S: NumericScalar> {
    Forward(&'a dyn LinearOperator<S>),
    Adjoint(&'a dyn AdjointOperator<S>),
}
impl<S: NumericScalar> Copy for Operator<'_, S> {}
impl<S: NumericScalar> Clone for Operator<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, S: NumericScalar> Operator<'a, S> {
    pub(super) fn linear(self) -> &'a dyn LinearOperator<S> {
        match self {
            Self::Forward(a) => a,
            Self::Adjoint(a) => a,
        }
    }
    /// Workspace request for the forward (and adjoint) applications.
    pub(super) fn request(self) -> Result<IterativeWorkspaceRequest> {
        match self {
            Self::Forward(a) => IterativeWorkspaceRequest::operator(a),
            Self::Adjoint(a) => IterativeWorkspaceRequest::adjoint(a),
        }
    }
}
#[derive(Clone)]
pub(super) struct Config<S: NumericScalar> {
    pub(super) tolerance: ResidualTolerance<S::Real>,
    pub(super) max_iterations: Budget,
    pub(super) execution: crate::LinalgExecutionOptions,
    pub(super) history: bool,
    pub(super) retries: usize,
    pub(super) relaxation: S::Real,
}
impl<S: NumericScalar> Config<S> {
    pub(super) fn new(tolerance: ResidualTolerance<S::Real>, limit: Budget) -> Self {
        Self {
            relaxation: S::Real::ONE,
            tolerance,
            max_iterations: limit,
            execution: crate::LinalgExecutionOptions::new(),
            history: false,
            retries: 0,
        }
    }
    /// Iteration allowance: the solver's limit, capped by an explicit execution budget.
    pub(super) fn limit(&self) -> Budget {
        self.execution.cap_iterations(self.max_iterations)
    }
    pub(super) fn requirements(
        &self,
        kind: StorageKind,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>> {
        IterativeRequirements::new(
            Shape {
                kind,
                rows: request.rows(),
                cols: request.cols(),
                operator_scratch: request.operator_scratch(),
                preconditioner_scratch: request.preconditioner_scratch(),
                columns: 1,
                cycle: 0,
                retained: 0,
            },
            self.history.then_some(self.limit()),
            Some(request),
            self.execution.byte_limit(),
        )
    }
}

/// Solve a square system with one biorthogonal or Richardson kernel through
/// the shared workspace driver.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn solve_into<S: NumericScalar, K: Kernel<S>>(
    config: &Config<S>,
    algorithm: scioxide_core::AlgorithmId,
    a: Operator<'_, S>,
    b: VectorView<'_, S>,
    preconditioning: Preconditioning<'_, S>,
    mut x: VectorViewMut<'_, S>,
    workspace: &mut IterativeWorkspace<S>,
    context: LinalgExecutionContext<'_, '_>,
) -> Result<IterativeSolveReport<S::Real>>
where
    Storage<S, K>: Family<S>,
{
    workspace.reject();
    validate_square(a.linear())?;
    if matches!(K::KIND, StorageKind::Bicg | StorageKind::Qmr) {
        preconditioning.adjoint(a.linear().cols())?;
    } else {
        preconditioning.validate_dimension(a.linear().cols())?;
    }
    let requirements =
        config.requirements(K::KIND, &a.request()?.with_preconditioner(preconditioning)?)?;
    workspace.run(
        context.with_options(&config.execution)?,
        &Job {
            algorithm,
            limit: config.limit(),
            requirements,
        },
        b,
        &mut x,
        |storage: &mut Storage<S, K>, b, x, kernels, solve| {
            storage.solve(config, a, b, preconditioning, x, kernels, solve)
        },
        IterativeProgress::of,
    )
}

/// Only common provider and accepted-state storage is shared. K owns exactly
/// the vectors required by its recurrence, never scratch for another method.
#[derive(Debug)]
pub(super) struct Storage<S: NumericScalar, K> {
    kernel: K,
    candidate: Vector<S>,
    accepted: Vector<S>,
    best: super::support::BestIterate<S>,
    residual: Vector<S>,
    product: Vector<S>,
    temporary: Vector<S>,
    operator_scratch: Vec<S>,
    inverse_scratch: Vec<S>,
    history: Option<Vec<S::Real>>,
}
pub(super) trait Kernel<S: NumericScalar>: Sized {
    const TRUE_RESIDUAL_EACH_STEP: bool = false;
    /// The kernel already computed b - A*x as part of its ordinary work.
    const ORIGINAL_RESIDUAL: bool = false;
    fn residual_norm(&self, c: &scioxide_core::OperatorContext<'_, '_, '_, S>) -> Result<S::Real>;
    /// Workspace storage kind of this recurrence.
    const KIND: StorageKind;
    /// Working vectors allocated by [`Self::new`].
    const VECTORS: usize;
    /// Whether reports carry restart and direction-loss diagnostics.
    const KRYLOV_DIAGNOSTICS: bool = true;
    fn new(n: usize) -> Result<Self>;
    fn start(&mut self, r: &Vector<S>) -> Result<()>;
    fn step(
        &mut self,
        a: &mut Actions<'_, S>,
        x: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool>;
}
pub(super) struct Actions<'a, S: NumericScalar> {
    kernels: crate::KernelContext<'a>,
    changed: bool,
    pub(super) relaxation: S::Real,
    operator: Operator<'a, S>,
    rhs: &'a Vector<S>,
    inverse: Preconditioning<'a, S>,
    temporary: &'a mut Vector<S>,
    operator_scratch: &'a mut [S],
    inverse_scratch: &'a mut [S],
}
impl<S: NumericScalar> Actions<'_, S> {
    fn apply(
        &mut self,
        v: &Vector<S>,
        out: &mut Vector<S>,
        adjoint: bool,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<bool> {
        if adjoint {
            let Operator::Adjoint(a) = self.operator else {
                return Err(crate::checked::unsupported(
                    scioxide_core::argument!("a"),
                    "adjoint capability required",
                ));
            };
            let n = a.apply_adjoint_workspace_len()?;
            if super::safety::apply_adjoint_scoped(
                a,
                v.as_slice(),
                finite::maximum(v.as_slice(), self.kernels.dispatch())?,
                self.temporary.as_mut_slice(),
                &mut self.operator_scratch[..n],
                c,
            )?
            .is_exhausted()
            {
                return Ok(false);
            }
            super::support::precondition(
                self.inverse,
                true,
                self.temporary,
                out,
                self.inverse_scratch,
                self.kernels,
                c,
            )
        } else {
            if !super::support::precondition(
                self.inverse,
                false,
                v,
                self.temporary,
                self.inverse_scratch,
                self.kernels,
                c,
            )? {
                return Ok(false);
            }
            let a = self.operator.linear();
            let n = a.apply_workspace_len()?;
            Ok(!super::safety::apply_operator_scoped(
                a,
                self.temporary.as_slice(),
                finite::maximum(self.temporary.as_slice(), self.kernels.dispatch())?,
                out.as_mut_slice(),
                &mut self.operator_scratch[..n],
                c,
            )?
            .is_exhausted())
        }
    }
    pub(super) fn update(
        &mut self,
        x: &mut Vector<S>,
        direction: &Vector<S>,
        alpha: S,
        scratch: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<bool> {
        if !super::support::precondition(
            self.inverse,
            false,
            direction,
            scratch,
            self.inverse_scratch,
            self.kernels,
            c,
        )? {
            return Ok(false);
        }
        self.changed |= finite::update(
            x.as_mut_slice(),
            alpha,
            scratch.as_slice(),
            S::ONE,
            self.kernels.dispatch(),
        )?
        .changed;
        Ok(true)
    }
}
impl<S: NumericScalar, K: Kernel<S>> Storage<S, K> {
    /// Exact payload of [`Self::new`]: the kernel's vectors, six shared
    /// vectors, provider scratch and history.
    pub(super) fn bytes(shape: &Shape, history: usize) -> Result<usize> {
        use super::workspace::{times, total};
        super::workspace::payload::<S>(
            total([
                times(K::VECTORS + 6, shape.cols)?,
                shape.operator_scratch,
                shape.preconditioner_scratch,
            ])?,
            history,
        )
    }

    pub(super) fn new(shape: &Shape, history: usize) -> Result<Self> {
        let n = shape.cols;
        Ok(Self {
            kernel: K::new(n)?,
            candidate: Vector::zeros(n)?,
            accepted: Vector::zeros(n)?,
            best: super::support::BestIterate::new(n)?,
            residual: Vector::zeros(n)?,
            product: Vector::zeros(n)?,
            temporary: Vector::zeros(n)?,
            operator_scratch: try_filled(
                shape.operator_scratch,
                S::ZERO,
                scioxide_core::quantity!("operator_scratch"),
            )?,
            inverse_scratch: try_filled(
                shape.preconditioner_scratch,
                S::ZERO,
                scioxide_core::quantity!("inverse_scratch"),
            )?,
            history: (history > 0)
                .then(|| try_with_capacity(history, scioxide_core::quantity!("workspace.history")))
                .transpose()?,
        })
    }

    pub(super) fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub(super) fn solve(
        &mut self,
        config: &Config<S>,
        a: Operator<'_, S>,
        b: &Vector<S>,
        preconditioning: Preconditioning<'_, S>,
        x: &mut Vector<S>,
        kernels: crate::KernelContext<'_>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<IterativeSolveReport<S::Real>> {
        let reference = finite::checked_norm(b.as_slice(), c.execution())?;
        let scale = super::residual_reference(reference);

        finite::copy(self.candidate.as_mut_slice(), x.as_slice())?;
        if let Some(h) = &mut self.history {
            h.clear();
        }
        let Some(mut verified) = super::support::true_residual_into_scoped(
            super::support::KrylovProblem {
                operator: a.linear(),
                rhs: b,
            },
            &self.candidate,
            &mut self.residual,
            &mut self.product,
            &mut self.operator_scratch[..a.linear().apply_workspace_len()?],
            kernels,
            c,
        )?
        else {
            return Err(evaluation_limit_error(c));
        };
        finite::copy(self.accepted.as_mut_slice(), x.as_slice())?;
        c.assess_accepted(super::checkpoint_trust(
            config.tolerance.is_satisfied(verified, scale)?,
        ))?;
        if config.history
            && let Some(h) = &mut self.history
        {
            h.push(verified);
        }
        self.best.reset(self.candidate.as_slice(), verified);
        let floor = super::safety::ResidualFloor::new(reference);
        self.kernel.start(&self.residual)?;

        let mut restarts = 0usize;
        let mut retries = 0usize;
        let mut last_retry = None;
        let mut lost = false;
        let mut accepted_steps = 0usize;
        let outcome = (|| -> Result<LinearSolveTermination> {
            loop {
                if config.tolerance.is_satisfied(verified, scale)? {
                    return Ok(LinearSolveTermination::Converged);
                }
                if let Some(reason) = c.stop_reason()? {
                    return Ok(LinearSolveTermination::ExecutionStopped(reason));
                }
                if c.counts().own().iterations() >= super::support::iteration_limit(c)
                    || !c.record_iteration()?.is_granted()
                {
                    return Ok(LinearSolveTermination::IterationLimit);
                }
                let mut actions = Actions {
                    changed: false,
                    kernels,
                    relaxation: config.relaxation,
                    operator: a,
                    rhs: b,
                    inverse: preconditioning,
                    temporary: &mut self.temporary,
                    operator_scratch: &mut self.operator_scratch,
                    inverse_scratch: &mut self.inverse_scratch,
                };
                // Bi-orthogonal recurrences (BiCG, CGS, QMR, TFQMR) can break
                // down when a shadow inner product vanishes. Recovery restarts
                // the kernel from the last verified iterate and its true
                // residual; if a retry would happen again without any accepted
                // step since the previous one, or the retry budget is spent,
                // the breakdown is reported instead of looping.
                let step = self.kernel.step(&mut actions, &mut self.candidate, c);
                let update_unchanged = !actions.changed;
                match step {
                    Err(StepFault::Breakdown) => {
                        lost = true;
                        let progress = accepted_steps;
                        if retries >= config.retries || last_retry == Some(progress) {
                            return Ok(LinearSolveTermination::Breakdown(
                                LinearSolveBreakdown::RecurrenceDenominator,
                            ));
                        }
                        retries += 1;
                        last_retry = Some(progress);
                        finite::copy(self.candidate.as_mut_slice(), self.accepted.as_slice())?;
                        self.kernel.start(&self.residual)?;
                        restarts += 1;
                        continue;
                    }
                    Err(StepFault::Error(error)) => return Err(error),
                    Ok(false) => return Ok(LinearSolveTermination::EvaluationLimit),
                    Ok(true) => {}
                }
                let estimate = self.kernel.residual_norm(c)?;
                let best_before = self.best.residual;
                if K::ORIGINAL_RESIDUAL {
                    self.best.observe(
                        self.candidate.as_slice(),
                        estimate,
                        c.counts().own().iterations(),
                    );
                }
                let apparent = config.tolerance.is_satisfied(estimate, scale)?;
                let periodic =
                    super::safety::should_recompute_true_residual(c.counts().own().iterations());
                let at_limit = c.counts().own().iterations() >= super::support::iteration_limit(c);
                let checkpoint = update_unchanged
                    || K::TRUE_RESIDUAL_EACH_STEP
                    || periodic
                    || apparent
                    || at_limit;
                if !checkpoint {
                    if let Some(history) = &mut self.history {
                        history.push(estimate);
                    }
                    continue;
                }
                let Some(next) = super::support::true_residual_into_scoped(
                    super::support::KrylovProblem {
                        operator: a.linear(),
                        rhs: b,
                    },
                    &self.candidate,
                    &mut self.residual,
                    &mut self.product,
                    &mut self.operator_scratch[..a.linear().apply_workspace_len()?],
                    kernels,
                    c,
                )?
                else {
                    return Ok(LinearSolveTermination::EvaluationLimit);
                };
                let unchanged =
                    update_unchanged || self.candidate.as_slice() == self.accepted.as_slice();
                self.best.observe(
                    self.candidate.as_slice(),
                    next,
                    c.counts().own().iterations(),
                );
                finite::copy(self.accepted.as_mut_slice(), self.candidate.as_slice())?;
                verified = next;
                accepted_steps += 1;
                let satisfied = config.tolerance.is_satisfied(verified, scale)?;
                c.commit_accepted(super::checkpoint_trust(satisfied))?;
                if config.history
                    && let Some(h) = &mut self.history
                {
                    h.push(verified);
                }
                if satisfied {
                    return Ok(LinearSolveTermination::Converged);
                }
                if unchanged {
                    return Ok(LinearSolveTermination::Stagnation);
                }
                // Methods that verify every step test the floor only at the
                // periodic checkpoints, which bounds the probe cost.
                if !at_limit
                    && (periodic || !K::TRUE_RESIDUAL_EACH_STEP)
                    && super::safety::floor_test_due(verified, best_before)
                {
                    match floor.reached(
                        a.linear(),
                        self.best.solution.as_slice(),
                        self.best.residual,
                        super::safety::FloorProbe {
                            input: self.temporary.as_mut_slice(),
                            image: self.product.as_mut_slice(),
                            workspace: &mut self.operator_scratch
                                [..a.linear().apply_workspace_len()?],
                        },
                        kernels,
                        c,
                    )? {
                        None => return Ok(LinearSolveTermination::EvaluationLimit),
                        Some(true) => return Ok(LinearSolveTermination::Stagnation),
                        Some(false) => {}
                    }
                }
                if K::TRUE_RESIDUAL_EACH_STEP {
                    self.kernel.start(&self.residual)?;
                }
            }
        })();
        let termination = match outcome {
            Ok(t) => t,
            Err(error) => match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. } => {
                    super::support::execution_stop(error)?
                }
                _ => return Err(error),
            },
        };
        if !termination.is_success() {
            self.accepted
                .as_mut_slice()
                .copy_from_slice(self.best.solution.as_slice());
            verified = self.best.residual;
        }
        if let Some(last) = self.history.as_mut().and_then(|h| h.last_mut()) {
            *last = verified;
        }
        let report = IterativeSolveReport::from_summary(solve_summary(
            c,
            termination,
            TrueResidual::against(verified, scale),
        ))?;
        let report = if K::KRYLOV_DIAGNOSTICS {
            report.with_krylov(KrylovDiagnostics {
                normal_residual: None,
                condition_estimate: None,
                rank_loss_detected: lost,
                restarts,
            })
        } else {
            report
        };
        finite::copy(x.as_mut_slice(), self.accepted.as_slice())?;
        Ok(report)
    }
}

macro_rules! applied {
    ($e:expr) => {
        if !$e? {
            return Ok(false);
        }
    };
}
fn vectors<S: NumericScalar, const N: usize>(n: usize) -> Result<[Vector<S>; N]> {
    let mut values = try_with_capacity(N, scioxide_core::quantity!("kernel_vectors"))?;
    for _ in 0..N {
        values.push(Vector::zeros(n)?);
    }
    values.try_into().map_err(|_| {
        crate::checked::inconsistent(scioxide_core::argument!("self"), "internal vector count")
    })
}

#[derive(Debug)]
pub(super) struct Bicg<S: NumericScalar> {
    v: [Vector<S>; 7],
    rho: S,
    steps: usize,
}
impl<S: NumericScalar> Kernel<S> for Bicg<S> {
    fn residual_norm(&self, c: &scioxide_core::OperatorContext<'_, '_, '_, S>) -> Result<S::Real> {
        finite::checked_norm(self.v[0].as_slice(), c.execution())
    }
    const KIND: StorageKind = StorageKind::Bicg;
    const VECTORS: usize = 7;
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            v: vectors(n)?,
            rho: S::ONE,
            steps: 0,
        })
    }
    fn start(&mut self, r: &Vector<S>) -> Result<()> {
        finite::copy(self.v[0].as_mut_slice(), r.as_slice())?;
        finite::copy(self.v[1].as_mut_slice(), r.as_slice())?;
        self.steps = 0;
        self.rho = S::ONE;
        Ok(())
    }
    fn step(
        &mut self,
        a: &mut Actions<'_, S>,
        x: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        let [r, shadow, p, q, ap, aq, temp] = &mut self.v;
        let rho = denominator(
            finite::dot(r.as_slice(), shadow.as_slice(), a.kernels.dispatch())?,
            shadow,
            r,
            c,
        )?;
        if self.steps == 0 {
            finite::copy(p.as_mut_slice(), r.as_slice())?;
            finite::copy(q.as_mut_slice(), shadow.as_slice())?;
        } else {
            let beta = divide(rho, self.rho)?;
            finite::update(
                p.as_mut_slice(),
                S::ONE,
                r.as_slice(),
                beta,
                a.kernels.dispatch(),
            )?;
            finite::update(
                q.as_mut_slice(),
                S::ONE,
                shadow.as_slice(),
                beta.conj(),
                a.kernels.dispatch(),
            )?;
        }
        applied!(a.apply(p, ap, false, c));
        let divisor = denominator(
            finite::dot(ap.as_slice(), q.as_slice(), a.kernels.dispatch())?,
            q,
            ap,
            c,
        )?;
        let alpha = divide(rho, divisor)?;
        applied!(a.apply(q, aq, true, c));
        applied!(a.update(x, p, alpha, temp, c));
        finite::update(
            r.as_mut_slice(),
            -alpha,
            ap.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        finite::update(
            shadow.as_mut_slice(),
            -alpha.conj(),
            aq.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        self.rho = rho;
        self.steps += 1;
        Ok(true)
    }
}
#[derive(Debug)]
pub(super) struct Cgs<S: NumericScalar> {
    v: [Vector<S>; 8],
    rho: S,
    steps: usize,
}
impl<S: NumericScalar> Kernel<S> for Cgs<S> {
    const ORIGINAL_RESIDUAL: bool = true;
    fn residual_norm(&self, c: &scioxide_core::OperatorContext<'_, '_, '_, S>) -> Result<S::Real> {
        finite::checked_norm(self.v[0].as_slice(), c.execution())
    }
    const KIND: StorageKind = StorageKind::Cgs;
    const VECTORS: usize = 8;
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            v: vectors(n)?,
            rho: S::ONE,
            steps: 0,
        })
    }
    fn start(&mut self, r: &Vector<S>) -> Result<()> {
        finite::copy(self.v[0].as_mut_slice(), r.as_slice())?;
        finite::copy(self.v[1].as_mut_slice(), r.as_slice())?;
        self.steps = 0;
        self.rho = S::ONE;
        Ok(())
    }
    fn step(
        &mut self,
        a: &mut Actions<'_, S>,
        x: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        let [r, shadow, u, p, q, v, h, temp] = &mut self.v;
        // The squared recurrence amplifies cancellation in its shadow products.
        // Twice-working-precision Dot2 resolves them; exact accumulation decides
        // only where the Dot2 bound cannot certify the breakdown guard.
        let rho = compensated_denominator(r, shadow, a.kernels)?;
        if self.steps == 0 {
            finite::copy(u.as_mut_slice(), r.as_slice())?;
            finite::copy(p.as_mut_slice(), r.as_slice())?;
        } else {
            let beta = divide(rho, self.rho)?;
            finite::copy(u.as_mut_slice(), r.as_slice())?;
            finite::update(
                u.as_mut_slice(),
                beta,
                q.as_slice(),
                S::ONE,
                a.kernels.dispatch(),
            )?;
            finite::update(
                p.as_mut_slice(),
                S::ONE,
                q.as_slice(),
                beta,
                a.kernels.dispatch(),
            )?;
            finite::update(
                p.as_mut_slice(),
                S::ONE,
                u.as_slice(),
                beta,
                a.kernels.dispatch(),
            )?;
        }
        applied!(a.apply(p, v, false, c));
        let divisor = compensated_denominator(v, shadow, a.kernels)?;
        let alpha = divide(rho, divisor)?;
        finite::copy(q.as_mut_slice(), u.as_slice())?;
        finite::update(
            q.as_mut_slice(),
            -alpha,
            v.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        finite::copy(h.as_mut_slice(), u.as_slice())?;
        finite::update(
            h.as_mut_slice(),
            S::ONE,
            q.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        applied!(a.update(x, h, alpha, temp, c));
        // Squaring the BiCG residual polynomial amplifies recurrence drift.
        // Spend the second product on b - A x instead of
        // recursively subtracting alpha * A M^-1 (u + q). Keep the directions
        // and fixed shadow; this is residual replacement, not a restart.
        let operator = a.operator.linear();
        let scratch = operator.apply_workspace_len()?;
        if super::support::true_residual_into_scoped(
            super::support::KrylovProblem {
                operator,
                rhs: a.rhs,
            },
            x,
            r,
            v,
            &mut a.operator_scratch[..scratch],
            a.kernels,
            c,
        )?
        .is_none()
        {
            return Ok(false);
        }
        self.rho = rho;
        self.steps += 1;
        Ok(true)
    }
}
#[derive(Debug)]
pub(super) struct Qmr<S: NumericScalar> {
    v: [Vector<S>; 12],
    epsilon: S,
    eta: S,
    gamma: S::Real,
    theta: S::Real,
    steps: usize,
}
impl<S: NumericScalar> Kernel<S> for Qmr<S> {
    fn residual_norm(&self, c: &scioxide_core::OperatorContext<'_, '_, '_, S>) -> Result<S::Real> {
        finite::checked_norm(self.v[10].as_slice(), c.execution())
    }
    const KIND: StorageKind = StorageKind::Qmr;
    const VECTORS: usize = 12;
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            v: vectors(n)?,
            epsilon: S::ONE,
            eta: -S::ONE,
            gamma: S::Real::ONE,
            theta: S::Real::ZERO,
            steps: 0,
        })
    }
    fn start(&mut self, r: &Vector<S>) -> Result<()> {
        finite::copy(self.v[0].as_mut_slice(), r.as_slice())?;
        finite::copy(self.v[1].as_mut_slice(), r.as_slice())?;
        finite::copy(self.v[10].as_mut_slice(), r.as_slice())?;
        self.v[11].view_mut().fill(S::ZERO);
        self.steps = 0;
        self.epsilon = S::ONE;
        self.eta = -S::ONE;
        self.gamma = S::Real::ONE;
        self.theta = S::Real::ZERO;
        Ok(())
    }
    fn step(
        &mut self,
        a: &mut Actions<'_, S>,
        x: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        let [vt, wt, v, w, p, q, ap, aq, d, temp, residual, image_d] = &mut self.v;
        let rho = finite::checked_norm(vt.as_slice(), c.execution())?;
        let xi = finite::checked_norm(wt.as_slice(), c.execution())?;
        finite::copy(v.as_mut_slice(), vt.as_slice())?;
        super::safety::normalized(finite::normalize(
            v.as_mut_slice(),
            rho,
            a.kernels.dispatch(),
        ))?;
        finite::copy(w.as_mut_slice(), wt.as_slice())?;
        super::safety::normalized(finite::normalize(
            w.as_mut_slice(),
            xi,
            a.kernels.dispatch(),
        ))?;
        let delta = denominator(
            finite::dot(v.as_slice(), w.as_slice(), a.kernels.dispatch())?,
            w,
            v,
            c,
        )?;
        if self.steps == 0 {
            finite::copy(p.as_mut_slice(), v.as_slice())?;
            finite::copy(q.as_mut_slice(), w.as_slice())?;
        } else {
            let ratio = divide(delta, self.epsilon)?;
            finite::update(
                p.as_mut_slice(),
                S::ONE,
                v.as_slice(),
                -S::from_real(xi) * ratio,
                a.kernels.dispatch(),
            )?;
            finite::update(
                q.as_mut_slice(),
                S::ONE,
                w.as_slice(),
                -S::from_real(rho) * ratio.conj(),
                a.kernels.dispatch(),
            )?;
        }
        applied!(a.apply(p, ap, false, c));
        let epsilon = denominator(
            finite::dot(ap.as_slice(), q.as_slice(), a.kernels.dispatch())?,
            q,
            ap,
            c,
        )?;
        let beta = divide(epsilon, delta)?;
        finite::copy(vt.as_mut_slice(), ap.as_slice())?;
        finite::update(
            vt.as_mut_slice(),
            -beta,
            v.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        applied!(a.apply(q, aq, true, c));
        finite::copy(wt.as_mut_slice(), aq.as_slice())?;
        finite::update(
            wt.as_mut_slice(),
            -beta.conj(),
            w.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        let next_rho = finite::checked_norm(vt.as_slice(), c.execution())?;
        let theta = divide(
            S::from_real(next_rho),
            S::from_real(self.gamma) * S::from_real(beta.abs()),
        )?
        .real_component();
        let gamma = S::Real::ONE / S::Real::ONE.hypot(theta);
        let ratio = gamma / self.gamma;
        let eta = -self.eta * divide(S::from_real(rho), beta)? * S::from_real(ratio * ratio);
        if self.steps == 0 {
            d.view_mut().fill(S::ZERO);
        } else {
            let f = self.theta * gamma;
            finite::scale(d.as_mut_slice(), S::from_real(f * f))?;
        }
        finite::update(
            d.as_mut_slice(),
            eta,
            p.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        let factor = if self.steps == 0 {
            S::Real::ZERO
        } else {
            let f = self.theta * gamma;
            f * f
        };
        finite::update(
            image_d.as_mut_slice(),
            eta,
            ap.as_slice(),
            S::from_real(factor),
            a.kernels.dispatch(),
        )?;
        finite::update(
            residual.as_mut_slice(),
            -S::ONE,
            image_d.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        applied!(a.update(x, d, S::ONE, temp, c));
        self.epsilon = epsilon;
        self.eta = eta;
        self.gamma = gamma;
        self.theta = theta;
        self.steps += 1;
        Ok(true)
    }
}
#[derive(Debug)]
pub(super) struct Tfqmr<S: NumericScalar> {
    v: [Vector<S>; 8],
    rho: S,
    alpha: S,
    eta: S,
    theta: S::Real,
    tau: S::Real,
    steps: usize,
}
impl<S: NumericScalar> Kernel<S> for Tfqmr<S> {
    fn residual_norm(&self, _c: &scioxide_core::OperatorContext<'_, '_, '_, S>) -> Result<S::Real> {
        crate::checked::finite_candidate(
            self.tau * crate::checked::rounded_count::<S::Real>(self.steps + 1)?.sqrt(),
        )
    }
    const KIND: StorageKind = StorageKind::Tfqmr;
    const VECTORS: usize = 8;
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            v: vectors(n)?,
            rho: S::ONE,
            alpha: S::ONE,
            eta: S::ZERO,
            theta: S::Real::ZERO,
            tau: S::Real::ZERO,
            steps: 0,
        })
    }
    fn start(&mut self, r: &Vector<S>) -> Result<()> {
        for i in 0..3 {
            finite::copy(self.v[i].as_mut_slice(), r.as_slice())?;
        }
        self.v[6].view_mut().fill(S::ZERO);
        self.steps = 0;
        self.theta = S::Real::ZERO;
        self.eta = S::ZERO;
        Ok(())
    }
    fn step(
        &mut self,
        a: &mut Actions<'_, S>,
        x: &mut Vector<S>,
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<bool> {
        let [shadow, w, u, next, v, uhat, d, temp] = &mut self.v;
        let even = self.steps.is_multiple_of(2);
        if self.steps == 0 {
            self.rho = denominator(
                finite::dot(w.as_slice(), shadow.as_slice(), a.kernels.dispatch())?,
                shadow,
                w,
                c,
            )?;
            self.tau = finite::checked_norm(w.as_slice(), c.execution())?;
            applied!(a.apply(u, v, false, c));
            finite::copy(uhat.as_mut_slice(), v.as_slice())?;
        } else if even {
            let rho = denominator(
                finite::dot(w.as_slice(), shadow.as_slice(), a.kernels.dispatch())?,
                shadow,
                w,
                c,
            )?;
            let beta = divide(rho, self.rho)?;
            finite::update(
                u.as_mut_slice(),
                S::ONE,
                w.as_slice(),
                beta,
                a.kernels.dispatch(),
            )?;
            finite::update(
                v.as_mut_slice(),
                S::ONE,
                uhat.as_slice(),
                beta,
                a.kernels.dispatch(),
            )?;
            finite::scale(v.as_mut_slice(), beta)?;
            applied!(a.apply(u, uhat, false, c));
            finite::update(
                v.as_mut_slice(),
                S::ONE,
                uhat.as_slice(),
                S::ONE,
                a.kernels.dispatch(),
            )?;
            self.rho = rho;
        } else {
            finite::copy(u.as_mut_slice(), next.as_slice())?;
            applied!(a.apply(u, uhat, false, c));
        }
        if even {
            let divisor = denominator(
                finite::dot(v.as_slice(), shadow.as_slice(), a.kernels.dispatch())?,
                shadow,
                v,
                c,
            )?;
            self.alpha = divide(self.rho, divisor)?;
            finite::copy(next.as_mut_slice(), u.as_slice())?;
            finite::update(
                next.as_mut_slice(),
                -self.alpha,
                v.as_slice(),
                S::ONE,
                a.kernels.dispatch(),
            )?;
        }
        finite::update(
            w.as_mut_slice(),
            -self.alpha,
            uhat.as_slice(),
            S::ONE,
            a.kernels.dispatch(),
        )?;
        finite::update(
            d.as_mut_slice(),
            S::ONE,
            u.as_slice(),
            divide(S::from_real(self.theta * self.theta) * self.eta, self.alpha)?,
            a.kernels.dispatch(),
        )?;
        let wn = finite::checked_norm(w.as_slice(), c.execution())?;
        let theta = divide(S::from_real(wn), S::from_real(self.tau))?.real_component();
        let cosine = S::Real::ONE / S::Real::ONE.hypot(theta);
        self.tau = wn * cosine;
        self.theta = theta;
        self.eta = S::from_real(cosine * cosine) * self.alpha;
        applied!(a.update(x, d, self.eta, temp, c));
        self.steps += 1;
        Ok(true)
    }
}

macro_rules! solver_api {
    ($solver:ident, $example:literal, $kernel:ident, $operator:ident, $variant:ident, $name:literal) => {
        #[doc = concat!($name, " solver with fixed right preconditioning.")]
        ///
        /// `A` must be square and may be nonsymmetric. The preconditioner `M⁻¹` is
        /// applied from the right and must not change during a solve. The solve
        /// converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`, with `‖b‖₂` replaced by
        /// one when `b = 0`. The true residual is recomputed every 50 iterations,
        /// when the residual estimate of the recurrence meets the tolerance, and at
        /// the end.
        ///
        /// A shadow inner product `r̂ᴴ v` counts as breakdown only when
        /// `|r̂ᴴ v| ≤ ε² ‖r̂‖₂ ‖v‖₂`, where `ε` is the machine epsilon. Breakdown ends
        /// the solve with
        /// [`Breakdown(RecurrenceDenominator)`](crate::LinearSolveBreakdown::RecurrenceDenominator)
        /// unless `with_breakdown_restarts` allows restarts.
        #[derive(Clone)]
        #[doc = $example]
        pub struct $solver<S: NumericScalar> {
            config: Config<S>,
        }
        impl<S: NumericScalar> core::fmt::Debug for $solver<S> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($solver))
                    .field("max_iterations", &self.config.max_iterations)
                    .field("execution", &self.config.execution)
                    .field("history", &self.config.history)
                    .finish_non_exhaustive()
            }
        }
        impl<S: NumericScalar> $solver<S> {
            /// Returns the execution options.
            pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
                self.config.execution
            }

            /// Sets the execution options: policy, product accuracy, time limit,
            /// byte limit and iteration budget.
            ///
            /// An iteration budget in `execution` caps `limit`. The accuracy setting
            /// applies to dense operator products; the recurrence arithmetic is as
            /// documented on the solver.
            pub const fn with_execution_options(
                mut self,
                execution: crate::LinalgExecutionOptions,
            ) -> Self {
                self.config.execution = execution;
                self
            }

            /// Creates a solver with residual tolerance `tolerance` and at most `limit` iterations.
            ///
            /// No history is recorded and breakdown restarts are disabled.
            pub fn new(tolerance: ResidualTolerance<S::Real>, limit: Budget) -> Self {
                Self {
                    config: Config::new(tolerance, limit),
                }
            }
            /// Enables or disables the residual history of up to `limit + 1` entries.
            ///
            /// Between checkpoints the entries are residual estimates of the
            /// recurrence; entries at checkpoints and the final entry are true
            /// residuals.
            #[must_use]
            pub fn with_history(mut self, enabled: bool) -> Self {
                self.config.history = enabled;
                self
            }
            /// Allows at most `limit` restarts after breakdown; the default is none.
            ///
            /// A restart continues from the last verified iterate and uses its true
            /// residual as the new shadow residual. A second breakdown before any
            /// accepted step ends the solve.
            #[must_use]
            pub fn with_breakdown_restarts(mut self, limit: Budget) -> Self {
                self.config.retries = limit.get();
                self
            }
            /// Returns the storage a solve of `request` needs.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if a size or byte count overflows `usize`.
            /// - `ResourceLimit` if the storage exceeds the byte limit of the
            ///   execution options.
            pub fn workspace_requirements(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeRequirements<S>> {
                self.config.requirements(StorageKind::$kernel, request)
            }
            /// Creates a workspace reserved for `request`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::workspace_requirements`].
            /// - `AllocationFailure` if memory cannot be reserved.
            pub fn workspace(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeWorkspace<S>> {
                let mut workspace = IterativeWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(request)?)?;
                Ok(workspace)
            }
            /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
            ///
            /// A solve that ends without meeting the tolerance, at the iteration or
            /// evaluation limit or by a breakdown, is an error. Use
            /// [`solve_with_options`](Self::solve_with_options) for a preconditioner, a
            /// start vector or the iterate of an unsuccessful solve.
            ///
            /// # Errors
            ///
            /// - The errors of [`solve_with_options`](Self::solve_with_options).
            /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge; its
            ///   diagnostics name the solver and carry the iteration count and the
            ///   residual norm.
            pub fn solve<'b>(
                &self,
                a: &dyn $operator<S>,
                b: impl Into<VectorView<'b, S>>,
            ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
                self.solve_with_options(a, b, crate::Preconditioning::None, None)?
                    .into_converged_solution()
            }
            /// Solves `A x = b` from the initial guess `x0`, or from zero when `x0` is `None`, and
            /// returns the iterate with its report.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::solve_into`].
            /// - `AllocationFailure` if the output or workspace cannot be allocated.
            pub fn solve_with_options<'b>(
                &self,
                a: &dyn $operator<S>,
                b: impl Into<VectorView<'b, S>>,
                preconditioning: Preconditioning<'_, S>,
                x0: Option<VectorView<'_, S>>,
            ) -> Result<IterativeSolveResult<S>> {
                let b = b.into();
                IterativeWorkspace::solve_owned(
                    a.cols(),
                    x0,
                    scioxide_core::algorithm_id!("linear", $name, 1),
                    |x, workspace| {
                        self.solve_into(
                            a,
                            b,
                            preconditioning,
                            x,
                            workspace,
                            ExecutionContext::sequential_native(),
                        )
                    },
                )
            }
            /// Solves `A x = b` in place, starting from the values in `x`.
            ///
            /// # Errors
            ///
            /// - The errors listed on
            ///   [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
            /// - `InvalidInput` (`UnsupportedConfiguration`) from BiCG and QMR if the
            ///   preconditioner is [`Preconditioning::Inverse`], which has no adjoint.
            pub fn solve_into<'b, 'scope, 'capability>(
                &self,
                a: &dyn $operator<S>,
                b: impl Into<VectorView<'b, S>>,
                preconditioning: Preconditioning<'_, S>,
                x: VectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<IterativeSolveReport<S::Real>> {
                solve_into::<S, $kernel<S>>(
                    &self.config,
                    scioxide_core::algorithm_id!("linear", $name, 1),
                    Operator::$variant(a),
                    b.into(),
                    preconditioning,
                    x,
                    workspace,
                    context.into(),
                )
            }
        }
    };
}
solver_api!(
    BicgSolver,
    r#"

BiCG is the biconjugate gradient method. It builds bases of the Krylov spaces of
`A` and `Aᴴ` with the two-sided Lanczos process. Each iteration costs one
product with `A` and one with `Aᴴ`, so `a` must be an
[`AdjointOperator`](scioxide_core::AdjointOperator). A preconditioner must be given
as [`Preconditioning::InverseAdjoint`]. The workspace holds 13 vectors of length
`n`. The residual norm need not decrease monotonically; [`QmrSolver`] smooths it
at the same cost, and [`BicgstabSolver`](crate::BicgstabSolver) avoids `Aᴴ`.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    BicgSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
    Preconditioning, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = BicgSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::adjoint(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- R. Fletcher, "Conjugate gradient methods for indefinite systems", in *Numerical
  Analysis, Dundee 1975*, Lecture Notes in Mathematics 506, Springer, 1976."#,
    Bicg,
    AdjointOperator,
    Adjoint,
    "BiCG"
);
solver_api!(
    CgsSolver,
    r#"

CGS is the conjugate gradient squared method. It applies the square of the BiCG
residual polynomial, so it needs no products with `Aᴴ`. Each iteration costs two
products with `A`; the workspace holds 14 vectors of length `n`. Convergence can
be faster than for BiCG but is more erratic, and rounding errors are amplified.
To limit that drift, the second product recomputes the true residual `b − A x`
in every iteration instead of updating it recursively; the directions
and the shadow residual are kept.

Cancellation in the shadow inner products `s = r̂ᴴ v` is amplified by the squared
polynomial. They are therefore computed with the Dot2 algorithm, in about twice
the working precision. Let `u = ε/2`, let `k` be the number of real products per
component (`n` for real and `2n` for complex scalars) plus 64 vector-lane
partial sums, `γ_k = k u/(1 − k u)`, and `η` the smallest subnormal number. The
computed `ŝ` satisfies

```text
|ŝ − s| ≤ E = 2u|ŝ| + 4γ_k² ‖r̂‖₂ ‖v‖₂ + kη,
```

so its relative error is about `2u` unless `|ŝ|` is below `γ_k² ‖r̂‖₂ ‖v‖₂ / u`.
When `[|ŝ| − E, |ŝ| + E]` contains the breakdown threshold, or the operands are
outside the normal range, an exact reduction decides the test and supplies the
product. Breakdown decisions therefore match those of an exact inner product.
The bound is wider for `f32` because `γ_k` is larger. All other recurrence
arithmetic uses `Standard` accuracy, whatever the execution options say.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    CgsSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
    Preconditioning, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = CgsSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- P. Sonneveld, "CGS, a fast Lanczos-type solver for nonsymmetric linear systems",
  *SIAM J. Sci. Stat. Comput.* 10(1), 1989.
- T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product", *SIAM J. Sci.
  Comput.* 26(6), 2005."#,
    Cgs,
    LinearOperator,
    Forward,
    "CGS"
);
solver_api!(
    QmrSolver,
    r#"

QMR is the quasi-minimal residual method. It uses the same two-sided Lanczos
bases as BiCG, without look-ahead, but chooses each iterate to minimize a
quasi-residual norm, which gives smoother convergence. Each iteration costs one
product with `A` and one with `Aᴴ`, so `a` must be an
[`AdjointOperator`](scioxide_core::AdjointOperator). A preconditioner must be given
as [`Preconditioning::InverseAdjoint`]. The workspace holds 18 vectors of length
`n`. [`TfqmrSolver`] gives similar smoothing without `Aᴴ`.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix, Preconditioning,
    QmrSolver, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = QmrSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::adjoint(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- R. W. Freund and N. M. Nachtigal, "QMR: a quasi-minimal residual method for
  non-Hermitian linear systems", *Numer. Math.* 60, 1991."#,
    Qmr,
    AdjointOperator,
    Adjoint,
    "QMR"
);
solver_api!(
    TfqmrSolver,
    r#"

TFQMR is the transpose-free quasi-minimal residual method. It applies the
quasi-minimal residual idea to the CGS recurrence, so it needs no products with
`Aᴴ` and converges more smoothly than CGS. Each iteration is one half step of
the method and costs one product with `A`; the workspace holds 14 vectors of
length `n`. Between checkpoints, the residual history records the bound
`τ √(k + 1)` on the residual norm after `k` half steps.

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix, Preconditioning,
    TfqmrSolver, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = TfqmrSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- R. W. Freund, "A transpose-free quasi-minimal residual algorithm for non-Hermitian
  linear systems", *SIAM J. Sci. Comput.* 14(2), 1993."#,
    Tfqmr,
    LinearOperator,
    Forward,
    "TFQMR"
);
crate::traits::impl_iterative_solver!(
    CgsSolver,
    _plan,
    scioxide_core::algorithm_id!("linear", "CGS", 1)
);
crate::traits::impl_iterative_solver!(
    TfqmrSolver,
    _plan,
    scioxide_core::algorithm_id!("linear", "TFQMR", 1)
);
