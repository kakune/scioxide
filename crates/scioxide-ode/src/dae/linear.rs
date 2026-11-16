//! Residual-Jacobian linear solves of the DAE integrator and its adjoint.
//!
//! The mixed Jacobian `a·∂F/∂y + b·∂F/∂ẏ` is supplied by the residual provider through
//! [`EffectiveJacobian`] and solved on the shared stiff [`LinearStorage`]: assembled and
//! factored on dense or sparse storage, or applied matrix-free by GMRES. Newton may
//! reuse raw derivative blocks and nearby factors. GMRES stage corrections are verified
//! against a fresh operator product; adjoint transposed solves, direct or iterative, are
//! verified against the matrix actually applied. Original residuals independently gate
//! stage acceptance.
use super::derivatives::Blocks;
use crate::OdeStageBackend;
use crate::safety::{finite, invalid};
use crate::stiff::linear::{LinearStorage, verify_backward_error, verify_defect};
use crate::stiff::{newton::NewtonReport, reuse};
use crate::termination::{Fault, Tunnel, admit};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AlgebraicScalar, EvaluationRole, LinearOperator, NumericScalar, RealScalar, SciOxideError,
    SolveContext, SolveState,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;

/// A directly supplied effective Jacobian, without an identity-mass assumption.
pub(crate) trait EffectiveJacobian<T: NumericScalar>: LinearOperator<T> {
    /// Acquire the original state and rate derivative blocks.
    fn dense(
        &mut self,
        output: &mut Matrix<T>,
        derivative_scratch: &mut Matrix<T>,
        differences: &mut Blocks<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault>;
    fn sparse(
        &mut self,
        pattern: scioxide_core::StructuralPatternView<'_>,
        output: &mut [T],
        derivative_scratch: &mut [T],
        differences: &mut Blocks<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault>;
    fn assemble_dense(
        &self,
        output: &mut Matrix<T>,
        state: &Matrix<T>,
        derivative: &Matrix<T>,
    ) -> Result<(), SciOxideError>;
    fn assemble_sparse(
        &self,
        pattern: scioxide_core::StructuralPatternView<'_>,
        output: &mut [T],
        state: &[T],
        derivative: &[T],
    ) -> Result<(), SciOxideError>;
    /// Resolve optional operator sources and return their attainable relative floor.
    fn prepare_product(
        &self,
        _zero: &[T],
        _output: &mut [T],
        _scratch: &mut [T],
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, Fault> {
        Ok(T::Real::ZERO)
    }
    fn preconditioner(&self) -> Option<&dyn scioxide_core::Preconditioner<T>>;
}

/// Identity and location of a Newton linearization within one integration epoch.
pub(crate) struct NewtonPoint<'a, T: NumericScalar> {
    pub(crate) time: T::Real,
    pub(crate) state: &'a [T],
    pub(crate) derivative: &'a [T],
    pub(crate) strategy: DerivativeStrategy,
    pub(crate) initialization: bool,
    pub(crate) inverse_step: T::Real,
}

#[derive(Clone, Copy)]
struct JacobianKey<S> {
    time: S,
    strategy: DerivativeStrategy,
}

/// Prepared storage for residual-Jacobian solves of dimension `n`.
pub(crate) struct DaeLinear<T: NumericScalar> {
    n: usize,
    differences: Blocks<T>,
    linear: LinearStorage<T>,
    jacobian: Matrix<T>,
    derivative: Matrix<T>,
    sparse_jacobian: Vec<T>,
    sparse_derivative: Vec<T>,
    point: Vec<T>,
    rate: Vec<T>,
    key: Option<JacobianKey<T::Real>>,
    factor_inverse: Option<T::Real>,
    factor_initialization: bool,
    refresh: bool,
    lagged_jacobian: bool,
    lagged_factor: bool,
    solution: Vec<T>,
    product: Vec<T>,
    scratch: Vec<T>,
    state: SolveState,
}

impl<T: NumericScalar> core::fmt::Debug for DaeLinear<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DaeLinear")
            .field("dimension", &self.n)
            .field("linear", &self.linear)
            .finish_non_exhaustive()
    }
}

impl<T: NumericScalar> DaeLinear<T> {
    /// Reserve storage; a Krylov `operator_scratch` covers one effective application.
    pub(crate) fn new(
        n: usize,
        backend: &OdeStageBackend,
        source_backend: &OdeStageBackend,
    ) -> Result<Self, SciOxideError> {
        if n == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "stage dimension must be positive",
            ));
        }
        let mut derivative = Matrix::empty();
        let mut jacobian = Matrix::empty();
        let sparse = match backend {
            OdeStageBackend::Dense => {
                derivative.try_resize_flat(n, n)?;
                jacobian.try_resize_flat(n, n)?;
                0
            }
            OdeStageBackend::Sparse { pattern, .. } => pattern.view().col_indices().len(),
            OdeStageBackend::Krylov { .. } => 0,
        };
        let linear = LinearStorage::new(n, backend)?;
        let operator = linear.iterative_scratch().0;
        let filled = |len, quantity| try_filled(len, T::ZERO, quantity);
        Ok(Self {
            n,
            differences: Blocks::new(n, source_backend)?,
            linear,
            jacobian,
            derivative,
            sparse_jacobian: filled(sparse, scioxide_core::quantity!("sparse_jacobian"))?,
            sparse_derivative: filled(sparse, scioxide_core::quantity!("sparse_derivative"))?,
            point: filled(n, scioxide_core::quantity!("dae_linearization_state"))?,
            rate: filled(n, scioxide_core::quantity!("dae_linearization_rate"))?,
            key: None,
            factor_inverse: None,
            factor_initialization: false,
            refresh: false,
            lagged_jacobian: false,
            lagged_factor: false,
            solution: filled(n, scioxide_core::quantity!("dae_solution"))?,
            product: filled(n, scioxide_core::quantity!("dae_product"))?,
            scratch: filled(operator.max(n), scioxide_core::quantity!("dae_scratch"))?,
            state: SolveState::new(),
        })
    }

    pub(crate) fn invalidate_factor(&mut self) {
        self.factor_inverse = None;
    }

    pub(crate) fn invalidate(&mut self) {
        self.key = None;
        self.invalidate_factor();
        self.refresh = false;
        self.lagged_jacobian = false;
        self.lagged_factor = false;
    }

    pub(crate) fn frozen(&self) -> bool {
        !self.linear.is_iterative()
    }

    pub(crate) fn requires_rate(&self) -> bool {
        self.lagged_jacobian || self.lagged_factor
    }

    pub(crate) fn observe_newton(
        &mut self,
        report: NewtonReport<T::Real>,
    ) -> Result<(), SciOxideError> {
        if reuse::slow(report)? {
            if self.lagged_factor {
                self.invalidate_factor();
            } else {
                self.refresh = true;
            }
        }
        Ok(())
    }

    /// Prepare one stage's frozen matrix. `None` requests a fresh exact transpose
    /// equation; Newton points permit reuse within the current integration epoch.
    pub(crate) fn prepare(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        point: Option<NewtonPoint<'_, T>>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        if matrix.rows() != self.n || matrix.cols() != self.n {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "effective residual Jacobian shape mismatch",
            )
            .into());
        }
        if self.linear.is_iterative() {
            return Ok(());
        }
        let limit = reuse::movement_limit()?;
        let keep = point.as_ref().is_some_and(|point| {
            !self.refresh
                && self.key.is_some_and(|key| {
                    key.strategy == point.strategy
                        && reuse::movement(key.time, point.time) <= limit
                        && reuse::within_movement(&self.point, point.state, limit)
                        && reuse::within_movement(&self.rate, point.derivative, limit)
                })
        });
        self.lagged_jacobian = keep
            && point.as_ref().is_some_and(|point| {
                self.key.is_some_and(|key| key.time != point.time)
                    || self.point != point.state
                    || self.rate != point.derivative
            });
        let inverse = point
            .as_ref()
            .map_or(T::Real::ONE, |point| point.inverse_step);
        let initialization = point.as_ref().is_some_and(|point| point.initialization);
        let nearby = keep
            && initialization == self.factor_initialization
            && self
                .factor_inverse
                .map(|old| reuse::nearby(inverse / old))
                .transpose()?
                .unwrap_or(false);
        self.lagged_factor = nearby && self.factor_inverse != Some(inverse);
        if nearby {
            return Ok(());
        }
        self.invalidate_factor();
        // Acquiring or assembling may fail after writing scratch. No partial entry
        // may be reused by a later attempt.
        let old_key = self.key.take();
        self.assemble_factor(matrix, !keep, context)?;
        self.factor_inverse = Some(inverse);
        self.factor_initialization = initialization;
        self.refresh = false;
        if keep {
            self.key = old_key;
        } else if let Some(point) = point {
            self.point.copy_from_slice(point.state);
            self.rate.copy_from_slice(point.derivative);
            self.key = Some(JacobianKey {
                time: point.time,
                strategy: point.strategy,
            });
        }
        Ok(())
    }

    /// Retry the current stage with its exact coefficient and the same raw blocks.
    pub(crate) fn refactor(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        inverse: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        let key = self.key.take();
        self.invalidate_factor();
        self.assemble_factor(matrix, false, context)?;
        self.key = key;
        self.factor_inverse = Some(inverse);
        self.lagged_factor = false;
        Ok(())
    }

    fn assemble_factor(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        refresh: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        match &mut self.linear {
            LinearStorage::Dense { matrix: output, .. } => {
                if refresh {
                    matrix.dense(
                        &mut self.jacobian,
                        &mut self.derivative,
                        &mut self.differences,
                        context,
                    )?;
                }
                matrix.assemble_dense(output, &self.jacobian, &self.derivative)?;
            }
            linear => {
                // Sparse value assembly propagates only `SciOxideError`; denials cross it in
                // the tunnel.
                let tunnel = Tunnel::default();
                let assembled = linear.assemble_sparse(|pattern, output| {
                    if refresh {
                        matrix
                            .sparse(
                                pattern,
                                &mut self.sparse_jacobian,
                                &mut self.sparse_derivative,
                                &mut self.differences,
                                context,
                            )
                            .map_err(|fault| tunnel.carry(fault))?;
                    }
                    matrix.assemble_sparse(
                        pattern,
                        output,
                        &self.sparse_jacobian,
                        &self.sparse_derivative,
                    )
                });
                tunnel.restore(assembled)?;
            }
        }
        admit(context, EvaluationRole::Factorization)?;
        self.linear.factor(context.execution())?;
        Ok(())
    }

    /// Apply the assembled matrix of the prepared factor, or the current operator, to
    /// `input`; the product is consistent with [`Self::solve`].
    pub(crate) fn apply(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        input: &[T],
        output: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        if matrix.rows() != self.n || input.len() != self.n || output.len() != self.n {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "effective residual Jacobian shape mismatch",
            )
            .into());
        }
        admit(context, EvaluationRole::LinearOperator)?;
        if self.linear.is_iterative() {
            let operator = self.linear.iterative_scratch().0;
            matrix
                .apply_into_scoped(
                    input,
                    output,
                    &mut self.scratch[..operator],
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require)?;
        } else {
            self.linear
                .apply_assembled(input, output, &mut self.scratch)?;
        }
        Ok(finite(output, scioxide_core::NonFiniteOrigin::Callback)?)
    }

    /// Solve a stage equation with the prepared factor or current operator. A nearby
    /// factor is a chord iteration, not an exact current Jacobian.
    ///
    /// GMRES targets the relative defect `relative`, and a fresh operator product then
    /// verifies it; `Ok(false)` reports a verified defect above the target, which the
    /// caller treats as a failed stage. Direct solves are backward stable and are not
    /// verified: their attainable relative defect is about the condition number times
    /// the unit roundoff, which on stiff steps in low precision exceeds any useful
    /// forcing target.
    pub(crate) fn solve_stage(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        rhs: &[T],
        output: &mut [T],
        relative: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, Fault> {
        self.check_solve(matrix, rhs, output)?;
        let attained = if self.linear.is_iterative() {
            // GMRES that stops short of its target fails the stage like a verified miss.
            match self.solve_iterative(matrix, rhs, relative, context)? {
                Ok(relative) => relative_defect(&self.product, rhs)? <= relative,
                Err(_) => false,
            }
        } else {
            admit(context, EvaluationRole::LinearSolve)?;
            self.linear
                .solve_direct(rhs, &mut self.solution, context.execution())?;
            finite(&self.solution, scioxide_core::NonFiniteOrigin::Callback)?;
            true
        };
        output.copy_from_slice(&self.solution);
        Ok(attained)
    }

    /// Solve using the prepared factor or current operator and verify the solution
    /// against `relative`: the relative defect `‖c − A x‖₂ / ‖c‖₂` of a Krylov
    /// solution, or the normwise backward error of a direct one (see
    /// [`verify_backward_error`]). Returns the measured value.
    pub(crate) fn solve(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        rhs: &[T],
        output: &mut [T],
        relative: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, Fault> {
        self.check_solve(matrix, rhs, output)?;
        let measured = if self.linear.is_iterative() {
            let relative = self
                .solve_iterative(matrix, rhs, relative, context)?
                .map_err(|reason| {
                    SciOxideError::convergence_failure(
                        reason,
                        scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                        "Krylov stage did not converge",
                    )
                })?;
            verify_defect(&self.product, rhs, relative)?
        } else {
            admit(context, EvaluationRole::LinearSolve)?;
            self.linear
                .solve_direct(rhs, &mut self.solution, context.execution())?;
            admit(context, EvaluationRole::LinearOperator)?;
            self.linear
                .apply_assembled(&self.solution, &mut self.product, &mut self.scratch)?;
            finite(&self.solution, scioxide_core::NonFiniteOrigin::Callback)?;
            finite(&self.product, scioxide_core::NonFiniteOrigin::Callback)?;
            verify_backward_error(
                &self.product,
                rhs,
                &self.solution,
                self.linear.assembled_norm_inf()?,
                relative,
            )?
        };
        output.copy_from_slice(&self.solution);
        Ok(measured)
    }

    fn check_solve(
        &self,
        matrix: &impl EffectiveJacobian<T>,
        rhs: &[T],
        output: &[T],
    ) -> Result<(), SciOxideError> {
        let n = self.n;
        if matrix.rows() != n || matrix.cols() != n || rhs.len() != n || output.len() != n {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "effective residual Jacobian shape mismatch",
            ));
        }
        finite(rhs, scioxide_core::NonFiniteOrigin::Callback)
    }

    /// GMRES into the solution buffer and its fresh operator product into the product
    /// buffer. Returns the relative target, raised to the operator's attainable floor,
    /// or the reason GMRES stopped short of it, in which case no product is formed.
    fn solve_iterative(
        &mut self,
        matrix: &mut impl EffectiveJacobian<T>,
        rhs: &[T],
        relative: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Result<T::Real, scioxide_core::ConvergenceReason>, Fault> {
        let n = self.n;
        let (operator, preconditioner_scratch) = self.linear.iterative_scratch();
        let preconditioner = matrix.preconditioner();
        if matrix.apply_workspace_len()? > operator
            || preconditioner.is_some_and(|p| p.dimension() != n)
            || preconditioner.map_or(
                Ok(0),
                scioxide_core::Preconditioner::precondition_workspace_len,
            )? > preconditioner_scratch
        {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "effective operator exceeds prepared capacity",
            )
            .into());
        }
        self.solution.fill(T::ZERO);
        let floor = matrix.prepare_product(
            &self.solution,
            &mut self.product,
            &mut self.scratch[..operator],
            context,
        )?;
        let relative = relative.max(floor);
        admit(context, EvaluationRole::LinearSolve)?;
        let identity = scioxide_core::SolveIdentity::new(
            context.identity().problem(),
            scioxide_core::algorithm_id!("ode", "dae-gmres", 1),
            n as u128,
            0,
        );
        let Self {
            linear,
            solution,
            state,
            ..
        } = &mut *self;
        let converged = scioxide_core::SolveScope::from(&mut *context).run(
            state,
            identity,
            scioxide_core::SolveBudgets::unbounded(),
            |child| {
                linear
                    .gmres(
                        &*matrix,
                        preconditioner,
                        rhs,
                        solution,
                        relative,
                        usize::MAX,
                        child,
                    )
                    .map(|outcome| outcome.map(|_| ()))
            },
        )?;
        if let Err(reason) = converged {
            return Ok(Err(reason));
        }
        admit(context, EvaluationRole::LinearOperator)?;
        matrix
            .apply_into_scoped(
                &self.solution,
                &mut self.product,
                &mut self.scratch[..operator],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite(&self.solution, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(&self.product, scioxide_core::NonFiniteOrigin::Callback)?;
        Ok(Ok(relative))
    }
}

/// Relative defect `‖applied − rhs‖ / ‖rhs‖` in the scaled two-norm of [`verify_defect`];
/// a nonzero defect of a zero right-hand side is infinite.
fn relative_defect<T: NumericScalar>(applied: &[T], rhs: &[T]) -> Result<T::Real, SciOxideError> {
    let mut defect = scioxide_core::ScaledNorm::new();
    let mut norm = scioxide_core::ScaledNorm::new();
    for (&actual, &expected) in applied.iter().zip(rhs) {
        defect.add_magnitude(crate::safety::state_difference(actual, expected)?.abs())?;
        norm.add_magnitude(expected.abs())?;
    }
    let (defect, norm) = (defect.finish()?, norm.finish()?);
    Ok(if defect == T::Real::ZERO {
        T::Real::ZERO
    } else {
        defect / norm
    })
}
