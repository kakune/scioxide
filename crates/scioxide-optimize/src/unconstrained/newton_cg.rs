//! Steihaug truncated conjugate gradient for the trust-region subproblem.
use super::trust_region::boundary_distance;
use crate::ObjectiveFunction;
use crate::context::SolveSession;
use crate::unconstrained::SecondOrderStorage;
use crate::utils::{dot, norm2, trial_component};

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};

fn reserve<S: RealScalar + NumericScalar<Real = S>>(
    w: &mut SecondOrderStorage<S>,
) -> Result<(), SciOxideError> {
    for v in [&mut w.cg_residual, &mut w.cg_direction, &mut w.cg_product] {
        scioxide_core::backend::try_reserve(
            v,
            w.dimension.saturating_sub(v.len()),
            scioxide_core::quantity!("newton_cg_vectors"),
        )?;
        v.resize(w.dimension, <S as scioxide_core::AlgebraicScalar>::ZERO);
    }
    Ok(())
}
pub(super) fn step<S, F>(
    f: &F,
    w: &mut SecondOrderStorage<S>,
    radius: S,
    limit: Budget,
    matrix_free: bool,
    session: &SolveSession<'_, '_, '_>,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    use scioxide_core::checked_float_quotient as quotient;
    reserve(w)?;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let gradient_norm = norm2(&w.gradient)?;
    w.direction.fill(zero);
    for ((r, d), &g) in w
        .cg_residual
        .iter_mut()
        .zip(&mut w.cg_direction)
        .zip(&w.gradient)
    {
        *r = quotient(g, gradient_norm)?;
        *d = -*r;
    }
    let mut rr = dot(&w.cg_residual, &w.cg_residual)?;
    let target =
        S::SQRT_EPSILON.max(S::try_from_f64_lossy(0.1)?.min(RealScalar::sqrt(gradient_norm)));
    w.newton.cg_iterations = 0;
    for _ in 0..limit.get() {
        w.newton.cg_iterations += 1;
        session.check()?;
        if matrix_free {
            f.hessian_vector_product_into(&w.x, &w.cg_direction, &mut w.cg_product)?;
        } else {
            crate::dense::matrix_vector_into(
                w.hessian.view(),
                &w.cg_direction,
                &mut w.cg_product,
                &mut w.gemv_scratch,
                session,
            )?;
        }
        for value in &mut w.cg_product {
            *value = quotient(*value, gradient_norm)?;
        }
        let curvature = dot(&w.cg_direction, &w.cg_product)?;
        let alpha = if curvature > zero {
            quotient(rr, curvature)?
        } else {
            zero
        };
        for ((trial, &p), &d) in w.trial_x.iter_mut().zip(&w.direction).zip(&w.cg_direction) {
            *trial = trial_component(p, alpha, d)?;
        }
        if curvature <= zero || norm2(&w.trial_x)? >= radius {
            let dn = norm2(&w.cg_direction)?;
            for ((p, d), (&current, &direction)) in w
                .rhs
                .iter_mut()
                .zip(&mut w.model_product)
                .zip(w.direction.iter().zip(&w.cg_direction))
            {
                *p = quotient(current, radius)?;
                *d = quotient(direction, dn)?;
            }
            let t = boundary_distance(&w.rhs, &w.model_product)?;
            let length = scioxide_core::checked_float_product(radius, t)?;
            let coefficient = quotient(length, dn)?;
            for (p, &d) in w.direction.iter_mut().zip(&w.cg_direction) {
                *p = trial_component(*p, coefficient, d)?;
            }
            w.newton.cg_termination = if curvature <= zero {
                NewtonCgTermination::NegativeCurvature
            } else {
                NewtonCgTermination::Boundary
            };
            return Ok(true);
        }
        w.direction.copy_from_slice(&w.trial_x);
        for (r, &hd) in w.cg_residual.iter_mut().zip(&w.cg_product) {
            *r = trial_component(*r, alpha, hd)?;
        }
        let next = dot(&w.cg_residual, &w.cg_residual)?;
        if RealScalar::sqrt(next) <= target {
            w.newton.cg_termination = NewtonCgTermination::ResidualSmall;
            return Ok(false);
        }
        let beta = quotient(next, rr)?;
        for (d, &r) in w.cg_direction.iter_mut().zip(&w.cg_residual) {
            *d = trial_component(-r, beta, *d)?;
        }
        rr = next;
    }
    // A bounded incomplete inner solve is still a valid model proposal.
    w.newton.cg_termination = NewtonCgTermination::IterationLimit;
    Ok(false)
}

/// Gives the reason the last truncated conjugate gradient subproblem solve stopped.
///
/// This describes the inner solve only; the termination of the minimization
/// is in [`OptimizeTermination`](crate::OptimizeTermination).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NewtonCgTermination {
    /// No subproblem was solved by conjugate gradients.
    #[default]
    NotRun,
    /// The scaled residual of the Newton equations fell below its tolerance.
    ResidualSmall,
    /// The step reached the trust-region boundary.
    Boundary,
    /// A direction of nonpositive curvature was followed to the boundary.
    NegativeCurvature,
    /// The limit on Hessian-vector products was reached.
    IterationLimit,
}
/// Reports the Hessian model of the last iteration of a Newton or trust-region solve.
///
/// The values describe the numerical model; they do not prove that the point
/// is a minimum.
#[derive(Debug, Clone, Copy)]
pub struct NewtonDiagnostics<S: RealScalar> {
    /// Inertia of the last factored Hessian before any shift.
    pub(crate) inertia: Option<scioxide_sparse::LdlInertia>,
    /// Diagonal shift in normalized Hessian units; zero means unmodified.
    pub(crate) normalized_shift: S,
    /// Maximum component of the scaled linear-system residual.
    pub(crate) linear_residual: Option<S>,
    /// Hessian-vector products of the last conjugate gradient solve.
    pub(crate) cg_iterations: usize,
    /// Outcome of the last conjugate gradient solve.
    pub(crate) cg_termination: NewtonCgTermination,
    /// Whether curvature was checked at the returned stationary point.
    pub(crate) second_order_checked: bool,
}

impl<S: RealScalar> NewtonDiagnostics<S> {
    /// Returns the inertia of the last factored Hessian before any shift, if one was factored.
    pub const fn inertia(self) -> Option<scioxide_sparse::LdlInertia> {
        self.inertia
    }
    /// Returns the diagonal shift `τ` added to the Hessian scaled by its largest
    /// entry; zero means no shift.
    pub const fn normalized_shift(self) -> S {
        self.normalized_shift
    }
    /// Returns the largest component of the scaled residual of the Newton
    /// equations, if they were solved by factorization.
    pub const fn linear_residual(self) -> Option<S> {
        self.linear_residual
    }
    /// Returns the number of Hessian-vector products in the last conjugate gradient solve.
    pub const fn cg_iterations(self) -> usize {
        self.cg_iterations
    }
    /// Returns the outcome of the last conjugate gradient solve.
    pub const fn cg_termination(self) -> NewtonCgTermination {
        self.cg_termination
    }
    /// Returns `true` if the curvature of the Hessian was checked at the returned point.
    pub const fn second_order_checked(self) -> bool {
        self.second_order_checked
    }
}
impl<S: RealScalar> Default for NewtonDiagnostics<S> {
    fn default() -> Self {
        Self {
            inertia: None,
            normalized_shift: S::ZERO,
            linear_residual: None,
            cg_iterations: 0,
            cg_termination: NewtonCgTermination::NotRun,
            second_order_checked: false,
        }
    }
}

/// Extract a negative direction from LDL congruence, without an eigensolver.
pub(super) fn stationary_direction<S: RealScalar + NumericScalar<Real = S>>(
    w: &mut SecondOrderStorage<S>,
    radius: S,
    session: &SolveSession<'_, '_, '_>,
    cached_direction: &mut Option<Option<S>>,
) -> Result<bool, SciOxideError> {
    use scioxide_core::checked_float_quotient as divide;
    let n = w.dimension;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let prepared = if let Some(prepared) = *cached_direction {
        prepared
    } else {
        w.ldl_workspace.try_reserve(
            &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::factor(
                    n,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        session.invoke(scioxide_core::EvaluationRole::Factorization, || {
            scioxide_linalg::LdlDecomposition::factor_into(
                w.hessian.view(),
                scioxide_linalg::HermitianInput::Exact,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut w.ldl_factor,
                &mut w.ldl_workspace,
                session.execution()?,
            )
        })?;
        let inertia = w.ldl_factor.inertia(scioxide_linalg::RankPolicy::Exact)?;
        w.newton.inertia = Some(inertia);
        if inertia.negative() == 0 {
            w.newton.second_order_checked = true;
            *cached_direction = Some(None);
            return Ok(false);
        }
        w.rhs.fill(zero);
        for block in w.ldl_factor.pivot_blocks() {
            let i = block.checked_range(n)?.start;
            let a = w.ldl_factor.diagonal()[i];
            if block.size() == 1 {
                if a < zero {
                    w.rhs[i] = one;
                    break;
                }
                continue;
            }
            let c = w.ldl_factor.diagonal()[i + 1];
            let b = w.ldl_factor.off_diagonal()[i];
            let scale = NumericScalar::abs(a)
                .max(NumericScalar::abs(b))
                .max(NumericScalar::abs(c));
            if scale == zero {
                continue;
            }
            let a = divide(a, scale)?;
            let b = divide(b, scale)?;
            let c = divide(c, scale)?;
            let low = (a + c) * S::HALF - RealScalar::hypot((a - c) * S::HALF, b);
            if low < zero {
                if b == zero {
                    w.rhs[if a < c { i } else { i + 1 }] = one;
                } else {
                    w.rhs[i] = b;
                    w.rhs[i + 1] = low - a;
                }
                break;
            }
        }
        let lower = w.ldl_factor.l_matrix();
        let triangular = scioxide_linalg::TriangularPlan::from_dense(
            lower.view(),
            scioxide_core::Triangle::Lower,
            scioxide_core::DiagonalKind::Unit,
        )?;
        session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
            triangular
                .solve_into(
                    scioxide_linalg::StructuredOperation::Transpose,
                    scioxide_linalg::VectorView::from_slice(&w.rhs),
                    &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                    w.factor_solution.view_mut(),
                    &mut w.ldl_workspace,
                    session.execution()?,
                )
                .and_then(|report| report.termination().require_accepted())
        })?;
        let length = norm2(w.factor_solution.as_slice())?;
        *cached_direction = Some(Some(length));
        Some(length)
    };
    let Some(length) = prepared else {
        return Ok(false);
    };
    for (i, &original) in w.ldl_factor.permutation().iter().enumerate() {
        w.direction[original] = scioxide_core::checked_float_product(
            radius,
            divide(w.factor_solution.as_slice()[i], length)?,
        )?;
    }
    if dot(&w.gradient, &w.direction)? > zero {
        for d in &mut w.direction {
            *d = -*d;
        }
    }
    crate::dense::matrix_vector_into(
        w.hessian.view(),
        &w.direction,
        &mut w.model_product,
        &mut w.gemv_scratch,
        session,
    )?;
    if dot(&w.direction, &w.model_product)? >= zero {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "LDL direction failed original-model curvature check",
        ));
    }

    w.newton.cg_termination = NewtonCgTermination::NegativeCurvature;
    Ok(true)
}
