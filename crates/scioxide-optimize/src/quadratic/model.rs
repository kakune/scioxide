//! One normalized constraint model, with coordinate bounds and original sources.
use super::common::{factor_qr, project_to_bounds};
use crate::context::SolveSession;
use crate::{
    LinearConstraintSense as Sense, OptimalityTolerances, OptimizeTermination, ProgrammingResult,
    QpProblem,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_filled, try_with_capacity};
use scioxide_core::{Bounds, Budget, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, RankPolicy};

pub(super) enum Normal<S> {
    General(Vec<S>),
    Coordinate { variable: usize, sign: S },
}
pub(super) enum Source {
    Row(usize),
    Lower(usize),
    Upper(usize),
    Fixed(usize),
}
pub(super) struct Row<S> {
    pub normal: Normal<S>,
    pub rhs: S,
    pub equality: bool,
    pub enabled: bool,
    source: Source,
    sign: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> Row<S> {
    pub(super) fn coefficient(&self, j: usize) -> S {
        match &self.normal {
            Normal::General(a) => a[j],
            Normal::Coordinate { variable, sign } => {
                if *variable == j {
                    *sign
                } else {
                    S::ZERO
                }
            }
        }
    }
    pub(super) fn value(&self, x: &[S]) -> Result<S, SciOxideError> {
        match &self.normal {
            Normal::General(a) => crate::utils::dot(a, x),
            Normal::Coordinate { variable, sign } => Ok(x[*variable] * *sign),
        }
    }
    pub(super) fn scale(&self, x: &[S]) -> Result<S, SciOxideError> {
        let mut sum = scioxide_core::ProductSum::new();
        sum.add_product(NumericScalar::abs(self.rhs), S::ONE)?;
        match &self.normal {
            Normal::General(a) => {
                for (&a, &x) in a.iter().zip(x) {
                    sum.add_product(NumericScalar::abs(a), NumericScalar::abs(x))?;
                }
            }
            Normal::Coordinate { variable, .. } => {
                sum.add_product(NumericScalar::abs(x[*variable]), S::ONE)?
            }
        }
        sum.finish()
    }
    pub(super) fn norm(&self) -> Result<S, SciOxideError> {
        match &self.normal {
            Normal::General(a) => crate::utils::norm2(a),
            Normal::Coordinate { .. } => Ok(S::ONE),
        }
    }
}

pub(super) struct Model<S> {
    pub rows: Vec<Row<S>>,
    pub general: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> Model<S> {
    pub(super) fn new(problem: &QpProblem<S>) -> Result<Self, SciOxideError> {
        let n = problem.dimension();
        let general = problem.constraints().map_or(0, |c| c.len());
        let capacity = checked_add(
            general,
            checked_add(n, n, scioxide_core::quantity!("qp_bounds"))?,
            scioxide_core::quantity!("qp_constraints"),
        )?;
        let mut rows = try_with_capacity(capacity, scioxide_core::quantity!("qp_constraints"))?;
        if let Some(c) = problem.constraints() {
            for i in 0..general {
                let sign = if c.senses()[i] == Sense::GreaterEqual {
                    -S::ONE
                } else {
                    S::ONE
                };
                let mut a = try_filled(n, S::ZERO, scioxide_core::quantity!("qp_general_row"))?;
                c.coefficients().write_row(i, &mut a)?;
                for v in &mut a {
                    *v *= sign;
                }
                rows.push(Row {
                    normal: Normal::General(a),
                    rhs: c.rhs()[i] * sign,
                    equality: c.senses()[i] == Sense::Equal,
                    enabled: true,
                    source: Source::Row(i),
                    sign,
                });
            }
        }
        if let Some(bounds) = problem.bounds() {
            for (variable, &(lo, hi)) in bounds.as_slice().iter().enumerate() {
                if let (Some(l), Some(u)) = (lo, hi)
                    && l == u
                {
                    rows.push(Row {
                        normal: Normal::Coordinate {
                            variable,
                            sign: S::ONE,
                        },
                        rhs: l,
                        equality: true,
                        enabled: true,
                        source: Source::Fixed(variable),
                        sign: S::ONE,
                    });
                    continue;
                }
                for (bound, sign, source) in [
                    (lo, -S::ONE, Source::Lower(variable)),
                    (hi, S::ONE, Source::Upper(variable)),
                ] {
                    if let Some(bound) = bound {
                        rows.push(Row {
                            normal: Normal::Coordinate { variable, sign },
                            rhs: sign * bound,
                            equality: false,
                            enabled: true,
                            source,
                            sign: S::ONE,
                        });
                    }
                }
            }
        }
        Ok(Self { rows, general })
    }

    pub(super) fn reduce_equalities(
        &mut self,
        n: usize,
        t: OptimalityTolerances<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let mut indices =
            try_with_capacity(self.rows.len(), scioxide_core::quantity!("qp_equalities"))?;
        indices.extend(
            self.rows
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.equality.then_some(i)),
        );
        let m = indices.len();
        if m == 0 {
            return Ok(true);
        }
        let mut a = Matrix::zeros(n.max(m), m)?;
        for (j, &i) in indices.iter().enumerate() {
            for k in 0..n {
                a.view_mut().set(k, j, self.rows[i].coefficient(k))?;
            }
        }
        let factor = factor_qr(&a, session)?;
        let policy = RankPolicy::Threshold(t.rank());
        let rank = factor.analyze_rank(policy)?.rank();
        let triangular = factor.to_triangular_matrix()?;
        let mut coordinates = try_filled(
            rank,
            S::ZERO,
            scioxide_core::quantity!("qp_equality_coordinates"),
        )?;
        for j in 0..m {
            let source = indices[factor.permutation()[j]];
            let mut sum = scioxide_core::RangeAccumulator::new();
            let mut magnitude = scioxide_core::ProductSum::new();
            magnitude.add_product(NumericScalar::abs(self.rows[source].rhs), S::ONE)?;
            for (i, &value) in coordinates.iter().enumerate().take(j.min(rank)) {
                let coefficient = triangular.view().get(i, j)?;
                sum.add_product(coefficient, value)?;
                magnitude
                    .add_product(NumericScalar::abs(coefficient), NumericScalar::abs(value))?;
            }
            let residual = scioxide_core::checked_difference(self.rows[source].rhs, sum.finish()?)?;
            if j < rank {
                coordinates[j] =
                    scioxide_core::checked_float_quotient(residual, triangular.view().get(j, j)?)?;
            } else if !t
                .feasibility()
                .is_satisfied(NumericScalar::abs(residual), magnitude.finish()?)?
            {
                return Ok(false);
            }
        }
        for &i in &indices {
            self.rows[i].enabled = false;
        }
        for &j in factor.permutation().iter().take(rank) {
            self.rows[indices[j]].enabled = true;
        }
        Ok(true)
    }

    pub(super) fn feasible(
        &self,
        x: &[S],
        t: OptimalityTolerances<S>,
    ) -> Result<bool, SciOxideError> {
        for row in &self.rows {
            let residual = scioxide_core::checked_difference(row.value(x)?, row.rhs)?;
            let violation = if row.equality {
                NumericScalar::abs(residual)
            } else {
                residual.max(S::ZERO)
            };
            if !t.feasibility().is_satisfied(violation, row.scale(x)?)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn initial(
        &self,
        problem: &QpProblem<S>,
        t: OptimalityTolerances<S>,
        limit: Budget,
        warm: Option<&super::QpWarmStart<S>>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let n = problem.dimension();
        let width = checked_add(n, 1, scioxide_core::quantity!("qp_phase_dimension"))?;
        let mut x = try_filled(width, S::ZERO, scioxide_core::quantity!("qp_initial_point"))?;
        if let Some(warm) = warm {
            x[..n].copy_from_slice(warm.point());
        } else if let Some(bounds) = problem.bounds() {
            project_to_bounds(&mut x[..n], bounds.as_slice());
        }
        if self.feasible(&x[..n], t)? {
            return Ok(ProgrammingResult {
                x,
                termination: OptimizeTermination::Converged,
                ..ProgrammingResult::empty()
            });
        }
        let count = self.rows[..self.general]
            .iter()
            .filter(|r| r.enabled)
            .map(|r| 1 + usize::from(r.equality))
            .sum();
        let mut a = Matrix::zeros(count, width)?;
        let mut b = try_with_capacity(count, scioxide_core::quantity!("qp_phase_rhs"))?;
        let mut senses = try_with_capacity(count, scioxide_core::quantity!("qp_phase_senses"))?;
        for row in self.rows[..self.general].iter().filter(|r| r.enabled) {
            for sign in [Some(S::ONE), row.equality.then_some(-S::ONE)]
                .into_iter()
                .flatten()
            {
                let i = b.len();
                for j in 0..n {
                    a.view_mut().set(i, j, row.coefficient(j) * sign)?;
                }
                a.view_mut().set(i, n, -S::ONE)?;
                b.push(row.rhs * sign);
                senses.push(Sense::LessEqual);
            }
        }
        let mut c = try_filled(
            width,
            S::ZERO,
            scioxide_core::quantity!("qp_phase_objective"),
        )?;
        c[n] = S::ONE;
        let mut bounds = try_with_capacity(width, scioxide_core::quantity!("qp_phase_bounds"))?;
        if let Some(original) = problem.bounds() {
            bounds.extend_from_slice(original.as_slice());
        } else {
            bounds.resize(n, (None, None));
        }
        bounds.push((Some(S::ZERO), None));
        let p = crate::LpProblem::new(c)?
            .with_constraints(crate::LinearConstraints::new(a, b, senses)?)?
            .with_bounds(Bounds::new(bounds)?)?;
        let solver = crate::SimplexSolver::default()
            .with_tolerances(t)
            .with_max_iterations(limit);
        let mut state = scioxide_core::SolveState::new();
        session.with_child(
            &mut state,
            solver.identity(&p),
            scioxide_core::SolveBudgets::unbounded()
                .with_limit(scioxide_core::BudgetKind::Iteration, limit),
            |context| {
                let child = SolveSession::new(context, scioxide_core::argument!("context"));
                let result = solver.solve_observed(&p, &child);
                if let Some(stop) = child.stopped() {
                    session.set_stopped(stop);
                }
                result
            },
        )
    }

    pub(super) fn diagnostics(
        &self,
        problem: &QpProblem<S>,
        x: &[S],
    ) -> Result<crate::OptimalityDiagnostics<S>, SciOxideError> {
        let mut primal = S::ZERO;
        let mut bounds = S::ZERO;
        for row in &self.rows {
            let residual = scioxide_core::checked_difference(row.value(x)?, row.rhs)?;
            let violation = if row.equality {
                NumericScalar::abs(residual)
            } else {
                residual.max(S::ZERO)
            };
            if matches!(row.normal, Normal::General(_)) {
                primal = primal.max(violation);
            } else {
                bounds = bounds.max(violation);
            }
        }
        Ok(crate::OptimalityDiagnostics {
            primal_feasibility: Some(primal),
            bound_violation: problem.bounds().map(|_| bounds),
            ..crate::OptimalityDiagnostics::unavailable()
        })
    }

    pub(super) fn assess(
        &self,
        problem: &QpProblem<S>,
        x: &[S],
        multipliers: &[S],
        scale: S,
        t: OptimalityTolerances<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<crate::verification::candidate::Assessment<S>, SciOxideError> {
        let n = problem.dimension();
        let mut dual = crate::OwnedDualCertificate {
            rows: try_filled(
                self.general,
                S::ZERO,
                scioxide_core::quantity!("qp_row_duals"),
            )?,
            lower: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_lower_duals"))?,
            upper: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_upper_duals"))?,
        };
        for (row, &value) in self.rows.iter().zip(multipliers) {
            let value = scioxide_core::checked_float_product(value, scale)?;
            match row.source {
                Source::Row(i) => dual.rows[i] = value * row.sign,
                Source::Lower(i) => dual.lower[i] = value,
                Source::Upper(i) => dual.upper[i] = value,
                Source::Fixed(i) => {
                    dual.lower[i] = (-value).max(S::ZERO);
                    dual.upper[i] = value.max(S::ZERO);
                }
            }
        }
        let evidence = crate::verification::OwnedProgrammingCertificate::Optimality(dual);
        let view = crate::QpVerificationProblem::from_problem(problem);
        let mut workspace = crate::VerificationWorkspace::for_qp(&view)?;
        let report = match crate::verification::kernel::qp_session(
            &view,
            evidence.borrow(x),
            t,
            &mut workspace,
            Some(session),
        ) {
            Ok(report) => report,
            Err(_) if session.stopped().is_some() => {
                return Ok(crate::verification::candidate::Assessment::unavailable());
            }
            Err(error) => return Err(error),
        };
        Ok(crate::verification::candidate::Assessment::new(
            evidence, report,
        ))
    }
}

pub(super) fn validate_warm<S: RealScalar + NumericScalar<Real = S>>(
    p: &QpProblem<S>,
    x: &[S],
    t: OptimalityTolerances<S>,
) -> Result<(), SciOxideError> {
    if x.len() != p.dimension() {
        return Err(SciOxideError::dimension_mismatch(
            [p.dimension()],
            [x.len()],
        ));
    }
    if !Model::new(p)?.feasible(x, t)? {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("self.warm_start"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "initial point is not feasible",
        ));
    }
    Ok(())
}
