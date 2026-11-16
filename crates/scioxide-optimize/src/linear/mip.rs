//! Bound-only relaxation state for branch and bound. The matrix and its
//! coordinate frame are prepared once; nodes only replace lower and upper bounds.
use super::{
    SimplexSolver,
    basis::Basis,
    simplex::basis_dual_into,
    standard_form::{StandardForm, standardize_bounded},
};
use crate::{
    LpProblem, OptimizeTermination, OwnedDualCertificate, OwnedProgrammingCertificate,
    ProgrammingResult, context::SolveSession,
};
use alloc::vec::Vec;
use scioxide_core::backend::{try_copy_slice, try_filled, try_with_capacity};
use scioxide_core::{
    Budget, NumericScalar, RealScalar, SciOxideError, checked_difference, checked_float_quotient,
};

/// Holds the relaxation storage that [`SimplexSolver`](crate::SimplexSolver)
/// uses as a [`MipRelaxationSolver`](crate::MipRelaxationSolver).
///
/// The objective and constraint rows are fixed when the workspace is prepared;
/// only the variable bounds change from node to node. A change of the active
/// cuts makes [`BranchAndBound`](crate::BranchAndBound) prepare a new
/// workspace. Each node reoptimizes
/// from the previous basis, which is factored by dense LU. Sparse (CSR)
/// constraint rows stay sparse.
pub struct MipSimplexWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    form: StandardForm<S>,
    basis: Basis<'static, S>,
    initialized: bool,
    reused: bool,
    root_indices: Vec<usize>,
    root_rows: Vec<usize>,
    lower: Vec<Option<S>>,
    upper: Vec<Option<S>>,
    point: Vec<S>,
    at_upper: Vec<bool>,
    values: Vec<S>,
    rhs: Vec<S>,
    dual: Vec<S>,
    column: Vec<S>,
    direction: Vec<S>,
    reduced: Vec<S>,
    verification: crate::VerificationWorkspace<S>,
    spare_duals: Vec<OwnedDualCertificate<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> MipSimplexWorkspace<S> {
    pub(crate) fn new(solver: &SimplexSolver<S>, p: &LpProblem<S>) -> Result<Self, SciOxideError> {
        let form = StandardForm::native(p)?;
        let m = form.rows;
        let n = form.columns;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let basis = Basis::new(&form, solver.basis_update_limit(), None, None)?;
        Ok(Self {
            form,
            basis,
            initialized: false,
            reused: false,
            root_indices: Vec::new(),
            root_rows: Vec::new(),
            lower: try_filled(n, Some(zero), scioxide_core::quantity!("mip_native_lower"))?,
            upper: try_filled(n, None, scioxide_core::quantity!("mip_native_upper"))?,
            point: try_filled(n, zero, scioxide_core::quantity!("mip_native_point"))?,
            at_upper: try_filled(n, false, scioxide_core::quantity!("mip_bound_states"))?,
            values: try_filled(m, zero, scioxide_core::quantity!("mip_basic_values"))?,
            rhs: try_filled(m, zero, scioxide_core::quantity!("mip_basis_rhs"))?,
            dual: try_filled(m, zero, scioxide_core::quantity!("mip_basis_dual"))?,
            column: try_filled(m, zero, scioxide_core::quantity!("mip_basis_column"))?,
            direction: try_filled(m, zero, scioxide_core::quantity!("mip_basis_direction"))?,
            reduced: try_filled(n, zero, scioxide_core::quantity!("mip_reduced_cost"))?,
            verification: crate::VerificationWorkspace::for_lp(
                &crate::LpVerificationProblem::from_problem(p),
            )?,
            // Search owns main and probe outputs. Interrupted candidates must
            // publish no evidence, while their payload remains reusable.
            spare_duals: try_with_capacity(2, scioxide_core::quantity!("mip_spare_evidence"))?,
        })
    }

    pub(crate) fn reused(&self) -> bool {
        self.reused
    }

    fn seed(
        &mut self,
        solver: &SimplexSolver<S>,
        p: &LpProblem<S>,
        output: &mut ProgrammingResult<S>,
        limit: Budget,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mut warm = super::SimplexWarmStart::new();
        *output = solver
            .clone()
            .with_max_iterations(limit)
            .reoptimize_in_context(p, &mut warm, context)?;
        output.try_reserve(p.dimension())?;
        if !output.is_success() {
            return Ok(());
        }
        let source = standardize_bounded(p)?;
        self.basis.indices.clear();
        for &column in warm.basis_indices() {
            self.basis
                .indices
                .push(source.original_column(column).ok_or_else(|| {
                    crate::validation::invalid(
                        scioxide_core::argument!("basis"),
                        "missing original column",
                    )
                })?);
        }
        self.basis.rows.clear();
        self.basis.rows.extend_from_slice(warm.retained_rows());
        self.basis.artificial_rows.clear();
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        self.basis.refactor(&self.form, &session)?;
        self.complete_rows(solver, &session)?;
        self.root_indices = try_copy_slice(
            &self.basis.indices,
            scioxide_core::quantity!("mip_root_basis"),
        )?;
        self.root_rows =
            try_copy_slice(&self.basis.rows, scioxide_core::quantity!("mip_root_rows"))?;
        self.initialized = true;
        Ok(())
    }

    // Cold standardization can eliminate fixed columns and consequently drop a
    // row that is independent in native coordinates. Complete the retained
    // basis by Schur complements before bounds can be relaxed again (siblings
    // and temporary repairs). Truly dependent rows stay omitted.
    fn complete_rows(
        &mut self,
        solver: &SimplexSolver<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        for row in 0..self.form.rows {
            if self.basis.rows.contains(&row) {
                continue;
            }
            let m = self.basis.rows.len();
            let mut chosen = None;
            let mut best = zero;
            for column in 0..self.form.columns {
                if self.basis.contains(column) {
                    continue;
                }
                self.basis.column(&self.form, column, &mut self.column[..m]);
                self.basis
                    .solve(&self.column[..m], &mut self.direction[..m], false, session)?;
                let a = self.form.coefficient(row, column);
                let mut residual = scioxide_core::ProductSum::new();
                residual.add_product(a, one)?;
                let mut scale = NumericScalar::abs(a);
                for (&j, &d) in self.basis.indices.iter().zip(&self.direction[..m]) {
                    let coefficient = self.form.coefficient(row, j);
                    residual.add_product(-coefficient, d)?;
                    scale = scale.max(NumericScalar::abs(coefficient));
                }
                let pivot = NumericScalar::abs(residual.finish()?);
                if pivot > best && pivot > solver.tolerances().pivot().threshold(scale)? {
                    chosen = Some(column);
                    best = pivot;
                }
            }
            if let Some(column) = chosen {
                self.basis.indices.push(column);
                self.basis.rows.push(row);
                self.basis.refactor(&self.form, session)?;
            }
        }
        Ok(())
    }

    pub(crate) fn solve_into(
        &mut self,
        solver: &SimplexSolver<S>,
        p: &LpProblem<S>,
        output: &mut ProgrammingResult<S>,
        limit: Budget,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if p.dimension() != self.form.transformed_variables
            || p.constraints().map_or(0, |c| c.len()) != self.form.rows
        {
            return Err(crate::validation::invalid(
                scioxide_core::argument!("problem"),
                "relaxation shape differs from its prepared workspace",
            ));
        }
        self.reused = self.initialized;
        if !self.initialized {
            return self.seed(solver, p, output, limit, context);
        }
        output.try_reserve(p.dimension())?;
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let n = self.form.columns;
        let original = p.dimension();
        for j in 0..original {
            let (lower, upper) = p.bounds().map_or((Some(zero), None), |b| b.as_slice()[j]);
            let encode = |v: S| -> Result<S, SciOxideError> {
                let scaled = v.scalbn(-self.form.column_exponents[j]);
                if !scaled.is_finite() || (v != zero && scaled == zero) {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Overflow,
                        "native bound scaling lost an endpoint",
                    ));
                }
                Ok(scaled)
            };
            self.lower[j] = lower.map(encode).transpose()?;
            self.upper[j] = upper.map(encode).transpose()?;
        }
        let cost_scale = self.form.objective.iter().fold(zero, |s, c| s.max(c.abs()));
        let cost_tol = solver.tolerances().reduced_cost().threshold(cost_scale)?;
        let feasibility = solver
            .tolerances()
            .feasibility()
            .threshold(self.form.rhs.iter().fold(one, |s, b| s.max(b.abs())))?;
        let pivot_tol = solver.tolerances().pivot().threshold(one)?;
        let mut iterations = 0;
        let mut restored_root = false;
        let mut stalls = 0;
        let mut bland = false;
        let mut refresh = true;
        output.verification = None;
        output.diagnostics = crate::OptimalityDiagnostics::unavailable();
        output.trust = scioxide_core::PartialResultTrust::DiagnosticOnly;
        output.iterations = 0;
        loop {
            session.check()?;
            let fresh = refresh;
            let m = self.basis.rows.len();
            for (target, &j) in self.rhs[..m].iter_mut().zip(&self.basis.indices) {
                *target = self.form.objective[j];
            }
            if fresh {
                self.basis.solve_checked(
                    &self.form,
                    &self.rhs[..m],
                    &mut self.dual[..m],
                    true,
                    &session,
                )?;
            } else {
                self.basis
                    .solve(&self.rhs[..m], &mut self.dual[..m], true, &session)?;
            }
            let mut admissible = true;
            for j in 0..n {
                self.reduced[j] = checked_difference(
                    self.form.objective[j],
                    self.basis.column_dot(&self.form, j, &self.dual[..m])?,
                )?;
                self.point[j] = if self.basis.contains(j) {
                    zero
                } else if self.reduced[j] > cost_tol {
                    if let Some(l) = self.lower[j] {
                        l
                    } else {
                        admissible &= self.reduced[j] <= cost_tol;
                        self.upper[j].unwrap_or(zero)
                    }
                } else if self.reduced[j] < -cost_tol {
                    if let Some(u) = self.upper[j] {
                        u
                    } else {
                        admissible = false;
                        self.lower[j].unwrap_or(zero)
                    }
                } else if self.at_upper[j] && self.upper[j].is_some() {
                    self.upper[j].unwrap_or(zero)
                } else {
                    self.lower[j].or(self.upper[j]).unwrap_or(zero)
                };
            }
            if !admissible {
                if restored_root {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "retained basis lost dual feasibility",
                    ));
                }
                self.basis.indices.clear();
                self.basis.indices.extend_from_slice(&self.root_indices);
                self.basis.rows.clear();
                self.basis.rows.extend_from_slice(&self.root_rows);
                self.basis.refactor(&self.form, &session)?;
                restored_root = true;
                refresh = true;
                continue;
            }
            for (target, &row) in self.rhs[..m].iter_mut().zip(&self.basis.rows) {
                let mut sum = scioxide_core::ProductSum::new();
                sum.add_product(self.form.rhs[row], one)?;
                for (j, a) in self.form.row_entries(row) {
                    if !self.basis.contains(j) {
                        sum.add_product(-a, self.point[j])?;
                    }
                }
                *target = sum.finish()?;
            }
            if fresh {
                self.basis.solve_checked(
                    &self.form,
                    &self.rhs[..m],
                    &mut self.values[..m],
                    false,
                    &session,
                )?;
            } else {
                self.basis
                    .solve(&self.rhs[..m], &mut self.values[..m], false, &session)?;
            }
            let mut leaving: Option<(usize, S, S)> = None;
            for (i, (&j, &v)) in self.basis.indices.iter().zip(&self.values[..m]).enumerate() {
                self.point[j] = v;
                let (violation, sign) = if self.lower[j].is_some_and(|l| v < l - feasibility) {
                    (self.lower[j].unwrap_or(zero) - v, one)
                } else if self.upper[j].is_some_and(|u| v > u + feasibility) {
                    (v - self.upper[j].unwrap_or(zero), -one)
                } else {
                    continue;
                };
                if leaving.is_none_or(|(k, best, _)| {
                    if bland {
                        j < self.basis.indices[k]
                    } else {
                        violation > best
                    }
                }) {
                    leaving = Some((i, violation, sign));
                }
            }
            self.form
                .reconstruct_solution_into(&self.point, &mut output.x)?;
            output.fun = crate::utils::dot(p.objective(), &output.x)?;
            output.iterations = iterations;
            let Some((row, violation, row_sign)) = leaving else {
                if !fresh {
                    refresh = true;
                    continue;
                }
                return self.finish(solver, p, output, OptimizeTermination::Converged, &session);
            };
            if iterations >= limit.get() {
                output.termination = OptimizeTermination::IterationLimit;
                self.retain_evidence(output);
                output.message.clear();
                output
                    .message
                    .push_str("native-bound dual simplex pivot limit");
                output.counts = session.accounting()?;
                return Ok(());
            }
            self.rhs[..m].fill(zero);
            self.rhs[row] = row_sign;
            self.basis
                .solve(&self.rhs[..m], &mut self.direction[..m], true, &session)?;
            let eligible = |j: usize, sign: S| {
                if self.basis.contains(j)
                    || self.lower[j] == self.upper[j] && self.lower[j].is_some()
                {
                    return false;
                }
                if sign > zero {
                    self.upper[j].is_none_or(|u| self.point[j] < u)
                } else {
                    self.lower[j].is_none_or(|l| self.point[j] > l)
                }
            };
            let mut relaxed: Option<S> = None;
            for j in 0..n {
                for sign in [one, -one] {
                    if !eligible(j, sign) {
                        continue;
                    }
                    let a = self.basis.column_dot(&self.form, j, &self.direction[..m])? * sign;
                    if a < -pivot_tol {
                        let r = checked_float_quotient(
                            (self.reduced[j] * sign).max(zero)
                                + if bland { zero } else { cost_tol },
                            -a,
                        )?;
                        relaxed = Some(relaxed.map_or(r, |old| old.min(r)));
                    }
                }
            }
            let mut chosen: Option<(usize, S, S, S)> = None;
            if let Some(relaxed) = relaxed {
                for j in 0..n {
                    for sign in [one, -one] {
                        if !eligible(j, sign) {
                            continue;
                        }
                        let a = self.basis.column_dot(&self.form, j, &self.direction[..m])? * sign;
                        if a >= -pivot_tol {
                            continue;
                        }
                        let ratio = checked_float_quotient((self.reduced[j] * sign).max(zero), -a)?;
                        if ratio <= relaxed
                            && chosen.is_none_or(|(k, _, best_ratio, strength)| {
                                if bland {
                                    ratio < best_ratio || ratio == best_ratio && j < k
                                } else {
                                    -a > strength || -a == strength && j < k
                                }
                            })
                        {
                            chosen = Some((j, sign, ratio, -a));
                        }
                    }
                }
            }
            let Some((j, _, ratio, _)) = chosen else {
                if !fresh {
                    self.basis.refactor(&self.form, &session)?;
                    refresh = true;
                    continue;
                }
                // A fresh factor rules out update drift before a Farkas claim.
                self.basis.refactor(&self.form, &session)?;
                self.rhs[..m].fill(zero);
                self.rhs[row] = -row_sign;
                self.basis.solve_checked(
                    &self.form,
                    &self.rhs[..m],
                    &mut self.dual[..m],
                    true,
                    &session,
                )?;
                return self.finish(solver, p, output, OptimizeTermination::Infeasible, &session);
            };
            if let Some(stop) = session.trial_iteration()? {
                output.termination = stop;
                self.retain_evidence(output);
                output.message.clear();
                output
                    .message
                    .push_str("native-bound dual simplex resource stop");
                output.counts = session.accounting()?;
                return Ok(());
            }
            self.basis.column(&self.form, j, &mut self.column[..m]);
            self.basis
                .solve(&self.column[..m], &mut self.direction[..m], false, &session)?;
            self.at_upper[self.basis.indices[row]] = row_sign < zero;
            self.at_upper[j] = false;
            let generation = self.basis.generation;
            self.basis
                .replace(row, j, &self.direction[..m], &self.form, &session)?;
            refresh = self.basis.generation != generation;
            super::simplex::mip_progress(
                &mut stalls,
                &mut bland,
                violation,
                ratio * violation,
                feasibility,
                cost_tol,
            );
            session.accept(scioxide_core::PartialResultTrust::DiagnosticOnly)?;
            iterations += 1;
        }
    }

    fn finish(
        &mut self,
        solver: &SimplexSolver<S>,
        p: &LpProblem<S>,
        output: &mut ProgrammingResult<S>,
        termination: OptimizeTermination,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let mut evidence = match output.evidence.take() {
            Some(
                OwnedProgrammingCertificate::Optimality(d)
                | OwnedProgrammingCertificate::Infeasibility(d),
            ) => d,
            _ => self
                .spare_duals
                .pop()
                .unwrap_or_else(|| OwnedDualCertificate::new(Vec::new(), Vec::new(), Vec::new())),
        };
        crate::storage::resize(
            &mut evidence.rows,
            p.constraints().map_or(0, |c| c.len()),
            zero,
        )?;
        crate::storage::resize(&mut evidence.lower, p.dimension(), zero)?;
        crate::storage::resize(&mut evidence.upper, p.dimension(), zero)?;
        basis_dual_into(
            p,
            &self.form,
            &self.basis.rows,
            &self.dual[..self.basis.rows.len()],
            termination == OptimizeTermination::Converged,
            &mut evidence.rows,
        )?;
        evidence.lower.fill(zero);
        evidence.upper.fill(zero);
        for j in 0..p.dimension() {
            let mut sum = scioxide_core::ProductSum::new();
            if termination == OptimizeTermination::Converged {
                sum.add_product(p.objective()[j], <S as scioxide_core::AlgebraicScalar>::ONE)?;
            }
            if let Some(c) = p.constraints() {
                for (row, &y) in evidence.rows.iter().enumerate() {
                    sum.add_product(c.coefficients().get(row, j)?, y)?;
                }
            }
            let reduced = sum.finish()?;
            let (l, u) = p.bounds().map_or((Some(zero), None), |b| b.as_slice()[j]);
            let active = |bound: Option<S>| -> Result<bool, SciOxideError> {
                match bound {
                    None => Ok(false),
                    Some(b) => Ok(termination == OptimizeTermination::Infeasible
                        || NumericScalar::abs(checked_difference(output.x[j], b)?)
                            <= solver.tolerances().feasibility().threshold(
                                NumericScalar::abs(output.x[j]).max(NumericScalar::abs(b)),
                            )?),
                }
            };
            if reduced >= zero && active(l)? {
                evidence.lower[j] = reduced;
            } else if reduced <= zero && active(u)? {
                evidence.upper[j] = -reduced;
            }
        }
        output.evidence = Some(if termination == OptimizeTermination::Infeasible {
            OwnedProgrammingCertificate::Infeasibility(evidence)
        } else {
            OwnedProgrammingCertificate::Optimality(evidence)
        });
        let view = crate::LpVerificationProblem::from_problem(p);
        let report = crate::verification::kernel::interrupted(
            crate::verification::kernel::lp_session(
                &view,
                output.certificate().ok_or_else(|| {
                    crate::validation::invalid(
                        scioxide_core::argument!("evidence"),
                        "missing retained evidence",
                    )
                })?,
                solver.tolerances(),
                &mut self.verification,
                Some(session),
            ),
            session,
        )?;
        output.diagnostics = report.diagnostics();
        output.termination = report.termination().unwrap_or(if report.is_accepted() {
            termination
        } else if termination == OptimizeTermination::Infeasible {
            OptimizeTermination::SuspectedInfeasible
        } else {
            OptimizeTermination::Stagnation
        });
        output.verification = Some(report);
        output.message.clear();
        output
            .message
            .push_str(if termination == OptimizeTermination::Infeasible {
                "native-bound dual simplex infeasibility witness"
            } else {
                "native-bound dual simplex optimum"
            });
        output.counts = session.accounting()?;
        output.algorithm = Some(session.algorithm()?);
        output.trust = if output.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        };
        session.assess(output.trust)?;
        Ok(())
    }

    fn retain_evidence(&mut self, output: &mut ProgrammingResult<S>) {
        if let Some(
            OwnedProgrammingCertificate::Optimality(dual)
            | OwnedProgrammingCertificate::Infeasibility(dual),
        ) = output.evidence.take()
            && self.spare_duals.len() < self.spare_duals.capacity()
        {
            self.spare_duals.push(dual);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {}
