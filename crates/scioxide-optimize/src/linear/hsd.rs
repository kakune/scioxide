//! Homogeneous self-dual predictor-corrector LP embedding.
use super::standard_form::StandardForm;
use crate::context::SolveSession;
use crate::utils::{dot, trial_component};
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{
    checked_float_product as multiply, checked_float_quotient as divide, checked_sum,
};
use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace, Matrix};

/// Refine a candidate on its active nonnegative face using a rank-revealing correction.
/// This is certificate preparation, not an iteration or an auxiliary LP solve.
pub(super) fn polish_point<S: RealScalar + NumericScalar<Real = S>>(
    form: &StandardForm<S>,
    point: &mut [S],
    rank: scioxide_linalg::RankPolicy<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let columns: Vec<_> = point
        .iter()
        .enumerate()
        .filter_map(|(j, &x)| (x > zero).then_some(j))
        .collect();
    if columns.is_empty() || columns.len() > form.rows {
        return Ok(());
    }
    let mut matrix = Matrix::zeros(form.rows, columns.len())?;
    for (col, &j) in columns.iter().enumerate() {
        for (row, value) in form.column_entries(j) {
            matrix.view_mut().set(row, col, value)?;
        }
    }
    let options = LinalgExecutionOptions::new();
    let mut factor = scioxide_linalg::SvdDecomposition::empty();
    let mut work = LinalgWorkspace::new();
    session.invoke(scioxide_core::EvaluationRole::Factorization, || {
        scioxide_linalg::SvdDecomposition::compute_into(
            &matrix,
            &options,
            &mut factor,
            &mut work,
            session.execution()?,
        )
    })?;
    let mut solution = scioxide_linalg::Vector::zeros(columns.len())?;
    let mut residual = try_filled(
        form.rows,
        zero,
        scioxide_core::quantity!("hsd_face_residual"),
    )?;
    for _ in 0..3 {
        for (i, value) in residual.iter_mut().enumerate() {
            *value = checked_sum(&[form.rhs[i], -form.row_dot(i, point)?])?;
        }
        session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
            factor.solve_into(
                scioxide_linalg::VectorView::from_slice(&residual),
                rank,
                zero,
                &options,
                solution.view_mut(),
                &mut work,
                session.execution()?,
            )
        })?;
        for (&j, &value) in columns.iter().zip(solution.as_slice()) {
            point[j] = checked_sum(&[point[j], value])?;
        }
    }
    Ok(())
}

/// Eliminate the diagonal block of `[-D⁻¹ Aᵀ; A 0]` without an augmented matrix.
#[allow(clippy::too_many_arguments)]
fn reduced_solve_once<S: RealScalar + NumericScalar<Real = S>>(
    factor: &mut crate::kkt::Factor<'_, S>,
    matrix: &Matrix<S>,
    form: &StandardForm<S>,
    weights: &[S],
    rhs: &[S],
    output: &mut [S],
    reduced_rhs: &mut [S],
    dual: &mut [S],
    session: &SolveSession<'_, '_, '_>,
    provider: Option<&mut dyn crate::KktProvider<S>>,
) -> Result<(), SciOxideError> {
    let n = form.columns;
    for (j, target) in output[..n].iter_mut().enumerate() {
        *target = multiply(weights[j], rhs[j])?;
    }
    for (i, target) in reduced_rhs.iter_mut().enumerate() {
        *target = checked_sum(&[
            rhs[n + i],
            crate::utils::compensated_products(|| {
                form.row_entries(i).map(|(j, a)| (a, output[j]))
            })?,
        ])?;
    }
    factor.solve_lent(provider, matrix, reduced_rhs, dual, session)?;
    for (j, target) in output[..n].iter_mut().enumerate() {
        *target = multiply(
            weights[j],
            checked_sum(&[
                -rhs[j],
                crate::utils::compensated_products(|| {
                    form.column_entries(j).map(|(i, a)| (a, dual[i]))
                })?,
            ])?,
        )?;
    }
    output[n..].copy_from_slice(dual);
    Ok(())
}

/// Refine in the original saddle equations, whose conditioning is better than
/// the normal equations. Corrections reuse the same reduced factorization.
#[allow(clippy::too_many_arguments)]
fn reduced_solve<S: RealScalar + NumericScalar<Real = S>>(
    factor: &mut crate::kkt::Factor<'_, S>,
    matrix: &Matrix<S>,
    form: &StandardForm<S>,
    weights: &[S],
    rhs: &[S],
    output: &mut [S],
    reduced_rhs: &mut [S],
    dual: &mut [S],
    residual: &mut [S],
    correction: &mut [S],
    session: &SolveSession<'_, '_, '_>,
    mut provider: Option<&mut dyn crate::KktProvider<S>>,
) -> Result<(), SciOxideError> {
    reduced_solve_once(
        factor,
        matrix,
        form,
        weights,
        rhs,
        output,
        reduced_rhs,
        dual,
        session,
        crate::kkt::lend(&mut provider),
    )?;
    let n = form.columns;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for attempt in 0..=3 {
        let mut defect = zero;
        let mut scale = zero;
        let output_norm = output.iter().fold(zero, |v, &x| v.max(x.abs()));
        let mut matrix_norm = zero;
        for j in 0..n {
            let diagonal = divide(output[j], weights[j])?;
            residual[j] = crate::utils::compensated_products(|| {
                [(rhs[j], one), (diagonal, one)]
                    .into_iter()
                    .chain(form.column_entries(j).map(|(i, a)| (-a, output[n + i])))
            })?;
            matrix_norm = matrix_norm.max(
                form.column_entries(j)
                    .fold(divide(one, weights[j])?, |v, (_, a)| v + a.abs()),
            );
            defect = defect.max(residual[j].abs());
            scale = scale.max(rhs[j].abs()).max(diagonal.abs());
        }
        for i in 0..form.rows {
            residual[n + i] = checked_sum(&[
                rhs[n + i],
                -crate::utils::compensated_products(|| {
                    form.row_entries(i).map(|(j, a)| (a, output[j]))
                })?,
            ])?;
            defect = defect.max(residual[n + i].abs());
            scale = scale.max(rhs[n + i].abs());
            matrix_norm = matrix_norm.max(form.row_entries(i).fold(zero, |v, (_, a)| v + a.abs()));
        }
        let normwise_threshold = scioxide_core::Scaled::from_value(matrix_norm)?
            .mul(scioxide_core::Scaled::from_value(output_norm)?)?
            .add(scioxide_core::Scaled::from_value(scale)?)?
            .mul(scioxide_core::Scaled::from_value(S::SQRT_EPSILON)?)?;
        let normwise_ok = scioxide_core::Scaled::from_value(defect)?.cmp_value(normwise_threshold)
            != core::cmp::Ordering::Greater;
        let threshold = multiply(S::EPSILON, scale)?;
        if defect <= threshold || ((S::MANTISSA_DIGITS > 24 || attempt == 3) && normwise_ok) {
            return Ok(());
        }
        if attempt == 3 {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "reduced LP direction failed original Newton residual check",
            ));
        }
        reduced_solve_once(
            factor,
            matrix,
            form,
            weights,
            residual,
            correction,
            reduced_rhs,
            dual,
            session,
            crate::kkt::lend(&mut provider),
        )?;
        session.cache_hit()?;
        for (value, &delta) in output.iter_mut().zip(correction.iter()) {
            *value = checked_sum(&[*value, delta])?;
        }
    }
    unreachable!("bounded refinement returns on its last attempt")
}

/// Embedding iterates and factor scratch; the KKT provider is borrowed per call.
pub(super) struct Embedding<S: RealScalar + NumericScalar<Real = S>> {
    factor: crate::kkt::Factor<'static, S>,
    pub x: Vec<S>,
    pub y: Vec<S>,
    pub s: Vec<S>,
    pub tau: S,
    pub kappa: S,
    pub rp: Vec<S>,
    pub rd: Vec<S>,
    pub gap: S,
    dx: Vec<S>,
    dy: Vec<S>,
    ds: Vec<S>,
    dt: S,
    dk: S,
    affine_x: Vec<S>,
    affine_s: Vec<S>,
    affine_t: S,
    affine_k: S,
    rhs: Vec<S>,
    solution: Vec<S>,
    border: Vec<S>,
    border_solution: Vec<S>,
    matrix: Matrix<S>,
    weighted: Matrix<S>,
    gram: Matrix<S>,
    weights: Vec<S>,
    reduced_rhs: Vec<S>,
    reduced_solution: Vec<S>,
    work: LinalgWorkspace<'static, S>,
    residual: Vec<S>,
    correction: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Embedding<S> {
    pub(super) fn new(form: &StandardForm<S>) -> Result<Self, SciOxideError> {
        let n = form.columns;
        let m = form.rows;
        let order = checked_add(n, m, scioxide_core::quantity!("hsd_kkt_order"))?;
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let o = <S as scioxide_core::AlgebraicScalar>::ONE;
        let vector = |len, value| try_filled(len, value, scioxide_core::quantity!("hsd_workspace"));
        let mut work = LinalgWorkspace::new();
        work.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &scioxide_linalg::DenseWorkspaceRequest::product(
                m,
                n,
                m,
                &LinalgExecutionOptions::new(),
            ),
        )?)?;
        Ok(Self {
            factor: crate::kkt::Factor::new(None, m, None)?.with_normwise_residual(),
            x: vector(n, o)?,
            y: vector(m, z)?,
            s: vector(n, o)?,
            tau: o,
            kappa: o,
            rp: vector(m, z)?,
            rd: vector(n, z)?,
            gap: z,
            dx: vector(n, z)?,
            dy: vector(m, z)?,
            ds: vector(n, z)?,
            dt: z,
            dk: z,
            affine_x: vector(n, z)?,
            affine_s: vector(n, z)?,
            affine_t: z,
            affine_k: z,
            rhs: vector(order, z)?,
            solution: vector(order, z)?,
            border: vector(order, z)?,
            border_solution: vector(order, z)?,
            matrix: Matrix::zeros(m, m)?,
            weighted: Matrix::zeros(m, n)?,
            gram: Matrix::zeros(m, m)?,
            weights: vector(n, z)?,
            reduced_rhs: vector(m, z)?,
            reduced_solution: vector(m, z)?,
            work,
            residual: vector(order, z)?,
            correction: vector(order, z)?,
        })
    }
    /// Serve every factorization through a borrowed provider instead of the built-in one.
    pub(super) fn release_builtin(&mut self) {
        self.factor.release_builtin();
    }
    pub(super) fn reset(&mut self) {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        self.x.fill(one);
        self.y.fill(zero);
        self.s.fill(one);
        self.tau = one;
        self.kappa = one;
        self.rp.fill(zero);
        self.rd.fill(zero);
        self.gap = zero;
    }
    pub(super) fn residuals(&mut self, f: &StandardForm<S>) -> Result<(), SciOxideError> {
        for (i, r) in self.rp.iter_mut().enumerate() {
            *r = trial_component(
                -crate::utils::compensated_products(|| {
                    f.row_entries(i).map(|(j, a)| (a, self.x[j]))
                })?,
                self.tau,
                f.rhs[i],
            )?;
        }
        for (j, r) in self.rd.iter_mut().enumerate() {
            *r = crate::utils::compensated_products(|| {
                [
                    (f.objective[j], self.tau),
                    (-self.s[j], <S as scioxide_core::AlgebraicScalar>::ONE),
                ]
                .into_iter()
                .chain(f.column_entries(j).map(|(i, a)| (-a, self.y[i])))
            })?;
        }
        self.gap = checked_sum(&[
            dot(&f.objective, &self.x)?,
            -dot(&f.rhs, &self.y)?,
            self.kappa,
        ])?;
        Ok(())
    }
    fn complementarity(&self) -> Result<S, SciOxideError> {
        divide(
            checked_sum(&[dot(&self.x, &self.s)?, multiply(self.tau, self.kappa)?])?,
            S::try_from_usize_exact(self.x.len() + 1)?,
        )
    }
    fn factor(
        &mut self,
        f: &StandardForm<S>,
        session: &SolveSession<'_, '_, '_>,
        mut provider: Option<&mut dyn crate::KktProvider<S>>,
    ) -> Result<(), SciOxideError> {
        let n = f.columns;
        let m = f.rows;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        self.weighted.as_mut_slice().fill(zero);
        for j in 0..n {
            self.weights[j] = divide(self.x[j], self.s[j])?;
            let root = RealScalar::sqrt(self.weights[j]);
            for (i, a) in f.column_entries(j) {
                self.weighted.as_mut_slice()[i * n + j] = multiply(a, root)?;
            }
        }
        scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
            self.weighted.view(),
            scioxide_core::Triangle::Lower,
            scioxide_linalg::StructuredOperation::Normal,
            one,
            zero,
            &LinalgExecutionOptions::new(),
            self.gram.view_mut(),
            &mut self.work,
            session.execution()?,
        )?;
        let mut assembly = crate::kkt::assembly::Assembly::new(&mut self.matrix, m)?;
        assembly.symmetric(0, m, |i, j| self.gram.view().get(i, j))?;
        self.border[..n].copy_from_slice(&f.objective);
        self.border[n..].copy_from_slice(&f.rhs);
        let scale = self
            .matrix
            .as_slice()
            .iter()
            .fold(zero, |a, &v| a.max(NumericScalar::abs(v)));
        // An ordinary full-rank system has no shift. Rank loss gets at most
        // six relative shifts, starting at one rounding unit of the Gram scale.
        let mut shift = zero;
        for attempt in 0..=6 {
            let inertia =
                self.factor
                    .factor_lent(crate::kkt::lend(&mut provider), &self.matrix, session)?;
            if crate::kkt::matches_inertia(inertia, m, 0)? {
                break;
            }
            if attempt == 6 || scale == zero {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "regularized LP normal system has unexpected inertia",
                ));
            }
            shift = if attempt == 0 {
                multiply(S::EPSILON, scale)?
            } else {
                multiply(shift, S::TWO)?
            };
            let mut assembly = crate::kkt::assembly::Assembly::retained(&mut self.matrix);
            assembly.diagonal(0, m, |i| checked_sum(&[self.gram.view().get(i, i)?, shift]))?;
        }
        reduced_solve(
            &mut self.factor,
            &self.matrix,
            f,
            &self.weights,
            &self.border,
            &mut self.border_solution,
            &mut self.reduced_rhs,
            &mut self.reduced_solution,
            &mut self.residual,
            &mut self.correction,
            session,
            provider,
        )
    }

    fn direction(
        &mut self,
        f: &StandardForm<S>,
        sigma: S,
        mu: S,
        corrector: bool,
        session: &SolveSession<'_, '_, '_>,
        provider: Option<&mut dyn crate::KktProvider<S>>,
    ) -> Result<(), SciOxideError> {
        let n = f.columns;
        let o = <S as scioxide_core::AlgebraicScalar>::ONE;
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let gamma = o - sigma;
        for j in 0..n {
            let correction = if corrector {
                multiply(self.affine_x[j], self.affine_s[j])?
            } else {
                z
            };
            let rc = checked_sum(&[
                multiply(sigma, mu)?,
                -multiply(self.x[j], self.s[j])?,
                -correction,
            ])?;
            self.ds[j] = rc;
            self.rhs[j] = checked_sum(&[multiply(gamma, self.rd[j])?, -divide(rc, self.x[j])?])?;
        }
        for i in 0..f.rows {
            self.rhs[n + i] = multiply(gamma, self.rp[i])?;
        }
        reduced_solve(
            &mut self.factor,
            &self.matrix,
            f,
            &self.weights,
            &self.rhs,
            &mut self.solution,
            &mut self.reduced_rhs,
            &mut self.reduced_solution,
            &mut self.residual,
            &mut self.correction,
            session,
            provider,
        )?;
        session.cache_hit()?;
        let correction = if corrector {
            multiply(self.affine_t, self.affine_k)?
        } else {
            z
        };
        let rt = checked_sum(&[
            multiply(sigma, mu)?,
            -multiply(self.tau, self.kappa)?,
            -correction,
        ])?;
        let v_solution = checked_sum(&[
            -dot(&f.objective, &self.solution[..n])?,
            dot(&f.rhs, &self.solution[n..])?,
        ])?;
        let v_border = checked_sum(&[
            -dot(&f.objective, &self.border_solution[..n])?,
            dot(&f.rhs, &self.border_solution[n..])?,
        ])?;
        let denominator = checked_sum(&[divide(self.kappa, self.tau)?, v_border])?;
        self.dt = divide(
            checked_sum(&[
                multiply(gamma, self.gap)?,
                divide(rt, self.tau)?,
                -v_solution,
            ])?,
            denominator,
        )?;
        self.dk = divide(trial_component(rt, -self.kappa, self.dt)?, self.tau)?;
        for j in 0..n {
            self.dx[j] = trial_component(self.solution[j], self.dt, self.border_solution[j])?;
            self.ds[j] = divide(
                trial_component(self.ds[j], -self.s[j], self.dx[j])?,
                self.x[j],
            )?;
        }
        for i in 0..f.rows {
            self.dy[i] =
                trial_component(self.solution[n + i], self.dt, self.border_solution[n + i])?;
        }
        Ok(())
    }
    fn step_length(&self) -> Result<S, SciOxideError> {
        let mut alpha = <S as scioxide_core::AlgebraicScalar>::ONE;
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        for (v, d) in self
            .x
            .iter()
            .zip(&self.dx)
            .chain(self.s.iter().zip(&self.ds))
            .chain([(&self.tau, &self.dt), (&self.kappa, &self.dk)])
        {
            if *d < z {
                alpha = alpha.min(divide(-*v, *d)?);
            }
        }
        Ok(alpha)
    }
    pub(super) fn advance(
        &mut self,
        f: &StandardForm<S>,
        session: &SolveSession<'_, '_, '_>,
        mut provider: Option<&mut dyn crate::KktProvider<S>>,
    ) -> Result<(), SciOxideError> {
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let o = <S as scioxide_core::AlgebraicScalar>::ONE;
        let mu = self.complementarity()?;
        self.factor(f, session, crate::kkt::lend(&mut provider))?;
        self.direction(f, z, mu, false, session, crate::kkt::lend(&mut provider))?;
        let alpha = self.step_length()?;
        let mut sum = scioxide_core::ProductSum::new();
        for j in 0..f.columns {
            sum.add_product(
                trial_component(self.x[j], alpha, self.dx[j])?,
                trial_component(self.s[j], alpha, self.ds[j])?,
            )?;
        }
        sum.add_product(
            trial_component(self.tau, alpha, self.dt)?,
            trial_component(self.kappa, alpha, self.dk)?,
        )?;
        let affine_mu = divide(sum.finish()?, S::try_from_usize_exact(f.columns + 1)?)?;
        let ratio = divide(affine_mu.max(z), mu)?.min(o);
        let sigma = multiply(multiply(ratio, ratio)?, ratio)?;
        self.affine_x.copy_from_slice(&self.dx);
        self.affine_s.copy_from_slice(&self.ds);
        self.affine_t = self.dt;
        self.affine_k = self.dk;
        self.direction(f, sigma, mu, true, session, provider)?;
        let mut alpha = multiply(self.step_length()?, S::try_from_f64_lossy(0.995)?)?;
        // Validate the entire trial before publication; no positivity floors.
        loop {
            let positive = self
                .x
                .iter()
                .zip(&self.dx)
                .chain(self.s.iter().zip(&self.ds))
                .chain([(&self.tau, &self.dt), (&self.kappa, &self.dk)])
                .try_fold(true, |valid, (&v, &d)| {
                    Ok::<_, SciOxideError>(valid && trial_component(v, alpha, d)? > z)
                })?;
            if positive {
                break;
            }
            alpha *= S::HALF;
            if alpha == z {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "no representable strictly positive step",
                ));
            }
        }
        for j in 0..f.columns {
            self.dx[j] = trial_component(self.x[j], alpha, self.dx[j])?;
            self.ds[j] = trial_component(self.s[j], alpha, self.ds[j])?;
        }
        for i in 0..f.rows {
            self.dy[i] = trial_component(self.y[i], alpha, self.dy[i])?;
        }
        let tau = trial_component(self.tau, alpha, self.dt)?;
        let kappa = trial_component(self.kappa, alpha, self.dk)?;
        session.accept(scioxide_core::PartialResultTrust::DiagnosticOnly)?;
        self.x.copy_from_slice(&self.dx);
        self.s.copy_from_slice(&self.ds);
        self.y.copy_from_slice(&self.dy);
        self.tau = tau;
        self.kappa = kappa;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{InteriorPointLp, LinearConstraintSense, LinearConstraints, LpProblem};

    fn form() -> StandardForm<f64> {
        let problem = LpProblem::new(alloc::vec![-1.0, -2.0])
            .unwrap()
            .with_constraints(
                LinearConstraints::new(
                    Matrix::from_vec(3, 2, alloc::vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap(),
                    alloc::vec![4.0, 3.0, 2.0],
                    alloc::vec![LinearConstraintSense::LessEqual; 3],
                )
                .unwrap(),
            )
            .unwrap();
        let mut form = super::super::standard_form::standardize(&problem).unwrap();
        form.equilibrate().unwrap();
        form
    }

    #[test]
    fn prepared_iterations_and_decoding_execute() {
        let form = form();
        let mut embedding = Embedding::new(&form).unwrap();
        let mut original = alloc::vec![0.0; 2];
        let identity =
            InteriorPointLp::default().identity(&LpProblem::new(alloc::vec![0.0, 0.0]).unwrap());
        crate::context::sequential(
            identity,
            scioxide_core::Budget::try_from(10).unwrap(),
            |context| {
                let session = SolveSession::new(context, scioxide_core::argument!("scope"));
                // Include the first factorization, not just warmed iterations.
                for _ in 0..4 {
                    embedding.residuals(&form).unwrap();
                    embedding.advance(&form, &session, None).unwrap();
                    form.reconstruct_solution_into(&embedding.x, &mut original)
                        .unwrap();
                }
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn reduced_border_matches_augmented_system() {
        let form = form();
        let mut embedding = Embedding::new(&form).unwrap();
        for (j, (x, s)) in embedding.x.iter_mut().zip(&mut embedding.s).enumerate() {
            *x = (j + 1) as f64 * 0.75;
            *s = (j + 2) as f64 * 0.5;
        }
        let identity =
            InteriorPointLp::default().identity(&LpProblem::new(alloc::vec![0.0, 0.0]).unwrap());
        crate::context::sequential(identity, scioxide_core::Budget::ONE, |context| {
            let session = SolveSession::new(context, scioxide_core::argument!("scope"));
            embedding.factor(&form, &session, None)?;
            let n = form.columns;
            let order = n + form.rows;
            let mut augmented = Matrix::zeros(order, order)?;
            let mut assembly = crate::kkt::assembly::Assembly::new(&mut augmented, order)?;
            assembly.diagonal(0, n, |j| Ok(-embedding.s[j] / embedding.x[j]))?;
            assembly.coupling(n, 0, form.rows, n, |i, j| Ok(form.coefficient(i, j)))?;
            let mut factor = crate::kkt::Factor::new(None, order, None)?.with_normwise_residual();
            let mut expected = alloc::vec![0.0; order];
            factor.solve(&augmented, &embedding.border, &mut expected, &session)?;
            for (&a, &b) in expected.iter().zip(&embedding.border_solution) {
                assert!((a - b).abs() <= 1e-11 * (1.0 + a.abs()));
            }
            // Both predictor and corrector use this same elimination, with
            // different right-hand sides and the same retained factors.
            for sign in [-1.0, 1.0] {
                for (j, value) in embedding.rhs.iter_mut().enumerate() {
                    *value = sign * (j + 1) as f64 * 0.125;
                }
                reduced_solve(
                    &mut embedding.factor,
                    &embedding.matrix,
                    &form,
                    &embedding.weights,
                    &embedding.rhs,
                    &mut embedding.solution,
                    &mut embedding.reduced_rhs,
                    &mut embedding.reduced_solution,
                    &mut embedding.residual,
                    &mut embedding.correction,
                    &session,
                    None,
                )?;
                factor.solve(&augmented, &embedding.rhs, &mut expected, &session)?;
                for (&a, &b) in expected.iter().zip(&embedding.solution) {
                    assert!((a - b).abs() <= 1e-11 * (1.0 + a.abs()));
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn relative_regularization_allocates_nothing_and_zero_slack_is_rejected() {
        let lp = LpProblem::new(alloc::vec![1.0_f64])
            .unwrap()
            .with_constraints(
                LinearConstraints::new(
                    Matrix::from_vec(2, 1, alloc::vec![1.0, 1.0]).unwrap(),
                    alloc::vec![1.0, 1.0],
                    alloc::vec![LinearConstraintSense::Equal; 2],
                )
                .unwrap(),
            )
            .unwrap();
        let mut form = super::super::standard_form::standardize(&lp).unwrap();
        form.equilibrate().unwrap();
        let mut embedding = Embedding::new(&form).unwrap();
        crate::context::sequential(
            InteriorPointLp::default().identity(&lp),
            scioxide_core::Budget::ONE,
            |context| {
                let session = SolveSession::new(context, scioxide_core::argument!("scope"));
                {
                    embedding.factor(&form, &session, None).unwrap();
                    embedding.factor(&form, &session, None).unwrap();
                };
                assert!(embedding.matrix.as_slice()[0] > embedding.gram.as_slice()[0]);
                embedding.s[0] = 0.0;
                assert!(embedding.factor(&form, &session, None).is_err());
                Ok(())
            },
        )
        .unwrap();
    }
}
