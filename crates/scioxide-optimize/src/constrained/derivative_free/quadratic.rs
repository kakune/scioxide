//! Minimum-Frobenius interpolation and Byrd--Omojokun composite trust steps.
//!
//! The normal step meets a projected Cauchy decrease in the linearized violation
//! least-squares problem. The tangential QP preserves that violation, uses the
//! model Lagrangian curvature, and enforces the remaining spherical trust region
//! through its scalar multiplier. All factors and candidate states are kept.

use super::{DerivativeFreeStorage, breakdown, dot};
use crate::constrained::sqp::{SqpSubproblem, SqpSubproblemObjective, solve_child};
use crate::context::SolveSession;
use crate::{
    DenseKktProvider, DenseSqpSubproblem, KktProvider, OptimalityTolerances, OptimizeTermination,
    SqpState,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_product, try_filled};
use scioxide_core::{
    Budget, EvaluationRole, ExecutionPolicy, FeasibilityTolerance, GradientTolerance,
    NumericScalar, RankThreshold, RealScalar, ResidualTolerance, SciOxideError, Tolerance,
};
use scioxide_linalg::{
    DirectSolveOptions, DirectSolveTermination, HouseholderDecomposition, HouseholderOrientation,
    HouseholderPivoting, HouseholderWorkspaceRequest, LinalgExecutionOptions, LinalgWorkspace,
    Matrix, RankPolicy, SvdWorkspaceRequest, ThinSvdDecomposition, VectorView, VectorViewMut,
};

pub(super) struct QuadraticWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    d: usize,
    rows: usize,
    directions: Vec<S>,
    hessians: Vec<S>,
    lagrange: Vec<S>,
    center: Vec<S>,
    previous_gradient: Vec<S>,
    axis_scales: Vec<S>,
    base_scales: Vec<S>,
    initial_radius: S,
    scale_ready: bool,
    interpolation_basis: Matrix<S>,
    interpolation_row: Vec<S>,
    linear_matrix: Matrix<S>,
    feature_matrix: Matrix<S>,
    unitary: Matrix<S>,
    reduced: Matrix<S>,
    linear_factor: HouseholderDecomposition<S>,
    linear_work: LinalgWorkspace<'static, S>,
    minimum_norm: ThinSvdDecomposition<S>,
    minimum_norm_work: LinalgWorkspace<'static, S>,
    interpolation_rhs: Vec<S>,
    null_rhs: Vec<S>,
    hessian_update: Vec<S>,
    gradient_update: Vec<S>,
    fitted: bool,
    version: usize,
    replacements: Vec<usize>,
    lagrangian: Matrix<S>,
    projection: Matrix<S>,
    product: Matrix<S>,
    projected: Matrix<S>,
    shifted: Matrix<S>,
    basis: Matrix<S>,
    scratch: Vec<S>,
    normal: Vec<S>,
    cauchy: Vec<S>,
    gradient: Vec<S>,
    values: Vec<S>,
    jacobian: Matrix<S>,
    normal_design: Matrix<S>,
    normal_target: Vec<S>,
    normal_active: Vec<bool>,
    bounds: Vec<(Option<S>, Option<S>)>,
    empty: Matrix<S>,
    provider: DenseSqpSubproblem<S>,
    curvature_provider: DenseKktProvider<S>,
    normal_state: SqpState<S>,
    tangent: SqpState<S>,
    best: SqpState<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> QuadraticWorkspace<S> {
    pub(super) fn new(d: usize, rows: usize) -> Result<Self, SciOxideError> {
        let directions =
            checked_product(&[2, d, d], scioxide_core::quantity!("quadratic_directions"))?;
        let functions = checked_add(rows, 1, scioxide_core::quantity!("quadratic_functions"))?;
        let hessians = checked_product(
            &[functions, d, d],
            scioxide_core::quantity!("quadratic_hessians"),
        )?;
        let features = checked_add(
            d,
            checked_product(
                &[
                    d,
                    checked_add(d, 1, scioxide_core::quantity!("quadratic_features"))?,
                ],
                scioxide_core::quantity!("quadratic_features"),
            )? / 2,
            scioxide_core::quantity!("quadratic_features"),
        )?;
        let lagrange = checked_product(
            &[2, d, features],
            scioxide_core::quantity!("quadratic_lagrange"),
        )?;
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        let mut linear_factor = HouseholderDecomposition::empty();
        linear_factor.try_reserve(2 * d, d, HouseholderOrientation::Qr)?;
        let mut linear_work = LinalgWorkspace::new();
        for request in [
            HouseholderWorkspaceRequest::factor(2 * d, d, HouseholderOrientation::Qr, &options),
            HouseholderWorkspaceRequest::q_matrix(2 * d, d, HouseholderOrientation::Qr, &options),
            HouseholderWorkspaceRequest::solve(2 * d, d, HouseholderOrientation::Qr, &options),
        ] {
            linear_work.try_reserve(&HouseholderDecomposition::<S>::workspace_requirements(
                &request,
            )?)?;
        }
        let reduced = Matrix::zeros(d, features - d)?;
        let mut minimum_norm_work = LinalgWorkspace::new();
        minimum_norm_work.try_reserve(&ThinSvdDecomposition::<S>::workspace_requirements(
            &SvdWorkspaceRequest::new(d, features - d, &options),
        )?)?;
        minimum_norm_work.try_reserve(&ThinSvdDecomposition::<S>::workspace_requirements(
            &SvdWorkspaceRequest::solve(d, features - d, &options),
        )?)?;
        let mut minimum_norm = ThinSvdDecomposition::empty();
        ThinSvdDecomposition::compute_into(
            &reduced,
            &options,
            &mut minimum_norm,
            &mut minimum_norm_work,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        let mut normal_state = SqpState::new();
        normal_state.reserve(d, 0)?;
        let mut tangent = SqpState::new();
        tangent.reserve(d, rows)?;
        let mut best = SqpState::new();
        best.reserve(d, rows)?;
        let vector = |n| try_filled(n, S::ZERO, scioxide_core::quantity!("quadratic_scratch"));
        Ok(Self {
            d,
            rows,
            directions: vector(directions)?,
            hessians: vector(hessians)?,
            lagrange: vector(lagrange)?,
            center: vector(d)?,
            previous_gradient: vector(d)?,
            axis_scales: vector(d)?,
            base_scales: vector(d)?,
            initial_radius: S::ONE,
            scale_ready: false,
            interpolation_basis: Matrix::zeros(2 * d, features)?,
            interpolation_row: vector(features)?,
            linear_matrix: Matrix::zeros(2 * d, d)?,
            feature_matrix: Matrix::zeros(2 * d, features - d)?,
            unitary: Matrix::zeros(2 * d, 2 * d)?,
            reduced,
            linear_factor,
            linear_work,
            minimum_norm,
            minimum_norm_work,
            interpolation_rhs: vector(2 * d)?,
            null_rhs: vector(d)?,
            hessian_update: vector(features - d)?,
            gradient_update: vector(d)?,
            fitted: false,
            version: 0,
            replacements: try_filled(
                2 * d + 1,
                0,
                scioxide_core::quantity!("quadratic_replacements"),
            )?,
            lagrangian: Matrix::zeros(d, d)?,
            projection: Matrix::zeros(d, d)?,
            product: Matrix::zeros(d, d)?,
            projected: Matrix::zeros(d, d)?,
            shifted: Matrix::zeros(d, d)?,
            basis: Matrix::zeros(d, d)?,
            scratch: vector(d)?,
            normal: vector(d)?,
            cauchy: vector(d)?,
            gradient: vector(d)?,
            values: vector(rows)?,
            jacobian: Matrix::zeros(rows, d)?,
            normal_design: Matrix::zeros(rows, d)?,
            normal_target: vector(rows)?,
            normal_active: try_filled(
                rows,
                false,
                scioxide_core::quantity!("quadratic_normal_active"),
            )?,
            bounds: try_filled(
                d,
                (None, None),
                scioxide_core::quantity!("quadratic_bounds"),
            )?,
            empty: Matrix::zeros(0, d)?,
            provider: DenseSqpSubproblem::new(d, rows, rows)?,
            curvature_provider: DenseKktProvider::new(d)?,
            normal_state,
            tangent,
            best,
        })
    }
    pub(super) fn matches(&self, d: usize, rows: usize) -> bool {
        self.d == d && self.rows == rows
    }
    pub(super) fn reset(&mut self) {
        self.hessians.fill(S::ZERO);
        self.replacements.fill(0);
        self.fitted = false;
        self.scale_ready = false;
        self.version = 0;
        self.tangent.multipliers.fill(S::ZERO);
        self.tangent.lower.fill(S::ZERO);
        self.tangent.upper.fill(S::ZERO);
    }
    pub(super) fn record_replacement(&mut self, index: usize) {
        self.replacements[index] += 1;
    }
    /// A local affine span controls gradient accuracy. Other interpolation
    /// points keep curvature information; cardinal growth controls their
    /// conditioning without forcing every point to the latest resolution.
    pub(super) fn local_geometry_target(
        &mut self,
        w: &DerivativeFreeStorage<S>,
    ) -> Result<Option<usize>, SciOxideError> {
        let quality = RankThreshold::new(S::ZERO, S::ONE / S::TEN)?;
        let mut count = 0;
        for k in 1..w.values.len() {
            let distance = w
                .free
                .iter()
                .map(|&j| (w.points[k * w.n + j] - w.points[j]).powi(2))
                .sum::<S>()
                .sqrt();
            if distance <= (S::TWO + S::ONE / S::TEN) * w.radius() {
                for (j, &coordinate) in w.free.iter().enumerate() {
                    self.scratch[j] = (w.points[k * w.n + coordinate] - w.points[coordinate])
                        / self.axis_scales[j];
                }
                count = add_basis(&mut self.basis, count, &mut self.scratch, quality)?;
            }
        }
        if count == self.d {
            return Ok(None);
        }
        let mut target = None;
        let mut largest = S::ZERO;
        for k in 1..w.values.len() {
            let distance = w
                .free
                .iter()
                .map(|&j| (w.points[k * w.n + j] - w.points[j]).powi(2))
                .sum::<S>()
                .sqrt();
            if distance <= (S::TWO + S::ONE / S::TEN) * w.radius() {
                continue;
            }
            for (j, &coordinate) in w.free.iter().enumerate() {
                self.scratch[j] =
                    (w.points[k * w.n + coordinate] - w.points[coordinate]) / self.axis_scales[j];
            }
            let original = crate::utils::norm2(&self.scratch)?;
            for _ in 0..2 {
                for i in 0..count {
                    let row = &self.basis.as_slice()[i * self.d..(i + 1) * self.d];
                    let coefficient = dot(&self.scratch, row);
                    for (v, &q) in self.scratch.iter_mut().zip(row) {
                        *v -= coefficient * q;
                    }
                }
            }
            let score = if original > S::ZERO {
                crate::utils::norm2(&self.scratch)? / original
            } else {
                S::ZERO
            };
            if score > largest {
                largest = score;
                target = Some(k);
            }
        }
        Ok(target.or(Some(w.values.len() - 1)))
    }
    pub(super) fn curvature(&self, function: usize, step: &[S]) -> S {
        (0..self.d)
            .map(|i| {
                step[i]
                    * dot(
                        &self.hessians[(function * self.d + i) * self.d
                            ..(function * self.d + i + 1) * self.d],
                        step,
                    )
            })
            .sum()
    }
    pub(super) fn lagrange(&self, row: usize, step: &[S]) -> S {
        let features = self.interpolation_row.len();
        let coefficients = &self.lagrange[row * features..(row + 1) * features];
        let mut value = (0..self.d)
            .map(|j| coefficients[j] * step[j] / self.axis_scales[j])
            .sum::<S>();
        let mut term = self.d;
        for i in 0..self.d {
            for j in i..self.d {
                value += coefficients[term]
                    * (step[i] / self.axis_scales[i])
                    * (step[j] / self.axis_scales[j])
                    * if i == j {
                        S::HALF
                    } else {
                        S::ONE / S::TWO.sqrt()
                    };
                term += 1;
            }
        }
        value
    }
    pub(super) fn fit(
        &mut self,
        w: &mut DerivativeFreeStorage<S>,
        rank: RankThreshold<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        if self.fitted && self.version == w.observations {
            w.diagnostics.poised = true;
            return Ok(true);
        }
        let d = self.d;
        let p = 2 * d;
        let features = self.interpolation_row.len();
        let q = features - d;
        if !self.scale_ready {
            for (j, &coordinate) in w.free.iter().enumerate() {
                self.base_scales[j] = (1..w.values.len())
                    .map(|k| {
                        NumericScalar::abs(w.points[k * w.n + coordinate] - w.points[coordinate])
                    })
                    .fold(S::ZERO, |a, b| a.max(b));
                if self.base_scales[j] == S::ZERO {
                    w.diagnostics.poised = false;
                    return Ok(false);
                }
            }
            self.initial_radius = w.radius();
            self.scale_ready = true;
        }
        // The Hessian-change norm uses fixed prepared coordinates. Uniform
        // radius normalization conditions the solve without changing that norm.
        for j in 0..d {
            self.axis_scales[j] = self.base_scales[j] * (w.radius() / self.initial_radius);
        }
        for k in 0..p {
            for (j, &coordinate) in w.free.iter().enumerate() {
                self.directions[k * d + j] = (w.points[(k + 1) * w.n + coordinate]
                    - w.points[coordinate])
                    / self.axis_scales[j];
            }
            self.linear_matrix.as_mut_slice()[k * d..(k + 1) * d]
                .copy_from_slice(&self.directions[k * d..(k + 1) * d]);
            self.interpolation_row[..d].copy_from_slice(&self.directions[k * d..(k + 1) * d]);
            let mut term = d;
            for i in 0..d {
                for j in i..d {
                    self.interpolation_row[term] = self.directions[k * d + i]
                        * self.directions[k * d + j]
                        * if i == j {
                            S::HALF
                        } else {
                            S::ONE / S::TWO.sqrt()
                        };
                    term += 1;
                }
            }
            self.feature_matrix.as_mut_slice()[k * q..(k + 1) * q]
                .copy_from_slice(&self.interpolation_row[d..]);
            let original = crate::utils::norm2(&self.interpolation_row)?;
            for _ in 0..2 {
                for previous in 0..k {
                    let basis = &self.interpolation_basis.as_slice()
                        [previous * features..(previous + 1) * features];
                    let coefficient = dot(&self.interpolation_row, basis);
                    for (v, &b) in self.interpolation_row.iter_mut().zip(basis) {
                        *v -= coefficient * b;
                    }
                }
            }
            let residual = crate::utils::norm2(&self.interpolation_row)?;
            if residual == S::ZERO || residual <= rank.threshold(original)? {
                w.diagnostics.poised = false;
                return Ok(false);
            }
            for (out, &v) in self.interpolation_basis.as_mut_slice()
                [k * features..(k + 1) * features]
                .iter_mut()
                .zip(&self.interpolation_row)
            {
                *out = v / residual;
            }
        }
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        session.invoke(EvaluationRole::Factorization, || {
            HouseholderDecomposition::factor_into(
                &self.linear_matrix,
                HouseholderOrientation::Qr,
                HouseholderPivoting::Column,
                &options,
                &mut self.linear_factor,
                &mut self.linear_work,
                session.execution()?,
            )
        })?;
        if self
            .linear_factor
            .analyze_rank(RankPolicy::Threshold(rank))?
            .rank()
            != d
        {
            w.diagnostics.poised = false;
            return Ok(false);
        }
        session.invoke(EvaluationRole::LinearOperator, || {
            self.linear_factor.q_matrix_into(
                &options,
                self.unitary.view_mut(),
                &mut self.linear_work,
                session.execution()?,
            )
        })?;
        // Project interpolation constraints onto the orthogonal complement of
        // the free linear coefficients. SVD gives the minimum-norm curvature
        // correction directly; no squared/normal interpolation system is formed.
        for i in 0..d {
            for j in 0..q {
                let mut sum = scioxide_core::ProductSum::new();
                for k in 0..p {
                    sum.add_product(
                        self.unitary.as_slice()[k * p + d + i],
                        self.feature_matrix.as_slice()[k * q + j],
                    )?;
                }
                self.reduced.as_mut_slice()[i * q + j] = sum.finish()?;
            }
        }
        session.invoke(EvaluationRole::Factorization, || {
            ThinSvdDecomposition::compute_into(
                &self.reduced,
                &options,
                &mut self.minimum_norm,
                &mut self.minimum_norm_work,
                session.execution()?,
            )
        })?;
        if self
            .minimum_norm
            .analyze_rank(RankPolicy::Threshold(rank))?
            .rank()
            != d
        {
            w.diagnostics.poised = false;
            return Ok(false);
        }
        for function in 0..=w.m {
            for i in 0..d {
                self.previous_gradient[i] = if self.fitted {
                    w.gradient[function * d + i]
                } else {
                    S::ZERO
                };
                if self.fitted {
                    for j in 0..d {
                        self.previous_gradient[i] = self.previous_gradient[i]
                            + self.hessians[(function * d + i) * d + j]
                                * (w.points[w.free[j]] - self.center[j]);
                    }
                }
            }
            for k in 0..p {
                for j in 0..d {
                    self.scratch[j] = self.axis_scales[j] * self.directions[k * d + j];
                }
                let difference = if function == 0 {
                    w.values[k + 1] - w.values[0]
                } else {
                    w.rows[(k + 1) * w.m + function - 1] - w.rows[function - 1]
                };
                self.interpolation_rhs[k] = difference
                    - dot(&self.previous_gradient, &self.scratch)
                    - S::HALF * self.curvature(function, &self.scratch);
            }
            if !self.solve_interpolation(rank, session)? {
                self.fitted = false;
                self.hessians.fill(S::ZERO);
                w.diagnostics.poised = false;
                return Ok(false);
            }
            for i in 0..d {
                w.gradient[function * d + i] =
                    self.previous_gradient[i] + self.gradient_update[i] / self.axis_scales[i];
            }
            let mut term = 0;
            for i in 0..d {
                for j in i..d {
                    let value = self.hessians[(function * d + i) * d + j]
                        + self.hessian_update[term]
                            * if i == j {
                                S::ONE
                            } else {
                                S::ONE / S::TWO.sqrt()
                            }
                            / self.axis_scales[i]
                            / self.axis_scales[j];
                    self.hessians[(function * d + i) * d + j] = value;
                    self.hessians[(function * d + j) * d + i] = value;
                    term += 1;
                }
            }
        }
        for k in 0..p {
            self.interpolation_rhs.fill(S::ZERO);
            self.interpolation_rhs[k] = S::ONE;
            if !self.solve_interpolation(rank, session)? {
                self.fitted = false;
                self.hessians.fill(S::ZERO);
                w.diagnostics.poised = false;
                return Ok(false);
            }
            self.lagrange[k * features..k * features + d].copy_from_slice(&self.gradient_update);
            self.lagrange[k * features + d..(k + 1) * features]
                .copy_from_slice(&self.hessian_update);
            for j in 0..d {
                w.geometry.as_mut_slice()[k * d + j] =
                    self.gradient_update[j] / self.axis_scales[j];
            }
        }
        self.fitted = true;
        self.version = w.observations;
        for (out, &coordinate) in self.center.iter_mut().zip(&w.free) {
            *out = w.points[coordinate];
        }
        crate::utils::validate_finite(&w.gradient)?;
        crate::utils::validate_finite(&self.hessians)?;
        w.diagnostics.poised = true;
        Ok(true)
    }
    fn solve_interpolation(
        &mut self,
        rank: RankThreshold<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let p = 2 * self.d;
        let q = self.hessian_update.len();
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        for i in 0..self.d {
            let mut sum = scioxide_core::ProductSum::new();
            for k in 0..p {
                sum.add_product(
                    self.unitary.as_slice()[k * p + self.d + i],
                    self.interpolation_rhs[k],
                )?;
            }
            self.null_rhs[i] = sum.finish()?;
        }
        session.invoke(EvaluationRole::LinearSolve, || {
            self.minimum_norm.solve_into(
                VectorView::from_slice(&self.null_rhs),
                RankPolicy::Threshold(rank),
                S::ZERO,
                &options,
                VectorViewMut::from_slice(&mut self.hessian_update),
                &mut self.minimum_norm_work,
                session.execution()?,
            )
        })?;
        for i in 0..p {
            let mut sum = scioxide_core::ProductSum::new();
            sum.add_product(self.interpolation_rhs[i], S::ONE)?;
            for j in 0..q {
                sum.add_product(
                    -self.feature_matrix.as_slice()[i * q + j],
                    self.hessian_update[j],
                )?;
            }
            self.interpolation_rhs[i] = sum.finish()?;
        }
        session
            .invoke(EvaluationRole::LinearSolve, || {
                self.linear_factor.solve_into(
                    VectorView::from_slice(&self.interpolation_rhs),
                    &DirectSolveOptions::new(RankPolicy::Exact),
                    VectorViewMut::from_slice(&mut self.gradient_update),
                    &mut self.linear_work,
                    session.execution()?,
                )
            })
            .map(|report| report.termination() == DirectSolveTermination::Accepted)
    }

    pub(super) fn step(
        &mut self,
        w: &mut DerivativeFreeStorage<S>,
        rank: RankThreshold<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<S, SciOxideError> {
        // Inner solves must resolve model changes well below the interpolation
        // gradient's sqrt-epsilon floor. epsilon^(3/4) is explicit and portable.
        let inner = S::SQRT_EPSILON * S::SQRT_EPSILON.sqrt();
        let tolerance = Tolerance::new(inner, inner)?;
        let tolerances = OptimalityTolerances::new(
            FeasibilityTolerance::from_tolerance(tolerance),
            GradientTolerance::from_tolerance(tolerance),
            ResidualTolerance::from_tolerance(tolerance),
            rank,
        );
        self.normal_step(w, tolerances, session)?;
        self.curvature_model(w, rank)?;
        for i in 0..self.d {
            self.gradient[i] = w.gradient[i] / w.objective_scale
                + dot(
                    &self.lagrangian.as_slice()[i * self.d..(i + 1) * self.d],
                    &self.normal,
                );
            let coordinate = w.free[i];
            let (lo, hi) = w.bounds[coordinate];
            self.bounds[i] = (
                lo.map(|b| b - w.points[coordinate] - self.normal[i]),
                hi.map(|b| b - w.points[coordinate] - self.normal[i]),
            );
        }
        for i in 0..w.m {
            self.values[i] = if i < w.eq {
                S::ZERO
            } else {
                (w.rows[i] / w.row_scales[i]
                    + dot(
                        &self.jacobian.as_slice()[i * self.d..(i + 1) * self.d],
                        &self.normal,
                    ))
                .min(S::ZERO)
            };
        }
        let base_shift = self.positive_shift(w.radius(), session)?;
        let radius = w.radius();
        if !self.tangent_solve(w.eq, base_shift, S::ZERO, tolerances, session)? {
            w.step.copy_from_slice(&self.normal);
            return Ok(self.prediction(w));
        }
        let mut norm = self.total_norm()?;
        if norm > radius {
            let mut lower = S::ZERO;
            // Normal gradient components are balanced by constraint multipliers;
            // tangential components determine the trust-boundary multiplier.
            for i in 0..self.d {
                self.scratch[i] = dot(
                    &self.projection.as_slice()[i * self.d..(i + 1) * self.d],
                    &self.gradient,
                );
            }
            let available = radius - crate::utils::norm2(&self.normal)?;
            let mut upper =
                (crate::utils::norm2(&self.scratch)? / available).max(S::SQRT_EPSILON / radius);
            let mut bracketed = false;
            for _ in 0..32 {
                if self.tangent_solve(w.eq, base_shift, upper, tolerances, session)? {
                    norm = self.total_norm()?;
                    if norm <= radius {
                        self.best.copy_from(&self.tangent);
                        bracketed = true;
                        break;
                    }
                }
                lower = upper;
                upper *= S::TWO;
                if !upper.is_finite() {
                    return Err(breakdown("quadratic trust multiplier overflow"));
                }
            }
            if !bracketed {
                w.step.copy_from_slice(&self.normal);
                return Ok(self.prediction(w));
            }
            // A first feasible multiplier can be much too large when the active
            // set changes. Establish a relative bracket before bisection.
            if lower == S::ZERO {
                for _ in 0..64 {
                    let candidate = upper * S::HALF;
                    if candidate == S::ZERO {
                        break;
                    }
                    if self.tangent_solve(w.eq, base_shift, candidate, tolerances, session)?
                        && self.total_norm()? <= radius
                    {
                        upper = candidate;
                        self.best.copy_from(&self.tangent);
                    } else {
                        lower = candidate;
                        break;
                    }
                }
            }
            for _ in 0..10 {
                let middle = lower * S::HALF + upper * S::HALF;
                if self.tangent_solve(w.eq, base_shift, middle, tolerances, session)?
                    && self.total_norm()? <= radius
                {
                    upper = middle;
                    self.best.copy_from(&self.tangent);
                } else {
                    lower = middle;
                }
            }
            self.tangent.copy_from(&self.best);
        }
        for (i, step) in w.step.iter_mut().enumerate() {
            *step = self.normal[i] + self.tangent.step[i];
        }
        let multiplier = self
            .tangent
            .multipliers
            .iter()
            .map(|&v| NumericScalar::abs(v))
            .sum::<S>();
        w.penalty = w.penalty.max(S::TWO * multiplier);
        crate::utils::validate_finite(&w.step)?;
        Ok(self.prediction(w))
    }

    fn prediction(&self, w: &DerivativeFreeStorage<S>) -> S {
        self.prediction_at(w, &w.step)
    }
    pub(super) fn prediction_at(&self, w: &DerivativeFreeStorage<S>, step: &[S]) -> S {
        let objective = -(dot(&w.gradient[..self.d], step) + S::HALF * self.curvature(0, step))
            / w.objective_scale;
        let mut before = S::ZERO;
        let mut after = S::ZERO;
        for i in 0..w.m {
            let old = w.rows[i] / w.row_scales[i];
            let next = (w.rows[i]
                + dot(&w.gradient[(i + 1) * self.d..(i + 2) * self.d], step)
                + S::HALF * self.curvature(i + 1, step))
                / w.row_scales[i];
            before = before.max(if i < w.eq {
                NumericScalar::abs(old)
            } else {
                old
            });
            after = after.max(if i < w.eq {
                NumericScalar::abs(next)
            } else {
                next
            });
        }
        (objective + w.penalty * (before - after)).max(S::ZERO)
    }

    fn normal_step(
        &mut self,
        w: &DerivativeFreeStorage<S>,
        tolerances: OptimalityTolerances<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let d = self.d;
        let radius = (S::ONE - S::ONE / (S::TWO * S::TWO + S::ONE)) * w.radius();
        self.normal.fill(S::ZERO);
        for i in 0..w.m {
            for j in 0..d {
                self.jacobian.as_mut_slice()[i * d + j] =
                    w.gradient[(i + 1) * d + j] / w.row_scales[i];
            }
        }
        for (i, &coordinate) in w.free.iter().enumerate() {
            let (lo, hi) = w.bounds[coordinate];
            self.bounds[i] = (
                lo.map(|v| v - w.points[coordinate]),
                hi.map(|v| v - w.points[coordinate]),
            );
        }
        self.normal_active.fill(false);
        for _ in 0..2 * d + 2 {
            let mut changed = false;
            for i in 0..w.m {
                let value = w.rows[i] / w.row_scales[i]
                    + dot(&self.jacobian.as_slice()[i * d..(i + 1) * d], &self.normal);
                let active = i < w.eq || value > S::ZERO;
                changed |= active != self.normal_active[i];
                self.normal_active[i] = active;
                self.normal_target[i] = if active {
                    -w.rows[i] / w.row_scales[i]
                } else {
                    S::ZERO
                };
                for j in 0..d {
                    self.normal_design.as_mut_slice()[i * d + j] = if active {
                        self.jacobian.as_slice()[i * d + j]
                    } else {
                        S::ZERO
                    };
                }
            }
            if !changed {
                break;
            }
            let problem = SqpSubproblem {
                objective: SqpSubproblemObjective::LeastSquares {
                    design: self.normal_design.view(),
                    target: &self.normal_target,
                },
                jacobian: self.empty.view(),
                values: &[],
                equalities: 0,
                bounds: &self.bounds,
                tolerances,
                max_iterations: Budget::ONE.saturating_add(999),
            };
            let report =
                match solve_child(&mut self.provider, problem, &mut self.normal_state, session) {
                    Ok(report) => report,
                    Err(error) if recoverable_model_error(&error) => break,
                    Err(error) => return Err(error),
                };
            if report.termination == OptimizeTermination::Converged {
                let length = crate::utils::norm2(&self.normal_state.step)?;
                let shrink = if length > radius {
                    radius / length
                } else {
                    S::ONE
                };
                for (out, &step) in self.cauchy.iter_mut().zip(&self.normal_state.step) {
                    *out = shrink * step;
                }
                if self.violation(w, &self.cauchy) < self.violation(w, &self.normal) {
                    self.normal.copy_from_slice(&self.cauchy);
                }
            }
        }
        // A projected Cauchy candidate prevents a nearly rank-deficient normal
        // least-squares direction losing sufficient decrease after truncation.
        self.scratch.fill(S::ZERO);
        for i in 0..w.m {
            let value = w.rows[i] / w.row_scales[i];
            let residual = if i < w.eq { value } else { value.max(S::ZERO) };
            for j in 0..d {
                self.scratch[j] = self.scratch[j] + self.jacobian.as_slice()[i * d + j] * residual;
            }
        }
        for j in 0..d {
            if self.bounds[j].0.is_some_and(|b| b >= S::ZERO) && self.scratch[j] > S::ZERO
                || self.bounds[j].1.is_some_and(|b| b <= S::ZERO) && self.scratch[j] < S::ZERO
            {
                self.scratch[j] = S::ZERO;
            }
        }
        let length = crate::utils::norm2(&self.scratch)?;
        if length > S::ZERO {
            let squared = dot(&self.scratch, &self.scratch);
            let curvature = (0..w.m)
                .filter(|&i| i < w.eq || w.rows[i] > S::ZERO)
                .map(|i| dot(&self.jacobian.as_slice()[i * d..(i + 1) * d], &self.scratch).powi(2))
                .sum::<S>();
            let mut alpha = radius / length;
            if curvature > S::ZERO {
                alpha = alpha.min(squared / curvature);
            }
            for (j, &g) in self.scratch.iter().enumerate() {
                if g > S::ZERO {
                    if let Some(lo) = self.bounds[j].0 {
                        alpha = alpha.min(-lo / g);
                    }
                } else if g < S::ZERO
                    && let Some(hi) = self.bounds[j].1
                {
                    alpha = alpha.min(-hi / g);
                }
            }
            for (step, &g) in self.cauchy.iter_mut().zip(&self.scratch) {
                *step = -alpha * g;
            }
            if self.violation(w, &self.cauchy) < self.violation(w, &self.normal) {
                self.normal.copy_from_slice(&self.cauchy);
            }
        }
        for (step, &(lo, hi)) in self.normal.iter_mut().zip(&self.bounds) {
            if let Some(lo) = lo {
                *step = step.max(lo);
            }
            if let Some(hi) = hi {
                *step = step.min(hi);
            }
        }
        Ok(())
    }
    fn violation(&self, w: &DerivativeFreeStorage<S>, step: &[S]) -> S {
        (0..w.m)
            .map(|i| {
                let v = w.rows[i] / w.row_scales[i]
                    + dot(
                        &self.jacobian.as_slice()[i * self.d..(i + 1) * self.d],
                        step,
                    );
                let residual = if i < w.eq { v } else { v.max(S::ZERO) };
                residual * residual
            })
            .sum()
    }

    fn curvature_model(
        &mut self,
        w: &DerivativeFreeStorage<S>,
        rank: RankThreshold<S>,
    ) -> Result<(), SciOxideError> {
        let d = self.d;
        for i in 0..d {
            for j in 0..=i {
                let mut value = self.hessians[i * d + j] / w.objective_scale;
                for row in 0..w.m {
                    value += self.tangent.multipliers[row]
                        * self.hessians[((row + 1) * d + i) * d + j]
                        / w.row_scales[row];
                }
                self.lagrangian.as_mut_slice()[i * d + j] = value;
                self.lagrangian.as_mut_slice()[j * d + i] = value;
            }
        }
        let mut count = 0;
        let largest = self
            .tangent
            .multipliers
            .iter()
            .chain(&self.tangent.lower)
            .chain(&self.tangent.upper)
            .fold(S::ZERO, |a, &b| a.max(NumericScalar::abs(b)));
        for row in 0..w.m {
            let norm = crate::utils::norm2(&self.jacobian.as_slice()[row * d..(row + 1) * d])?;
            if row < w.eq
                || self.tangent.multipliers[row] > S::EPSILON * largest
                    && w.rows[row] / w.row_scales[row] >= -S::TWO * w.radius() * norm
            {
                self.scratch
                    .copy_from_slice(&self.jacobian.as_slice()[row * d..(row + 1) * d]);
                count = add_basis(&mut self.basis, count, &mut self.scratch, rank)?;
            }
        }
        for (j, &coordinate) in w.free.iter().enumerate() {
            for (endpoint, lambda, sign) in [
                (w.bounds[coordinate].0, self.tangent.lower[j], -S::ONE),
                (w.bounds[coordinate].1, self.tangent.upper[j], S::ONE),
            ] {
                if let Some(b) = endpoint
                    && lambda > S::EPSILON * largest
                    && NumericScalar::abs(w.points[coordinate] - b) <= S::TWO * w.radius()
                {
                    self.scratch.fill(S::ZERO);
                    self.scratch[j] = sign;
                    count = add_basis(&mut self.basis, count, &mut self.scratch, rank)?;
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                self.projection.as_mut_slice()[i * d + j] = if i == j { S::ONE } else { S::ZERO }
                    - (0..count)
                        .map(|k| {
                            self.basis.as_slice()[k * d + i] * self.basis.as_slice()[k * d + j]
                        })
                        .sum::<S>();
            }
        }
        for i in 0..d {
            for j in 0..d {
                self.product.as_mut_slice()[i * d + j] = (0..d)
                    .map(|k| {
                        self.lagrangian.as_slice()[i * d + k]
                            * self.projection.as_slice()[k * d + j]
                    })
                    .sum();
            }
        }
        for i in 0..d {
            for j in 0..=i {
                let value = (0..d)
                    .map(|k| {
                        self.projection.as_slice()[i * d + k] * self.product.as_slice()[k * d + j]
                    })
                    .sum::<S>()
                    + if i == j { S::ONE } else { S::ZERO }
                    - self.projection.as_slice()[i * d + j];
                self.projected.as_mut_slice()[i * d + j] = value;
                self.projected.as_mut_slice()[j * d + i] = value;
            }
        }
        crate::utils::validate_finite(self.projected.as_slice())
    }
    fn positive_shift(
        &mut self,
        radius: S,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<S, SciOxideError> {
        let mut shift = S::ZERO;
        let scale = (0..self.d)
            .map(|i| {
                self.projected.as_slice()[i * self.d..(i + 1) * self.d]
                    .iter()
                    .map(|&v| NumericScalar::abs(v))
                    .sum::<S>()
            })
            .fold(S::ZERO, |a, b| a.max(b));
        for _ in 0..12 {
            self.shifted
                .as_mut_slice()
                .copy_from_slice(self.projected.as_slice());
            for i in 0..self.d {
                self.shifted.as_mut_slice()[i * self.d + i] =
                    self.shifted.as_slice()[i * self.d + i] + shift;
            }
            let inertia = session.invoke(EvaluationRole::Factorization, || {
                self.curvature_provider
                    .factor(self.shifted.view(), session.execution()?)
            })?;
            if inertia.positive() == self.d && inertia.negative() == 0 && inertia.zero() == 0 {
                return Ok(shift);
            }
            shift = if shift == S::ZERO {
                S::SQRT_EPSILON
                    * if scale > S::ZERO {
                        scale
                    } else {
                        S::ONE / radius
                    }
            } else {
                S::TEN * shift
            };
        }
        Err(breakdown(
            "quadratic model regularization did not produce positive curvature",
        ))
    }
    fn tangent_solve(
        &mut self,
        equalities: usize,
        base: S,
        trust: S,
        tolerances: OptimalityTolerances<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        self.shifted
            .as_mut_slice()
            .copy_from_slice(self.projected.as_slice());
        for i in 0..self.d {
            self.shifted.as_mut_slice()[i * self.d + i] =
                self.shifted.as_slice()[i * self.d + i] + base + trust;
            self.scratch[i] = self.gradient[i] + trust * self.normal[i];
        }
        let problem = SqpSubproblem {
            objective: SqpSubproblemObjective::Quadratic {
                hessian: self.shifted.view(),
                gradient: &self.scratch,
            },
            jacobian: self.jacobian.view(),
            values: &self.values,
            equalities,
            bounds: &self.bounds,
            tolerances,
            max_iterations: Budget::ONE.saturating_add(999),
        };
        match solve_child(&mut self.provider, problem, &mut self.tangent, session) {
            Ok(report) => Ok(report.termination == OptimizeTermination::Converged),
            Err(error) if recoverable_model_error(&error) => Ok(false),
            Err(error) => Err(error),
        }
    }
    fn total_norm(&mut self) -> Result<S, SciOxideError> {
        for i in 0..self.d {
            self.scratch[i] = self.normal[i] + self.tangent.step[i];
        }
        crate::utils::norm2(&self.scratch)
    }
}

// A failed finite model factorization permits the already prepared projected
// Cauchy/TRSTLP safeguard. Execution, callback, shape, and range errors propagate.
fn recoverable_model_error(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
            | scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Breakdown
                    | scioxide_core::NumericalIssue::IllConditioned
                    | scioxide_core::NumericalIssue::LossOfSignificance,
                ..
            }
    )
}

fn add_basis<S: RealScalar + NumericScalar<Real = S>>(
    basis: &mut Matrix<S>,
    count: usize,
    candidate: &mut [S],
    rank: RankThreshold<S>,
) -> Result<usize, SciOxideError> {
    if count >= candidate.len() {
        return Ok(count);
    }
    let original = crate::utils::norm2(candidate)?;
    for _ in 0..2 {
        for k in 0..count {
            let row = &basis.as_slice()[k * candidate.len()..(k + 1) * candidate.len()];
            let projection = dot(candidate, row);
            for (value, &q) in candidate.iter_mut().zip(row) {
                *value -= projection * q;
            }
        }
    }
    let residual = crate::utils::norm2(candidate)?;
    if residual == S::ZERO || residual <= rank.threshold(original)? {
        return Ok(count);
    }
    let d = candidate.len();
    for (out, &value) in basis.as_mut_slice()[count * d..(count + 1) * d]
        .iter_mut()
        .zip(candidate.iter())
    {
        *out = value / residual;
    }
    Ok(count + 1)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::{Cobyqa, ConstrainedProblem, Constraints, ObjectiveFn, OptimizeResult};
    use scioxide_core::ExecutionContext;
    #[test]
    fn backward_interpolation_points_are_replaced_during_rotated_solves() {
        let solver = Cobyqa::<f64>::default();
        let f = ObjectiveFn::new(3, |x: &[f64]| {
            Ok((x[0] + 2.0 * x[1] - 1.0).powi(2)
                + 0.2 * (x[1] - x[2] + 0.3).powi(2)
                + 3.0 * (x[2] - 0.7).powi(2))
        });
        let free = Constraints::new(3).unwrap();
        let problem = ConstrainedProblem::new(&f, &free).unwrap();
        let mut used = [false; 7];
        for start in [[3.0, -2.0, 1.0], [-1.0, 3.0, -4.0], [0.0, 0.0, 0.0]] {
            let mut w = solver.workspace(&problem).unwrap();
            let mut output = OptimizeResult::empty();
            solver
                .minimize_into(
                    &problem,
                    &start,
                    &mut output,
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(output.fun() < 1e-6, "{output:?}");
            for (slot, &count) in w
                .storage
                .quadratic
                .as_ref()
                .unwrap()
                .replacements
                .iter()
                .enumerate()
            {
                used[slot] |= count > 0;
            }
        }
        assert!(
            used[1..].iter().all(|&v| v),
            "unreplaced interpolation points: {used:?}"
        );
    }
}
