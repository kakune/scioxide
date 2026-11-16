//! Levin collocation, solved in transposed form for quadrature weights that
//! apply to every amplitude.
use super::number;
use scioxide_core::{
    Complex, EvaluationRole, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError,
    SolveContext,
};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, RankPolicy, Vector,
};
type LevinWeights<S> = ([Complex<S>; 25], [Complex<S>; 25], S);
type CollocationWeights<S> = ([Complex<S>; 25], S);

struct Collocation<S: RealScalar>
where
    Complex<S>: NumericScalar<Real = S>,
{
    matrix: Matrix<Complex<S>>,
    factor: HouseholderDecomposition<Complex<S>>,
    workspace: LinalgWorkspace<'static, Complex<S>>,
    rhs: Vector<Complex<S>>,
    weights: Vector<Complex<S>>,
}
impl<S: RealScalar> Collocation<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn new(n: usize) -> Result<Self, SciOxideError> {
        let matrix = Matrix::identity(n)?;
        let mut factor = HouseholderDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        // Both sides of the factor's transactional swap must be reserved.
        for _ in 0..2 {
            HouseholderDecomposition::factor_into(
                &matrix,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut factor,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )?;
        }
        workspace.try_reserve(
            &scioxide_linalg::HouseholderDecomposition::<Complex<S>>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                    n,
                    n,
                    scioxide_linalg::HouseholderOrientation::Qr,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        Ok(Self {
            matrix,
            factor,
            workspace,
            rhs: Vector::zeros(n)?,
            weights: Vector::zeros(n)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn weights(
        &mut self,
        cosines: &[[S; 25]; 25],
        derivatives: &[S; 25],
        half: S,
        omega: S,
        phases: [S; 2],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<CollocationWeights<S>>, SciOxideError> {
        // Levin: find p = sum c_k T_k with p' + i omega g' p = f at the collocation points;
        // then the integral of f e^{i omega g} is [p e^{i omega g}] at the endpoints. The
        // result is linear in f, so instead of solving for p per amplitude we solve the
        // transposed system M w = r once, where M[k][j] is the Levin operator applied to T_k
        // at point j and r_k = T_k(1) e^{i phi(b)} - T_k(-1) e^{i phi(a)} (T_k(+-1) = (+-1)^k).
        // Then `sum_j w_j f(x_j)` is the Levin estimate for any amplitude, so all vector
        // components share one factorization. The operator is written in t in [-1, 1],
        // hence the `half` chain-rule factor on the omega term and on the final weights.
        // The 13-point system uses every second node (stride 2) of the 25-point grid, so the
        // low-order estimate reuses the same samples.
        let n = self.rhs.len();
        let stride = 24 / (n - 1);
        let left = Complex::new(phases[0].cos(), phases[0].sin());
        let right = Complex::new(phases[1].cos(), phases[1].sin());
        for k in 0..n {
            self.rhs.as_mut_slice()[k] = if k % 2 == 0 {
                right - left
            } else {
                right + left
            };
        }
        for j in 0..n {
            let index = j * stride;
            let t = cosines[1][index];
            let mut previous = S::ZERO;
            let mut current = S::ONE;
            for (k, cosine_row) in cosines.iter().enumerate().take(n) {
                // T'_k(t) = k U_{k-1}(t), including the exact endpoint limits.
                let derivative = if k == 0 {
                    S::ZERO
                } else {
                    number::<S>(k)? * current
                };
                self.matrix.as_mut_slice()[k * n + j] = Complex::new(
                    derivative,
                    omega * half * derivatives[index] * cosine_row[index],
                );
                if k > 0 {
                    let next = number::<S>(2)? * t * current - previous;
                    previous = current;
                    current = next;
                }
            }
        }
        // A nonfinite operator entry, rank deficiency, a rejected solve or an
        // unacceptable residual or condition estimate leaves the region without
        // collocation weights; the caller bisects it instead of failing the call.
        if !self
            .matrix
            .as_slice()
            .iter()
            .all(|entry| entry.has_finite_components())
        {
            return Ok(None);
        }
        super::rule::charge(context, EvaluationRole::Factorization)?;
        HouseholderDecomposition::factor_into(
            &self.matrix,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut self.factor,
            &mut self.workspace,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        let rank = self
            .factor
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)?;
        if !rank.is_full_rank() {
            return Ok(None);
        }
        super::rule::charge(context, EvaluationRole::LinearSolve)?;
        let report = self.factor.solve_into(
            &self.rhs,
            &(scioxide_linalg::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
                .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()),
            self.weights.view_mut(),
            &mut self.workspace,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        if report.termination() != scioxide_linalg::DirectSolveTermination::Accepted {
            return Ok(None);
        }
        // Rank-revealing QR alone does not bound the weight error, so the solve is also
        // required to have a small normwise backward error (residual relative to
        // |M||w| + |r|); the condition estimate then converts that into a forward bound.
        let reciprocal = self
            .factor
            .condition_estimate_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut self.workspace,
                scioxide_core::ExecutionContext::sequential(),
            )?
            .reciprocal()
            .value();
        let mut residual = S::ZERO;
        let mut scale = S::ZERO;
        for k in 0..n {
            let mut sum = Complex::new(S::ZERO, S::ZERO);
            let mut mass = S::ZERO;
            for j in 0..n {
                let product = self.matrix.as_slice()[k * n + j] * self.weights.as_slice()[j];
                sum += product;
                mass += product.abs();
            }
            residual = residual.max((sum - self.rhs.as_slice()[k]).abs());
            scale = scale.max(mass + self.rhs.as_slice()[k].abs());
        }
        let backward = if scale == S::ZERO {
            S::ZERO
        } else {
            residual / scale
        };
        if !(backward <= number::<S>(128)? * number::<S>(n)? * S::EPSILON && reciprocal > S::ZERO) {
            return Ok(None);
        }
        let mut weights = [Complex::new(S::ZERO, S::ZERO); 25];
        for (j, &weight) in self.weights.as_slice().iter().enumerate() {
            weights[j * stride] = Complex::new(weight.re * half, weight.im * half);
        }
        // Heuristic forward roundoff contribution, explicitly not a certificate.
        let error = crate::kernel::checked_intermediate(
            (backward + number::<S>(8)? * S::EPSILON) / reciprocal,
        )?;
        Ok(Some((weights, error)))
    }
}
pub(super) struct LevinWorkspace<S: RealScalar>
where
    Complex<S>: NumericScalar<Real = S>,
{
    high: Collocation<S>,
    low: Collocation<S>,
}
impl<S: RealScalar> LevinWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    pub(super) fn new() -> Result<Self, SciOxideError> {
        Ok(Self {
            high: Collocation::new(25)?,
            low: Collocation::new(13)?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn weights(
        &mut self,
        cosines: &[[S; 25]; 25],
        derivatives: &[S; 25],
        half: S,
        omega: S,
        phases: [S; 2],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<LevinWeights<S>>, SciOxideError> {
        let Some((high, a)) =
            self.high
                .weights(cosines, derivatives, half, omega, phases, context)?
        else {
            return Ok(None);
        };
        let Some((low, b)) =
            self.low
                .weights(cosines, derivatives, half, omega, phases, context)?
        else {
            return Ok(None);
        };
        Ok(Some((high, low, a.max(b))))
    }
}
