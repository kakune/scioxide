//! Least-squares trust-region steps from a QR factorization, with the
//! Levenberg–Marquardt parameter chosen by Moré's safeguarded iteration.
use super::*;
use crate::least_squares_trust_region::{
    LeastSquaresTrustRegionReport, StepQuery, select, validate,
};

/// Fold identity damping into a copied R and solve its triangular system.
#[allow(clippy::too_many_arguments)]
pub(super) fn damped_step<S: RealScalar + NumericScalar<Real = S>>(
    packed: crate::MatrixView<'_, S>,
    rhs: &[S],
    diagonal: Option<&[S]>,
    permutation: &[usize],
    damping: S,
    r: &mut [S],
    solution: &mut [S],
    row: &mut [S],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = solution.len();
    r.fill(S::ZERO);
    for i in 0..n {
        for j in i..n {
            r[i * n + j] = packed.get_element(i, j);
        }
    }
    solution.copy_from_slice(&rhs[..n]);
    if damping > S::ZERO {
        let root = damping.sqrt();
        for index in 0..n {
            execution.check()?;
            row.fill(S::ZERO);
            row[index] = root * diagonal.map_or(S::ONE, |d| d[permutation[index]]);
            let mut extra = S::ZERO;
            for pivot in index..n {
                if row[pivot] == S::ZERO {
                    continue;
                }
                let (rotation, norm) =
                    crate::rotation::Rotation::generate(r[pivot * n + pivot], row[pivot])?;
                if !norm.is_finite() {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Overflow,
                        "regularized triangular factor overflowed",
                    ));
                }
                for column in pivot..n {
                    (r[pivot * n + column], row[column]) =
                        rotation.pair(r[pivot * n + column], row[column])?;
                }
                (solution[pivot], extra) = rotation.pair(solution[pivot], extra)?;
            }
        }
    }
    for i in (0..n).rev() {
        execution.check()?;
        let diagonal = r[i * n + i];
        // Keep representable substitutions in scalar arithmetic. Tiny or
        // overflowing numerators retain the exponent-separated replay below.
        let ordinary = (|| {
            let mut sum = scioxide_core::CompensatedSum::ZERO;
            sum.add(solution[i]).ok()?;
            for j in i + 1..n {
                let a = -r[i * n + j];
                let b = solution[j];
                let product = a * b;
                if scioxide_core::backend::standard::product_needs_retry(a, b, product) {
                    return None;
                }
                sum.add(product).ok()?;
            }
            scioxide_core::backend::standard::ordinary_quotient(sum.total().ok()?, diagonal)
        })();
        if let Some(value) = ordinary {
            solution[i] = value;
            continue;
        }
        let left = crate::FiniteVectorView::from_workspace(crate::VectorView::from_slice(
            &r[i * n + i + 1..(i + 1) * n],
        ));
        let right = crate::FiniteVectorView::from_workspace(crate::VectorView::from_slice(
            &solution[i + 1..n],
        ));
        let (sum, exponent) = crate::reduction::compensated_dot(left, right, solution[i], -S::ONE)?;
        let e = crate::kernel::standard::exponent(diagonal.abs())?;
        solution[i] = crate::kernel::standard::restore(sum / diagonal.scalbn(-e), exponent - e)?;
    }
    Ok(())
}

impl<S: RealScalar + NumericScalar<Real = S>> HouseholderDecomposition<S> {
    /// Solves the trust-region subproblem `min ‖A p − b‖₂` subject to
    /// `‖D p‖₂ ≤ radius`, with `D = diag(diagonal)`, into `output`.
    ///
    /// The solution is `p(λ)` with `(AᵀA + λ DᵀD) p = Aᵀ b`. If `‖D p(0)‖₂` is
    /// at most `1.1 · radius`, then `λ = 0`. Otherwise Moré's safeguarded Newton
    /// iteration on `1/‖D p(λ)‖₂` keeps `λ`
    /// between a lower bound (the Newton step from `λ = 0`) and the upper bound
    /// `‖D⁻¹ Aᵀ b‖₂ / radius`, and stops when `‖D p(λ)‖₂` is within 10% of
    /// `radius`, or after 10 iterations. `hint`, clamped to the bounds, is the
    /// starting `λ`, for example the value from the previous step. Each `p(λ)`
    /// reuses `R` and `Qᵀ b` and eliminates `√λ D` with Givens rotations, so the
    /// normal equations are never formed; an iteration costs `O(n²)`.
    ///
    /// The factorization must be real QR with `m ≥ n`. If it fails the rank or
    /// condition checks of `acceptance`, the result has that status and no step,
    /// and `output` is unchanged. Reserve
    /// [`HouseholderWorkspaceRequest::damped_solve`](crate::HouseholderWorkspaceRequest::damped_solve)
    /// in `workspace` to avoid allocation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `radius` is not finite and positive,
    ///   `hint` is not finite and nonnegative, or an entry of `diagonal` is not
    ///   finite and positive.
    /// - `InvalidInput` (`UnsupportedConfiguration`) for LQ or for `m < n`.
    /// - `DimensionMismatch` if `rhs` does not have length `m`, or `diagonal`
    ///   or `output` does not have length `n`.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if an entry of `Qᵀ b` or of a damped
    ///   step `p(λ)` with `λ > 0` overflows.
    /// - Limit errors from the options in `acceptance`, workspace errors and
    ///   `ExecutionStopped`, as for [`Self::solve_damped_into`].
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # References
    ///
    /// - J. J. Moré, The Levenberg–Marquardt algorithm: implementation and
    ///   theory, in *Numerical Analysis*, Lecture Notes in Mathematics 630,
    ///   Springer, 1978, 105–116.
    /// - J. J. Moré, B. S. Garbow and K. E. Hillstrom, *User Guide for
    ///   MINPACK-1*, Argonne National Laboratory report ANL-80-74, 1980.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_trust_region_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        diagonal: &[S],
        radius: S,
        hint: S,
        acceptance: &DirectSolveOptions<S>,
        output: crate::VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<LeastSquaresTrustRegionReport<S>, SciOxideError> {
        workspace.begin_attempt();
        validate(radius, hint)?;
        if self.orientation != HouseholderOrientation::Qr || self.rows < self.cols {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "trust-region solves require tall or square QR geometry",
            ));
        }
        crate::checked::expect_shape([self.cols], [diagonal.len()])?;
        for &value in diagonal {
            if !value.is_finite() || value <= S::ZERO {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("diagonal"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "diagonal scales must be finite and positive",
                ));
            }
        }
        let options = acceptance.execution_options();
        let context = context.into().with_options(&options)?;
        context.execution().check()?;
        let rhs = rhs.into();
        crate::checked::expect_shape([self.rows], [rhs.len()])?;
        crate::checked::expect_shape([self.cols], [output.len()])?;
        let requirements = Self::workspace_requirements(
            &HouseholderWorkspaceRequest::damped_solve(self.rows, self.cols, &options)
                .with_residual(acceptance.residual()),
        )?;
        let rhs = rhs.try_finite_dispatched(context.kernel().dispatch())?;
        workspace.try_reserve(&requirements)?;
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "qr_trust_region", 1),
                self.rows.max(self.cols) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, kernels, solve| {
                let admission =
                    self.admit_solve(acceptance, workspace, kernels.scoped(&mut *solve))?;
                if admission.termination() != DirectSolveTermination::Accepted {
                    return Ok(LeastSquaresTrustRegionReport::new(admission, None));
                }
                crate::solve_support::admit_iteration(
                    Some(kernels.execution()),
                    Some(solve),
                    "QR trust-region budget exhausted",
                )?;
                let n = self.cols;
                workspace.with_prefix(n * n + self.rows + 3 * n, |storage, _| {
                    let (r, rest) = crate::workspace::split_mut(storage, n * n)?;
                    let (qtr, rest) = crate::workspace::split_mut(rest, self.rows)?;
                    let (candidate, rest) = crate::workspace::split_mut(rest, n)?;
                    let (row, derivative) = crate::workspace::split_mut(rest, n)?;
                    let derivative = &mut derivative[..n];
                    qtr.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(out, value)| *out = value);
                    self.qr.apply_qh_vector_in_place(qtr)?;
                    let packed = self.packed_factors();
                    let step = select(
                        radius,
                        hint,
                        || {
                            let mut gradient = scioxide_core::ScaledNorm::new();
                            for j in 0..n {
                                let mut sum = scioxide_core::RangeAccumulator::new();
                                for (i, &value) in qtr[..=j].iter().enumerate() {
                                    sum.add_product(packed.get_element(i, j), value)?;
                                }
                                gradient.add(sum.finish()? / diagonal[self.permutation()[j]])?;
                            }
                            gradient.finish()
                        },
                        |query| match query {
                            StepQuery::Norm(damping) => {
                                kernels.execution().check()?;
                                damped_step(
                                    packed,
                                    qtr,
                                    Some(diagonal),
                                    self.permutation(),
                                    damping,
                                    r,
                                    candidate,
                                    row,
                                    kernels.execution(),
                                )?;
                                let mut norm = scioxide_core::ScaledNorm::new();
                                for (i, &value) in candidate.iter().enumerate() {
                                    norm.add(value * diagonal[self.permutation()[i]])?;
                                }
                                norm.finish()
                            }
                            StepQuery::NewtonScale { norm, .. } => {
                                // Solve R_λᵀ y = Dᵀ D p / ‖D p‖₂ by forward substitution.
                                kernels.execution().check()?;
                                for i in 0..n {
                                    let scale = diagonal[self.permutation()[i]];
                                    let mut sum = scioxide_core::RangeAccumulator::new();
                                    sum.add((candidate[i] * scale / norm) * scale)?;
                                    for j in 0..i {
                                        sum.add_product(-r[j * n + i], derivative[j])?;
                                    }
                                    derivative[i] = sum.finish()? / r[i * n + i];
                                }
                                let mut scale = scioxide_core::ScaledNorm::new();
                                for &value in derivative.iter() {
                                    scale.add(value)?;
                                }
                                scale.finish()
                            }
                        },
                    )?;
                    let residual = if acceptance.residual() == crate::Estimation::Estimate {
                        for (column, &original) in self.permutation().iter().enumerate() {
                            row[original] = candidate[column];
                        }
                        self.residual_columns(
                            crate::MatrixView::from(rhs.as_view()),
                            crate::MatrixView::from_slice(n, 1, row)?,
                            kernels,
                        )?
                    } else {
                        None
                    };
                    let admission = DirectSolveReport::new(
                        admission.termination(),
                        admission.rank(),
                        admission.condition(),
                        residual,
                    );
                    kernels.execution().check()?;
                    for (column, &original) in self.permutation().iter().enumerate() {
                        output.data[original * output.stride] = candidate[column];
                    }
                    Ok(LeastSquaresTrustRegionReport::new(admission, Some(step)))
                })
            },
        )
    }
}
