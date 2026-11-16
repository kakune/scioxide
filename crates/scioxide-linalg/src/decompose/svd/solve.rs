//! Shared solves over the vectors and spectrum of either typed SVD result.

use crate::kernel::standard;
use crate::{FiniteVectorView, MatrixView, RankPolicy};
use scioxide_core::NumericScalar;
use scioxide_core::{AlgebraicScalar, RealScalar, SciOxideError};

/// Borrowed mathematical factors; callers establish shape and finiteness once.
pub(crate) struct SpectralView<'a, S: NumericScalar> {
    pub(crate) u: MatrixView<'a, S>,
    pub(crate) values: &'a [S::Real],
    pub(crate) vt: MatrixView<'a, S>,
}

impl<S: NumericScalar> SpectralView<'_, S> {
    /// Solve in a fresh workspace attempt. Validation and reservation come first,
    /// and the right-hand side counts as one iteration.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub(crate) fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        rank_policy: RankPolicy<S::Real>,
        damping: S::Real,
        options: &crate::LinalgExecutionOptions,
        mut output: crate::VectorViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let (rows, cols) = (self.u.rows(), self.vt.cols());
        let rhs = rhs.into();
        crate::checked::expect_shape([rows], [rhs.len()])?;
        crate::checked::expect_shape([cols], [output.len()])?;
        crate::checked::validate_input(damping, "damping")?;
        if damping < S::Real::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("damping"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "damping must be nonnegative",
            ));
        }
        let rhs = rhs.try_finite_dispatched(context.kernel().dispatch())?;
        let threshold = self.threshold(rank_policy)?;
        workspace.try_reserve(&crate::ThinSvdDecomposition::<S>::workspace_requirements(
            &crate::SvdWorkspaceRequest::solve(rows, cols, options),
        )?)?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "svd_solve", 1),
                rows,
                cols,
            ),
            options.solve_budgets(),
            |workspace, kernels, solve| {
                crate::solve_support::admit_iteration(
                    Some(kernels.execution()),
                    Some(solve),
                    "SVD solve budget exhausted",
                )?;
                if self.values.is_empty() {
                    output.iter_mut().for_each(|value| *value = S::ZERO);
                    return Ok(());
                }
                workspace.with_direct(|storage, _, _, _| {
                    let (coefficients, candidate) =
                        crate::workspace::split_mut(storage, self.values.len())?;
                    let candidate = &mut candidate[..cols];
                    self.solve(rhs, threshold, damping, coefficients, candidate)?;
                    kernels.execution().check()?;
                    output
                        .iter_mut()
                        .zip(candidate.iter())
                        .for_each(|(out, &value)| *out = value);
                    Ok(())
                })
            },
        )
    }

    /// Form the pseudoinverse in a fresh workspace attempt. Each output column
    /// counts as one iteration.
    pub(crate) fn pseudoinverse_into<'scope, 'capability>(
        &self,
        rank_policy: RankPolicy<S::Real>,
        options: &crate::LinalgExecutionOptions,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let (rows, cols) = (self.u.rows(), self.vt.cols());
        crate::checked::expect_shape([cols, rows], [output.rows(), output.cols()])?;
        let threshold = self.threshold(rank_policy)?;
        workspace.try_reserve(&crate::ThinSvdDecomposition::<S>::workspace_requirements(
            &crate::SvdWorkspaceRequest::pseudoinverse(rows, cols, options),
        )?)?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "svd_pseudoinverse", 1),
                rows,
                cols,
            ),
            options.solve_budgets(),
            |workspace, kernels, solve| {
                if output.is_empty() {
                    return Ok(());
                }
                workspace.with_direct(|storage, _, _, _| {
                    let (coefficients, storage) =
                        crate::workspace::split_mut(storage, self.values.len())?;
                    let (candidate, column) = crate::workspace::split_mut(storage, rows * cols)?;
                    let column = &mut column[..cols];
                    for col in 0..rows {
                        crate::solve_support::admit_iteration(
                            Some(kernels.execution()),
                            Some(&mut *solve),
                            "SVD pseudoinverse budget exhausted",
                        )?;
                        self.inverse_column(col, threshold, coefficients, column)?;
                        for row in 0..cols {
                            candidate[row * rows + col] = column[row];
                        }
                    }
                    kernels.execution().check()?;
                    for row in 0..cols {
                        for col in 0..rows {
                            output.set_element(row, col, candidate[row * rows + col]);
                        }
                    }
                    Ok(())
                })
            },
        )
    }
    pub(crate) fn threshold(&self, policy: RankPolicy<S::Real>) -> Result<S::Real, SciOxideError> {
        super::spectrum_rank(self.values, self.u.rows(), self.vt.cols(), policy)
            .map(|rank| rank.threshold())
    }

    pub(crate) fn solve(
        &self,
        rhs: FiniteVectorView<'_, S>,
        threshold: S::Real,
        damping: S::Real,
        coefficients: &mut [S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        for (index, (&singular, coefficient)) in
            self.values.iter().zip(coefficients.iter_mut()).enumerate()
        {
            *coefficient = if singular <= threshold {
                S::ZERO
            } else {
                let left = FiniteVectorView::from_workspace(self.u.col(index)?.conjugate());
                let (dot, exponent) =
                    crate::reduction::compensated_dot(left, rhs, S::ZERO, S::ONE)?;
                filtered(dot, exponent, singular, damping)?
            };
        }
        self.combine(coefficients, output)
    }

    pub(crate) fn inverse_column(
        &self,
        column: usize,
        threshold: S::Real,
        coefficients: &mut [S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        for (index, (&singular, coefficient)) in
            self.values.iter().zip(coefficients.iter_mut()).enumerate()
        {
            *coefficient = if singular <= threshold {
                S::ZERO
            } else {
                filtered(
                    self.u.get_element(column, index).conj(),
                    0,
                    singular,
                    S::Real::ZERO,
                )?
            };
        }
        self.combine(coefficients, output)
    }

    fn combine(&self, coefficients: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        let coefficients =
            FiniteVectorView::from_workspace(crate::VectorView::from_slice(coefficients));
        for (column, value) in output.iter_mut().enumerate() {
            let left = FiniteVectorView::from_workspace(self.vt.col(column)?.conjugate());
            let (sum, exponent) =
                crate::reduction::compensated_dot(left, coefficients, S::ZERO, S::ONE)?;
            *value = standard::restore(sum, exponent)?;
        }
        Ok(())
    }
}

pub(crate) fn filtered<S: NumericScalar>(
    dot: S,
    exponent: i32,
    singular: S::Real,
    damping: S::Real,
) -> Result<S, SciOxideError> {
    let se = standard::exponent(singular)?;
    let root = damping.sqrt();
    let e = if root == S::Real::ZERO {
        se
    } else {
        se.max(standard::exponent(root)?)
    };
    let sigma = singular.scalbn(-e);
    let lambda = root.scalbn(-e);
    let denominator = sigma * sigma + lambda * lambda;
    let multiplier = singular.scalbn(-se) / denominator;
    let de = standard::exponent(standard::magnitude(dot))?;
    standard::restore(
        standard::scale(dot, -de) * S::from_real(multiplier),
        exponent + de + se - 2 * e,
    )
}

fn identity(
    algorithm: scioxide_core::AlgorithmId,
    rows: usize,
    cols: usize,
) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        algorithm,
        ((rows as u128) << 64) | cols as u128,
        0,
    )
}
