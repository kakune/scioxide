//! Reordering of a Schur decomposition, selected eigenvalues first, with condition estimates.
use super::schur::for_each_schur_block;
use super::support::{DirectLengths, direct_requirements, identity, square, times, total};
use super::{SchurDecomposition, SchurReorderOptions, SchurReorderReport};
use crate::kernel::frames::extended;
use crate::kernel::sylvester::solve;
use crate::workspace::Extended;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::{take_mut, take_square};
use crate::{ConditionEstimate, Estimation, LinalgWorkspace, MatrixView};
use scioxide_core::AlgebraicScalar;
use scioxide_core::NumericScalar;
use scioxide_core::{
    Dimensions, EstimateReliability, ExecutionContext, ExecutionPolicy, ReliableEstimate,
    SciOxideError, SolveContext,
};
use scioxide_core::{Scaled, ScaledValue};

const EXHAUSTED: &str = "Schur reordering work budget exhausted";

// Candidate T and Q, real restandardization copies, block starts/sizes and
// selection flags, and the Sylvester right-hand side for estimates.
pub(super) fn reorder_lengths<S: NumericScalar>(
    n: usize,
    estimates: Estimation,
) -> Result<DirectLengths, SciOxideError> {
    let matrix = square(n)?;
    Ok(DirectLengths {
        scalars: total(&[matrix, matrix])?,
        reals: if S::IS_REAL {
            total(&[matrix, matrix])?
        } else {
            0
        },
        indices: times(3, n)?,
        extended: match estimates {
            Estimation::Skip => 0,
            Estimation::Estimate => times(n / 2, n - n / 2)?,
        },
    })
}

impl<S: NumericScalar> SchurDecomposition<S> {
    /// Returns the first `dimension` columns of `Q`, an orthonormal basis of
    /// the invariant subspace of the leading `dimension` eigenvalues of `T`.
    ///
    /// After [`Self::reorder`], pass [`SchurReorderReport::selected_dimension`]
    /// to get the subspace of the selected eigenvalues.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` exceeds the order or splits a real 2×2 block.
    pub fn invariant_subspace(&self, dimension: usize) -> Result<MatrixView<'_, S>, SciOxideError> {
        super::support::validate_subspace(self.q.rows(), dimension, self.blocks())?;
        self.q.view().block(0, 0, self.q.rows(), dimension)
    }

    /// Returns a copy with the selected eigenvalues moved to the leading block of `T`.
    ///
    /// `selection[i]` selects the eigenvalue at diagonal position `i`.
    /// Selecting either member of a real 2×2 block selects both. The selected
    /// blocks keep their relative order, and so do the others. Adjacent
    /// diagonal blocks are swapped by orthogonal (unitary) similarity
    /// transformations computed from a small Sylvester equation. Each swap is
    /// accepted only if it passes weak and strong backward-error tests, and
    /// moved 2×2 blocks are standardized again. The leading columns of the new
    /// `Q` span the invariant subspace of the selected eigenvalues (see
    /// [`Self::invariant_subspace`]). With [`Estimation::Estimate`], the report
    /// also contains condition estimates.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `selection.len()` is not the order.
    /// - `NumericalInstability` with
    ///   [`IllConditioned`](scioxide_core::NumericalIssue::IllConditioned) if
    ///   two adjacent blocks cannot be swapped stably.
    /// - `NumericalInstability` if the cluster condition estimate underflows or
    ///   `T` overflows when scaled back.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, SchurDecomposition};
    /// let a = Matrix::from_rows(&[[1.0_f64, 1.0], [0.0, 3.0]])?;
    /// let schur = SchurDecomposition::decompose(&a)?;
    /// // Move the eigenvalue 3 to the leading position.
    /// let (reordered, _report) = schur.reorder(&[false, true])?;
    /// assert!((reordered.eigenvalues()[0].re - 3.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - Z. Bai and J. W. Demmel, "On swapping diagonal blocks in real Schur
    ///   form", *Linear Algebra Appl.* 186, 1993.
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §7.6.
    pub fn reorder(
        &self,
        selection: &[bool],
    ) -> Result<(Self, SchurReorderReport<S::Real>), SciOxideError> {
        self.reorder_with_options(selection, &crate::SchurReorderOptions::default())
    }

    /// Runs [`reorder`](Self::reorder) with `options`, which set whether condition estimates are
    /// computed and the execution limits.
    ///
    /// ```
    /// use scioxide_linalg::{Estimation, Matrix, SchurDecomposition, SchurReorderOptions};
    /// let a = Matrix::from_rows(&[[1.0_f64, 1.0], [0.0, 3.0]])?;
    /// let schur = SchurDecomposition::decompose(&a)?;
    /// let options = SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate);
    /// let (reordered, report) = schur.reorder_with_options(&[false, true], &options)?;
    /// assert!((reordered.eigenvalues()[0].re - 3.0).abs() < 1e-12);
    /// assert!(report.separation().is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`reorder`](Self::reorder), and:
    ///
    /// - `ConvergenceFailure` if the work budget of `options` is exhausted.
    pub fn reorder_with_options(
        &self,
        selection: &[bool],
        options: &SchurReorderOptions,
    ) -> Result<(Self, SchurReorderReport<S::Real>), SciOxideError> {
        let mut output = self.try_clone()?;
        let report = output.reorder_in_place(
            selection,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }

    /// Moves the selected eigenvalues to the leading block of `T` in place.
    ///
    /// This is the reusable form of [`Self::reorder`], which describes the
    /// method. Size the workspace with
    /// [`SchurWorkspaceRequest::reorder`](crate::SchurWorkspaceRequest::reorder).
    /// Swaps and Sylvester operator applications count toward the work budget
    /// of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::reorder`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn reorder_in_place<'scope, 'a>(
        &mut self,
        selection: &[bool],
        options: &SchurReorderOptions,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<SchurReorderReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        scope.kernel().execution().check()?;
        let n = self.t.rows();
        if selection.len() != n {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(n),
                Dimensions::one(selection.len()),
            ));
        }
        let requirements = direct_requirements::<S>(
            reorder_lengths::<S>(n, options.estimates)?,
            options.execution.byte_limit(),
        )?;
        workspace.try_reserve(&requirements)?;
        // Standardization can split blocks, so publication may need n blocks.
        self.reserve(n)?;
        let options = *options;
        workspace.run_scope(
            scope,
            identity(
                scioxide_core::algorithm_id!("linalg", "schur_reorder", 1),
                n,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, _, solve_context| {
                workspace.with_direct(|mut scalars, mut reals, mut indices, child| {
                    let mut scratch = ReorderScratch {
                        t: take_square(&mut scalars, n)?,
                        q: take_square(&mut scalars, n)?,
                        real_t: take_mut(&mut reals, if S::IS_REAL { square(n)? } else { 0 })?,
                        real_q: take_mut(&mut reals, if S::IS_REAL { square(n)? } else { 0 })?,
                        blocks: take_mut(&mut indices, 2 * n)?.as_chunks_mut::<2>().0,
                        selected: take_mut(&mut indices, n)?,
                        rhs: child.extended_mut(),
                    };
                    scratch.t.as_mut_slice().copy_from_slice(self.t.as_slice());
                    scratch.q.as_mut_slice().copy_from_slice(self.q.as_slice());
                    let report = reorder_candidate(
                        &mut scratch,
                        &self.blocks,
                        selection,
                        &options,
                        solve_context,
                    )?;
                    solve_context.execution().check()?;
                    let _ = solve_context.counts().total()?;
                    self.publish(scratch.t.as_slice(), scratch.q.as_slice(), n)?;
                    Ok(report)
                })
            },
        )
    }
}

struct ReorderScratch<'w, S: NumericScalar> {
    t: ScratchMatrix<'w, S>,
    q: ScratchMatrix<'w, S>,
    real_t: &'w mut [S::Real],
    real_q: &'w mut [S::Real],
    blocks: &'w mut [[usize; 2]],
    selected: &'w mut [usize],
    rhs: &'w mut [Extended<S>],
}

fn reorder_candidate<S: NumericScalar>(
    scratch: &mut ReorderScratch<'_, S>,
    blocks: &[super::SchurBlock],
    selection: &[bool],
    options: &SchurReorderOptions,
    solve_context: &mut SolveContext<'_, '_>,
) -> Result<SchurReorderReport<S::Real>, SciOxideError> {
    let n = scratch.t.rows();
    let count = blocks.len();
    let scale = crate::kernel::standard::slice_maximum(scratch.t.as_slice());
    super::normalize_slice(scratch.t.as_mut_slice(), scale);
    for (target, block) in scratch.blocks.iter_mut().zip(blocks) {
        *target = [block.start(), block.size()];
    }
    let mut iterations = 0;
    let (selected_dimension, swaps) = super::support::reorder_blocks(
        &mut scratch.blocks[..count],
        scratch.selected,
        selection,
        || {
            super::support::reorder_step(
                &mut iterations,
                options.iteration_budget().get(),
                solve_context,
                EXHAUSTED,
            )
        },
        |start, left, right| {
            super::schur_swap::swap(&mut scratch.t, &mut scratch.q, start, left, right)
        },
    )?;
    // Swaps leave moved 2x2 blocks in a valid but non-standard shape;
    // restandardize them so the eigenvalues can be read from T.
    if S::IS_REAL {
        for (v, s) in scratch.real_t.iter_mut().zip(scratch.t.as_slice()) {
            *v = s.real_component();
        }
        for (v, s) in scratch.real_q.iter_mut().zip(scratch.q.as_slice()) {
            *v = s.real_component();
        }
        for &[start, size] in &scratch.blocks[..count] {
            if size == 2 {
                super::real_schur::standardize_pair(
                    scratch.real_t,
                    n,
                    start,
                    Some(&mut *scratch.real_q),
                )?;
            }
        }
        for (v, &s) in scratch
            .t
            .as_mut_slice()
            .iter_mut()
            .zip(scratch.real_t.iter())
        {
            *v = S::from_real(s);
        }
        for (v, &s) in scratch
            .q
            .as_mut_slice()
            .iter_mut()
            .zip(scratch.real_q.iter())
        {
            *v = S::from_real(s);
        }
    }
    for_each_schur_block(scratch.t.as_slice(), n, |_, _| {})?;
    let (condition, separation) = match options.estimates {
        Estimation::Estimate => {
            let (condition, sep) = estimate(
                &scratch.t,
                selected_dimension,
                scratch.rhs,
                &mut iterations,
                options,
                solve_context,
            )?;
            let sep = sep.mul(Scaled::from_value(scale)?)?.round()?;
            (
                Some(ConditionEstimate::new(
                    condition,
                    EstimateReliability::ModelBased,
                )?),
                Some(ReliableEstimate::new(sep, EstimateReliability::ModelBased)?),
            )
        }
        Estimation::Skip => (None, None),
    };
    for value in scratch.t.as_mut_slice() {
        *value = crate::checked::mul(*value, S::from_real(scale))?;
    }
    for_each_schur_block(scratch.t.as_slice(), n, |_, _| {})?;
    Ok(SchurReorderReport {
        selected: selected_dimension,
        swaps,
        iterations,
        condition,
        separation,
    })
}

fn estimate<S: NumericScalar>(
    t: &ScratchMatrix<'_, S>,
    m: usize,
    rhs: &mut [ScaledValue<scioxide_core::Complex<S::Real>>],
    iterations: &mut usize,
    options: &SchurReorderOptions,
    solve_context: &mut SolveContext<'_, '_>,
) -> Result<(S::Real, Scaled<S::Real>), SciOxideError> {
    let n = t.rows();
    let cols = n - m;
    let count = m * cols;
    if count == 0 {
        let mut norm = Scaled::ZERO;
        for j in 0..n {
            let mut sum = Scaled::ZERO;
            for i in 0..n {
                sum = sum.add(extended(t.get_element(i, j))?.magnitude()?)?;
            }
            if !sum.abs_le(norm) {
                norm = sum;
            }
        }
        return Ok((S::Real::ONE, norm));
    }
    // xTRSEN estimates: with `T = [T11 T12; 0 T22]`, solve
    // `T11 R - R T22 = T12`; the spectral projector norm gives the cluster's
    // reciprocal condition `1 / sqrt(1 + ||R||_F^2)`, and
    // `sep(T11, T22) = 1 / ||Sylv^{-1}||` is estimated by the 1-norm
    // estimator below.
    let rhs = &mut rhs[..count];
    super::support::projector_rhs(t.try_view()?, m, rhs)?;
    let ([condition], [separation]) =
        super::support::projector_estimates(rhs, |rhs, adjoint, _| {
            super::support::reorder_step(
                iterations,
                options.iteration_budget().get(),
                solve_context,
                EXHAUSTED,
            )?;
            solve(t, m, rhs, adjoint)
        })?;
    Ok(separation.map_or((S::Real::ZERO, Scaled::ZERO), |s| (condition, s)))
}
