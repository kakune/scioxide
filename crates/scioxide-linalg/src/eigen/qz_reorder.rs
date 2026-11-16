//! Reordering of a generalized Schur decomposition, selected eigenvalues first, with diagnostics.
use super::qz::{for_each_generalized_value, normalize, restore, round_real};
use super::qz_kernel::Pencil;
use super::support::{DirectLengths, direct_requirements, identity, square, times};
use super::{GeneralizedSchurDecomposition, GeneralizedSchurReorderReport, SchurReorderOptions};
use crate::kernel::frames::extended;
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

const EXHAUSTED: &str = "QZ reordering budget exhausted";

// Candidate S, T, Q and Z; block (start, size) pairs, selection and
// infinite-eigenvalue flags; both coupled Sylvester right-hand sides.
pub(super) fn reorder_lengths(
    n: usize,
    estimates: Estimation,
) -> Result<DirectLengths, SciOxideError> {
    Ok(DirectLengths {
        scalars: times(4, square(n)?)?,
        reals: 0,
        indices: times(4, n)?,
        extended: match estimates {
            Estimation::Skip => 0,
            Estimation::Estimate => times(2, times(n / 2, n - n / 2)?)?,
        },
    })
}

/// Arena-level selected-first reordering of a staged pencil with Q and Z.
///
/// The block structure and infinite eigenvalues are read from the staged
/// `(S, T)`. `indices` holds at least `4 n` entries and `rhs` at least
/// `2 floor(n/2) ceil(n/2)` values when estimates are requested. Work units
/// are charged to `solve`; failure leaves the staged pencil indeterminate.
pub(crate) fn qz_reorder<S: NumericScalar>(
    pencil: &mut Pencil<'_, S>,
    selection: &[bool],
    mut indices: &mut [usize],
    rhs: &mut [Extended<S>],
    options: &SchurReorderOptions,
    solve: &mut SolveContext<'_, '_>,
) -> Result<GeneralizedSchurReorderReport<S::Real>, SciOxideError> {
    let n = pencil.s.rows();
    let (blocks, _) = take_mut(&mut indices, 2 * n)?.as_chunks_mut::<2>();
    let selected = take_mut(&mut indices, n)?;
    let infinite = take_mut(&mut indices, n)?;
    let mut count = 0;
    for_each_generalized_value(&pencil.s, &pencil.t, |block, values| {
        blocks[count] = [block.start(), block.size()];
        for (flag, value) in infinite[block.start()..].iter_mut().zip(values) {
            *flag = usize::from(value.is_infinite());
        }
        count += 1;
    })?;
    let sa = normalize(pencil.s.as_mut_slice())?;
    let sb = normalize(pencil.t.as_mut_slice())?;
    let mut iterations = 0;
    let (selected_dimension, swaps) = super::support::reorder_blocks(
        &mut blocks[..count],
        selected,
        selection,
        || {
            super::support::reorder_step(
                &mut iterations,
                options.iteration_budget().get(),
                solve,
                EXHAUSTED,
            )
        },
        |start, left, right| {
            super::qz_swap::swap(pencil, infinite, start, left, right)?;
            infinite[start..start + left + right].rotate_left(left);
            Ok(())
        },
    )?;
    // Adjacent swaps disturb the diagonal real 2x2 T blocks. Restore their
    // canonical shape, applying the same transformations to Q and Z.
    if S::IS_REAL {
        for &[start, size] in &blocks[..count] {
            if size == 2 {
                super::qz_kernel::canonical_pair(pencil, start)?;
            }
        }
    }
    restore(pencil.s.as_mut_slice(), sa)?;
    restore(pencil.t.as_mut_slice(), sb)?;
    // Swaps update Q and Z with plain rotations; check them once here.
    for value in [pencil.q.as_ref(), pencil.z.as_ref()]
        .into_iter()
        .flatten()
        .flat_map(|m| m.as_slice())
    {
        crate::checked::finite_candidate(*value)?;
    }
    count = 0;
    for_each_generalized_value(&pencil.s, &pencil.t, |block, _| {
        blocks[count] = [block.start(), block.size()];
        count += 1;
    })?;
    let mut report = GeneralizedSchurReorderReport {
        selected: selected_dimension,
        swaps,
        iterations: 0,
        left: None,
        right: None,
        separation: None,
    };
    if options.estimates == Estimation::Estimate {
        let (left, right, separation) = estimate(
            &pencil.s,
            &pencil.t,
            &blocks[..count],
            selected_dimension,
            rhs,
            || {
                super::support::reorder_step(
                    &mut iterations,
                    options.iteration_budget().get(),
                    solve,
                    EXHAUSTED,
                )
            },
        )?;
        report.left = Some(ConditionEstimate::new(
            left,
            EstimateReliability::ModelBased,
        )?);
        report.right = Some(ConditionEstimate::new(
            right,
            EstimateReliability::ModelBased,
        )?);
        report.separation = Some([
            ReliableEstimate::new(separation[0], EstimateReliability::ModelBased)?,
            ReliableEstimate::new(separation[1], EstimateReliability::ModelBased)?,
        ]);
    }
    report.iterations = iterations;
    Ok(report)
}

impl<S: NumericScalar> GeneralizedSchurDecomposition<S> {
    /// Returns the first `dimension` columns of `Q`, an orthonormal basis of
    /// the left deflating subspace of the leading `dimension` eigenvalues.
    ///
    /// After [`Self::reorder`], pass
    /// [`GeneralizedSchurReorderReport::selected_dimension`] to get the
    /// subspace of the selected eigenvalues.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` exceeds the order or splits a real 2×2 block.
    pub fn left_deflating_subspace(
        &self,
        dimension: usize,
    ) -> Result<MatrixView<'_, S>, SciOxideError> {
        self.subspace(dimension, true)
    }
    /// Returns the first `dimension` columns of `Z`, an orthonormal basis of
    /// the right deflating subspace of the leading `dimension` eigenvalues.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` exceeds the order or splits a real 2×2 block.
    pub fn right_deflating_subspace(
        &self,
        dimension: usize,
    ) -> Result<MatrixView<'_, S>, SciOxideError> {
        self.subspace(dimension, false)
    }
    fn subspace(&self, m: usize, left: bool) -> Result<MatrixView<'_, S>, SciOxideError> {
        let n = self.s.rows();
        super::support::validate_subspace(n, m, self.blocks())?;
        (if left { &self.q } else { &self.z })
            .view()
            .block(0, 0, n, m)
    }

    /// Returns a copy with the selected eigenvalues moved to the leading blocks of `(S, T)`.
    ///
    /// `selection[i]` selects the eigenvalue at diagonal position `i`.
    /// Selecting either member of a real 2×2 block selects both. The selected
    /// blocks keep their relative order, and so do the others. Adjacent
    /// diagonal blocks are exchanged by orthogonal (unitary) equivalence
    /// transformations computed from a small coupled Sylvester system. Each
    /// exchange is accepted only if it passes weak and strong backward-error
    /// tests, and exact zeros of `T` for infinite eigenvalues are kept. The
    /// leading columns of the new `Q` and `Z` span the deflating subspaces of
    /// the selected eigenvalues. With [`Estimation::Estimate`], the report
    /// also contains condition estimates.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `selection.len()` is not the order.
    /// - `NumericalInstability` with
    ///   [`IllConditioned`](scioxide_core::NumericalIssue::IllConditioned) if
    ///   two adjacent blocks cannot be exchanged stably.
    /// - `NumericalInstability` if a condition estimate underflows or a result
    ///   overflows when scaled back.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{GeneralizedSchurDecomposition, Matrix};
    /// let a = Matrix::from_rows(&[[1.0_f64, 1.0], [0.0, 3.0]])?;
    /// let b = Matrix::<f64>::identity(2)?;
    /// let qz = GeneralizedSchurDecomposition::decompose(&a, &b)?;
    /// // Move the eigenvalue 3 to the leading position.
    /// let (reordered, _report) = qz.reorder(&[false, true])?;
    /// assert!((reordered.eigenvalues()[0].ratio()?.re - 3.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// B. Kågström and P. Poromaa, "Computing eigenspaces with specified
    /// eigenvalues of a regular matrix pair (A, B) and condition estimation:
    /// theory, algorithms and software", *Numer. Algorithms* 12, 1996.
    pub fn reorder(
        &self,
        selection: &[bool],
    ) -> Result<(Self, GeneralizedSchurReorderReport<S::Real>), SciOxideError> {
        self.reorder_with_options(selection, &crate::SchurReorderOptions::default())
    }

    /// Runs [`reorder`](Self::reorder) with `options`, which set whether condition estimates are
    /// computed and the execution limits.
    ///
    /// ```
    /// use scioxide_linalg::{Estimation, GeneralizedSchurDecomposition, Matrix, SchurReorderOptions};
    /// let a = Matrix::from_rows(&[[1.0_f64, 1.0], [0.0, 3.0]])?;
    /// let qz = GeneralizedSchurDecomposition::decompose(&a, &Matrix::<f64>::identity(2)?)?;
    /// let options = SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate);
    /// let (reordered, report) = qz.reorder_with_options(&[false, true], &options)?;
    /// assert!((reordered.eigenvalues()[0].ratio()?.re - 3.0).abs() < 1e-12);
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
    ) -> Result<(Self, GeneralizedSchurReorderReport<S::Real>), SciOxideError> {
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

    /// Moves the selected eigenvalues to the leading blocks of `(S, T)` in place.
    ///
    /// This is the reusable form of [`Self::reorder`], which describes the
    /// method. Size the workspace with
    /// [`GeneralizedSchurWorkspaceRequest::reorder`](crate::GeneralizedSchurWorkspaceRequest::reorder).
    /// Swaps and coupled Sylvester operator applications count toward the
    /// work budget of `options`.
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
    ) -> Result<GeneralizedSchurReorderReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        scope.kernel().execution().check()?;
        let n = self.s.rows();
        if selection.len() != n {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(n),
                Dimensions::one(selection.len()),
            ));
        }
        let requirements = direct_requirements::<S>(
            reorder_lengths(n, options.estimates)?,
            options.execution.byte_limit(),
        )?;
        workspace.try_reserve(&requirements)?;
        // Reserve the maximum number of blocks before transactional staging.
        self.reserve(n)?;
        let options = *options;
        workspace.run_scope(
            scope,
            identity(scioxide_core::algorithm_id!("linalg", "qz_reorder", 1), n),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, _, solve| {
                workspace.with_direct(|mut scalars, _, indices, child| {
                    let mut pencil = Pencil {
                        s: take_square(&mut scalars, n)?,
                        t: take_square(&mut scalars, n)?,
                        q: Some(take_square(&mut scalars, n)?),
                        z: Some(take_square(&mut scalars, n)?),
                    };
                    for (target, source) in [(&mut pencil.s, &self.s), (&mut pencil.t, &self.t)]
                        .into_iter()
                        .chain(pencil.q.as_mut().map(|q| (q, &self.q)))
                        .chain(pencil.z.as_mut().map(|z| (z, &self.z)))
                    {
                        target.as_mut_slice().copy_from_slice(source.as_slice());
                    }
                    let report = qz_reorder(
                        &mut pencil,
                        selection,
                        indices,
                        child.extended_mut(),
                        &options,
                        solve,
                    )?;
                    solve.execution().check()?;
                    let _ = solve.counts().total()?;
                    let counts = self
                        .report
                        .map_or_else(scioxide_core::SolveCounts::new, |report| report.counts());
                    let _ = self.publish(&pencil, counts)?;
                    Ok(report)
                })
            },
        )
    }
}

type Estimates<R> = (R, R, [R; 2]);
fn estimate<S: NumericScalar>(
    s: &ScratchMatrix<'_, S>,
    t: &ScratchMatrix<'_, S>,
    blocks: &[[usize; 2]],
    m: usize,
    rhs: &mut [ScaledValue<scioxide_core::Complex<S::Real>>],
    mut admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<Estimates<S::Real>, SciOxideError> {
    let n = s.rows();
    let cols = n - m;
    let count = m * cols;
    if count == 0 {
        let mut norm = Scaled::ZERO;
        for a in [s, t] {
            for &v in a.as_slice() {
                let mag = extended(v)?.magnitude()?;
                norm = norm.add(mag.mul(mag)?)?;
            }
        }
        let norm = norm.sqrt()?.round()?;
        return Ok((S::Real::ONE, S::Real::ONE, [norm; 2]));
    }
    // xTGSEN-style estimates: solving the coupled Sylvester system with
    // right-hand side `(S12, T12)` gives `(R, L)`; the reciprocal norms of
    // the right and left spectral projectors are `1 / sqrt(1 + ||R||_F^2)`
    // and `1 / sqrt(1 + ||L||_F^2)`. The two Difu/Difl separations are
    // estimated from the operator and its reversed-role counterpart.
    let rhs = &mut rhs[..2 * count];
    for (target, matrix) in rhs.chunks_exact_mut(count).zip([s, t]) {
        super::support::projector_rhs(matrix.try_view()?, m, target)?;
    }
    let ([right, left], separations) =
        super::support::projector_estimates::<_, 2, 2>(rhs, |rhs, adjoint, reverse| {
            admit()?;
            crate::kernel::generalized_sylvester::solve(s, t, blocks, m, rhs, adjoint, reverse)
        })?;
    let [forward, backward] = separations;
    Ok((
        left,
        right,
        [
            forward.map_or(Ok(S::Real::ZERO), round_real)?,
            backward.map_or(Ok(S::Real::ZERO), round_real)?,
        ],
    ))
}
