//! General (non-Hermitian) dense eigenvalues and eigenvectors.
//!
//! The matrix is scaled, balanced and reduced to Hessenberg form (see
//! `hessenberg`); real eigenvalue-only problems skip the reduction when the
//! balanced matrix is already upper Hessenberg. Real input stays real and uses
//! Francis double-shift QR in the real region of a
//! `LinalgWorkspace<Complex<R>>`; complex vectors are formed only in the
//! output. Complex input uses single-shift complex QR. Eigenvectors of every
//! requested side come from one Schur form (`schur_vectors`) and are
//! back-transformed with matrix products. The residuals `A V − V Λ` and the
//! linear independence of the vectors are checked in balanced coordinates;
//! residuals in input coordinates are checked after balancing is undone with
//! rescaling against overflow. QR steps are counted in a nested solve.

mod balance;
mod vectors;
use super::real_schur::{HessenbergProblem, SWEEP, real_hessenberg_eigenvalues};
use crate::GeneralEigenvaluesResult;
use crate::workspace::{take_mut, take_square};
use balance::balance;
use core::cmp::Ordering;
pub(crate) use vectors::right_vectors_from_schur;
use vectors::{entry, member, vectors_candidate};

use scioxide_core::AlgebraicScalar;
use scioxide_core::Complex;
use scioxide_core::checked_float_quotient;
use scioxide_core::{
    EstimateReliability, ExecutionContext, ExecutionPolicy, NonFiniteOrigin, NumericScalar,
    NumericalIssue, RealScalar, SciOxideError, SolveContext, Tolerance,
};

use crate::kernel::{GemmConfig, GemmOp, WorkspaceGemmProblem, gemm_workspace_with_context};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::{
    Accuracy, ConditionEstimate, DenseWorkspaceRequest, FiniteMatrixView, KernelContext,
    LinalgExecutionOptions, LinalgRequirements, LinalgSlices, LinalgWorkspace, MatrixView,
    MatrixViewMut, SchurOptions,
};

use super::hessenberg;
use super::schur::{complex_schur_form, for_each_schur_block, real_schur_form};
use super::schur_vectors::{self, Side};
use super::support::{
    DirectLengths, direct_requirements, identity, square, square_finite, times, total,
};
use super::types::reserve;
use super::{
    EigenvalueOrder, GeneralEigenDecomposition, GeneralEigenOptions, GeneralEigenvaluesBuffer,
    GeneralEigenvectorSelection, normalize_slice,
};

/// Eigenvectors per back-transformation product.
const BLOCK: usize = 32;

// Values-only staging: the Hessenberg matrix, reduction scratch and balancing
// scale live in the real arena for real input, and in the complex scalar
// arena (with the candidate values) for complex input.
pub(super) fn values_lengths<S: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<DirectLengths, SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    Ok(if S::IS_REAL {
        DirectLengths {
            scalars: n,
            reals: total(&[
                square(n)?,
                hessenberg::Scratch::<S::Real>::len(n, policy)?,
                n,
                times(SWEEP, n)?,
            ])?,
            indices: n,
            extended: 0,
        }
    } else {
        DirectLengths {
            scalars: total(&[
                square(n)?,
                hessenberg::Scratch::<Complex<S::Real>>::len(n, policy)?,
                n,
            ])?,
            reals: n,
            indices: n,
            extended: 0,
        }
    })
}

/// Requested eigenvector sides.
fn sides(selection: GeneralEigenvectorSelection) -> usize {
    usize::from(selection.includes_left()) + usize::from(selection.includes_right())
}

/// Product scratch of the back-transformation and residual products.
fn products<W: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<LinalgRequirements, SciOxideError> {
    let options = LinalgExecutionOptions::new().with_execution_policy(policy);
    [
        DenseWorkspaceRequest::product(BLOCK.min(n), n, n, &options),
        DenseWorkspaceRequest::product(n, n, n, &options),
    ]
    .iter()
    .try_fold(LinalgRequirements::default(), |requirements, request| {
        Ok(requirements.max(&crate::kernel::product_requirements::<W>(request)?))
    })
}

/// Working-scalar arena of a vector job: the normalized input, the Schur
/// factor (later the residual product), the Schur vectors, Schur-coordinate
/// vectors (later the independence basis), one vector matrix per requested
/// side, work rows for the QR sweeps and the vector solves, the Hessenberg
/// scratch and product scratch.
fn stage_len<W: NumericScalar>(
    n: usize,
    sides: usize,
    policy: ExecutionPolicy,
) -> Result<usize, SciOxideError> {
    total(&[
        times(4 + sides, square(n)?)?,
        times(SWEEP, n)?,
        hessenberg::Scratch::<W>::len(n, policy)?,
        products::<W>(n, policy)?.scalar_len()?,
    ])
}

// Vector staging: real input stages its matrices in the real arena, complex
// input in the scalar arena. Per-vector values and phase factors are
// complex; scale, residuals and conditions are real; the balancing
// permutation and the output order are indices.
pub(super) fn vectors_lengths<S: NumericScalar>(
    n: usize,
    selection: GeneralEigenvectorSelection,
    policy: ExecutionPolicy,
) -> Result<DirectLengths, SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let sides = sides(selection);
    let both = usize::from(selection == GeneralEigenvectorSelection::Both);
    let complex = times(2 + sides, n)?;
    let real = times(2 + sides + both, n)?;
    Ok(if S::IS_REAL {
        DirectLengths {
            scalars: complex,
            reals: total(&[real, stage_len::<S::Real>(n, sides, policy)?])?,
            indices: times(2, n)?,
            extended: 0,
        }
    } else {
        DirectLengths {
            scalars: total(&[complex, stage_len::<Complex<S::Real>>(n, sides, policy)?])?,
            reals: real,
            indices: times(2, n)?,
            extended: 0,
        }
    })
}

fn values_candidate<'w, S, R>(
    matrix: FiniteMatrixView<'_, S>,
    options: &GeneralEigenOptions<R>,
    mut scalars: &'w mut [Complex<R>],
    mut reals: &'w mut [R],
    mut indices: &'w mut [usize],
    solve: &mut SolveContext<'_, '_>,
) -> Result<&'w [Complex<R>], SciOxideError>
where
    S: NumericScalar<Real = R>,
    R: RealScalar,
    Complex<R>: NumericScalar<Real = R>,
{
    let n = matrix.rows();
    let reduction_scale = matrix.maximum;
    let values = take_mut(&mut scalars, n)?;
    let scale = take_mut(&mut reals, n)?;
    let permutation = take_mut(&mut indices, n)?;
    // Values only: real input uses the real double-shift QR (half the
    // storage and roughly a quarter of the complex flops, with conjugate
    // pairs produced exactly); complex input uses complex Schur QR. An
    // input that is already upper Hessenberg skips the reduction.
    let policy = options.execution_options().execution_policy();
    let balanced_scale;
    if S::IS_REAL {
        let hessenberg = take_mut(&mut reals, square(n)?)?;
        let mut reduction = hessenberg::Scratch::carve(&mut reals, n, policy)?;
        let sweep = take_mut(&mut reals, times(SWEEP, n)?)?;
        copy_input(matrix.as_view(), hessenberg, reduction_scale);
        balance(hessenberg, n, scale, permutation, options.balancing())?;
        balanced_scale = normalize_balanced(hessenberg);
        if !is_real_upper_hessenberg(hessenberg, n) {
            let mut reduced = ScratchMatrix::new(&mut *hessenberg, n, n)?;
            hessenberg::reduce(&mut reduced, &mut reduction, policy, solve.execution())?;
            hessenberg::clear_reflectors(&mut reduced, hessenberg::Packing::Hessenberg);
        }
        real_hessenberg_eigenvalues(
            HessenbergProblem {
                matrix: hessenberg,
                values: Some(&mut *values),
                vectors: None,
                work: sweep,
            },
            n,
            options.deflation_tolerance(),
            super::qr_iteration_budget(options.iteration_budget(), n)?,
            solve,
        )?;
    } else {
        let mut hessenberg = take_square(&mut scalars, n)?;
        let mut reduction = hessenberg::Scratch::carve(&mut scalars, n, policy)?;
        copy_input(matrix.as_view(), hessenberg.as_mut_slice(), reduction_scale);
        balance(
            hessenberg.as_mut_slice(),
            n,
            scale,
            permutation,
            options.balancing(),
        )?;
        balanced_scale = normalize_balanced(hessenberg.as_mut_slice());
        complex_schur_form(
            &mut hessenberg,
            None,
            &mut reduction,
            &schur_options(options),
            solve,
        )?;
        values
            .iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = hessenberg.as_slice()[index * n + index]);
    }
    rescale_complex_values(values, [balanced_scale, reduction_scale])?;
    values.sort_unstable_by(|left, right| compare_complex(*left, *right, options.ordering()));
    validate_complex_output(values)?;
    Ok(values)
}

/// Schur controls of a general eigen job.
fn schur_options<R: RealScalar>(options: &GeneralEigenOptions<R>) -> SchurOptions<R> {
    SchurOptions::new()
        .with_execution_options(options.execution_options())
        .with_deflation_tolerance(options.deflation_tolerance())
}

/// Per-eigenvalue candidate data in Schur order. Phase, residual and
/// condition regions of sides that were not requested are empty.
struct Small<'w, R: RealScalar> {
    /// Eigenvalues in normalized units.
    values: &'w mut [Complex<R>],
    /// Eigenvalues in input units.
    eigenvalues: &'w mut [Complex<R>],
    right_phases: &'w mut [Complex<R>],
    left_phases: &'w mut [Complex<R>],
    scale: &'w mut [R],
    right_residuals: &'w mut [R],
    left_residuals: &'w mut [R],
    eigenvalue_conditions: &'w mut [R],
    eigenvector_conditions: &'w mut [R],
    permutation: &'w mut [usize],
    /// Schur index of each output eigenvalue.
    order: &'w mut [usize],
}

/// Arenas left after the per-eigenvalue regions.
type Rest<'w, R> = (&'w mut [Complex<R>], &'w mut [R]);

impl<'w, R: RealScalar> Small<'w, R> {
    fn carve(
        mut scalars: &'w mut [Complex<R>],
        mut reals: &'w mut [R],
        mut indices: &'w mut [usize],
        n: usize,
        selection: GeneralEigenvectorSelection,
    ) -> Result<(Self, Rest<'w, R>), SciOxideError> {
        let right = if selection.includes_right() { n } else { 0 };
        let left = if selection.includes_left() { n } else { 0 };
        let both = if selection == GeneralEigenvectorSelection::Both {
            n
        } else {
            0
        };
        let small = Self {
            values: take_mut(&mut scalars, n)?,
            eigenvalues: take_mut(&mut scalars, n)?,
            right_phases: take_mut(&mut scalars, right)?,
            left_phases: take_mut(&mut scalars, left)?,
            scale: take_mut(&mut reals, n)?,
            right_residuals: take_mut(&mut reals, right)?,
            left_residuals: take_mut(&mut reals, left)?,
            eigenvalue_conditions: take_mut(&mut reals, both)?,
            eigenvector_conditions: take_mut(&mut reals, n)?,
            permutation: take_mut(&mut indices, n)?,
            order: take_mut(&mut indices, n)?,
        };
        Ok((small, (scalars, reals)))
    }
}

/// Working-scalar staging of a vector job (see [`stage_len`]). Vector
/// matrices hold one vector per row.
struct Stage<'w, W: NumericScalar> {
    /// Balanced input = `balance_scale * t` before the Schur reduction.
    balance_scale: W::Real,
    a: &'w mut [W],
    t: &'w mut [W],
    z: &'w mut [W],
    x: &'w mut [W],
    right: &'w mut [W],
    left: &'w mut [W],
    work: &'w mut [W],
    reduction: hessenberg::Scratch<'w, W>,
    gemm: LinalgWorkspace<'w, W>,
}

impl<'w, W: NumericScalar> Stage<'w, W> {
    fn carve(
        mut rest: &'w mut [W],
        n: usize,
        selection: GeneralEigenvectorSelection,
        policy: ExecutionPolicy,
    ) -> Result<Self, SciOxideError> {
        let matrix = square(n)?;
        let side = |included: bool| if included { matrix } else { 0 };
        let products = products::<W>(n, policy)?;
        Ok(Self {
            balance_scale: W::Real::ONE,
            a: take_mut(&mut rest, matrix)?,
            t: take_mut(&mut rest, matrix)?,
            z: take_mut(&mut rest, matrix)?,
            x: take_mut(&mut rest, matrix)?,
            right: take_mut(&mut rest, side(selection.includes_right()))?,
            left: take_mut(&mut rest, side(selection.includes_left()))?,
            work: take_mut(&mut rest, times(SWEEP, n)?)?,
            reduction: hessenberg::Scratch::carve(&mut rest, n, policy)?,
            gemm: LinalgWorkspace::from_slices(
                LinalgSlices {
                    scalars: take_mut(&mut rest, products.scalar_len()?)?,
                    ..LinalgSlices::default()
                },
                &products,
            )?,
        })
    }
}

/// Copy and normalize a logical view into working-scalar row-major storage.
fn copy_input<S: NumericScalar, W: NumericScalar<Real = S::Real>>(
    source: MatrixView<'_, S>,
    target: &mut [W],
    scale: S::Real,
) {
    let normalized = scale == S::Real::ZERO || scale == S::Real::ONE;
    let inverse = if normalized {
        S::Real::ONE
    } else {
        S::Real::ONE / scale
    };
    for (row, values) in target.chunks_exact_mut(source.cols().max(1)).enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            let entry = source.get_element(row, column);
            let entry = W::from_components(entry.real_component(), entry.imaginary_component());
            *value = if normalized {
                entry
            } else if inverse.is_finite() {
                entry * W::from_real(inverse)
            } else {
                crate::rotation::divide_real(entry, scale)
            };
        }
    }
}

/// Stage the normalized input `a` and its balanced copy in `t`.
fn stage_input<S, W, R>(
    matrix: FiniteMatrixView<'_, S>,
    options: &GeneralEigenOptions<R>,
    stage_data: &mut Stage<'_, W>,
    small: &mut Small<'_, R>,
) -> Result<(), SciOxideError>
where
    S: NumericScalar<Real = R>,
    W: NumericScalar<Real = R>,
    R: RealScalar,
{
    let n = matrix.rows();
    copy_input(matrix.as_view(), stage_data.a, matrix.maximum);
    stage_data.t.copy_from_slice(stage_data.a);
    balance(
        stage_data.t,
        n,
        small.scale,
        small.permutation,
        options.balancing(),
    )?;
    stage_data.balance_scale = normalize_balanced(stage_data.t);
    Ok(())
}

/// Balancing can shrink an entire matrix far below the absolute QR and pivot
/// thresholds. Normalize again so those thresholds refer to the balanced
/// problem, recording the scale separately from the similarity factors.
fn normalize_balanced<W: NumericScalar>(matrix: &mut [W]) -> W::Real {
    let maximum = crate::kernel::standard::slice_maximum(matrix);
    let scale = if maximum == W::Real::ZERO {
        W::Real::ONE
    } else {
        maximum
    };
    normalize_slice(matrix, scale);
    scale
}

fn is_real_upper_hessenberg<R: RealScalar>(matrix: &[R], order: usize) -> bool {
    (2..order).all(|row| (0..row - 1).all(|column| matrix[row * order + column] == R::ZERO))
}

fn rescale_complex_values<R: RealScalar>(
    values: &mut [Complex<R>],
    scales: [R; 2],
) -> Result<(), SciOxideError> {
    if scales == [R::ONE; 2] {
        return Ok(());
    }
    values.iter_mut().try_for_each(|value| {
        value.re = scioxide_core::checked_float_product_factors(&[value.re, scales[0], scales[1]])?;
        value.im = scioxide_core::checked_float_product_factors(&[value.im, scales[0], scales[1]])?;
        Ok(())
    })
}

fn compare_complex<R: RealScalar>(
    left: Complex<R>,
    right: Complex<R>,
    ordering: EigenvalueOrder,
) -> Ordering {
    let real = left.re.partial_cmp(&right.re).unwrap_or(Ordering::Equal);
    let ordered_real = match ordering {
        EigenvalueOrder::Ascending => real,
        EigenvalueOrder::Descending => real.reverse(),
    };
    ordered_real.then_with(|| right.im.partial_cmp(&left.im).unwrap_or(Ordering::Equal))
}

fn validate_complex_output<R: RealScalar>(values: &[Complex<R>]) -> Result<(), SciOxideError> {
    values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
        .then_some(())
        .ok_or_else(|| {
            SciOxideError::non_finite(NonFiniteOrigin::Output, "computed eigenvalue is not finite")
        })
}

fn validate_relative_residuals<R: RealScalar>(
    residuals: &[R],
    options: &GeneralEigenOptions<R>,
) -> Result<(), SciOxideError> {
    let tolerance = options.residual_tolerance().threshold(R::ONE)?
        * crate::checked::rounded_count::<R>(residuals.len().max(1))?
        * R::try_from_f64_exact(32.0)?;
    residuals
        .iter()
        .enumerate()
        .find(|(_, residual)| **residual > tolerance)
        .map_or(Ok(()), |(index, residual)| {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::DefectiveEigenbasis,
                alloc::format!(
                    "eigenvector {index} relative residual {residual:?} exceeds {tolerance:?}"
                ),
            ))
        })
}

fn compute_eigenvector_conditions<R: RealScalar>(
    eigenvalues: &[Complex<R>],
    value_scale: R,
    matrix_norm: R,
    conditions: &mut [R],
) -> Result<(), SciOxideError> {
    for ((index, eigenvalue), slot) in eigenvalues.iter().enumerate().zip(conditions) {
        let separation = eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(other, value)| (other != index).then_some((*eigenvalue - *value).norm()))
            .fold(R::MAX, R::min);
        let reciprocal = if eigenvalues.len() <= 1 {
            R::ONE
        } else if matrix_norm == R::ZERO {
            R::ZERO
        } else {
            scioxide_core::checked_product_quotient(separation, value_scale, matrix_norm)?
                .min(R::ONE)
        };
        let _ = ConditionEstimate::new(reciprocal, EstimateReliability::Heuristic)?;
        *slot = reciprocal;
    }
    Ok(())
}

impl<R: RealScalar> GeneralEigenvaluesResult<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Computes all eigenvalues of a general real or complex square matrix.
    ///
    /// The eigenvalues are complex and sorted by [`GeneralEigenOptions::ordering`].
    /// For a real matrix, complex eigenvalues come in exact conjugate pairs.
    /// The input is read in place, including strided and conjugated views.
    ///
    /// The matrix is scaled by its largest entry magnitude, balanced (see
    /// [`GeneralEigenBalancing`](crate::GeneralEigenBalancing)), reduced to
    /// upper Hessenberg form by Householder reflections, and iterated to Schur
    /// form. Real input uses
    /// Francis double-shift QR in real arithmetic; complex input uses
    /// single-shift complex QR. The cost is O(n³) time and O(n²) memory. Use
    /// [`GeneralEigenDecomposition::decompose`] for eigenvectors and
    /// [`SchurDecomposition::decompose`](crate::SchurDecomposition::decompose)
    /// for the Schur form.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if `matrix` contains NaN or infinity.
    /// - `ConvergenceFailure` if QR iteration exhausts its iteration budget.
    /// - `NumericalInstability` if balancing or scaling an eigenvalue back to
    ///   the input units overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{GeneralEigenvaluesResult, Matrix};
    /// // The rotation [[0, -1], [1, 0]] has eigenvalues ±i, listed +i first.
    /// let a = Matrix::from_vec(2, 2, vec![0.0_f64, -1.0, 1.0, 0.0])?;
    /// let values = GeneralEigenvaluesResult::compute(&a)?;
    /// assert!((values.eigenvalues()[0].im - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §7.4–7.5.
    /// - B. N. Parlett and C. Reinsch, "Balancing a matrix for calculation of
    ///   eigenvalues and eigenvectors", *Numer. Math.* 13, 1969.
    pub fn compute<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::compute_with_options(matrix, &crate::GeneralEigenOptions::default())
    }

    /// Runs [`compute`](Self::compute) with `options`, which set the balancing, the ordering, the
    /// eigenvector selection, the tolerances and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`compute`](Self::compute), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn compute_with_options<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        options: &GeneralEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = GeneralEigenvaluesBuffer::empty();
        let report = GeneralEigenvaluesBuffer::compute_into(
            matrix,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(Self { output, report })
    }
}

impl<R: RealScalar> GeneralEigenvaluesBuffer<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Computes all eigenvalues of a general square matrix into `output`.
    ///
    /// This is the reusable form of [`GeneralEigenvaluesResult::compute`],
    /// which describes the method. Size the workspace with a
    /// [`GeneralEigenWorkspaceRequest`](crate::GeneralEigenWorkspaceRequest).
    /// QR steps count toward the budget set by
    /// [`LinalgExecutionOptions::with_iteration_budget`] and appear in the
    /// `nested()` counts of a calling `SolveContext`, without using its own
    /// budget. Neither the Schur vectors nor the full Schur form are formed.
    ///
    /// # Errors
    ///
    /// - The errors of [`GeneralEigenvaluesResult::compute`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn compute_into<'scope, 'm, 'a, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        options: &GeneralEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, Complex<R>>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let kernels = scope.kernel();
        kernels.execution().check()?;
        let finite = square_finite(matrix.into(), kernels.dispatch())?;
        let n = finite.rows();
        let requirements = direct_requirements::<Complex<R>>(
            values_lengths::<S>(n, options.execution_options().execution_policy())?,
            options.execution_options().byte_limit(),
        )?;
        workspace.try_reserve(&requirements)?;
        reserve(
            &mut output.eigenvalues,
            n,
            scioxide_core::quantity!("output.eigenvalues"),
        )?;
        let options = *options;
        workspace.run_scope(
            scope,
            identity(
                scioxide_core::algorithm_id!("linalg", "general_eigenvalues", 1),
                n,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, _, solve| {
                workspace.with_direct(|scalars, reals, indices, _| {
                    let values =
                        values_candidate(finite, &options, scalars, reals, indices, solve)?;
                    let _ = solve.counts().total()?;
                    output.eigenvalues.clear();
                    output.eigenvalues.extend_from_slice(values);
                    Ok(())
                })?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )
    }
}
impl<R: RealScalar> GeneralEigenDecomposition<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Computes the eigenvalues and left or right eigenvectors of a general square matrix.
    ///
    /// [`GeneralEigenOptions::with_eigenvectors`] selects left, right or both
    /// sides; the conventions are stated on [`GeneralEigenDecomposition`].
    ///
    /// The matrix is scaled, balanced, reduced to Hessenberg form and iterated
    /// to Schur form `T = Zᴴ B Z` of the balanced matrix `B`, as in
    /// [`GeneralEigenvaluesResult::compute`]; a real matrix keeps a real
    /// quasi-triangular `T`. The eigenvectors of `T` are found by back
    /// substitution (right) or forward substitution (left), transformed by `Z`
    /// and the balancing, and normalized. Both sides come from the same Schur
    /// form. The cost is O(n³) time and O(n²) memory. Use
    /// [`Self::decompose_into`] to reuse storage or run in parallel.
    ///
    /// The result is checked before it is returned. Each relative residual
    /// must satisfy the bound of
    /// [`GeneralEigenOptions::with_residual_tolerance`], and the eigenvectors
    /// must be numerically linearly independent in balanced coordinates. A
    /// defective or nearly defective matrix fails this check.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if `matrix` contains NaN or infinity.
    /// - `ConvergenceFailure` if QR iteration exhausts its iteration budget.
    /// - `NumericalInstability` if balancing or scaling an eigenvalue back to
    ///   the input units overflows.
    /// - `NumericalInstability` with
    ///   [`DefectiveEigenbasis`](scioxide_core::NumericalIssue::DefectiveEigenbasis)
    ///   if the eigenvectors are numerically linearly dependent or a residual
    ///   exceeds its bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{GeneralEigenDecomposition, Matrix};
    /// // The triangular matrix [1 2; 0 3] has the eigenvalues 1 and 3.
    /// let a = Matrix::from_rows(&[[1.0_f64, 2.0], [0.0, 3.0]])?;
    /// let decomposition = GeneralEigenDecomposition::decompose(&a)?;
    /// let mut values: Vec<f64> = decomposition.eigenvalues().iter().map(|z| z.re).collect();
    /// values.sort_by(f64::total_cmp);
    /// assert!((values[0] - 1.0).abs() < 1e-14 && (values[1] - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §7.4–7.6.
    pub fn decompose<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::decompose_with_options(matrix, &crate::GeneralEigenOptions::default())
    }

    /// Runs [`decompose`](Self::decompose) with `options`, which set the balancing, the ordering,
    /// the eigenvector selection, the tolerances and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`decompose`](Self::decompose), and limit errors from `options` as described
    /// in [Execution](crate#execution).
    pub fn decompose_with_options<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        options: &GeneralEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::decompose_into(
            matrix,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes a general eigendecomposition into `output`.
    ///
    /// This is the reusable form of [`Self::decompose`], which describes the
    /// method and the checks. Size the workspace with a
    /// [`GeneralEigenWorkspaceRequest`](crate::GeneralEigenWorkspaceRequest).
    /// QR steps count toward the budget set by
    /// [`LinalgExecutionOptions::with_iteration_budget`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::decompose`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn decompose_into<'scope, 'm, 'a, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        options: &GeneralEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, Complex<R>>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let kernels = scope.kernel();
        kernels.execution().check()?;
        let finite = square_finite(matrix.into(), kernels.dispatch())?;
        let n = finite.rows();
        let selection = options.eigenvectors();
        let policy = options.execution_options().execution_policy();
        let requirements = direct_requirements::<Complex<R>>(
            vectors_lengths::<S>(n, selection, policy)?,
            options.execution_options().byte_limit(),
        )?;
        workspace.try_reserve(&requirements)?;
        output.reserve(n, selection)?;
        let options = *options;
        let report = workspace.run_scope(
            scope,
            identity(
                scioxide_core::algorithm_id!("linalg", "general_eigenvectors", 1),
                n,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, _, solve| {
                workspace.with_direct(|scalars, reals, indices, _| {
                    let (mut small, (scalars, reals)) =
                        Small::carve(scalars, reals, indices, n, selection)?;
                    if S::IS_REAL {
                        let mut stage = Stage::<R>::carve(reals, n, selection, policy)?;
                        stage_input(finite, &options, &mut stage, &mut small)?;
                        real_schur_form(
                            stage.t,
                            stage.z,
                            stage.work,
                            n,
                            &mut stage.reduction,
                            &schur_options(&options),
                            solve,
                        )?;
                        vectors_candidate(finite, &options, &mut stage, &mut small, solve)?;
                        let _ = solve.counts().total()?;
                        output.publish(&stage, &small, &options)
                    } else {
                        let mut stage = Stage::<Complex<R>>::carve(scalars, n, selection, policy)?;
                        stage_input(finite, &options, &mut stage, &mut small)?;
                        complex_schur_form(
                            &mut ScratchMatrix::new(&mut *stage.t, n, n)?,
                            Some(&mut ScratchMatrix::new(&mut *stage.z, n, n)?),
                            &mut stage.reduction,
                            &schur_options(&options),
                            solve,
                        )?;
                        vectors_candidate(finite, &options, &mut stage, &mut small, solve)?;
                        let _ = solve.counts().total()?;
                        output.publish(&stage, &small, &options)
                    }
                })?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )?;
        output.report = Some(report);
        Ok(report)
    }

    pub(crate) fn reserve(
        &mut self,
        n: usize,
        selection: GeneralEigenvectorSelection,
    ) -> Result<(), SciOxideError> {
        let quantity = scioxide_core::quantity!("output.eigen_storage");
        reserve(&mut self.eigenvalues, n, quantity)?;
        if selection.includes_left() {
            self.left_eigenvectors.try_reserve_shape(n, n)?;
            reserve(&mut self.left_residuals, n, quantity)?;
        }
        if selection.includes_right() {
            self.right_eigenvectors.try_reserve_shape(n, n)?;
            reserve(&mut self.right_residuals, n, quantity)?;
        }
        if selection == GeneralEigenvectorSelection::Both {
            reserve(&mut self.eigenvalue_conditions, n, quantity)?;
        }
        reserve(&mut self.balancing.scale_factors, n, quantity)?;
        reserve(&mut self.balancing.permutation, n, quantity)?;
        reserve(&mut self.eigenvector_conditions, n, quantity)
    }

    // Copy a validated candidate in output order. Storage was reserved
    // before the scope and condition values were validated by the
    // candidate, so nothing here fails after the first write; complex
    // vectors are formed here from their real or complex rows.
    fn publish<W: NumericScalar<Real = R>>(
        &mut self,
        stage: &Stage<'_, W>,
        small: &Small<'_, R>,
        options: &GeneralEigenOptions<R>,
    ) -> Result<(), SciOxideError> {
        let n = small.values.len();
        let selection = options.eigenvectors();
        let order = &*small.order;
        let gather = |target: &mut alloc::vec::Vec<R>, source: &[R], requested: bool| {
            target.clear();
            if requested {
                target.extend(order.iter().map(|&index| source[index]));
            }
        };
        let expand = |target: &mut crate::Matrix<Complex<R>>,
                      vectors: &[W],
                      factors: &[Complex<R>],
                      requested: bool|
         -> Result<(), SciOxideError> {
            if !requested {
                return target.try_resize_flat(0, 0);
            }
            target.try_resize_flat(n, n)?;
            let output = target.as_mut_slice();
            for (column, &index) in order.iter().enumerate() {
                let (start, size, conjugate) = member::<W>(small.values, index);
                for row in 0..n {
                    let value = entry(vectors, n, (start, size), row) * factors[start];
                    output[row * n + column] = if conjugate { value.conj() } else { value };
                }
            }
            Ok(())
        };
        expand(
            &mut self.right_eigenvectors,
            stage.right,
            small.right_phases,
            selection.includes_right(),
        )?;
        expand(
            &mut self.left_eigenvectors,
            stage.left,
            small.left_phases,
            selection.includes_left(),
        )?;
        gather(
            &mut self.right_residuals,
            small.right_residuals,
            selection.includes_right(),
        );
        gather(
            &mut self.left_residuals,
            small.left_residuals,
            selection.includes_left(),
        );
        self.eigenvalues.clear();
        self.eigenvalues
            .extend(order.iter().map(|&index| small.eigenvalues[index]));
        self.balancing.strategy = options.balancing();
        self.balancing.scale_factors.clear();
        self.balancing.scale_factors.extend_from_slice(small.scale);
        self.balancing.permutation.clear();
        self.balancing
            .permutation
            .extend_from_slice(small.permutation);
        self.eigenvector_conditions.clear();
        for &index in order {
            self.eigenvector_conditions.push(ConditionEstimate::new(
                small.eigenvector_conditions[index],
                EstimateReliability::Heuristic,
            )?);
        }
        self.eigenvalue_conditions.clear();
        if selection == GeneralEigenvectorSelection::Both {
            for &index in order {
                self.eigenvalue_conditions.push(ConditionEstimate::new(
                    small.eigenvalue_conditions[index],
                    EstimateReliability::ModelBased,
                )?);
            }
        }
        self.selection = selection;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;
    use alloc::vec;

    fn complex_zero<R: RealScalar>() -> Complex<R> {
        Complex::new(R::ZERO, R::ZERO)
    }

    fn complex_one<R: RealScalar>() -> Complex<R> {
        Complex::new(R::ONE, R::ZERO)
    }

    fn with_solve<T>(work: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded();
        let mut solve = state
            .begin_solve(
                ExecutionContext::sequential(),
                identity(scioxide_core::algorithm_id!("test", "general", 1), 0),
                &mut budgets,
            )
            .unwrap();
        work(&mut solve)
    }

    fn cast<R: RealScalar>(value: f64) -> R {
        R::try_from_f64_exact(value).unwrap_or(R::ZERO)
    }

    fn complex_matrix<R: RealScalar>(order: usize, values: &[(f64, f64)]) -> Matrix<Complex<R>>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        Matrix::from_vec(
            order,
            order,
            values
                .iter()
                .map(|&(real, imaginary)| Complex::new(cast(real), cast(imaginary)))
                .collect(),
        )
        .unwrap()
    }

    fn scaled_complex_norm<R: RealScalar>(values: impl IntoIterator<Item = Complex<R>>) -> R {
        let (scale, sum_squares) = values
            .into_iter()
            .flat_map(|value| [value.re, value.im])
            .fold((R::ZERO, R::ONE), |(scale, sum), value| {
                let magnitude = value.abs();
                if magnitude == R::ZERO {
                    (scale, sum)
                } else if scale < magnitude {
                    let ratio = scale / magnitude;
                    (magnitude, R::ONE + sum * ratio * ratio)
                } else {
                    let ratio = magnitude / scale;
                    (scale, sum + ratio * ratio)
                }
            });
        if scale == R::ZERO {
            scale
        } else {
            scale * sum_squares.sqrt()
        }
    }

    fn normalized_difference<R: RealScalar>(
        actual: &Matrix<Complex<R>>,
        expected: &Matrix<Complex<R>>,
    ) -> R
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let difference = scaled_complex_norm(
            actual
                .as_slice()
                .iter()
                .zip(expected.as_slice())
                .map(|(&actual, &expected)| actual - expected),
        );
        let scale = scaled_complex_norm(expected.as_slice().iter().copied());
        if scale == R::ZERO {
            difference
        } else {
            difference / scale
        }
    }

    fn unitarity_residual<R: RealScalar>(matrix: &Matrix<Complex<R>>) -> R
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let order = matrix.rows();
        let residual = scaled_complex_norm((0..order).flat_map(|left| {
            (0..order).map(move |right| {
                let inner = (0..order).fold(complex_zero(), |sum, row| {
                    sum + matrix.get_element(row, left).conj() * matrix.get_element(row, right)
                });
                inner
                    - if left == right {
                        complex_one()
                    } else {
                        complex_zero()
                    }
            })
        }));
        let dimension = crate::checked::rounded_count::<R>(order.max(1)).unwrap();
        residual / dimension.sqrt()
    }

    fn structure_residual<R: RealScalar>(matrix: &Matrix<Complex<R>>, band: usize) -> R
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let residual = scaled_complex_norm((0..matrix.rows()).flat_map(|row| {
            (0..matrix.cols())
                .filter(move |&column| row > column + band)
                .map(move |column| matrix.get_element(row, column))
        }));
        let scale = scaled_complex_norm(matrix.as_slice().iter().copied());
        if scale == R::ZERO {
            residual
        } else {
            residual / scale
        }
    }

    fn assert_factorization<R: RealScalar>(
        fixture: &str,
        source: &Matrix<Complex<R>>,
        factor: &Matrix<Complex<R>>,
        vectors: &Matrix<Complex<R>>,
        structure_band: usize,
    ) where
        Complex<R>: NumericScalar<Real = R>,
    {
        let adjoint = vectors.view().adjoint().try_to_matrix().unwrap();
        let reconstructed = vectors
            .view()
            .mul_mat(factor)
            .and_then(|product| product.view().mul_mat(&adjoint))
            .unwrap();
        let dimension = crate::checked::rounded_count::<R>(source.rows().max(1)).unwrap();
        let bound = cast::<R>(512.0) * dimension * <R as RealScalar>::EPSILON;
        for (metric, residual) in [
            (
                "reconstruction",
                normalized_difference(&reconstructed, source),
            ),
            ("unitarity", unitarity_residual(vectors)),
            ("structure", structure_residual(factor, structure_band)),
        ] {
            assert!(
                residual.is_finite() && residual <= bound,
                "{fixture}: {metric} residual {residual:?} exceeds {bound:?}"
            );
        }
    }

    fn scale_matrix<R: RealScalar>(matrix: &Matrix<Complex<R>>, scale: R) -> Matrix<Complex<R>>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let scale = Complex::new(scale, R::ZERO);
        Matrix::from_vec(
            matrix.rows(),
            matrix.cols(),
            matrix
                .as_slice()
                .iter()
                .map(|&value| value * scale)
                .collect(),
        )
        .unwrap()
    }

    fn exercise_complex_schur_invariants<R: RealScalar>()
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let non_normal = complex_matrix::<R>(
            4,
            &[
                (4.0, 1.0),
                (2.0, -1.0),
                (1.0, 0.5),
                (3.0, 1.0),
                (1.0, -0.5),
                (3.0, 2.0),
                (2.0, -1.5),
                (1.0, 0.25),
                (3.0, 1.0),
                (1.0, -1.0),
                (2.0, 0.5),
                (4.0, -0.5),
                (2.0, -1.0),
                (1.0, 0.75),
                (3.0, 1.25),
                (1.0, -2.0),
            ],
        );
        let clustered = R::ONE + cast::<R>(32.0) * <R as RealScalar>::EPSILON;
        let fixtures = vec![
            ("empty", Matrix::zeros(0, 0).unwrap()),
            ("singleton", complex_matrix(1, &[(-3.0, 2.0)])),
            (
                "normal",
                complex_matrix(
                    3,
                    &[
                        (4.0, 0.0),
                        (1.0, 2.0),
                        (0.5, -1.0),
                        (1.0, -2.0),
                        (3.0, 0.0),
                        (-0.5, 0.25),
                        (0.5, 1.0),
                        (-0.5, -0.25),
                        (2.0, 0.0),
                    ],
                ),
            ),
            ("non-normal", non_normal.clone()),
            (
                "repeated and clustered",
                Matrix::from_vec(
                    4,
                    4,
                    vec![
                        Complex::new(R::ONE, R::ZERO),
                        Complex::new(cast(2.0), cast(-1.0)),
                        Complex::new(cast(0.5), cast(0.25)),
                        Complex::new(cast(-0.75), cast(0.5)),
                        complex_zero(),
                        Complex::new(R::ONE, R::ZERO),
                        Complex::new(cast(1.5), cast(-0.5)),
                        Complex::new(cast(0.25), cast(1.0)),
                        complex_zero(),
                        complex_zero(),
                        Complex::new(clustered, R::ZERO),
                        Complex::new(cast(2.0), cast(0.5)),
                        complex_zero(),
                        complex_zero(),
                        complex_zero(),
                        Complex::new(cast(2.0), R::ZERO),
                    ],
                )
                .unwrap(),
            ),
            (
                "small finite scale",
                scale_matrix(&non_normal, R::MIN_POSITIVE.sqrt()),
            ),
            (
                "large finite scale",
                scale_matrix(&non_normal, R::MAX.sqrt()),
            ),
        ];

        fixtures.into_iter().for_each(|(fixture, matrix)| {
            let mut normalized = matrix.clone();
            normalize_slice(
                normalized.as_mut_slice(),
                crate::kernel::standard::slice_maximum(matrix.as_slice()),
            );
            let order = normalized.rows();
            let policy = ExecutionPolicy::Sequential;
            let mut arena = vec![
                complex_zero();
                hessenberg::Scratch::<Complex<R>>::len(order, policy).unwrap()
            ];
            let mut rest = arena.as_mut_slice();
            let mut reduction = hessenberg::Scratch::carve(&mut rest, order, policy).unwrap();

            let mut hessenberg = normalized.clone();
            let mut hessenberg_vectors = Matrix::zeros(order, order).unwrap();
            let execution = ExecutionContext::sequential();
            hessenberg::reduce(&mut hessenberg, &mut reduction, policy, execution).unwrap();
            hessenberg::form_q(
                &hessenberg,
                hessenberg::Packing::Hessenberg,
                &mut reduction,
                &mut hessenberg_vectors,
                policy,
                execution,
            )
            .unwrap();
            hessenberg::clear_reflectors(&mut hessenberg, hessenberg::Packing::Hessenberg);
            assert_factorization(fixture, &normalized, &hessenberg, &hessenberg_vectors, 1);

            let mut schur = normalized.clone();
            let mut schur_vectors = Matrix::zeros(order, order).unwrap();
            with_solve(|solve| {
                complex_schur_form(
                    &mut schur,
                    Some(&mut schur_vectors),
                    &mut reduction,
                    &schur_options(&GeneralEigenOptions::default()),
                    solve,
                )
            })
            .unwrap();
            assert_factorization(fixture, &normalized, &schur, &schur_vectors, 0);
        });
    }

    #[test]
    fn complex_schur_form_preserves_f32_invariants() {
        exercise_complex_schur_invariants::<f32>();
    }

    #[test]
    fn complex_schur_form_preserves_f64_invariants() {
        exercise_complex_schur_invariants::<f64>();
    }

    #[test]
    fn upper_hessenberg_detection_preserves_structured_inputs() {
        let hessenberg = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0];
        assert!(is_real_upper_hessenberg(&hessenberg, 3));

        let dense = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 7.0, 8.0];
        assert!(!is_real_upper_hessenberg(&dense, 3));
    }

    #[test]
    fn real_rotation_has_conjugate_values_and_complex_vectors() {
        let matrix = Matrix::from_vec(2, 2, vec![0.0_f64, -1.0, 1.0, 0.0]).unwrap();
        let values = GeneralEigenvaluesResult::<f64>::compute(&matrix).unwrap();
        assert!((values.eigenvalues()[0] - Complex::new(0.0, 1.0)).norm() < 1.0e-12);
        assert!((values.eigenvalues()[1] - Complex::new(0.0, -1.0)).norm() < 1.0e-12);
        let decomposition = GeneralEigenDecomposition::<f64>::decompose(&matrix).unwrap();
        assert_residual(&matrix, &decomposition, 1.0e-12);
    }

    #[test]
    fn complex_triangular_matrix_has_expected_values() {
        let matrix = Matrix::from_vec(
            3,
            3,
            vec![
                Complex::new(3.0_f64, 2.0),
                Complex::new(4.0, -1.0),
                Complex::new(2.0, 0.0),
                complex_zero(),
                Complex::new(-1.0, 0.5),
                Complex::new(7.0, 1.0),
                complex_zero(),
                complex_zero(),
                Complex::new(2.0, -3.0),
            ],
        )
        .unwrap();
        let result = GeneralEigenDecomposition::<f64>::decompose(&matrix).unwrap();
        assert_residual(&matrix, &result, 1.0e-11);
    }

    #[test]
    fn defective_values_succeed_but_vectors_report_defect() {
        let matrix = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 0.0, 1.0]).unwrap();
        let values = GeneralEigenvaluesResult::<f64>::compute(&matrix).unwrap();
        assert_eq!(
            values.eigenvalues(),
            &[Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)]
        );
        let error = GeneralEigenDecomposition::<f64>::decompose(&matrix).unwrap_err();
        assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::DefectiveEigenbasis,
                ..
            }
        ));
    }

    #[test]
    fn dense_real_matrix_has_small_right_residuals() {
        let matrix = Matrix::from_vec(
            4,
            4,
            vec![
                4.0_f64, -2.0, 1.0, 3.0, 1.0, 5.0, -3.0, 2.0, 2.0, 1.0, -1.0, 4.0, -2.0, 3.0, 2.0,
                1.0,
            ],
        )
        .unwrap();
        let values = GeneralEigenvaluesResult::<f64>::compute(&matrix).unwrap();
        let decomposition = GeneralEigenDecomposition::<f64>::decompose(&matrix).unwrap();
        values
            .eigenvalues()
            .iter()
            .zip(decomposition.eigenvalues())
            .for_each(|(left, right)| {
                assert!(
                    (*left - *right).norm() < 1.0e-10,
                    "values-only {left:?} differs from decomposition {right:?}"
                );
            });
        assert_residual(&matrix, &decomposition, 1.0e-9);
    }

    #[test]
    fn balancing_permutation_restores_right_eigenvector_rows() {
        let matrix =
            Matrix::from_vec(3, 3, vec![5.0_f64, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]).unwrap();
        let decomposition = GeneralEigenDecomposition::<f64>::decompose(&matrix).unwrap();
        assert_residual(&matrix, &decomposition, 1.0e-11);
    }

    #[test]
    fn nonnormal_problem_reports_matched_left_and_right_diagnostics() {
        let matrix = Matrix::from_vec(
            3,
            3,
            vec![1.0_f64, 10.0, -2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0],
        )
        .unwrap();
        let options =
            GeneralEigenOptions::default().with_eigenvectors(GeneralEigenvectorSelection::Both);
        let decomposition =
            GeneralEigenDecomposition::<f64>::decompose_with_options(&matrix, &options).unwrap();
        assert_eq!(
            decomposition.eigenvector_selection(),
            GeneralEigenvectorSelection::Both
        );
        assert_eq!(decomposition.left_eigenvectors().unwrap().cols(), 3);
        assert_eq!(decomposition.right_eigenvectors().unwrap().cols(), 3);
        assert!(
            decomposition
                .left_residuals()
                .unwrap()
                .iter()
                .chain(decomposition.right_residuals().unwrap())
                .all(|residual| *residual < 1e-12)
        );
        assert_eq!(
            decomposition
                .eigenvalue_condition_estimates()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(decomposition.eigenvector_condition_estimates().len(), 3);
        assert_eq!(decomposition.balancing().scale_factors().len(), 3);
        assert_eq!(decomposition.balancing().permutation().len(), 3);
    }

    #[test]
    fn complex_left_only_job_omits_right_vectors() {
        let zero = complex_zero();
        let matrix = Matrix::from_vec(
            3,
            3,
            vec![
                Complex::new(1.0_f32, 1.0),
                Complex::new(4.0, -2.0),
                zero,
                zero,
                Complex::new(2.0, -1.0),
                Complex::new(0.0, 3.0),
                zero,
                zero,
                Complex::new(-1.0, 2.0),
            ],
        )
        .unwrap();
        let options =
            GeneralEigenOptions::default().with_eigenvectors(GeneralEigenvectorSelection::Left);
        let decomposition =
            GeneralEigenDecomposition::<f32>::decompose_with_options(&matrix, &options).unwrap();
        assert!(decomposition.left_eigenvectors().is_some());
        assert!(decomposition.right_eigenvectors().is_none());
        assert!(
            decomposition
                .left_residuals()
                .unwrap()
                .iter()
                .all(|residual| *residual < 1e-5)
        );
    }

    fn assert_residual<S: NumericScalar<Real = f64>>(
        matrix: &Matrix<S>,
        decomposition: &GeneralEigenDecomposition<f64>,
        tolerance: f64,
    ) {
        let order = matrix.rows();
        let vectors = decomposition.right_eigenvectors().unwrap();
        for column in 0..order {
            for row in 0..order {
                let applied: Complex<f64> = (0..order).fold(complex_zero::<f64>(), |sum, inner| {
                    sum + crate::eigen::into_complex(matrix.get_element(row, inner))
                        * vectors.get_element(inner, column)
                });
                let expected =
                    decomposition.eigenvalues()[column] * vectors.get_element(row, column);
                assert!((applied - expected).norm() < tolerance);
            }
        }
    }
}
