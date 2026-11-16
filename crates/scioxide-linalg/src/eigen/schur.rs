//! Dense Schur factorizations and selected invariant subspaces.
//
// Arena-level kernels shared with plan families (matrix functions, Sylvester
// and Lyapunov plans), reachable as `crate::eigen::<name>`:
//
//   schur_kernel_lengths::<S>(n, policy, operation) -> SchurKernelLengths
//       Direct scalar and real scratch beyond the caller's T and Q.
//   schur_kernel(t, q, n, scalars, reals, &SchurOptions, solve, operation)
//       `t` holds a finite row-major `n x n` input on entry and the Schur
//       form `T` in input units on exit; `q` receives unitary `Q` with
//       `A = Q T Q^H`. QR steps are charged to `solve`.
//   for_each_schur_block(t, n, operation, visit)
//       Visit the diagonal blocks of a Schur form and their eigenvalues,
//       rejecting unrepresentable eigenvalues.
use super::hessenberg;
use super::schur_types::{SchurBlock, SchurDecomposition, SchurOptions};
use super::support::{
    DirectLengths, copy_view, direct_requirements, identity, square, square_finite, times, total,
};
use super::types::reserve;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::take_mut;
use crate::{LinalgWorkspace, MatrixView};
use scioxide_core::NumericScalar;
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError, SolveContext,
};

/// Direct scratch of [`schur_kernel`] beyond the caller's T and Q.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SchurKernelLengths {
    pub(crate) scalars: usize,
    pub(crate) reals: usize,
}

/// Scratch lengths of the Schur kernel for scalar domain `S`; `policy` is the
/// execution policy of the [`SchurOptions`] later passed to [`schur_kernel`].
pub(crate) fn schur_kernel_lengths<S: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<SchurKernelLengths, SciOxideError> {
    Ok(if S::IS_REAL {
        // Real Hessenberg/Schur form, real Schur vectors and the reduction.
        let matrix = square(n)?;
        SchurKernelLengths {
            scalars: 0,
            reals: total(&[
                matrix,
                matrix,
                hessenberg::Scratch::<S::Real>::len(n, policy)?,
                times(super::real_schur::SWEEP, n)?,
            ])?,
        }
    } else {
        SchurKernelLengths {
            scalars: hessenberg::Scratch::<S>::len(n, policy)?,
            reals: 0,
        }
    })
}

/// Arena-level Schur factorization `A = Q T Q^H` on caller scratch.
///
/// `t` holds a finite row-major input on entry. Real input gives the real
/// quasi-triangular form through the real double-shift QR, complex input the
/// triangular form. Balancing is disabled because Q must be unitary. The
/// iteration runs on `A / scale`; T is scaled back while Q is scale-free.
pub(crate) fn schur_kernel<S: NumericScalar>(
    t: &mut [S],
    q: &mut [S],
    n: usize,
    mut scalars: &mut [S],
    mut reals: &mut [S::Real],
    options: &SchurOptions<S::Real>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let scale = crate::kernel::standard::slice_maximum(t);
    super::normalize_slice(t, scale);
    let policy = options.execution_options().execution_policy();
    if S::IS_REAL {
        let real_t = take_mut(&mut reals, square(n)?)?;
        let real_q = take_mut(&mut reals, square(n)?)?;
        let mut reduction = hessenberg::Scratch::carve(&mut reals, n, policy)?;
        let sweep = take_mut(&mut reals, times(super::real_schur::SWEEP, n)?)?;
        for (dst, src) in real_t.iter_mut().zip(t.iter()) {
            *dst = src.real_component();
        }
        real_schur_form(real_t, real_q, sweep, n, &mut reduction, options, solve)?;
        for (dst, &src) in t.iter_mut().zip(real_t.iter()) {
            *dst = S::from_real(src);
        }
        for (dst, &src) in q.iter_mut().zip(real_q.iter()) {
            *dst = S::from_real(src);
        }
    } else {
        let mut reduction = hessenberg::Scratch::carve(&mut scalars, n, policy)?;
        complex_schur_form(
            &mut ScratchMatrix::new(&mut *t, n, n)?,
            Some(&mut ScratchMatrix::new(&mut *q, n, n)?),
            &mut reduction,
            options,
            solve,
        )?;
    }
    restore(t, q, scale)
}

/// Real Schur form `A = Z T Z^T` of the normalized real candidate held in
/// `t`: Hessenberg reduction, Z from its reflectors, then the double-shift QR
/// with accumulated vectors. T is quasi-triangular with standardized 2x2
/// blocks; bulge residue below the subdiagonal is cleared. `sweep` holds
/// `SWEEP * n` reals.
pub(super) fn real_schur_form<R: RealScalar>(
    t: &mut [R],
    z: &mut [R],
    sweep: &mut [R],
    n: usize,
    reduction: &mut hessenberg::Scratch<'_, R>,
    options: &SchurOptions<R>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let policy = options.execution_options().execution_policy();
    let mut reduced = ScratchMatrix::new(&mut *t, n, n)?;
    let mut vectors = ScratchMatrix::new(&mut *z, n, n)?;
    hessenberg::reduce(&mut reduced, reduction, policy, solve.execution())?;
    hessenberg::form_q(
        &reduced,
        hessenberg::Packing::Hessenberg,
        reduction,
        &mut vectors,
        policy,
        solve.execution(),
    )?;
    hessenberg::clear_reflectors(&mut reduced, hessenberg::Packing::Hessenberg);
    super::real_schur::real_hessenberg_eigenvalues(
        super::real_schur::HessenbergProblem {
            matrix: t,
            values: None,
            vectors: Some(z),
            work: sweep,
        },
        n,
        options.tolerance,
        super::qr_iteration_budget(options.iteration_budget(), n)?,
        solve,
    )?;
    // Bulge entries below the first subdiagonal are operational QR scratch.
    for row in 2..n {
        t[row * n..row * n + row - 1].fill(R::ZERO);
    }
    Ok(())
}

/// Complex Schur form `A = Z T Z^H` of a normalized complex candidate:
/// Hessenberg reduction, Z (when requested) from its reflectors, then
/// single-shift QR with accumulated vectors.
pub(super) fn complex_schur_form<S: NumericScalar, M: MatrixStorage<S>>(
    t: &mut M,
    mut z: Option<&mut M>,
    reduction: &mut hessenberg::Scratch<'_, S>,
    options: &SchurOptions<S::Real>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let policy = options.execution_options().execution_policy();
    hessenberg::reduce(t, reduction, policy, solve.execution())?;
    if let Some(z) = z.as_deref_mut() {
        hessenberg::form_q(
            t,
            hessenberg::Packing::Hessenberg,
            reduction,
            z,
            policy,
            solve.execution(),
        )?;
    }
    hessenberg::clear_reflectors(t, hessenberg::Packing::Hessenberg);
    super::complex_schur::schur_iteration(
        t,
        z,
        &options.tolerance,
        super::qr_iteration_budget(options.iteration_budget(), t.rows())?,
        solve,
    )
}

fn restore<S: NumericScalar>(t: &mut [S], q: &[S], scale: S::Real) -> Result<(), SciOxideError> {
    for value in t.iter_mut() {
        *value = crate::checked::mul(*value, S::from_real(scale))?;
    }
    for &value in q {
        crate::checked::finite_candidate(value)?;
    }
    Ok(())
}

/// Visit diagonal blocks and eigenvalues of a Schur form, in block order.
///
/// A nonzero subdiagonal of real input marks a standardized 2x2 block (equal
/// diagonals, `b c < 0`) whose eigenvalues are `a +- i sqrt(|b|) sqrt(|c|)`,
/// positive imaginary part first.
pub(crate) fn for_each_schur_block<S: NumericScalar>(
    t: &[S],
    n: usize,
    mut visit: impl FnMut(SchurBlock, &[Complex<S::Real>]),
) -> Result<(), SciOxideError> {
    let at = |row: usize, col: usize| t[row * n + col];
    let mut k = 0;
    while k < n {
        let pair = S::IS_REAL && k + 1 < n && at(k + 1, k) != S::ZERO;
        let (block, values) = if pair {
            let real = at(k, k).real_component();
            let im = at(k, k + 1).abs().sqrt() * at(k + 1, k).abs().sqrt();
            (
                SchurBlock::RealTwoByTwo { start: k },
                [Complex::new(real, im), Complex::new(real, -im)],
            )
        } else {
            let value = crate::eigen::into_complex(at(k, k));
            (SchurBlock::OneByOne { start: k }, [value, value])
        };
        let values = &values[..block.size()];
        if values
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "Schur eigenvalue is not representable",
            ));
        }
        visit(block, values);
        k += block.size();
    }
    Ok(())
}

pub(super) fn decompose_lengths<S: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<DirectLengths, SciOxideError> {
    let matrix = square(n)?;
    let kernel = schur_kernel_lengths::<S>(n, policy)?;
    Ok(DirectLengths {
        scalars: total(&[matrix, matrix, kernel.scalars])?,
        reals: kernel.reals,
        indices: 0,
        extended: 0,
    })
}

impl<S: NumericScalar> crate::workspace::accounting::StorageBytes for SchurDecomposition<S> {
    fn storage_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        crate::workspace::accounting::sum([
            self.t.storage_bytes()?,
            self.q.storage_bytes()?,
            self.values.storage_bytes()?,
            self.blocks.storage_bytes()?,
        ])
    }
}
impl<S: NumericScalar> SchurDecomposition<S> {
    /// Computes the Schur decomposition `A = Q T Qᴴ` of a square matrix.
    ///
    /// The conventions for `T`, its blocks and the eigenvalues are stated on
    /// [`SchurDecomposition`]. The matrix is scaled by its largest entry
    /// magnitude, reduced to upper Hessenberg form by Householder reflections,
    /// and iterated to Schur form with accumulated transformations: Francis
    /// double-shift QR for real input and single-shift QR for complex input.
    /// No balancing is applied, because it would make `Q` non-unitary. The
    /// cost is O(n³) time and O(n²) memory. Use
    /// [`GeneralEigenvaluesResult::compute`](crate::GeneralEigenvaluesResult::compute)
    /// when only the eigenvalues are needed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if `matrix` contains NaN or infinity.
    /// - `ConvergenceFailure` if QR iteration exhausts its iteration budget.
    /// - `NumericalInstability` if `T` or an eigenvalue overflows when scaled
    ///   back to the input units.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, SchurDecomposition};
    /// // [[0, -2], [1, 0]] has eigenvalues ±i√2, held in one real 2×2 block.
    /// let a = Matrix::from_vec(2, 2, vec![0.0_f64, -2.0, 1.0, 0.0])?;
    /// let schur = SchurDecomposition::decompose(&a)?;
    /// assert_eq!(schur.blocks().len(), 1);
    /// assert!((schur.eigenvalues()[0].im - 2.0_f64.sqrt()).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - J. G. F. Francis, "The QR transformation, parts I and II", *Comput. J.*
    ///   4, 1961–1962.
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §7.4–7.5.
    pub fn decompose<'a>(matrix: impl Into<MatrixView<'a, S>>) -> Result<Self, SciOxideError> {
        Self::decompose_with_options(matrix, &crate::SchurOptions::default())
    }

    /// Runs [`decompose`](Self::decompose) with `options`, which set the deflation tolerance and
    /// the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`decompose`](Self::decompose), and limit errors from `options` as described
    /// in [Execution](crate#execution).
    pub fn decompose_with_options<'a>(
        matrix: impl Into<MatrixView<'a, S>>,
        options: &SchurOptions<S::Real>,
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

    /// Computes the Schur decomposition `A = Q T Qᴴ` into `output`.
    ///
    /// This is the reusable form of [`Self::decompose`], which describes the
    /// method. Size the workspace with
    /// [`SchurWorkspaceRequest::new`](crate::SchurWorkspaceRequest::new). The
    /// QR bulge chase runs on one thread. Each attempted QR step counts toward
    /// the budget set by [`crate::LinalgExecutionOptions::with_iteration_budget`]
    /// and appears in the `nested()` counts of a calling `SolveContext`.
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
    pub fn decompose_into<'scope, 'm, 'a>(
        matrix: impl Into<MatrixView<'m, S>>,
        options: &SchurOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let kernels = scope.kernel();
        kernels.execution().check()?;
        let finite = square_finite(matrix.into(), kernels.dispatch())?;
        let n = finite.rows();
        let requirements = direct_requirements::<S>(
            decompose_lengths::<S>(n, options.execution_options().execution_policy())?,
            options.execution_options().byte_limit(),
        )?;
        workspace.try_reserve(&requirements)?;
        output.reserve(n)?;
        let options = *options;
        let report = workspace.run_scope(
            scope,
            identity(scioxide_core::algorithm_id!("linalg", "schur", 1), n),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, _, solve| {
                workspace.with_direct(|mut scalars, reals, _, _| {
                    let t = take_mut(&mut scalars, square(n)?)?;
                    let q = take_mut(&mut scalars, square(n)?)?;
                    copy_view(finite.as_view(), t);
                    schur_kernel(t, q, n, scalars, reals, &options, solve)?;
                    for_each_schur_block(t, n, |_, _| {})?;
                    solve.execution().check()?;
                    let _ = solve.counts().total()?;
                    output.publish(t, q, n)
                })?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )?;
        output.report = Some(report);
        Ok(report)
    }

    pub(crate) fn action_storage(
        &self,
        plan: &mut crate::workspace::accounting::StoragePlan,
        n: usize,
    ) -> Result<(), SciOxideError> {
        plan.matrix(&self.t, n, n)?;
        plan.matrix(&self.q, n, n)?;
        plan.buffer(&self.values, n)?;
        plan.buffer(&self.blocks, n)
    }

    /// Reserve output storage for order `n` without changing the current factors.
    pub(crate) fn reserve(&mut self, n: usize) -> Result<(), SciOxideError> {
        self.t.try_reserve_shape(n, n)?;
        self.q.try_reserve_shape(n, n)?;
        reserve(
            &mut self.values,
            n,
            scioxide_core::quantity!("output.eigenvalues"),
        )?;
        reserve(
            &mut self.blocks,
            n,
            scioxide_core::quantity!("output.blocks"),
        )
    }

    // Copy a validated candidate into reserved storage and rebuild derived
    // blocks and eigenvalues from T.
    pub(super) fn publish(&mut self, t: &[S], q: &[S], n: usize) -> Result<(), SciOxideError> {
        self.t.try_resize_flat(n, n)?;
        self.q.try_resize_flat(n, n)?;
        self.t.as_mut_slice().copy_from_slice(t);
        self.q.as_mut_slice().copy_from_slice(q);
        self.values.clear();
        self.blocks.clear();
        let (values, blocks) = (&mut self.values, &mut self.blocks);
        for_each_schur_block(t, n, |block, block_values| {
            blocks.push(block);
            values.extend_from_slice(block_values);
        })
    }

    // Fallible copy used by owned transformations.
    pub(super) fn try_clone(&self) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        let n = self.t.rows();
        output.reserve(n)?;
        output.t.try_copy_from(&self.t)?;
        output.q.try_copy_from(&self.q)?;
        output.values.extend_from_slice(&self.values);
        output.blocks.extend_from_slice(&self.blocks);
        output.report = self.report;
        Ok(output)
    }
}
