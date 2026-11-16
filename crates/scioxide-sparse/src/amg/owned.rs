use super::cycle::{Coarse, Engine, Layout, Source, prepare};
use super::*;
use crate::{SparseCholesky, SparseSymmetricLimits, SparseSymmetricSymbolic};
/// Applies smoothed-aggregation multigrid as a preconditioner: an
/// [`AmgHierarchy`], a sparse Cholesky factor of its coarsest matrix, and a
/// cycle.
///
/// Each application runs one [`AmgCycle`] from a zero initial guess, with the
/// Jacobi smoothers of the hierarchy. `apply_into` writes the result for each
/// right-hand side, and the type implements the core
/// [`Preconditioner`](scioxide_core::Preconditioner) trait for Krylov solvers. The
/// coarse factor uses the [`Accuracy`](scioxide_core::Accuracy) of the call. Each
/// right-hand side counts as one iteration.
///
/// # Examples
///
/// ```
/// use scioxide_core::{BlockVectorView, ExecutionContext};
/// use scioxide_sparse::{
///     AmgCycleOptions, AmgHierarchy, AmgLimits, AmgPreconditioner, AmgSetupOptions, CsrMatrix,
///     SparseSolveOptions, SparseSymmetricLimits,
/// };
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![2., -1., -1., 2., -1., -1., 2.0_f64],
/// )?;
/// let options = SparseSolveOptions::new();
/// let hierarchy = AmgHierarchy::build(
///     a.view(),
///     BlockVectorView::new(&[1.; 3], 3, 1)?,
///     AmgSetupOptions::new().with_coarse_size(1),
///     AmgLimits::new(4, 1000, 1 << 20),
///     options,
///     ExecutionContext::sequential(),
/// )?;
/// let p = AmgPreconditioner::build(
///     hierarchy,
///     AmgCycleOptions::new(),
///     SparseSymmetricLimits::new(16, 0, 1 << 16),
///     options,
///     ExecutionContext::sequential(),
/// )?;
/// let mut workspace = p.workspace(1)?;
/// let mut correction = [0.0; 3];
/// p.apply_into(
///     &[1.0, 0.0, 1.0],
///     options,
///     &mut correction,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!(correction.iter().all(|x| *x > 0.0));
/// // A NaN input is rejected and the output keeps its previous value.
/// assert!(
///     p.apply_into(
///         &[f64::NAN, 0.0, 0.0],
///         options,
///         &mut correction,
///         &mut workspace,
///         ExecutionContext::sequential()
///     )
///     .is_err()
/// );
/// assert!(correction.iter().all(|x| *x > 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct AmgPreconditioner<S: NumericScalar> {
    hierarchy: AmgHierarchy<S>,
    coarse: SparseCholesky<S>,
    options: AmgCycleOptions,
    layout: Layout,
}
impl<S: NumericScalar> AmgPreconditioner<S> {
    /// Builds smoothed-aggregation multigrid for the Hermitian positive
    /// definite matrix `A = matrix`, which stores both triangles.
    ///
    /// The hierarchy is built by [`AmgHierarchy::build`] with the constant
    /// vector as the only near-null-space mode, [`AmgSetupOptions::new`]
    /// (coarsening stops at 16 rows; strength threshold 1/16) and
    /// [`AmgLimits::default`] (at most 64 levels, no entry or byte limit). The
    /// coarsest matrix is factored by [`SparseCholesky::new`], and each
    /// application runs one cycle of [`AmgCycleOptions::new`]: a V cycle with
    /// one pre- and one post-smoothing sweep. The configured form is
    /// [`AmgHierarchy::build`] followed by [`build`](Self::build).
    ///
    /// ```
    /// use scioxide_sparse::{AmgPreconditioner, sparse_diagonals};
    ///
    /// // tridiag(−1, 2, −1) of order 3 is below the coarse size: one exact level.
    /// let a = sparse_diagonals(3, 3, &[-1, 0, 1], &[&[-1.0_f64; 2], &[2.0; 3], &[-1.0; 2]])?;
    /// let z = AmgPreconditioner::new(&a)?.apply(&[1.0, 0.0, 1.0])?;
    /// assert!(z.iter().all(|v| (v - 1.0).abs() < 1e-14));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`AmgHierarchy::build`] and [`build`](Self::build)
    ///   other than `ResourceLimit`.
    pub fn new(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let n = matrix.rows();
        let ones =
            scioxide_core::backend::try_filled(n, S::ONE, scioxide_core::quantity!("amg_modes"))?;
        let hierarchy = AmgHierarchy::build(
            matrix.view(),
            BlockVectorView::new(&ones, n, 1)?,
            AmgSetupOptions::new(),
            AmgLimits::default(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential_native(),
        )?;
        let coarse = SparseCholesky::new(hierarchy.coarse_matrix())?;
        let cycle = AmgCycleOptions::new();
        let layout = prepare(
            Source::Owned(&hierarchy),
            Coarse::Cholesky(&coarse),
            cycle,
            scioxide_core::argument!("hierarchy"),
        )?;
        Ok(Self {
            hierarchy,
            coarse,
            options: cycle,
            layout,
        })
    }
    /// Creates the preconditioner from `hierarchy`: factors its coarsest matrix
    /// by [`SparseCholesky`](crate::SparseCholesky) with a fill-reducing
    /// ordering, and prepares the cycle.
    ///
    /// `limits` bounds the Cholesky analysis, factor and scratch; the hierarchy
    /// and the application scratch are not counted. `options` sets the
    /// [`Accuracy`](scioxide_core::Accuracy) and iteration limit of the
    /// factorization.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the Cholesky factor exceeds `limits`.
    /// - `NumericalInstability` if the coarsest matrix is not positive definite
    ///   or a value overflows.
    /// - `ConvergenceFailure` if the iteration limit is reached;
    ///   `ExecutionStopped` on cancellation or a deadline.
    /// - `SizeOverflow`, `AllocationFailure` if storage cannot be sized or
    ///   allocated.
    pub fn build<'s, 'c>(
        hierarchy: AmgHierarchy<S>,
        cycle: AmgCycleOptions,
        limits: SparseSymmetricLimits,
        options: SparseSolveOptions,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<Self, SciOxideError> {
        let a = hierarchy.coarse_matrix();
        let symbolic = SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            crate::SparseOrdering::default(),
            None,
            limits,
        )?;
        let coarse = SparseCholesky::factor_scoped(symbolic, a.view(), options, scope)?;
        let layout = prepare(
            Source::Owned(&hierarchy),
            Coarse::Cholesky(&coarse),
            cycle,
            scioxide_core::argument!("hierarchy"),
        )?;
        Ok(Self {
            hierarchy,
            coarse,
            options: cycle,
            layout,
        })
    }
    /// Returns the hierarchy.
    pub const fn hierarchy(&self) -> &AmgHierarchy<S> {
        &self.hierarchy
    }
    /// Returns the cycle options.
    pub const fn cycle_options(&self) -> AmgCycleOptions {
        self.options
    }
    fn engine(&self) -> Engine<'_, S> {
        Engine {
            source: Source::Owned(&self.hierarchy),
            coarse: Coarse::Cholesky(&self.coarse),
            options: self.options,
            layout: &self.layout,
        }
    }
}
impl<S: NumericScalar> crate::factor_solve::Application<S> for AmgPreconditioner<S> {
    fn dimension(&self) -> usize {
        self.engine().dimension()
    }
    fn work_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.engine().work_len())
    }
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        work: &mut [S],
        options: SparseSolveOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.engine()
            .apply_column(rhs, candidate, work, options, execution)
    }
}
crate::factor_solve::impl_applications!(AmgPreconditioner<S>);

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use scioxide_core::Preconditioner;

    fn star() -> CsrMatrix<f64> {
        // A star centred on row zero forces a nonidentity fill-reducing order.
        CsrMatrix::from_raw(
            5,
            5,
            alloc::vec![0, 5, 7, 9, 11, 13],
            alloc::vec![0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4],
            alloc::vec![
                5.0_f64, -1., -1., -1., -1., -1., 2., -1., 2., -1., 2., -1., 2.
            ],
        )
        .unwrap()
    }

    fn preconditioner(a: &CsrMatrix<f64>) -> AmgPreconditioner<f64> {
        let context = ExecutionContext::sequential();
        let hierarchy = AmgHierarchy::build(
            a.view(),
            BlockVectorView::new(&[1.; 5], 5, 1).unwrap(),
            AmgSetupOptions::new(),
            AmgLimits::new(4, 1000, 1_000_000),
            SparseSolveOptions::new(),
            context,
        )
        .unwrap();
        AmgPreconditioner::build(
            hierarchy,
            AmgCycleOptions::new(),
            SparseSymmetricLimits::new(64, 0, 1_000_000),
            SparseSolveOptions::new(),
            context,
        )
        .unwrap()
    }

    #[test]
    fn coarse_inverse_preserves_original_coordinates_after_amd() {
        let p = preconditioner(&star());
        assert!(
            p.coarse
                .permutation()
                .iter()
                .enumerate()
                .any(|(i, &j)| i != j)
        );
        let mut output = [0.; 5];
        let mut work = alloc::vec![0.; p.precondition_workspace_len().unwrap()];
        // Independently manufactured A * [1,2,3,4,5].
        p.precondition_into(&[-9., 3., 5., 7., 9.], &mut output, &mut work)
            .unwrap();
        for (&actual, expected) in output.iter().zip([1., 2., 3., 4., 5.]) {
            assert!((actual - expected).abs() <= 64. * f64::EPSILON * expected);
        }
    }

    #[test]
    fn applications_reuse_the_prepared_cycle_without_revalidation() {
        let n = 24;
        let mut builder = crate::SparseBuilder::new(n, n);
        for i in 0..n {
            builder.push(i, i, 2.0_f64).unwrap();
            if i > 0 {
                builder.push(i, i - 1, -1.).unwrap();
                builder.push(i - 1, i, -1.).unwrap();
            }
        }
        let a = builder
            .finish_csr(
                crate::DuplicatePolicy::Reject,
                crate::ExplicitZeroPolicy::Drop,
            )
            .unwrap();
        let context = ExecutionContext::sequential();
        let hierarchy = AmgHierarchy::build(
            a.view(),
            BlockVectorView::new(&alloc::vec![1.; n], n, 1).unwrap(),
            AmgSetupOptions::new().with_coarse_size(4),
            AmgLimits::new(8, 10_000, 1 << 24),
            SparseSolveOptions::new(),
            context,
        )
        .unwrap();
        let mut p = AmgPreconditioner::build(
            hierarchy,
            AmgCycleOptions::new(),
            SparseSymmetricLimits::new(64, 0, 1 << 20),
            SparseSolveOptions::new(),
            context,
        )
        .unwrap();
        assert!(p.hierarchy.levels.len() > 1);
        // Validation rejects a zero Jacobi inverse-diagonal entry. Poisoning the
        // prepared hierarchy shows that applications never rerun it.
        p.hierarchy.levels[0].inverse[0] = 0.;
        assert!(
            p.hierarchy
                .cycle(&p.coarse, AmgCycleOptions::new())
                .is_err()
        );
        let mut workspace = p.workspace(1).unwrap();
        let mut output = alloc::vec![0.; n];
        for _ in 0..3 {
            p.apply_into(
                &alloc::vec![1.; n],
                SparseSolveOptions::new(),
                &mut output,
                &mut workspace,
                context,
            )
            .unwrap();
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }
}
