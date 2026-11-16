//! Complete CS decomposition by an SVD and one-sided Jacobi on the first block column.
use super::cs_kernel::{PrerotationScratch, cs_sweep, identity, left_factor, norm, prerotate};
use super::util::orthogonality;
use super::util::reserve_vec;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix, ScratchVec};
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{KernelContext, LinalgExecutionOptions, LinalgWorkspace, Matrix, MatrixView};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, NumericScalar, NumericalIssue, RealScalar,
    SciOxideError, SolveContext,
};
/// Diagnostics of a [`CsdDecomposition`], recomputed from the factors.
///
/// The values are computed estimates, not verified bounds.
#[derive(Debug, Clone, Copy)]
pub struct CsdReport<R: RealScalar> {
    /// Input Frobenius unitarity error.
    pub(crate) input_unitarity_error: R,
    /// Full four-block Frobenius reconstruction error.
    pub(crate) residual: R,
    /// Maximum Frobenius orthogonality error of the four unitary factors.
    pub(crate) orthogonality_error: R,
    /// Completed coupled Jacobi sweeps.
    pub(crate) sweeps: usize,
}

impl<R: RealScalar> CsdReport<R> {
    /// Returns the unitarity error of the input, `‖Xᴴ X − I‖_F`.
    pub const fn input_unitarity_error(&self) -> R {
        self.input_unitarity_error
    }

    /// Returns the reconstruction residual `‖X − diag(U₁, U₂) CS diag(V₁, V₂)ᴴ‖_F`.
    pub const fn residual(&self) -> R {
        self.residual
    }

    /// Returns the largest `‖Wᴴ W − I‖_F` over the factors `W = U₁, U₂, V₁, V₂`.
    pub const fn orthogonality_error(&self) -> R {
        self.orthogonality_error
    }

    /// Returns the number of Jacobi sweeps.
    pub const fn sweeps(&self) -> usize {
        self.sweeps
    }
}

/// Complete CS decomposition `X = diag(U₁, U₂) CS diag(V₁, V₂)ᴴ` of an `n × n` unitary matrix.
///
/// `X` is partitioned after `p` rows and `q` columns. `U₁` (`p × p`), `U₂` (`(n−p) × (n−p)`),
/// `V₁` (`q × q`) and `V₂` (`(n−q) × (n−q)`) are unitary. With `r = min(p, n−p, q, n−q)`,
/// the `n × n` middle matrix `CS` has the block form
///
/// ```text
///               q columns         n−q columns
///         ⎡ I_a  0   0   │  0    0    0   ⎤
///   p     ⎢ 0    C   0   │  0   −S    0   ⎥
///   rows  ⎢ 0    0   0   │  0    0   −I_b ⎥
///         ⎢──────────────┼────────────────⎥
///   n−p   ⎢ 0    0   0   │  I_e  0    0   ⎥
///   rows  ⎢ 0    S   0   │  0    C    0   ⎥
///         ⎣ 0    0   I_d │  0    0    0   ⎦
/// ```
///
/// - `C = diag(cos θ₁, …, cos θᵣ)` and `S = diag(sin θ₁, …, sin θᵣ)`, with principal angles
///   `0 ≤ θ₁ ≤ … ≤ θᵣ ≤ π/2`.
/// - `a = min(p, q) − r`, `b = min(p, n−q) − r`, `d = min(n−p, q) − r` and
///   `e = min(n−p, n−q) − r` are the sizes of the identity blocks. Blocks of size zero are
///   absent.
///
/// The method rotates the first `q` columns by the right singular vectors of their row block
/// with fewer rows, which makes the columns of both row blocks orthogonal up to roundoff, and
/// then runs one-sided Jacobi sweeps (usually one or two) until the top `p` rows and the
/// bottom `n − p` rows each have mutually orthogonal columns. This gives `V₁`. The block
/// column norms give the cosines and sines, which are renormalized so that
/// `cos² θ + sin² θ = 1` and keep full relative accuracy when small. `U₁` and `U₂` are
/// completed from the normalized columns, and `V₂` is recovered from the last `n − q`
/// columns of `X` without a second Jacobi iteration. The SVD and each sweep cost `O(n q²)`.
///
/// The input must be unitary to within `‖Xᴴ X − I‖_F ≤ 32 n ε`. The factors are accepted only
/// if the reconstruction residual and every orthogonality error are at most `128 n ε`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{CsdDecomposition, Matrix};
/// let x = Matrix::from_vec(2, 2, vec![0.8_f64, -0.6, 0.6, 0.8])?;
/// let csd = CsdDecomposition::factor(&x, 1, 1)?;
/// // X is the plane rotation by θ with cos θ = 0.8 and sin θ = 0.6.
/// assert!((csd.cosines()[0] - 0.8).abs() < 1e-14);
/// assert!((csd.sines()[0] - 0.6).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CsdDecomposition<S: NumericScalar> {
    u1: Matrix<S>,
    u2: Matrix<S>,
    v1: Matrix<S>,
    v2: Matrix<S>,
    cs: Matrix<S>,
    angles: Vec<S::Real>,
    cosines: Vec<S::Real>,
    sines: Vec<S::Real>,
    report: Option<CsdReport<S::Real>>,
}
impl<S: NumericScalar> CsdDecomposition<S> {
    /// Returns an empty decomposition for use as the output of [`factor_into`](Self::factor_into).
    pub fn empty() -> Self {
        Self {
            u1: Matrix::empty(),
            u2: Matrix::empty(),
            v1: Matrix::empty(),
            v2: Matrix::empty(),
            cs: Matrix::empty(),
            angles: Vec::new(),
            cosines: Vec::new(),
            sines: Vec::new(),
            report: None,
        }
    }
    /// Returns the unitary factor `U₁` (`p × p`).
    pub fn u1_matrix(&self) -> &Matrix<S> {
        &self.u1
    }
    /// Returns the unitary factor `U₂` (`(n−p) × (n−p)`).
    pub fn u2_matrix(&self) -> &Matrix<S> {
        &self.u2
    }
    /// Returns the unitary factor `V₁` (`q × q`), not its adjoint.
    pub fn v1_matrix(&self) -> &Matrix<S> {
        &self.v1
    }
    /// Returns the unitary factor `V₂` (`(n−q) × (n−q)`), not its adjoint.
    pub fn v2_matrix(&self) -> &Matrix<S> {
        &self.v2
    }
    /// Returns the `n × n` middle matrix `CS`, including the identity blocks.
    pub fn cs_matrix(&self) -> &Matrix<S> {
        &self.cs
    }
    /// Returns the `r` principal angles `θⱼ ∈ [0, π/2]` in nondecreasing order.
    pub fn angles(&self) -> &[S::Real] {
        &self.angles
    }
    /// Returns `cos θⱼ ≥ 0`, computed from the norms of the top blocks.
    pub fn cosines(&self) -> &[S::Real] {
        &self.cosines
    }
    /// Returns `sin θⱼ ≥ 0`, computed from the norms of the bottom blocks.
    pub fn sines(&self) -> &[S::Real] {
        &self.sines
    }
    /// Returns the diagnostics, or `None` for an [`empty`](Self::empty) result.
    pub fn report(&self) -> Option<&CsdReport<S::Real>> {
        self.report.as_ref()
    }
    /// Computes the CS decomposition of `x` with a `p × q` upper-left block.
    ///
    /// # Errors
    ///
    /// The errors of [`factor_into`](Self::factor_into).
    pub fn factor<'m>(
        x: impl Into<MatrixView<'m, S>>,
        p: usize,
        q: usize,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(x, p, q, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the product accuracy and the limits
    /// described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        x: impl Into<MatrixView<'m, S>>,
        p: usize,
        q: usize,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            x,
            p,
            q,
            &options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            &mut output,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Computes the CS decomposition of `x` with a `p × q` upper-left block into `output`.
    ///
    /// Size the workspace with [`CsdWorkspaceRequest`]. Preparation counts as one iteration
    /// and each Jacobi sweep as one more toward the iteration limit of `options`. Without
    /// a limit, at most 40 sweeps run. The iterations of the nested SVD are counted in a
    /// child scope, and the iteration limit also bounds them.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` is not square.
    /// - `InvalidInput` if `p > n` or `q > n`, or if `‖Xᴴ X − I‖_F > 32 n ε`.
    /// - `NonFinite` if an entry of `x` is NaN or infinite.
    /// - `ConvergenceFailure` if the nested SVD or the Jacobi iteration exceeds the
    ///   iteration limit.
    /// - `NumericalInstability` if the residual or orthogonality check fails, or if the
    ///   completion of `U₁` or `U₂` breaks down.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn factor_into<'m, 'scope, 'capability>(
        x: impl Into<MatrixView<'m, S>>,
        p: usize,
        q: usize,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(options)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let x = x.into();
        let n = x.rows();
        if x.cols() != n {
            return Err(SciOxideError::dimension_mismatch([n, n], [n, x.cols()]));
        }
        let request = CsdWorkspaceRequest::new(n, p, q, options);
        let requirements = Self::workspace_requirements(&request)?;
        let x = x.try_finite_dispatched(kernels.dispatch())?;
        let unitarity_error = orthogonality(x.as_view(), kernels)?;
        if unitarity_error
            > crate::checked::dimension_epsilon::<S::Real>(n.max(1))?
                * S::Real::try_from_usize_exact(32)?
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "input is not orthogonal/unitary within 32 n epsilon",
            ));
        }
        workspace.try_reserve(&requirements)?;
        output.reserve(n, p, q)?;
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("linalg", "csd", 1),
            n as u128,
            ((p as u128) << 64) | q as u128,
        );
        workspace.run_scope(
            context,
            identity,
            options.solve_budgets(),
            |workspace, kernels, solve| {
                workspace.with_prefixes(prefix(n, p, q)?, |scalars, reals, indices, child| {
                    let mut work = CsdScratch::new(scalars, reals, indices, n, p, q)?;
                    let report = compute(
                        AdmittedInput {
                            x: x.as_view(),
                            unitarity_error,
                        },
                        p,
                        q,
                        &mut work,
                        child,
                        *options,
                        kernels,
                        solve,
                    )?;
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    for (out, candidate) in [
                        (&mut output.u1, &work.candidate.u1),
                        (&mut output.u2, &work.candidate.u2),
                        (&mut output.v1, &work.candidate.v1),
                        (&mut output.v2, &work.candidate.v2),
                        (&mut output.cs, &work.candidate.cs),
                    ] {
                        out.try_resize_flat(candidate.rows(), candidate.cols())?;
                        out.as_mut_slice().copy_from_slice(candidate.as_slice());
                    }
                    for (out, values) in [
                        (&mut output.angles, &work.candidate.angles),
                        (&mut output.cosines, &work.candidate.cosines),
                        (&mut output.sines, &work.candidate.sines),
                    ] {
                        out.clear();
                        out.extend_from_slice(values);
                    }
                    output.report = Some(report);
                    Ok(())
                })
            },
        )
    }
    fn reserve(&mut self, n: usize, p: usize, q: usize) -> Result<(), SciOxideError> {
        for (a, r) in [
            (&mut self.u1, p),
            (&mut self.u2, n - p),
            (&mut self.v1, q),
            (&mut self.v2, n - q),
            (&mut self.cs, n),
        ] {
            a.try_reserve_shape(r, r)?;
        }
        let r = p.min(n - p).min(q).min(n - q);
        for v in [&mut self.angles, &mut self.cosines, &mut self.sines] {
            reserve_vec(v, r, scioxide_core::quantity!("angle_buffers"))?;
        }
        Ok(())
    }
}
impl<S: NumericScalar> Default for CsdDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}
/// Workspace request for the CS decomposition of an `n × n` matrix with a `p × q` upper-left block.
#[derive(Debug, Clone, Copy)]
pub struct CsdWorkspaceRequest {
    n: usize,
    p: usize,
    q: usize,
    options: LinalgExecutionOptions,
}
impl CsdWorkspaceRequest {
    /// Returns a request for an `n × n` matrix partitioned after `p` rows and `q` columns.
    pub const fn new(n: usize, p: usize, q: usize, options: &LinalgExecutionOptions) -> Self {
        Self {
            n,
            p,
            q,
            options: *options,
        }
    }
}

/// Scratch lengths of an `n x n` partition after `p` rows and `q` columns, and
/// the number of principal angles.
fn geometry(n: usize, p: usize, q: usize) -> Result<([usize; 13], usize), SciOxideError> {
    validate_partition(n, p, q)?;
    let square = |a, b| {
        scioxide_core::backend::checked_product(&[a, b], scioxide_core::quantity!("csd_scratch"))
    };
    let largest = p.max(n - p);
    Ok((
        [
            square(n, n)?,
            square(n, q)?,
            square(q, q)?,
            square(n, q)?,
            square(largest, largest)?,
            square(largest, q)?,
            square(n, n)?,
            n,
            square(p, p)?,
            square(n - p, n - p)?,
            square(q, q)?,
            square(n - q, n - q)?,
            square(n, n)?,
        ],
        p.min(n - p).min(q).min(n - q),
    ))
}

/// Scalar, real and index scratch lengths of the CS stage.
fn prefix(n: usize, p: usize, q: usize) -> Result<[usize; 3], SciOxideError> {
    let (lengths, r) = geometry(n, p, q)?;
    let add = |a: usize, b: usize| {
        a.checked_add(b)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("csd_scratch")))
    };
    Ok([
        lengths.into_iter().try_fold(0, add)?,
        [q, r, r, r].into_iter().try_fold(0, add)?,
        add(n, q)?,
    ])
}

impl<S: NumericScalar> CsdDecomposition<S> {
    /// Returns the workspace size for [`factor_into`](Self::factor_into), including the nested
    /// SVD.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `p > n` or `q > n`.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &CsdWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let CsdWorkspaceRequest { n, p, q, options } = *request;
        let [scalars, reals, indices] = prefix(n, p, q)?;
        // The SVD of the smaller row block of the first block column runs in
        // the arena after the CS scratch.
        let block = crate::SvdWorkspaceRequest::new(p.min(n - p), q, &options);
        let mut requirements = crate::ThinSvdDecomposition::<S>::workspace_requirements(&block)?
            .max(&crate::SvdDecomposition::<S>::workspace_requirements(
                &block,
            )?);
        let add = |a: usize, b: usize| {
            a.checked_add(b).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("csd_scratch"))
            })
        };
        *requirements.scalar_mut(ScalarRegion::Direct) =
            add(requirements.scalar(ScalarRegion::Direct), scalars)?;
        *requirements.real_mut(RealRegion::Direct) =
            add(requirements.real(RealRegion::Direct), reals)?;
        requirements.indices = add(requirements.indices, indices)?;
        requirements.byte_limit = options.byte_limit();
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }
    /// Allocates a workspace sized by [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// The errors of [`workspace_requirements`](Self::workspace_requirements), and
    /// workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{CsdDecomposition, CsdWorkspaceRequest, LinalgExecutionOptions};
    /// let request = CsdWorkspaceRequest::new(4, 2, 2, &LinalgExecutionOptions::new());
    /// let work = CsdDecomposition::<f64>::workspace(&request)?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &CsdWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut work = LinalgWorkspace::new();
        work.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(work)
    }
}
struct CsdCandidate<'a, S: NumericScalar> {
    u1: ScratchMatrix<'a, S>,
    u2: ScratchMatrix<'a, S>,
    v1: ScratchMatrix<'a, S>,
    v2: ScratchMatrix<'a, S>,
    cs: ScratchMatrix<'a, S>,
    angles: ScratchVec<'a, S::Real>,
    cosines: ScratchVec<'a, S::Real>,
    sines: ScratchVec<'a, S::Real>,
}
struct CsdScratch<'a, S: NumericScalar> {
    input: ScratchMatrix<'a, S>,
    columns: ScratchMatrix<'a, S>,
    z: ScratchMatrix<'a, S>,
    sorted: ScratchMatrix<'a, S>,
    basis: ScratchMatrix<'a, S>,
    embedding: ScratchMatrix<'a, S>,
    product: ScratchMatrix<'a, S>,
    norms: ScratchVec<'a, S::Real>,
    order: ScratchVec<'a, usize>,
    mapping: ScratchVec<'a, usize>,
    vector: ScratchVec<'a, S>,
    candidate: CsdCandidate<'a, S>,
}
impl<'a, S: NumericScalar> CsdScratch<'a, S> {
    fn new(
        mut scalars: &'a mut [S],
        reals: &'a mut [S::Real],
        indices: &'a mut [usize],
        n: usize,
        p: usize,
        q: usize,
    ) -> Result<Self, SciOxideError> {
        let (lengths, r) = geometry(n, p, q)?;
        let mut take = |length| crate::workspace::take_mut(&mut scalars, length);
        let input = ScratchMatrix::new(take(lengths[0])?, 0, 0)?;
        let columns = ScratchMatrix::new(take(lengths[1])?, 0, 0)?;
        let z = ScratchMatrix::new(take(lengths[2])?, 0, 0)?;
        let sorted = ScratchMatrix::new(take(lengths[3])?, 0, 0)?;
        let basis = ScratchMatrix::new(take(lengths[4])?, 0, 0)?;
        let embedding = ScratchMatrix::new(take(lengths[5])?, 0, 0)?;
        let product = ScratchMatrix::new(take(lengths[6])?, 0, 0)?;
        let vector = ScratchVec::new(take(lengths[7])?, n)?;
        let u1 = ScratchMatrix::new(take(lengths[8])?, 0, 0)?;
        let u2 = ScratchMatrix::new(take(lengths[9])?, 0, 0)?;
        let v1 = ScratchMatrix::new(take(lengths[10])?, 0, 0)?;
        let v2 = ScratchMatrix::new(take(lengths[11])?, 0, 0)?;
        let cs = ScratchMatrix::new(take(lengths[12])?, 0, 0)?;
        let (norms, rest) = crate::workspace::split_mut(reals, q)?;
        let (angles, rest) = crate::workspace::split_mut(rest, r)?;
        let (cosines, sines) = crate::workspace::split_mut(rest, r)?;
        let (order, mapping) = crate::workspace::split_mut(indices, q)?;
        Ok(Self {
            input,
            columns,
            z,
            sorted,
            basis,
            embedding,
            product,
            vector,
            norms: ScratchVec::new(norms, 0)?,
            order: ScratchVec::new(order, 0)?,
            mapping: ScratchVec::new(mapping, 0)?,
            candidate: CsdCandidate {
                u1,
                u2,
                v1,
                v2,
                cs,
                angles: ScratchVec::new(angles, 0)?,
                cosines: ScratchVec::new(cosines, 0)?,
                sines: ScratchVec::new(&mut sines[..r], 0)?,
            },
        })
    }
}
fn validate_partition(n: usize, p: usize, q: usize) -> Result<(), SciOxideError> {
    if p > n || q > n {
        return Err(SciOxideError::invalid_input(
            if p > n {
                scioxide_core::argument!("p")
            } else {
                scioxide_core::argument!("q")
            },
            scioxide_core::InvalidInputReason::OutOfRange,
            "partition sizes must not exceed matrix order",
        ));
    }
    Ok(())
}

/// A finite square input checked before the epoch, with its measured unitarity defect.
struct AdmittedInput<'a, S: NumericScalar> {
    x: MatrixView<'a, S>,
    unitarity_error: S::Real,
}

fn failure(counts: scioxide_core::SolveCounts) -> SciOxideError {
    SciOxideError::convergence_failure(
        ConvergenceReason::IterationLimit,
        ConvergenceDiagnostics::new().with_counts(counts),
        "no complete CS candidate",
    )
}

#[allow(clippy::too_many_arguments)]
fn compute<'capability, S: NumericScalar>(
    input: AdmittedInput<'_, S>,
    p: usize,
    q: usize,
    w: &mut CsdScratch<'_, S>,
    child: &mut LinalgWorkspace<'_, S>,
    options: LinalgExecutionOptions,
    context: KernelContext<'capability>,
    solve: &mut SolveContext<'_, 'capability>,
) -> Result<CsdReport<S::Real>, SciOxideError> {
    let AdmittedInput {
        x,
        unitarity_error: input_unitarity_error,
    } = input;
    let n = x.rows();
    w.input.try_resize_flat(n, n)?;
    for i in 0..n {
        for j in 0..n {
            w.input.set_element(i, j, x.get(i, j)?);
        }
    }
    let epsilon = crate::checked::dimension_epsilon::<S::Real>(n.max(1))?;
    // CS decomposition of a unitary `X = [X11 X12; X21 X22]` (p rows / q
    // columns in the first blocks): one-sided Jacobi on the first q columns
    // finds V1 so that `[X11; X21] V1` has mutually orthogonal columns in
    // both row blocks. Their block norms are the cosines and sines, U1/U2
    // are completed from the normalized columns, and V2 is recovered from
    // the second block column as `V2 = X[:, q..]^H P[:, q..]` with
    // `P = diag(U1, U2) CS`, rather than by a second Jacobi iteration.
    crate::solve_support::admit_iteration(
        Some(context.execution()),
        Some(solve),
        "no complete CS candidate",
    )?;
    w.columns.try_resize_flat(n, q)?;
    identity(&mut w.z, q)?;
    for i in 0..n {
        for j in 0..q {
            w.columns.set_element(i, j, w.input.get_element(i, j));
        }
    }
    let mut sweeps = 0;
    let mut converged = q < 2;
    if !converged {
        // `sorted` and `embedding` are free until the factors are assembled.
        prerotate(
            &mut w.columns,
            &mut w.z,
            p,
            PrerotationScratch {
                block: &mut w.embedding,
                product: &mut w.sorted,
            },
            options,
            child,
            context,
            solve,
        )?;
        for _ in 0..options
            .iteration_budget()
            .map_or(40, |budget| budget.get().saturating_sub(1))
        {
            crate::solve_support::admit_iteration(
                Some(context.execution()),
                Some(solve),
                "no complete CS candidate",
            )?;
            sweeps += 1;
            if cs_sweep(&mut w.columns, &mut w.z, p, epsilon, context)? {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return Err(failure(solve.counts()));
    }
    assemble_factors(w, p, q, context)?;
    let (residual, orthogonality_error) = validate_factors(w, q, epsilon, context)?;
    Ok(CsdReport {
        input_unitarity_error,
        residual,
        orthogonality_error,
        sweeps,
    })
}

fn assemble_factors<S: NumericScalar>(
    w: &mut CsdScratch<'_, S>,
    p: usize,
    q: usize,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    // Block sizes of the canonical CS form: r genuine angle pairs, plus
    // identity blocks a, b, d, e where one partition dimension exceeds the
    // other (angle exactly 0 or pi/2). Columns are ordered by increasing
    // angle `atan2(||bottom||, ||top||)`, which puts cosine-one columns first
    // and sine-one columns last, matching that layout.
    let n = w.input.rows();
    let r = p.min(n - p).min(q).min(n - q);
    let a = p.min(q) - r;
    let b = p.min(n - q) - r;
    let d = (n - p).min(q) - r;
    let e = (n - p).min(n - q) - r;
    w.order.resize(q, 0)?;
    for (i, v) in w.order.iter_mut().enumerate() {
        *v = i;
    }
    w.norms.resize(q, S::Real::ZERO)?;
    for (col, angle) in w.norms.iter_mut().enumerate() {
        *angle = norm(&w.columns, col, p, n)?.atan2(norm(&w.columns, col, 0, p)?);
    }
    w.order.sort_unstable_by(|&i, &j| {
        w.norms[i]
            .partial_cmp(&w.norms[j])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    w.sorted.try_resize_flat(n, q)?;
    w.candidate.v1.try_resize_flat(q, q)?;
    for (j, &col) in w.order.iter().enumerate() {
        for i in 0..n {
            w.sorted.set_element(i, j, w.columns.get_element(i, col));
        }
        for i in 0..q {
            w.candidate.v1.set_element(i, j, w.z.get_element(i, col));
        }
    }
    let candidate = &mut w.candidate;
    for values in [
        &mut candidate.angles,
        &mut candidate.cosines,
        &mut candidate.sines,
    ] {
        values.resize(r, S::Real::ZERO)?;
    }
    for j in 0..r {
        // Both block norms are measured and renormalized so that
        // `c^2 + s^2 = 1` holds to rounding; small values of either are kept
        // with full relative accuracy instead of being derived as
        // `sqrt(1 - x^2)`.
        let c = norm(&w.sorted, a + j, 0, p)?;
        let s = norm(&w.sorted, a + j, p, n)?;
        let length = c.hypot(s);
        candidate.cosines[j] = c / length;
        candidate.sines[j] = s / length;
        candidate.angles[j] = s.atan2(c);
    }
    // Canonical first-column embeddings: top [I_a,C,0], bottom [0;S;I_d].
    w.norms.resize(q, S::Real::ZERO)?;
    for (top, start, rows) in [(true, 0, p), (false, p, n - p)] {
        for (j, v) in w.norms.iter_mut().enumerate() {
            *v = if (top && j < a + r) || (!top && j >= a) {
                norm(&w.sorted, j, start, start + rows)?
            } else {
                S::Real::ZERO
            };
        }
        left_factor(
            &w.sorted,
            start,
            rows,
            crate::decompose::cs_kernel::LeftFactorScratch {
                norms: &w.norms,
                order: &mut w.order,
                vector: &mut w.vector[..rows],
            },
            &mut w.basis,
            &mut w.embedding,
            context,
        )?;
        let u = if top {
            &mut candidate.u1
        } else {
            &mut candidate.u2
        };
        w.mapping.resize(rows, rows)?;
        w.mapping.fill(rows);
        for j in 0..q {
            if (top && j < a + r) || (!top && j >= a) {
                let dest = if top { j } else { e + j - a };
                if let Some(source) = (0..rows).find(|&i| w.embedding.get_element(i, j) != S::ZERO)
                {
                    w.mapping[dest] = source;
                }
            }
        }
        for dest in 0..rows {
            if w.mapping[dest] == rows {
                let source = (0..rows)
                    .find(|source| !w.mapping.contains(source))
                    .ok_or_else(|| {
                        SciOxideError::numerical_instability(
                            NumericalIssue::Breakdown,
                            "basis permutation incomplete",
                        )
                    })?;
                w.mapping[dest] = source;
            }
        }
        // `mapping[dest]` names the completed basis vector that must sit in
        // column `dest` so that the canonical CS layout lines up with the
        // Jacobi columns; the unused completion vectors fill the remaining
        // slots in order.
        u.try_resize_flat(rows, rows)?;
        for i in 0..rows {
            for (j, &source) in w.mapping.iter().enumerate() {
                u.set_element(i, j, w.basis.get_element(i, source));
            }
        }
    }
    candidate.cs.try_resize_flat(n, n)?;
    candidate.cs.as_mut_slice().fill(S::ZERO);
    for i in 0..a {
        candidate.cs.set_element(i, i, S::ONE);
    }
    for i in 0..d {
        candidate.cs.set_element(p + e + r + i, a + r + i, S::ONE);
    }
    for i in 0..e {
        candidate.cs.set_element(p + i, q + i, S::ONE);
    }
    for i in 0..b {
        candidate.cs.set_element(a + r + i, q + e + r + i, -S::ONE);
    }
    for i in 0..r {
        let c = S::from_real(candidate.cosines[i]);
        let s = S::from_real(candidate.sines[i]);
        candidate.cs.set_element(a + i, a + i, c);
        candidate.cs.set_element(p + e + i, a + i, s);
        candidate.cs.set_element(a + i, q + e + i, -s);
        candidate.cs.set_element(p + e + i, q + e + i, c);
    }
    // P = diag(U1,U2) CS. The remaining columns span the supplied right complement.
    w.product.try_resize_flat(n, n)?;
    for i in 0..n {
        context.execution().check()?;
        let (u, row, offset) = if i < p {
            (&candidate.u1, i, 0)
        } else {
            (&candidate.u2, i - p, p)
        };
        for j in 0..n {
            let value = (0..u.cols()).fold(S::ZERO, |sum, k| {
                sum + u.get_element(row, k) * candidate.cs.get_element(offset + k, j)
            });
            w.product.set_element(i, j, value);
        }
    }
    candidate.v2.try_resize_flat(n - q, n - q)?;
    for i in 0..n - q {
        context.execution().check()?;
        for j in 0..n - q {
            let value = (0..n).fold(S::ZERO, |sum, k| {
                sum + w.input.get_element(k, q + i).conj() * w.product.get_element(k, q + j)
            });
            candidate.v2.set_element(i, j, value);
        }
    }
    Ok(())
}

fn validate_factors<S: NumericScalar>(
    w: &CsdScratch<'_, S>,
    q: usize,
    epsilon: S::Real,
    context: KernelContext<'_>,
) -> Result<(S::Real, S::Real), SciOxideError> {
    let n = w.input.rows();
    let candidate = &w.candidate;
    // Frobenius norm of the reconstruction defect, one stop check per row.
    let residual = crate::kernel::standard::try_norm((0..n).map(|i| {
        context.execution().check()?;
        crate::kernel::standard::norm_workspace((0..n).map(|j| {
            let (v, col, offset) = if j < q {
                (&candidate.v1, j, 0)
            } else {
                (&candidate.v2, j - q, q)
            };
            let value = (0..v.cols()).fold(S::ZERO, |sum, k| {
                sum + w.product.get_element(i, offset + k) * v.get_element(col, k).conj()
            });
            w.input.get_element(i, j) - value
        }))
    }))?;
    let mut orthogonality_error = S::Real::ZERO;
    for u in [&candidate.u1, &candidate.u2, &candidate.v1, &candidate.v2] {
        orthogonality_error = orthogonality_error.max(orthogonality(u.try_view()?, context)?);
    }
    let bound = epsilon * S::Real::try_from_usize_exact(128)?;
    if residual > bound || orthogonality_error > bound {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Cancellation,
            "CS reconstruction or orthogonality exceeds roundoff allowance",
        ));
    }
    Ok((residual, orthogonality_error))
}
