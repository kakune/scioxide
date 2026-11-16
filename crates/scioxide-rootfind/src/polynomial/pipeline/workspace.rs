//! Workspace of polynomial root solves.
use super::*;
/// Holds the scratch storage of [`crate::polynomial_roots_into`] and the
/// diagnostics of the last solve.
///
/// A workspace is reserved for one degree and one method. Create it with
/// [`Self::workspace`], and reserve it again with [`Self::try_reserve`] and
/// [`Self::workspace_requirements`] before solving a polynomial of another
/// degree or with another method. For companion methods it holds both a real
/// and a complex matrix, so real and complex coefficients of the same degree
/// share one workspace. `Auto` above degree 64 needs only O(n) storage. The
/// general rules are in
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
#[derive(Debug)]
pub struct PolynomialRootWorkspace<S: RealScalar>
where
    Complex<S>: NumericScalar<Real = S>,
{
    pub(crate) coefficients: Vec<Complex<S>>,
    pub(crate) deflated: Vec<Complex<S>>,
    pub(crate) current: Vec<Complex<S>>,
    pub(crate) next: Vec<Complex<S>>,
    pub(crate) candidates: Vec<Complex<S>>,
    pub(crate) best: Vec<Complex<S>>,
    pub(crate) evaluations: Vec<Option<CheckedBasisEvaluation<S>>>,
    pub(crate) seed_radii: Vec<S>,
    pub(crate) root_radii: Vec<S>,
    pub(crate) hull: Vec<usize>,
    pub(crate) scratch_diagnostics: Vec<PolynomialRootDiagnostic<S>>,
    pub(crate) diagnostics: Vec<PolynomialRootDiagnostic<S>>,
    pub(crate) matrix: Matrix<Complex<S>>,
    pub(crate) real_matrix: Matrix<S>,
    pub(crate) real_coefficients: Vec<S>,
    pub(crate) eigen_result: GeneralEigenvaluesBuffer<S>,
    pub(crate) eigen_workspace: LinalgWorkspace<'static, Complex<S>>,
    pub(crate) prepared: Option<(usize, PolynomialRootMethod)>,
    pub(crate) shell: crate::attempt::Shell<PolynomialRootReport<S>>,
}
impl<S: RealScalar> PolynomialRootWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            deflated: Vec::new(),
            current: Vec::new(),
            next: Vec::new(),
            candidates: Vec::new(),
            best: Vec::new(),
            evaluations: Vec::new(),
            seed_radii: Vec::new(),
            root_radii: Vec::new(),
            hull: Vec::new(),
            scratch_diagnostics: Vec::new(),
            diagnostics: Vec::new(),
            matrix: Matrix::empty(),
            real_matrix: Matrix::empty(),
            real_coefficients: Vec::new(),
            eigen_result: GeneralEigenvaluesBuffer::empty(),
            eigen_workspace: LinalgWorkspace::new(),
            prepared: None,
            shell: crate::attempt::Shell::new(),
        }
    }
    /// Returns the workspace size for polynomials of degree `degree` solved
    /// with `options`.
    ///
    /// The size depends on the degree and on [`PolynomialRootOptions::method`]:
    /// companion methods, and `Auto` through degree 64, need O(n²) storage for
    /// the matrix and its eigenvalue workspace, the other methods O(n). The
    /// eigenvalue workspace follows
    /// [`PolynomialRootOptions::execution_policy`], and the size is limited by
    /// [`PolynomialRootOptions::workspace_byte_limit`]. Pass the result to
    /// [`Self::try_reserve`] or [`Self::check`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`degree`, `OutOfRange`) if `degree` is zero.
    /// - `SizeOverflow` if a buffer size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds
    ///   [`PolynomialRootOptions::workspace_byte_limit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_rootfind::{PolynomialRootOptions, PolynomialRootWorkspace};
    ///
    /// // `Auto` forms a companion matrix through degree 64 and none above.
    /// let options = PolynomialRootOptions::<f64>::default();
    /// let companion = PolynomialRootWorkspace::workspace_requirements(64, &options)?;
    /// let iterative = PolynomialRootWorkspace::workspace_requirements(65, &options)?;
    /// assert_eq!(companion.companion_dimension(), 64);
    /// assert_eq!(iterative.companion_dimension(), 0);
    /// assert!(iterative.required_bytes()? < companion.required_bytes()?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        degree: usize,
        options: &PolynomialRootOptions<S>,
    ) -> Result<PolynomialRootWorkspaceRequirements<S>, SciOxideError> {
        let mut requirements = Self::requirements(degree, options.method)?;
        requirements.limit = options.byte_limit;
        requirements.policy = options.policy;
        let bytes = requirements.required_bytes()?;
        if bytes > requirements.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                requirements.limit,
            ));
        }
        Ok(requirements)
    }
    /// Creates a workspace reserved for polynomials of degree `degree` solved
    /// with `options`.
    ///
    /// The workspace serves [`crate::polynomial_roots_into`] for this degree
    /// and [`PolynomialRootOptions::method`] without allocating. It equals
    /// [`Self::new`] reserved with [`Self::try_reserve`] for
    /// [`Self::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`
    /// if the storage cannot be allocated.
    pub fn workspace(
        degree: usize,
        options: &PolynomialRootOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(degree, options)?)?;
        Ok(workspace)
    }
    /// Returns the per-root diagnostics of the last solve that wrote its
    /// output, aligned with that output.
    pub fn diagnostics(&self) -> &[PolynomialRootDiagnostic<S>] {
        &self.diagnostics
    }
    /// Computes the storage for `degree` and `method`: one stored evaluation
    /// and two diagnostic records per root, and `degree + 1` hull indices.
    /// # Errors
    /// Rejects zero degree and overflowing sizes.
    pub(crate) fn requirements(
        degree: usize,
        method: PolynomialRootMethod,
    ) -> Result<PolynomialRootWorkspaceRequirements<S>, SciOxideError> {
        if degree == 0 {
            return Err(invalid(
                scioxide_core::argument!("degree"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "degree must be positive",
            ));
        }
        let companion = driver::primary_method(degree, method) == PolynomialRootMethod::Companion;
        let matrix = if companion {
            checked_mul(degree, degree, scioxide_core::quantity!("root_matrix"))?
        } else {
            0
        };
        let coefficients = checked_mul(
            checked_add(degree, 1, scioxide_core::quantity!("coefficients"))?,
            if method == PolynomialRootMethod::Laguerre {
                2
            } else {
                1
            },
            scioxide_core::quantity!("coefficient_scratch"),
        )?;
        Ok(PolynomialRootWorkspaceRequirements {
            scalar: core::marker::PhantomData,
            degree,
            method,
            limit: usize::MAX,
            policy: scioxide_core::ExecutionPolicy::Sequential,
            real_elements: checked_add(
                checked_mul(degree, 2, scioxide_core::quantity!("root_radii"))?,
                checked_add(
                    matrix,
                    if companion { degree + 1 } else { 0 },
                    scioxide_core::quantity!("real_companion"),
                )?,
                scioxide_core::quantity!("real_scratch"),
            )?,
            complex_elements: checked_add(
                checked_add(
                    coefficients,
                    checked_mul(degree, 4, scioxide_core::quantity!("root_scratch"))?,
                    scioxide_core::quantity!("root_scratch"),
                )?,
                matrix,
                scioxide_core::quantity!("root_matrix"),
            )?,
            companion_dimension: if companion { degree } else { 0 },
            evaluation_records: degree,
            diagnostic_records: checked_mul(
                degree,
                2,
                scioxide_core::quantity!("diagnostic_records"),
            )?,
            index_elements: checked_add(degree, 1, scioxide_core::quantity!("seed_hull"))?,
        })
    }
    /// Reserves storage for the degree and method of `requirements`.
    ///
    /// Does nothing if the workspace is already reserved for them.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a buffer size overflows `usize`.
    /// - `ResourceLimit` if the storage would exceed the byte limit.
    /// - `AllocationFailure` if the storage cannot be allocated. The workspace
    ///   must then be reserved again before it is used.
    pub fn try_reserve(
        &mut self,
        requirements: &PolynomialRootWorkspaceRequirements<S>,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        let degree = requirements.degree;
        let method = requirements.method;

        self.prepared = None;
        let zero = Complex::new(S::ZERO, S::ZERO);
        resize(
            &mut self.coefficients,
            degree + 1,
            zero,
            scioxide_core::quantity!("coefficients"),
        )?;
        resize(
            &mut self.deflated,
            if method == PolynomialRootMethod::Laguerre {
                degree + 1
            } else {
                0
            },
            zero,
            scioxide_core::quantity!("deflation"),
        )?;
        for buffer in [
            &mut self.current,
            &mut self.next,
            &mut self.candidates,
            &mut self.best,
        ] {
            resize(
                buffer,
                degree,
                zero,
                scioxide_core::quantity!("root_scratch"),
            )?;
        }
        resize(
            &mut self.evaluations,
            degree,
            None,
            scioxide_core::quantity!("root_evaluations"),
        )?;
        resize(
            &mut self.seed_radii,
            degree,
            S::ZERO,
            scioxide_core::quantity!("seed_radii"),
        )?;
        resize(
            &mut self.root_radii,
            degree,
            S::ZERO,
            scioxide_core::quantity!("root_radii"),
        )?;
        resize(
            &mut self.hull,
            degree + 1,
            0,
            scioxide_core::quantity!("seed_hull"),
        )?;
        resize(
            &mut self.scratch_diagnostics,
            degree,
            PolynomialRootDiagnostic::unevaluated(),
            scioxide_core::quantity!("root_diagnostics"),
        )?;
        resize(
            &mut self.diagnostics,
            degree,
            PolynomialRootDiagnostic::unevaluated(),
            scioxide_core::quantity!("root_diagnostics"),
        )?;
        if requirements.companion_dimension > 0 {
            resize(
                &mut self.real_coefficients,
                degree + 1,
                S::ZERO,
                scioxide_core::quantity!("real_coefficients"),
            )?;
            self.real_matrix.try_resize_flat(degree, degree)?;
            self.eigen_workspace
                .try_reserve(&eigen_requirements::<S>(degree, requirements.policy)?)?;
            // Warm the shared eigen workspace and its output on a diagonal
            // companion-sized matrix, which deflates without QR steps.
            self.matrix.try_resize_flat(degree, degree)?;
            let one = Complex::new(S::ONE, S::ZERO);
            self.matrix
                .as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(index, value)| {
                    *value = if index % (degree + 1) == 0 { one } else { zero };
                });
            GeneralEigenvaluesBuffer::compute_into(
                &self.matrix,
                &GeneralEigenOptions::default().with_execution_options(
                    scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(requirements.policy),
                ),
                &mut self.eigen_result,
                &mut self.eigen_workspace,
                scioxide_core::ExecutionContext::sequential(),
            )?;
        }
        self.prepared = Some((degree, method));
        Ok(())
    }
}
impl<S: RealScalar> PolynomialRootWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Returns whether the workspace is reserved for the degree and method of
    /// `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a buffer size overflows `usize`.
    /// - `ResourceLimit` if the storage already held, grown to the request,
    ///   would exceed the byte limit.
    pub fn check(
        &self,
        requirements: &PolynomialRootWorkspaceRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let retained = self.projected_bytes(requirements)?;
        if retained > requirements.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                retained,
                requirements.limit,
            ));
        }
        let eigen_ready = if requirements.companion_dimension > 0 {
            self.eigen_workspace.check(&eigen_requirements::<S>(
                requirements.companion_dimension,
                requirements.policy,
            )?)? == scioxide_core::Capacity::Sufficient
        } else {
            true
        };
        Ok(
            if eigen_ready && self.prepared == Some((requirements.degree, requirements.method)) {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    fn projected_bytes(
        &self,
        r: &PolynomialRootWorkspaceRequirements<S>,
    ) -> Result<usize, SciOxideError> {
        use crate::workspace::{buffer, sum};
        let q = scioxide_core::quantity!("workspace_bytes");
        let degree = r.degree;
        let coefficient = checked_mul(degree + 1, core::mem::size_of::<Complex<S>>(), q)?;
        let roots = checked_mul(degree, core::mem::size_of::<Complex<S>>(), q)?;
        let diagnostics = checked_mul(
            degree,
            core::mem::size_of::<PolynomialRootDiagnostic<S>>(),
            q,
        )?;
        let eigen = r.companion_dimension;
        let shared = if eigen > 0 {
            self.eigen_workspace
                .reservation_bytes(&eigen_requirements::<S>(eigen, r.policy)?)?
        } else {
            self.eigen_workspace.retained_bytes()?
        };
        sum([
            buffer(&self.real_coefficients).and_then(|b| {
                Ok(b.max(checked_mul(
                    if eigen > 0 { degree + 1 } else { 0 },
                    core::mem::size_of::<S>(),
                    q,
                )?))
            }),
            self.real_matrix.retained_bytes().and_then(|b| {
                Ok(b.max(checked_mul(
                    checked_mul(eigen, eigen, q)?,
                    core::mem::size_of::<S>(),
                    q,
                )?))
            }),
            buffer(&self.coefficients).map(|b| b.max(coefficient)),
            buffer(&self.deflated).map(|b| {
                b.max(if r.method == PolynomialRootMethod::Laguerre {
                    coefficient
                } else {
                    0
                })
            }),
            buffer(&self.current).map(|b| b.max(roots)),
            buffer(&self.next).map(|b| b.max(roots)),
            buffer(&self.candidates).map(|b| b.max(roots)),
            buffer(&self.best).map(|b| b.max(roots)),
            buffer(&self.evaluations).and_then(|b| {
                Ok(b.max(checked_mul(
                    r.evaluation_records,
                    core::mem::size_of::<Option<CheckedBasisEvaluation<S>>>(),
                    q,
                )?))
            }),
            buffer(&self.seed_radii)
                .and_then(|b| Ok(b.max(checked_mul(degree, core::mem::size_of::<S>(), q)?))),
            buffer(&self.root_radii)
                .and_then(|b| Ok(b.max(checked_mul(degree, core::mem::size_of::<S>(), q)?))),
            buffer(&self.hull).and_then(|b| {
                Ok(b.max(checked_mul(degree + 1, core::mem::size_of::<usize>(), q)?))
            }),
            buffer(&self.diagnostics).map(|b| b.max(diagnostics)),
            buffer(&self.scratch_diagnostics).map(|b| b.max(diagnostics)),
            self.matrix.retained_bytes().and_then(|b| {
                Ok(b.max(checked_mul(
                    checked_mul(eigen, eigen, q)?,
                    core::mem::size_of::<Complex<S>>(),
                    q,
                )?))
            }),
            self.eigen_result.retained_bytes().and_then(|b| {
                Ok(b.max(checked_mul(eigen, core::mem::size_of::<Complex<S>>(), q)?))
            }),
            Ok(shared),
        ])
    }

    /// Returns the record of the most recent call, including a call rejected
    /// before any numerical work.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<PolynomialRootReport<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the report of the most recent call if it produced one, or
    /// `None` if that call returned an error. A report from an earlier call is never
    /// returned.
    pub fn last_accepted(&self) -> Option<&PolynomialRootReport<S>> {
        self.shell.progress()
    }
}
crate::attempt::attempted!(
    PolynomialRootWorkspace<S>,
    [S: RealScalar, Complex<S>: NumericScalar<Real = S>],
    PolynomialRootReport<S>
);
impl<S: RealScalar> PolynomialRootWorkspaceRequirements<S> {
    /// Returns the workspace size in bytes, including the eigenvalue
    /// workspace of companion methods.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(&self) -> Result<usize, SciOxideError>
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let q = scioxide_core::quantity!("workspace_bytes");
        let mut bytes = checked_mul(self.complex_elements, core::mem::size_of::<Complex<S>>(), q)?;
        bytes = checked_add(
            bytes,
            checked_mul(
                self.evaluation_records,
                core::mem::size_of::<Option<CheckedBasisEvaluation<S>>>(),
                q,
            )?,
            q,
        )?;
        bytes = checked_add(
            bytes,
            checked_mul(self.real_elements, core::mem::size_of::<S>(), q)?,
            q,
        )?;
        bytes = checked_add(
            bytes,
            checked_mul(self.index_elements, core::mem::size_of::<usize>(), q)?,
            q,
        )?;
        bytes = checked_add(
            bytes,
            checked_mul(
                self.diagnostic_records,
                core::mem::size_of::<PolynomialRootDiagnostic<S>>(),
                q,
            )?,
            q,
        )?;
        if self.companion_dimension > 0 {
            bytes = checked_add(
                bytes,
                eigen_requirements::<S>(self.companion_dimension, self.policy)?
                    .required_bytes::<Complex<S>>()?,
                q,
            )?;
            bytes = checked_add(
                bytes,
                checked_mul(
                    self.companion_dimension,
                    core::mem::size_of::<Complex<S>>(),
                    q,
                )?,
                q,
            )?;
        }
        Ok(bytes)
    }
}

fn eigen_requirements<S: RealScalar>(
    degree: usize,
    policy: scioxide_core::ExecutionPolicy,
) -> Result<scioxide_linalg::LinalgRequirements, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let request = scioxide_linalg::GeneralEigenWorkspaceRequest::new(
        degree,
        &GeneralEigenOptions::<S>::default().with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(policy),
        ),
    );
    Ok(
        GeneralEigenvaluesBuffer::<S>::workspace_requirements::<Complex<S>>(&request)?.max(
            &GeneralEigenvaluesBuffer::<S>::workspace_requirements::<S>(&request)?,
        ),
    )
}

impl<S: RealScalar> Default for PolynomialRootWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn default() -> Self {
        Self::new()
    }
}
