use super::*;
use scioxide_core::{AlgebraicScalar, Invocation, OperatorContext};

use crate::{IntervalJacobianOperator, IntervalLinearOperator};

/// Borrowed claim about the roots of a nonlinear system in a box, checked by
/// [`SystemRootVerifier::verify`].
#[derive(Debug, Clone, Copy)]
pub struct SystemRootCertificateView<'a, S: NumericScalar> {
    /// Box `X` to search, one rectangle per unknown.
    pub region: &'a [IntervalBox<S::Real>],
    /// Point `c ∈ X`, usually an approximate root.
    pub center: &'a [S],
    /// Claim to prove.
    pub claim: RootClaim,
}
/// Mutable region, center and claim of a [`SystemRootCertificate`].
pub struct SystemRootCertificateViewMut<'a, S: NumericScalar> {
    /// Box `X` to search, one rectangle per unknown.
    pub region: &'a mut [IntervalBox<S::Real>],
    /// Point `c ∈ X`, usually an approximate root.
    pub center: &'a mut [S],
    /// Claim to prove.
    pub claim: &'a mut RootClaim,
}
/// Owned claim about the roots of a nonlinear system in a box.
///
/// The storage proves nothing by itself: keep the report of the verification.
///
/// ```
/// use scioxide_validated::{SystemRootCertificate, RootClaim};
///
/// let mut certificate = SystemRootCertificate::<f64>::new(2)?;
/// *certificate.view_mut().claim = RootClaim::Exists;
/// assert_eq!(certificate.view().claim, RootClaim::Exists);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SystemRootCertificate<S: NumericScalar> {
    region: Vec<IntervalBox<S::Real>>,
    center: Vec<S>,
    claim: RootClaim,
}
impl<S: NumericScalar> SystemRootCertificate<S> {
    /// Allocates a claim of `Unique` for `order` unknowns, with a zero region
    /// and center.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(order: usize) -> Result<Self> {
        Ok(Self {
            region: try_filled(order, zero()?, scioxide_core::quantity!("region"))?,
            center: try_filled(order, S::ZERO, scioxide_core::quantity!("center"))?,
            claim: RootClaim::Unique,
        })
    }
    /// Borrows the region, center and claim.
    pub fn view(&self) -> SystemRootCertificateView<'_, S> {
        SystemRootCertificateView {
            region: &self.region,
            center: &self.center,
            claim: self.claim,
        }
    }
    /// Borrows the region, center and claim for writing; a changed claim must
    /// be verified again.
    pub fn view_mut(&mut self) -> SystemRootCertificateViewMut<'_, S> {
        SystemRootCertificateViewMut {
            region: &mut self.region,
            center: &mut self.center,
            claim: &mut self.claim,
        }
    }
}

const KRAWCZYK: AlgorithmId = scioxide_core::algorithm_id!("validated-roots", "krawczyk", 1);

/// Proves or refutes claims about the roots of a nonlinear system `F(x) = 0`
/// in a box, and isolates its roots, with the Krawczyk operator.
///
/// `F: ℝⁿ → ℝⁿ` or `ℂⁿ → ℂⁿ` and its Jacobian `J` are given by one
/// [`IntervalJacobianOperator`]. The preconditioner `R ≈ J(c)⁻¹` is an
/// [`IntervalLinearOperator`] that must stand for a single matrix, for
/// example a [`LuIntervalInverse`](crate::LuIntervalInverse) of the Jacobian
/// at an approximate root. The verifier does not trust `R`.
///
/// # Method
///
/// For a box `X` with center `c`, the Krawczyk operator is
///
/// ```text
/// K(X) = c − R·F(c) + (I − R·J(X))(X − c).
/// ```
///
/// The verifier proves `Empty` if `0 ∉ F(X)` or if `K(X)` misses `X` in some
/// component, `Unique` if `‖I − R·J(X)‖ < 1` and `K(X) ⊆ X`, and `Exists` if
/// `F(c)` is exactly zero. A verification costs `2n + 3` operator evaluations
/// and `n` iterations for `n` unknowns, plus any work inside the operators,
/// and O(n) storage besides the operators' scratch; no Jacobian or inverse
/// matrix is formed.
///
/// ```
/// use scioxide_core::{ExecutionContext, StepTolerance};
/// use scioxide_linalg::MatrixView;
/// use scioxide_validated::{
///     DenseIntervalOperator, ProofRequest, SystemRootCertificateView, RootClaim, ValidatedOptions,
///     SystemRootVerifier, interval_enclosure,
/// };
///
/// // F(x) = x − 1/2 has the unique root 1/2; the Jacobian is 1 and R = 1.
/// let (a, r) = ([1.0_f64], [1.0_f64]);
/// let shift = [interval_enclosure(-0.5, -0.5)?];
/// struct Affine<'a>(
///     DenseIntervalOperator<'a, f64>,
///     &'a [scioxide_validated::IntervalBox<f64>],
/// );
/// # use scioxide_core::{SciOxideError, ResidualOperator, JacobianOperator, LinearOperator};
/// # use scioxide_validated::{IntervalBox, IntervalJacobianOperator, IntervalLinearOperator,
/// #     IntervalResidualOperator};
/// # impl ResidualOperator<f64> for Affine<'_> {
/// #     fn input_dimension(&self) -> usize { 1 }
/// #     fn output_dimension(&self) -> usize { 1 }
/// #     fn residual_workspace_len(&self) -> Result<usize, SciOxideError> { Ok(0) }
/// #     fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
/// #         y[0] = x[0] - 0.5; Ok(())
/// #     }
/// # }
/// # impl JacobianOperator<f64> for Affine<'_> {
/// #     fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> { Ok(0) }
/// #     fn jvp_into(&self, _: &[f64], v: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
/// #         y[0] = v[0]; Ok(())
/// #     }
/// # }
/// # impl IntervalResidualOperator<f64> for Affine<'_> {
/// #     fn residual_enclosure_workspace_len(&self) -> Result<usize, SciOxideError> {
/// #         self.0.enclosure_workspace_len()
/// #     }
/// #     fn residual_enclosure_into(&self, x: &[IntervalBox<f64>], y: &mut [IntervalBox<f64>],
/// #         w: &mut [IntervalBox<f64>]) -> Result<(), SciOxideError> {
/// #         self.0.apply_enclosure_into(x, Some(self.1), y, w)
/// #     }
/// # }
/// # impl IntervalJacobianOperator<f64> for Affine<'_> {
/// #     fn jvp_enclosure_workspace_len(&self) -> Result<usize, SciOxideError> {
/// #         self.0.enclosure_workspace_len()
/// #     }
/// #     fn jvp_enclosure_into(&self, _: &[IntervalBox<f64>], v: &[IntervalBox<f64>],
/// #         y: &mut [IntervalBox<f64>], w: &mut [IntervalBox<f64>])
/// #         -> Result<(), SciOxideError> { self.0.apply_enclosure_into(v, None, y, w) }
/// # }
/// let identity = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &a)?)?;
/// let function = Affine(identity, &shift);
/// let inverse = DenseIntervalOperator::point(MatrixView::from_slice(1, 1, &r)?)?;
/// let verifier = SystemRootVerifier::new(&function, &inverse, ValidatedOptions::new())?;
/// let (region, center) = ([interval_enclosure(0.0, 1.0)?], [0.25]);
/// let proposal = SystemRootCertificateView {
///     region: &region,
///     center: &center,
///     claim: RootClaim::Unique,
/// };
/// assert!(verifier.verify(proposal)?.is_proved());
/// let mut workspace = verifier.workspace(ProofRequest::proofs(1))?;
/// assert!(
///     verifier.verify_scoped(proposal, &mut workspace, ExecutionContext::sequential())?
///         .is_proved()
/// );
/// let (report, tree) = verifier.isolate(&region, StepTolerance::default(), 15)?;
/// assert!(report.is_complete() && report.isolated() == 1);
/// assert!(verifier.verify_isolation(&region, tree.view())?.is_complete());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct SystemRootVerifier<'a, S: NumericScalar> {
    function: &'a (dyn IntervalJacobianOperator<S> + Sync),
    inverse: &'a (dyn IntervalLinearOperator<S> + Sync),
    n: usize,
    residual_scratch: usize,
    jacobian_scratch: usize,
    inverse_scratch: usize,
    options: ValidatedOptions,
}
impl<S: NumericScalar> core::fmt::Debug for SystemRootVerifier<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SystemRootVerifier")
            .field("order", &self.n)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl<'a, S: NumericScalar> SystemRootVerifier<'a, S> {
    /// Creates a verifier for the map `function` and the preconditioner `inverse`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if an operator is not square or the two differ in
    ///   dimension.
    /// - Errors of the operators' scratch-length methods.
    pub fn new(
        function: &'a (dyn IntervalJacobianOperator<S> + Sync),
        inverse: &'a (dyn IntervalLinearOperator<S> + Sync),
        options: ValidatedOptions,
    ) -> Result<Self> {
        let n = function.input_dimension();
        crate::shell::exact(n, function.output_dimension())?;
        crate::shell::exact(n, inverse.rows())?;
        crate::shell::exact(n, inverse.cols())?;
        Ok(Self {
            function,
            inverse,
            n,
            residual_scratch: function.residual_enclosure_workspace_len()?,
            jacobian_scratch: function.jvp_enclosure_workspace_len()?,
            inverse_scratch: inverse.enclosure_workspace_len()?,
            options,
        })
    }
    /// Returns the number of unknowns `n`. For `n = 0` the empty tuple is the
    /// unique root.
    pub const fn order(&self) -> usize {
        self.n
    }
    /// Returns the options.
    pub const fn options(&self) -> &ValidatedOptions {
        &self.options
    }
    pub(super) const fn identity(&self) -> SolveIdentity {
        identity(KRAWCZYK, self.n)
    }
    fn scratch(&self) -> usize {
        self.residual_scratch
            .max(self.jacobian_scratch)
            .max(self.inverse_scratch)
    }
    /// Returns the workspace size for `request`: independent proofs, or an
    /// isolation into a tree of `request.capacity()` nodes.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements(&self, request: ProofRequest) -> Result<RootRequirements> {
        let shape = workspace::Shape {
            order: self.n,
            scratch: self.scratch(),
            tree_order: if request.capacity() > 0 { self.n } else { 0 },
            nodes: request.capacity(),
            ..workspace::Shape::default()
        };
        RootRequirements::new::<S>(shape, request.proof_count(), self.options.byte_limit())
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if it cannot be sized or reserved.
    pub fn workspace(&self, request: ProofRequest) -> Result<RootWorkspace<S>> {
        let mut workspace = RootWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Check that the providers of `self` are unchanged.
    pub(super) fn providers(&self) -> Result<()> {
        if self.function.input_dimension() != self.n
            || self.function.output_dimension() != self.n
            || self.function.residual_enclosure_workspace_len()? != self.residual_scratch
            || self.function.jvp_enclosure_workspace_len()? != self.jacobian_scratch
        {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "root provider shape or requirements changed",
            ));
        }
        crate::protocol::validate_linear_provider(self.inverse, self.n, self.inverse_scratch)
    }
    /// Proves or refutes the claim of `certificate` with the Krawczyk test.
    ///
    /// Allocates a workspace and runs sequentially in a new solve. An unbounded
    /// enclosure from an operator makes the proof inconclusive.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the region or center does not have length `n`.
    /// - `NonFinite` if the center contains NaN or infinity.
    /// - `InvalidInput` if the region is empty or unbounded or has a nonzero
    ///   imaginary part for a real system, the center lies outside it, an
    ///   operator changed shape or returned an empty or invalid enclosure, or
    ///   the [floating-point environment](crate#floating-point-environment) is
    ///   not supported.
    /// - Errors of the operators.
    pub fn verify(
        &self,
        certificate: SystemRootCertificateView<'_, S>,
    ) -> Result<RootVerificationReport<S::Real>> {
        self.verify_scoped(
            certificate,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify`], and those of
    ///   [`RootWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_scoped<'s, 'c>(
        &self,
        certificate: SystemRootCertificateView<'_, S>,
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<RootVerificationReport<S::Real>> {
        workspace.core.run(
            scope.into(),
            self.identity(),
            &self.options,
            || {
                let requirements = self.workspace_requirements(ProofRequest::proofs(1))?;
                self.validate_certificate(certificate)?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, _, _, _, context| self.verify_in(certificate, &mut lanes[0], context),
            RootVerificationReport::is_proved,
        )
    }
    /// Verify inside an existing epoch, mapping stops and committing decided claims.
    pub(super) fn verify_in(
        &self,
        certificate: SystemRootCertificateView<'_, S>,
        lane: &mut workspace::Lane<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<RootVerificationReport<S::Real>> {
        let mut scratch = lane.system(self.n, self.scratch());
        let result = self.check(certificate, &mut scratch, context);
        finish(result, context)
    }
    /// Validate a proposal's shape against this verifier.
    fn shape(&self, certificate: SystemRootCertificateView<'_, S>) -> Result<()> {
        crate::shell::exact(self.n, certificate.center.len())?;
        crate::shell::exact(self.n, certificate.region.len())
    }
    fn validate_certificate(&self, c: SystemRootCertificateView<'_, S>) -> Result<()> {
        crate::environment::check()?;
        self.providers()?;
        self.shape(c)?;
        validate_boxes::<S>(
            c.region,
            true,
            scioxide_core::argument!("certificate.region"),
        )?;
        for (&center, &region) in c.center.iter().zip(c.region) {
            if !contains(region, point_enclosure(center)?) {
                return Err(invalid(
                    scioxide_core::argument!("certificate.center"),
                    "root center must lie in its region",
                ));
            }
        }
        Ok(())
    }
    /// Verifies independent claims, writing one report per certificate.
    ///
    /// The proofs run in parallel under the options' execution policy, and the
    /// reports do not depend on the policy or the number of workers. On error,
    /// `reports` is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `reports` and `certificates` differ in length.
    /// - The errors of [`Self::verify_scoped`] for any certificate, and errors
    ///   of the executor.
    pub fn verify_batch_into<'s, 'c>(
        &self,
        certificates: &[SystemRootCertificateView<'_, S>],
        reports: &mut [RootVerificationReport<S::Real>],
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<()> {
        let count = reports.len();
        workspace
            .core
            .verify_batch(
                scope.into(),
                self.identity(),
                &self.options,
                || {
                    let requirements =
                        self.workspace_requirements(ProofRequest::proofs(certificates.len()))?;
                    crate::shell::exact(certificates.len(), count)?;
                    certificates
                        .iter()
                        .try_for_each(|&c| self.validate_certificate(c))?;
                    Ok(requirements)
                },
                reports,
                |_, _| Ok(crate::shell::Prepared::Ready(())),
                &|index, lane, _, _, context| self.verify_in(certificates[index], lane, context),
            )
            .map(|_| ())
    }
    /// Krawczyk proof of the `certificate` argument, using the `work` argument.
    fn check(
        &self,
        c: SystemRootCertificateView<'_, S>,
        w: &mut workspace::SystemScratch<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<RootVerificationReport<S::Real>> {
        crate::environment::check()?;
        self.providers()?;
        for (&center, point) in c.center.iter().zip(w.point.iter_mut()) {
            *point = point_enclosure(center)?;
        }
        if let Some(reason) = stop(context)? {
            return Ok(RootVerificationReport::unknown(reason));
        }
        if self.n == 0 {
            return Ok(RootVerificationReport::fact(c.claim, RootClaim::Unique));
        }
        macro_rules! apply {
            ($role:expr,$out:ident,$call:expr) => {{
                if let Some(reason) = admit(context, $role)? {
                    return Ok(RootVerificationReport::unknown(reason));
                }
                w.$out.fill(sentinel());
                if let Some(reason) = invoked($call?) {
                    return Ok(RootVerificationReport::unknown(reason));
                }
                self.providers()?;
                crate::environment::check()?;
                validate_boxes::<S>(&w.$out, false, scioxide_core::argument!("self"))?;
                if w.$out.iter().any(|&v| !unpack(v).finite()) {
                    return Ok(RootVerificationReport::unknown(RootInconclusive::Unbounded));
                }
            }};
        }
        apply!(
            EvaluationRole::Residual,
            a,
            self.function
                .residual_enclosure_into_scoped(
                    c.region,
                    w.a,
                    &mut w.provider[..self.residual_scratch],
                    context
                )
                .map(Invocation::Evaluated)
        );
        if w.a.iter().any(|&v| !unpack(v).contains_zero()) {
            return Ok(RootVerificationReport::fact(c.claim, RootClaim::Empty));
        }
        apply!(
            EvaluationRole::Residual,
            a,
            self.function
                .residual_enclosure_into_scoped(
                    w.point,
                    w.a,
                    &mut w.provider[..self.residual_scratch],
                    context
                )
                .map(Invocation::Evaluated)
        );
        let exact_root = w.a.iter().all(|&v| unpack(v).is_zero());
        apply!(
            EvaluationRole::Preconditioner,
            r,
            self.inverse.apply_enclosure_into_scoped(
                w.a,
                None,
                w.r,
                &mut w.provider[..self.inverse_scratch],
                &mut OperatorContext::new(context)
            )
        );
        for ((slot, &p), &r) in w.image.iter_mut().zip(w.point.iter()).zip(w.r.iter()) {
            *slot = pack(unpack(p).sub(unpack(r))?);
        }
        w.rows.fill(S::Real::ZERO);
        for j in 0..self.n {
            if let Some(reason) = iteration(context)? {
                return Ok(RootVerificationReport::unknown(reason));
            }
            w.direction.fill(zero()?);
            w.direction[j] = point_enclosure(S::ONE)?;
            apply!(
                EvaluationRole::Jacobian,
                a,
                self.function
                    .jvp_enclosure_into_scoped(
                        c.region,
                        w.direction,
                        w.a,
                        &mut w.provider[..self.jacobian_scratch],
                        context
                    )
                    .map(Invocation::Evaluated)
            );
            apply!(
                EvaluationRole::Preconditioner,
                r,
                self.inverse.apply_enclosure_into_scoped(
                    w.a,
                    None,
                    w.r,
                    &mut w.provider[..self.inverse_scratch],
                    &mut OperatorContext::new(context)
                )
            );
            let delta = unpack(c.region[j]).sub(unpack(w.point[j]))?;
            for (((slot, &basis), &product), row) in w
                .image
                .iter_mut()
                .zip(w.direction.iter())
                .zip(w.r.iter())
                .zip(w.rows.iter_mut())
            {
                let e = defect_entry(basis, product, row)?;
                *slot = pack(unpack(*slot).add(e.mul(delta)?)?);
            }
        }
        let q = w.rows.iter().copied().fold(S::Real::ZERO, RealScalar::max);
        let mut report = if c
            .region
            .iter()
            .zip(w.image.iter())
            .any(|(&x, &k)| disjoint(x, k))
        {
            RootVerificationReport::fact(c.claim, RootClaim::Empty)
        } else if q < S::Real::ONE
            && c.region
                .iter()
                .zip(w.image.iter())
                .all(|(&x, &k)| contains(x, k))
        {
            RootVerificationReport::fact(c.claim, RootClaim::Unique)
        } else if exact_root {
            RootVerificationReport::fact(c.claim, RootClaim::Exists)
        } else {
            RootVerificationReport::unknown(if q.is_finite() {
                RootInconclusive::Reason(RootProofFailure::Inequality)
            } else {
                RootInconclusive::Unbounded
            })
        };
        report.defect = Some(q);
        Ok(report)
    }
}
