//! Interval extensions of scalar, linear and nonlinear maps.
//!
//! Each trait method that receives a solve context ends in `_scoped` and takes
//! the context last. The verifier counts each call toward the evaluation
//! budget before it makes it; the method charges the context only for nested
//! work, for example by passing it to a `*_scoped` function of another crate.
//! Implement the unsuffixed method; the scoped default forwards to it. Override
//! the scoped method when a nested solve needs the caller’s context.
use crate::DecoratedInterval;
use scioxide_core::{
    Invocation, JacobianOperator, LinearOperator, NumericScalar, OperatorContext, RealScalar,
    ResidualOperator, SciOxideError, SolveContext,
};

/// Rectangle `{x + iy : x ∈ re, y ∈ im}` in the complex plane, enclosing a
/// real or complex value.
///
/// For real problems the imaginary interval must be exactly zero.
pub type IntervalBox<R> = scioxide_core::Complex<scioxide_core::ExtendedInterval<R>>;

/// Recheck the square shape and scratch length of a linear provider.
pub(crate) fn validate_linear_provider<S: NumericScalar>(
    provider: &(impl IntervalLinearOperator<S> + ?Sized),
    order: usize,
    scratch: usize,
) -> Result<(), SciOxideError> {
    if provider.rows() != order
        || provider.cols() != order
        || provider.enclosure_workspace_len()? != scratch
    {
        return Err(crate::shell::invalid(
            scioxide_core::argument!("self"),
            "provider shape or requirements changed",
        ));
    }
    Ok(())
}

/// Interval extension of a real function `f`, used by [`ScalarRootVerifier`](crate::ScalarRootVerifier).
///
/// For every input `X`, the result must contain `{f(x) : x ∈ X}`, including the
/// rounding errors of the evaluation, and its decoration must describe whether
/// `f` is defined and continuous on `X`. Evaluating `f` with the arithmetic of
/// [`DecoratedInterval`] meets both requirements. The function must not change
/// during a proof. A verifier cannot detect an extension that breaks these
/// rules; see [Interval extensions you provide](crate#interval-extensions-you-provide).
///
/// Closures `Fn(DecoratedInterval<R>) -> Result<DecoratedInterval<R>, SciOxideError>`
/// implement this trait.
///
/// ```
/// use scioxide_core::SciOxideError;
/// use scioxide_validated::{DecoratedInterval as D, IntervalScalarFunction};
///
/// let square = |x: D<f64>| x.square();
/// fn takes(_: &dyn IntervalScalarFunction<f64>) {}
/// takes(&square);
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait IntervalScalarFunction<R: RealScalar> {
    /// Returns an enclosure of `f` on `input`.
    ///
    /// # Errors
    ///
    /// Propagates implementation errors unchanged.
    fn enclose(&self, input: DecoratedInterval<R>) -> Result<DecoratedInterval<R>, SciOxideError>;

    /// Returns an enclosure of `f` on `input` inside the caller's solve.
    ///
    /// Override this method to charge nested work to `context`.
    ///
    /// # Errors
    ///
    /// Propagates implementation errors unchanged.
    fn enclose_scoped(
        &self,
        input: DecoratedInterval<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<DecoratedInterval<R>, SciOxideError> {
        let _ = context;
        self.enclose(input)
    }
}
impl<R, F> IntervalScalarFunction<R> for F
where
    R: RealScalar,
    F: Fn(DecoratedInterval<R>) -> Result<DecoratedInterval<R>, SciOxideError> + ?Sized,
{
    fn enclose(&self, input: DecoratedInterval<R>) -> Result<DecoratedInterval<R>, SciOxideError> {
        self(input)
    }
}

/// Interval extension of a linear map, or of a fixed set of linear maps, used
/// by [`LinearVerifier`](crate::LinearVerifier) and
/// [`SystemRootVerifier`](crate::SystemRootVerifier).
///
/// The operator stands for a nonempty set `𝒜` of matrices, for example all
/// matrices whose entries lie in given intervals. Its point application
/// ([`LinearOperator`]) approximates one member. Each enclosure must contain
/// `Ax + c` for every `A ∈ 𝒜`, every vector `x` in the input boxes and every
/// `c` in the addend boxes (zero when absent), including rounding errors. The
/// operator must not change during a proof, and a verifier cannot detect one
/// that breaks these rules. [`DenseIntervalOperator`](crate::DenseIntervalOperator),
/// [`SparseIntervalOperator`](crate::SparseIntervalOperator) and
/// [`LuIntervalInverse`](crate::LuIntervalInverse) implement this trait.
///
/// Inputs are bounded. An unbounded output makes the proof inconclusive and an
/// empty output is an error. The slices never overlap. On error the output
/// must be unchanged; the scratch slice may hold anything after a call.
pub trait IntervalLinearOperator<S: NumericScalar>: LinearOperator<S> {
    /// Returns the length of the scratch slice that the enclosure methods need.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length overflows `usize`.
    fn enclosure_workspace_len(&self) -> Result<usize, SciOxideError>;

    /// Writes an enclosure of `{Ax + c}` to `output` without allocating.
    ///
    /// The addend lets a residual `b − Ax` be enclosed without rounding `Ax`
    /// first.
    ///
    /// # Errors
    ///
    /// The implementation returns an error, leaving `output` unchanged, for
    /// slices of the wrong length, invalid input boxes or an unsupported
    /// floating-point environment.
    fn apply_enclosure_into(
        &self,
        input: &[IntervalBox<S::Real>],
        addend: Option<&[IntervalBox<S::Real>]>,
        output: &mut [IntervalBox<S::Real>],
        workspace: &mut [IntervalBox<S::Real>],
    ) -> Result<(), SciOxideError>;

    /// Writes an enclosure of `{Ax + c}` to `output` inside a solve, which
    /// supplies the execution context and counts nested work.
    ///
    /// The verifier has already counted this call; the implementation charges
    /// `context` only for additional work. As for
    /// [`LinearOperator::apply_into_scoped`], only `Evaluated(())` means that
    /// `output` holds the result; the other [`Invocation`] variants report a
    /// refused budget or a stop. The default returns a pending stop, and
    /// otherwise calls [`Self::apply_enclosure_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::apply_enclosure_into`], and errors of nested work.
    fn apply_enclosure_into_scoped(
        &self,
        input: &[IntervalBox<S::Real>],
        addend: Option<&[IntervalBox<S::Real>]>,
        output: &mut [IntervalBox<S::Real>],
        workspace: &mut [IntervalBox<S::Real>],
        context: &mut OperatorContext<'_, '_, '_, S>,
    ) -> Result<Invocation<()>, SciOxideError> {
        if let Some(reason) = context.stop_reason()? {
            return Ok(Invocation::ExecutionStopped(reason));
        }
        self.apply_enclosure_into(input, addend, output, workspace)
            .map(Invocation::Evaluated)
    }
}

/// Interval extension of a continuous map `F: ℂⁿ → ℂⁿ` or `ℝⁿ → ℝⁿ`, used by
/// [`SystemRootVerifier`](crate::SystemRootVerifier).
///
/// For every box `X`, the output must contain `{F(x) : x ∈ X}`, including
/// rounding errors. Input boxes are bounded and nonempty, with zero imaginary
/// parts for real maps. An empty or incomplete output is an error; an
/// unbounded output makes the proof inconclusive. The map must not change
/// during a proof and must leave the output unchanged on error. A verifier
/// cannot detect an extension that breaks these rules. The slices never
/// overlap.
pub trait IntervalResidualOperator<T: NumericScalar>: ResidualOperator<T> {
    /// Returns the length of the scratch slice that
    /// [`Self::residual_enclosure_into_scoped`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length overflows `usize`.
    fn residual_enclosure_workspace_len(&self) -> Result<usize, SciOxideError>;
    /// Writes an enclosure of `F` on the box `domain` to `output` without
    /// allocating.
    ///
    /// The verifier counts this evaluation; override the scoped method for nested work.
    ///
    /// # Errors
    ///
    /// Errors of the implementation propagate unchanged to the caller of the
    /// verifier.
    fn residual_enclosure_into(
        &self,
        domain: &[IntervalBox<T::Real>],
        output: &mut [IntervalBox<T::Real>],
        scratch: &mut [IntervalBox<T::Real>],
    ) -> Result<(), SciOxideError>;

    /// Evaluates the enclosure inside the caller's solve; override for nested work.
    ///
    /// # Errors
    ///
    /// Propagates implementation errors unchanged.
    fn residual_enclosure_into_scoped(
        &self,
        domain: &[IntervalBox<T::Real>],
        output: &mut [IntervalBox<T::Real>],
        scratch: &mut [IntervalBox<T::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let _ = context;
        self.residual_enclosure_into(domain, output, scratch)
    }
}

/// Interval extension of the Jacobian `J = F′` of a continuously
/// differentiable map, used by [`SystemRootVerifier`](crate::SystemRootVerifier).
///
/// The output must contain `J(x)v` for every `x` in the box `domain` and every
/// `v` in the box `direction`, including rounding errors. `F` must be
/// differentiable on a neighborhood of the box. A complex map must be
/// holomorphic; write a map that is not as a real map of twice the dimension.
/// The residual and Jacobian extensions must describe the same map. The point
/// products of [`JacobianOperator`] are approximations and are not used as
/// proof.
pub trait IntervalJacobianOperator<T: NumericScalar>:
    IntervalResidualOperator<T> + JacobianOperator<T>
{
    /// Returns the length of the scratch slice that
    /// [`Self::jvp_enclosure_into_scoped`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length overflows `usize`.
    fn jvp_enclosure_workspace_len(&self) -> Result<usize, SciOxideError>;
    /// Writes an enclosure of `{J(x)v : x ∈ domain, v ∈ direction}` to
    /// `output` without allocating.
    ///
    /// Budget accounting, slices and errors follow
    /// [`IntervalResidualOperator::residual_enclosure_into_scoped`].
    ///
    /// # Errors
    ///
    /// Errors of the implementation propagate unchanged to the caller of the
    /// verifier.
    fn jvp_enclosure_into(
        &self,
        domain: &[IntervalBox<T::Real>],
        direction: &[IntervalBox<T::Real>],
        output: &mut [IntervalBox<T::Real>],
        scratch: &mut [IntervalBox<T::Real>],
    ) -> Result<(), SciOxideError>;

    /// Evaluates the enclosure inside the caller's solve; override for nested work.
    ///
    /// # Errors
    ///
    /// Propagates implementation errors unchanged.
    fn jvp_enclosure_into_scoped(
        &self,
        domain: &[IntervalBox<T::Real>],
        direction: &[IntervalBox<T::Real>],
        output: &mut [IntervalBox<T::Real>],
        scratch: &mut [IntervalBox<T::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let _ = context;
        self.jvp_enclosure_into(domain, direction, output, scratch)
    }
}
