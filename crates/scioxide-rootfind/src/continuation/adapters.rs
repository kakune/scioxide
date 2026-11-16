//! Adapters that build continuation and deflation problems from other maps.
use super::*;
use core::cell::RefCell;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, NumericScalar};
/// Rewrites a complex map `G: ℂⁿ → ℂⁿ` as a real map `ℝ²ⁿ → ℝ²ⁿ`.
///
/// Each complex coordinate `zₖ = aₖ + i bₖ` becomes the pair `(aₖ, bₖ)`, in
/// that interleaved order, for both inputs and outputs. The Jacobian–vector
/// product of the real map is the complex product `G'(z) v`, which is correct
/// only when `G` is holomorphic. For a map that is not holomorphic, implement
/// the real map directly.
///
/// [`Self::square`] gives a square map for [`DeflatedProblem`] or the system
/// solvers. [`Self::with_real_parameter`] adds a real parameter `λ` as the last
/// input, giving a map `ℝ²ⁿ⁺¹ → ℝ²ⁿ` for [`ContinuationProblem`]; `λ` is
/// passed to `G` as the complex number `λ + 0i`.
///
/// The adapter holds O(n) conversion buffers. It is not reentrant: a call made
/// from inside the wrapped provider is an `InvalidInput` error.
pub struct RealifiedProblem<'a, R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    provider: &'a dyn JacobianOperator<Complex<R>>,
    parameter: bool,
    storage: RefCell<RealificationStorage<R>>,
}
struct RealificationStorage<R: RealScalar> {
    point: Vec<Complex<R>>,
    direction: Vec<Complex<R>>,
    output: Vec<Complex<R>>,
    scratch: Vec<Complex<R>>,
}
impl<'a, R: RealScalar> RealifiedProblem<'a, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Wraps a square complex map `G: ℂⁿ → ℂⁿ`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `provider` has no outputs or is not square.
    /// - `SizeOverflow` or `AllocationFailure` if the buffers cannot be allocated.
    /// - Any error from the provider's workspace-length queries.
    pub fn square(provider: &'a dyn JacobianOperator<Complex<R>>) -> Result<Self, SciOxideError> {
        Self::new(provider, false)
    }
    /// Wraps a map `G: ℂⁿ × ℝ → ℂⁿ` whose last input is a real parameter.
    ///
    /// `provider` takes `n + 1` complex inputs; the imaginary part of the last
    /// one is always zero.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `provider` has no outputs or does not have exactly
    ///   one more input than outputs.
    /// - `SizeOverflow` or `AllocationFailure` if the buffers cannot be allocated.
    /// - Any error from the provider's workspace-length queries.
    pub fn with_real_parameter(
        provider: &'a dyn JacobianOperator<Complex<R>>,
    ) -> Result<Self, SciOxideError> {
        Self::new(provider, true)
    }
    /// Checks the shape of `provider` and allocates the conversion buffers.
    fn new(
        provider: &'a dyn JacobianOperator<Complex<R>>,
        parameter: bool,
    ) -> Result<Self, SciOxideError> {
        let n = provider.output_dimension();
        let m = checked_size(
            n,
            usize::from(parameter),
            scioxide_core::quantity!("input_dimension"),
        )?;
        if n == 0 || provider.input_dimension() != m {
            return Err(invalid(
                scioxide_core::argument!("provider"),
                InvalidInputReason::Inconsistent,
                "complex adapter shape mismatch",
            ));
        }
        n.checked_mul(2)
            .and_then(|v| v.checked_add(usize::from(parameter)))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("real_coordinates"))
            })?;
        let make =
            |len, quantity| scioxide_core::backend::try_filled(len, Complex::<R>::ZERO, quantity);
        Ok(Self {
            provider,
            parameter,
            storage: RefCell::new(RealificationStorage {
                point: make(m, scioxide_core::quantity!("storage.point"))?,
                direction: make(m, scioxide_core::quantity!("storage.direction"))?,
                output: make(n, scioxide_core::quantity!("storage.output"))?,
                scratch: make(
                    provider
                        .residual_workspace_len()?
                        .max(provider.jvp_workspace_len()?),
                    scioxide_core::quantity!("storage.scratch"),
                )?,
            }),
        })
    }
    /// Evaluates the residual, or the Jacobian–vector product when `v` is given.
    fn evaluate(
        &self,
        x: &[R],
        v: Option<&[R]>,
        y: &mut [R],
        scratch: &mut [R],
        mut c: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        let argument = if x.len() != self.input_dimension() {
            Some(scioxide_core::argument!("x"))
        } else if y.len() != self.output_dimension() {
            Some(scioxide_core::argument!("y"))
        } else if !scratch.is_empty() {
            Some(scioxide_core::argument!("s"))
        } else if v.is_some_and(|v| v.len() != x.len()) {
            Some(scioxide_core::argument!("v"))
        } else {
            None
        };
        if let Some(argument) = argument {
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "realification input/output/scratch shape mismatch",
            ));
        }
        finite(x, NonFiniteOrigin::Input)?;
        if let Some(v) = v {
            finite(v, NonFiniteOrigin::Input)?;
        }
        let mut storage = self.storage.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "reentrant realification adapter",
            )
        })?;
        let RealificationStorage {
            point,
            direction,
            output,
            scratch,
        } = &mut *storage;
        let n = output.len();
        for i in 0..n {
            point[i] = Complex::new(x[2 * i], x[2 * i + 1]);
            if let Some(v) = v {
                direction[i] = Complex::new(v[2 * i], v[2 * i + 1]);
            }
        }
        if self.parameter {
            point[n] = Complex::new(x[2 * n], R::ZERO);
            if let Some(v) = v {
                direction[n] = Complex::new(v[2 * n], R::ZERO);
            }
        }
        output.fill(Complex::new(R::NAN, R::NAN));
        if v.is_some() {
            let len = self.provider.jvp_workspace_len()?;
            match c.as_mut() {
                Some(c) => self.provider.jvp_into_scoped(
                    point,
                    direction,
                    output,
                    &mut scratch[..len],
                    c,
                )?,
                None => self
                    .provider
                    .jvp_into(point, direction, output, &mut scratch[..len])?,
            }
        } else {
            let len = self.provider.residual_workspace_len()?;
            match c.as_mut() {
                Some(c) => {
                    self.provider
                        .residual_into_scoped(point, output, &mut scratch[..len], c)?
                }
                None => self
                    .provider
                    .residual_into(point, output, &mut scratch[..len])?,
            }
        }
        finite(output, NonFiniteOrigin::Callback)?;
        for (slot, z) in y.chunks_exact_mut(2).zip(output.iter()) {
            slot[0] = z.re;
            slot[1] = z.im;
        }
        Ok(())
    }
}
impl<R: RealScalar> ResidualOperator<R> for RealifiedProblem<'_, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn input_dimension(&self) -> usize {
        2 * self.provider.output_dimension() + usize::from(self.parameter)
    }
    fn output_dimension(&self) -> usize {
        2 * self.provider.output_dimension()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(x, None, y, s, None)
    }
    fn residual_into_scoped(
        &self,
        x: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, None, y, s, Some(c))
    }
}
impl<R: RealScalar> JacobianOperator<R> for RealifiedProblem<'_, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[R], v: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(v), y, s, None)
    }
    fn jvp_into_scoped(
        &self,
        x: &[R],
        v: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(v), y, s, Some(c))
    }
}

/// Forms the linear homotopy between a start system `G` and a target system `F`.
///
/// ```text
/// H(x, λ) = (1 − λ) G(x) + λ F(x),   x ∈ ℝⁿ, λ ∈ ℝ
/// ```
///
/// `H(·, 0) = G` and `H(·, 1) = F`, so following the solution curve of `H = 0`
/// from a known root of `G` at `λ = 0` to `λ = 1` gives a root of `F`. `H` is a
/// map `ℝⁿ⁺¹ → ℝⁿ` for [`ContinuationProblem`]. Complex systems can be used
/// after conversion with [`RealifiedProblem::square`].
///
/// A residual of `H` costs one residual of `G` and of `F`. A Jacobian–vector
/// product costs one product and one residual of each. The scratch length,
/// reported by the residual and Jacobian workspace queries, is `3n` plus the
/// largest scratch length of the two maps.
///
/// # References
///
/// A. J. Sommese and C. W. Wampler, *The Numerical Solution of Systems of
/// Polynomials Arising in Engineering and Science*, World Scientific, 2005.
pub struct LinearHomotopy<'a, R: RealScalar> {
    start: &'a dyn JacobianOperator<R>,
    target: &'a dyn JacobianOperator<R>,
    n: usize,
    scratch: usize,
}
impl<'a, R: RealScalar> LinearHomotopy<'a, R> {
    /// Creates the homotopy from the start system `start` and the target system `target`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if either map is empty or not square, or the two maps
    ///   have different dimensions.
    /// - `SizeOverflow` if the scratch length overflows `usize`.
    /// - Any error from the maps' workspace-length queries.
    pub fn new(
        start: &'a dyn JacobianOperator<R>,
        target: &'a dyn JacobianOperator<R>,
    ) -> Result<Self, SciOxideError> {
        let n = start.input_dimension();
        if n == 0
            || start.output_dimension() != n
            || target.input_dimension() != n
            || target.output_dimension() != n
        {
            return Err(invalid(
                if n == 0 || start.output_dimension() != n {
                    scioxide_core::argument!("start")
                } else {
                    scioxide_core::argument!("target")
                },
                InvalidInputReason::Inconsistent,
                "homotopy maps must be matching square systems",
            ));
        }
        let scratch = start
            .residual_workspace_len()?
            .max(start.jvp_workspace_len()?)
            .max(target.residual_workspace_len()?)
            .max(target.jvp_workspace_len()?);
        checked_size(n, 1, scioxide_core::quantity!("input_dimension"))?;
        checked_size(
            n.checked_mul(3).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("state_vectors"))
            })?,
            scratch,
            scioxide_core::quantity!("residual_workspace_len"),
        )?;
        Ok(Self {
            start,
            target,
            n,
            scratch,
        })
    }
    /// Evaluates the residual, or the Jacobian–vector product when `v` is given.
    fn evaluate(
        &self,
        x: &[R],
        v: Option<&[R]>,
        y: &mut [R],
        s: &mut [R],
        mut c: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        let n = self.n;
        let argument = if x.len() != n + 1 {
            Some(scioxide_core::argument!("x"))
        } else if y.len() != n {
            Some(scioxide_core::argument!("y"))
        } else if s.len() != self.residual_workspace_len()? {
            Some(scioxide_core::argument!("s"))
        } else if v.is_some_and(|v| v.len() != n + 1) {
            Some(scioxide_core::argument!("v"))
        } else {
            None
        };
        if let Some(argument) = argument {
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "homotopy shape mismatch",
            ));
        }
        finite(x, NonFiniteOrigin::Input)?;
        let (a, s) = s.split_at_mut(n);
        let (b, s) = s.split_at_mut(n);
        let (out, s) = s.split_at_mut(n);
        let lambda = x[n];
        if let Some(v) = v {
            finite(v, NonFiniteOrigin::Input)?;
            invoke_jvp(self.start, &x[..n], &v[..n], a, s, &mut c)?;
            invoke_jvp(self.target, &x[..n], &v[..n], b, s, &mut c)?;
            for i in 0..n {
                out[i] = affine((R::ONE - lambda) * a[i], lambda, b[i])?;
            }
            invoke_residual(self.start, &x[..n], a, s, &mut c)?;
            invoke_residual(self.target, &x[..n], b, s, &mut c)?;
            for i in 0..n {
                out[i] = affine(out[i], v[n], difference(b[i], a[i])?)?;
            }
        } else {
            invoke_residual(self.start, &x[..n], a, s, &mut c)?;
            invoke_residual(self.target, &x[..n], b, s, &mut c)?;
            for i in 0..n {
                out[i] = affine((R::ONE - lambda) * a[i], lambda, b[i])?;
            }
        }
        finite(out, NonFiniteOrigin::Intermediate)?;
        y.copy_from_slice(out);
        Ok(())
    }
}
/// Counts one nested provider call against `c`, when present.
fn nested_admit(
    c: &mut Option<&mut SolveContext<'_, '_>>,
    role: EvaluationRole,
) -> Result<(), SciOxideError> {
    if let Some(c) = c {
        crate::context::Session::new(c).admit(role)?;
    }
    Ok(())
}
/// Evaluates the residual and checks that it is finite.
///
/// With a context the call is counted and may stop on a limit; the stop is
/// returned as the context's stop error. Without a context the call is not
/// counted. Callback errors are returned unchanged.
pub(super) fn invoke_residual<R: RealScalar>(
    p: &dyn ResidualOperator<R>,
    x: &[R],
    y: &mut [R],
    s: &mut [R],
    c: &mut Option<&mut SolveContext<'_, '_>>,
) -> Result<(), SciOxideError> {
    let Some(c) = c else {
        return crate::residual::evaluate_unadmitted(p, x, y, s);
    };
    match crate::residual::evaluate_scoped_raw(p, EvaluationRole::Residual, x, y, s, c) {
        Ok(result) => {
            result?;
            finite(y, NonFiniteOrigin::Callback)
        }
        Err(KernelFault::Stopped(reason)) => Err(crate::context::stop_error(c, reason)),
        Err(KernelFault::Error(error)) => Err(error),
    }
}
/// Evaluates a counted Jacobian–vector product and checks that it is finite.
pub(super) fn invoke_jvp<R: RealScalar>(
    p: &dyn JacobianOperator<R>,
    x: &[R],
    v: &[R],
    y: &mut [R],
    s: &mut [R],
    c: &mut Option<&mut SolveContext<'_, '_>>,
) -> Result<(), SciOxideError> {
    nested_admit(c, EvaluationRole::LinearOperator)?;
    y.fill(R::NAN);
    let s = crate::residual::scratch_slice(s, p.jvp_workspace_len()?)?;
    match c {
        Some(c) => p.jvp_into_scoped(x, v, y, s, c)?,
        None => p.jvp_into(x, v, y, s)?,
    }
    finite(y, NonFiniteOrigin::Callback)
}
impl<R: RealScalar> ResidualOperator<R> for LinearHomotopy<'_, R> {
    fn input_dimension(&self) -> usize {
        self.n + 1
    }
    fn output_dimension(&self) -> usize {
        self.n
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(3 * self.n + self.scratch)
    }
    fn residual_into(&self, x: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(x, None, y, s, None)
    }
    fn residual_into_scoped(
        &self,
        x: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, None, y, s, Some(c))
    }
}
impl<R: RealScalar> JacobianOperator<R> for LinearHomotopy<'_, R> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.residual_workspace_len()
    }
    fn jvp_into(&self, x: &[R], v: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(v), y, s, None)
    }
    fn jvp_into_scoped(
        &self,
        x: &[R],
        v: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.evaluate(x, Some(v), y, s, Some(c))
    }
}
