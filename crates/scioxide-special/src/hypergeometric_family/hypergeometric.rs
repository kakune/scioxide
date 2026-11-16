//! Stored hypergeometric parameter sets and the plain hypergeometric entry points.
use crate::hypergeometric_family::hypergeometric_kernel::{self as kernel, real_parts};
use crate::hypergeometric_family::hypergeometric_value::Kind;
use crate::result::{Evaluation, Forms, Publish};
use crate::support::HYPERGEOMETRIC_ERRORS as ERRORS;
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::is_real;
use crate::support::widen_numeric as widen;
use crate::support::{C, ZERO, real};
use crate::util::scoped;
use crate::{HypergeometricOptions, SpecialFunctionResult};
use alloc::vec::Vec;
use scioxide_core::algorithm_component;
use scioxide_core::{
    Argument, DomainReason, NumericScalar, SciOxideError, SolveContext, SolveScope,
};

/// Evaluation of a hypergeometric function, converted to a result on demand.
type HypergeometricEvaluation<T> = crate::complex_support::ScaledEvaluation<T>;

/// Names the function that a [`HypergeometricFunction`] evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypergeometricFamily {
    /// The generalized hypergeometric function `ₚF_q(a; b; z)` of [`crate::hyp_pfq`]. The
    /// shapes `₀F₁`, `₁F₁`, `₂F₁` and `₂F₀` use the continuations of their named functions.
    Pfq,
    /// Tricomi's confluent hypergeometric function `U(a, b, z)` of [`crate::hyperu`].
    U,
}
/// Selects the first derivative that [`HypergeometricFunction::evaluate`] computes with the value.
///
/// Derivatives are analytic, not finite differences. Parameter indices are zero-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypergeometricDerivative {
    /// Computes the value only.
    None,
    /// Computes the derivative with respect to the argument `z`.
    Argument,
    /// Computes the derivative with respect to numerator parameter `aᵢ` (`a` for `U`).
    Numerator(usize),
    /// Computes the derivative with respect to denominator parameter `bⱼ` (`b` for `U`).
    Denominator(usize),
}

/// Stores the parameters of `ₚF_q(a; b; z)` or `U(a, b, z)` for repeated evaluation.
///
/// The value holds the validated `p` numerator and `q` denominator parameters, the
/// regularization flag and the options, in `O(p + q)` storage; it precomputes nothing else.
/// Build it once with [`Self::pfq`] or [`Self::u`] and evaluate it at many arguments `z`. It
/// offers what the plain functions do not: the regularized `ₚF_q`, a first derivative with
/// respect to `z` or to one parameter, and evaluation without allocation.
///
/// [`Self::evaluate`] allocates a [`HypergeometricWorkspace`] for each call;
/// [`Self::evaluate_scoped`] reuses one from [`Self::workspace`] and allocates nothing. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces) for the shared rules.
///
/// The domains, branch cuts and accuracy are those of [`crate::hyp_pfq`] and the named
/// functions it covers, and of [`crate::hyperu`]. Integer parameter gaps and continuation
/// shifts count toward the iteration budget of the stored
/// [`HypergeometricOptions`](crate::HypergeometricOptions); they must convert exactly to
/// `usize` and to `f64`, or evaluation returns `ScalarConversion`.
#[derive(Debug, Clone)]
pub struct HypergeometricFunction<T: NumericScalar> {
    family: HypergeometricFamily,
    numerator: Vec<C>,
    denominator: Vec<C>,
    regularized: bool,
    options: HypergeometricOptions,
    marker: core::marker::PhantomData<T>,
}
/// Holds the `O(p + q)` scratch of [`HypergeometricFunction::evaluate_scoped`].
///
/// Create it with [`HypergeometricFunction::workspace`]; it fits only functions with the same
/// numbers of numerator and denominator parameters, and evaluation does not resize it.
#[derive(Debug, Clone)]
pub struct HypergeometricWorkspace<T: NumericScalar> {
    numerator: Vec<Jet>,
    denominator: Vec<Jet>,
    marker: core::marker::PhantomData<T>,
}

fn parameters<T: NumericScalar>(values: &[T]) -> Result<Vec<C>, SciOxideError> {
    values.iter().try_fold(
        scioxide_core::backend::try_with_capacity(
            values.len(),
            scioxide_core::quantity!("hypergeometric_parameters"),
        )?,
        |mut result, &x| {
            result.push(widen(x, "hypergeometric parameter x must be finite")?);
            Ok(result)
        },
    )
}

use crate::support::invalid;

impl<T: NumericScalar> HypergeometricFunction<T> {
    /// Prepares `ₚF_q(a; b; z)` with numerator parameters `a` and denominator parameters `b`.
    ///
    /// [`crate::hyp_pfq`] defines the series, its domains and its accuracy; the shapes `₀F₁`,
    /// `₁F₁`, `₂F₁` and `₂F₀` have the domains and branch cuts of [`crate::hyp0f1`],
    /// [`crate::hyp1f1`], [`crate::hyp2f1`] and [`crate::hyp2f0`]. With `regularized = true` each
    /// Pochhammer symbol `(bⱼ)ₖ` in the denominator is replaced by `Γ(bⱼ + k)`, which divides the
    /// function by `Π Γ(bⱼ)` and makes it finite at nonpositive integer `bⱼ`. The argument domain
    /// is checked by [`Self::evaluate`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is not finite.
    /// - `DomainError` (`Pole`) with argument `denominator` if `regularized` is false and some
    ///   `bⱼ` is a nonpositive integer, unless a numerator parameter is a nonpositive integer
    ///   greater than `bⱼ`, so that the series ends before the zero denominator.
    /// - `AllocationFailure` if the parameter storage cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_special::{HypergeometricDerivative, HypergeometricFunction};
    ///
    /// // ₁F₁(1; 1; z) = eᶻ, whose derivative is also eᶻ
    /// let f = HypergeometricFunction::pfq(&[1.0_f64], &[1.0], false, Default::default())?;
    /// for z in [0.25, 0.5, 1.0] {
    ///     let result = f.evaluate(z, HypergeometricDerivative::Argument)?;
    ///     assert!((result.try_value()? - z.exp()).abs() < 1e-13);
    ///     assert!((result.try_derivative()? - z.exp()).abs() < 1e-12);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn pfq(
        numerator: &[T],
        denominator: &[T],
        regularized: bool,
        options: HypergeometricOptions,
    ) -> Result<Self, SciOxideError> {
        Self::pfq_checked(
            numerator,
            denominator,
            regularized,
            options,
            scioxide_core::argument!("denominator"),
        )
    }

    /// Stores `ₚF_q` parameters and reports a denominator pole with argument `poles`.
    fn pfq_checked(
        numerator: &[T],
        denominator: &[T],
        regularized: bool,
        options: HypergeometricOptions,
        poles: Argument,
    ) -> Result<Self, SciOxideError> {
        let function = Self {
            family: HypergeometricFamily::Pfq,
            numerator: parameters(numerator)?,
            denominator: parameters(denominator)?,
            regularized,
            options,
            marker: core::marker::PhantomData,
        };
        let mut workspace = function.workspace()?;
        function.fill(&mut workspace, HypergeometricDerivative::None)?;
        kernel::validate(
            &workspace.numerator,
            &workspace.denominator,
            regularized,
            poles,
        )?;
        Ok(function)
    }

    /// Prepares Tricomi's confluent hypergeometric function `U(a, b, z)`.
    ///
    /// The normalization, domains, branch cut and accuracy are those of [`crate::hyperu`]. The
    /// argument domain is checked by [`Self::evaluate`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` or `b` is not finite.
    /// - `AllocationFailure` if the parameter storage cannot be allocated.
    pub fn u(a: T, b: T, options: HypergeometricOptions) -> Result<Self, SciOxideError> {
        Ok(Self {
            family: HypergeometricFamily::U,
            numerator: parameters(&[a])?,
            denominator: parameters(&[b])?,
            regularized: false,
            options,
            marker: core::marker::PhantomData,
        })
    }
    /// Returns the function this value evaluates.
    pub const fn family(&self) -> HypergeometricFamily {
        self.family
    }
    /// Allocates a workspace that fits these parameter counts.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the scratch cannot be allocated.
    pub fn workspace(&self) -> Result<HypergeometricWorkspace<T>, SciOxideError> {
        Ok(HypergeometricWorkspace {
            numerator: scioxide_core::backend::try_filled(
                self.numerator.len(),
                Jet::default(),
                scioxide_core::quantity!("hypergeometric_numerator_scratch"),
            )?,
            denominator: scioxide_core::backend::try_filled(
                self.denominator.len(),
                Jet::default(),
                scioxide_core::quantity!("hypergeometric_denominator_scratch"),
            )?,
            marker: core::marker::PhantomData,
        })
    }

    fn fill(
        &self,
        scratch: &mut HypergeometricWorkspace<T>,
        derivative: HypergeometricDerivative,
    ) -> Result<(), SciOxideError> {
        if scratch.numerator.len() != self.numerator.len()
            || scratch.denominator.len() != self.denominator.len()
        {
            return Err(invalid(
                scioxide_core::argument!("scratch"),
                "workspace does not match the parameter counts",
            ));
        }
        match derivative {
            HypergeometricDerivative::Numerator(i) if i >= self.numerator.len() => {
                return Err(invalid(
                    scioxide_core::argument!("derivative"),
                    "numerator index out of range",
                ));
            }
            HypergeometricDerivative::Denominator(i) if i >= self.denominator.len() => {
                return Err(invalid(
                    scioxide_core::argument!("derivative"),
                    "denominator index out of range",
                ));
            }
            _ => {}
        }
        for (is_numerator, (source, dest)) in [
            (true, (&self.numerator, &mut scratch.numerator)),
            (false, (&self.denominator, &mut scratch.denominator)),
        ] {
            for (i, (&v, out)) in source.iter().zip(dest.iter_mut()).enumerate() {
                let selector = if is_numerator {
                    HypergeometricDerivative::Numerator(i)
                } else {
                    HypergeometricDerivative::Denominator(i)
                };
                *out = if derivative == selector {
                    Jet::variable(v)
                } else {
                    Jet::constant(v)
                };
            }
        }
        Ok(())
    }
    /// Evaluates the stored function at `z`, and optionally one first derivative.
    ///
    /// The definitions, branch cuts, domains and accuracy are those of the function the value
    /// was built for ([`Self::pfq`] or [`Self::u`]). An argument derivative holds the parameters
    /// fixed. A derivative can have a smaller domain than the value: for `₂F₁` at `z = 1` the
    /// derivative needs `c − a − b > 1`. The result is a detailed result
    /// (see [detailed results](crate#detailed-results)); when the iteration budget runs out, it
    /// carries an unsuccessful termination and whatever diagnostics are available. Each call
    /// allocates a workspace; [`Self::evaluate_scoped`] reuses one.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `z` is not finite.
    /// - `InvalidInput` (`OutOfRange`) if a derivative index is not a valid parameter index.
    /// - `DomainError` (`OutsideDomain`) if `z` is outside the domain of the stored function,
    ///   for example `|z| ≥ 1` for a general `ₚF_q` with `p = q + 1`.
    /// - `DomainError` (`Pole`) with argument `denominator` if a parameter derivative of a
    ///   denominator parameter at a nonpositive integer is requested.
    /// - `ConvergenceFailure` if the budget runs out before any estimate is available.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn evaluate(
        &self,
        z: T,
        derivative: HypergeometricDerivative,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        self.evaluate_for(z, derivative, &mut self.workspace()?, None)?
            .publish()
    }
    /// Evaluates like [`Self::evaluate`] inside a solve scope, reusing `scratch`.
    ///
    /// The stored iteration budget bounds the evaluation, its work is counted in the scope, and
    /// the scope's cancellation and deadline are checked. No allocation occurs. See
    /// [scoped evaluation](crate#scoped-evaluation) and
    /// [execution contexts](scioxide_core#execution-contexts).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate`] other than `AllocationFailure`.
    /// - `InvalidInput` (`OutOfRange`) with argument `scratch` if `scratch` does not fit the
    ///   parameter counts.
    /// - `ExecutionStopped` if the scope is cancelled or its deadline passes.
    pub fn evaluate_scoped<'scope, 'cap>(
        &self,
        z: T,
        derivative: HypergeometricDerivative,
        scratch: &mut HypergeometricWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<T>, SciOxideError> {
        scoped(scope, self.options.iteration_budget(), |cx| {
            self.evaluate_for(z, derivative, scratch, Some(cx))
        })?
        .publish()
    }
    /// Evaluate before projecting the value or forming the detailed result.
    fn evaluate_for(
        &self,
        z: T,
        derivative: HypergeometricDerivative,
        scratch: &mut HypergeometricWorkspace<T>,
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<HypergeometricEvaluation<T>, SciOxideError> {
        self.fill(scratch, derivative)?;
        let mut work = Work::with_context(self.options.iteration_budget().get(), context);
        evaluate(
            self.family,
            &scratch.numerator,
            &scratch.denominator,
            widen(z, "hypergeometric parameter z must be finite")?,
            self.regularized,
            derivative,
            &mut work,
        )
    }
}

fn evaluate<T: NumericScalar>(
    family: HypergeometricFamily,
    a: &[Jet],
    b: &[Jet],
    z: C,
    regularized: bool,
    derivative: HypergeometricDerivative,
    work: &mut Work<'_, '_, '_>,
) -> Result<HypergeometricEvaluation<T>, SciOxideError> {
    let result = compute::<T>(family, a, b, z, regularized, derivative, work);
    report(result, derivative, work)
}

fn compute<T: NumericScalar>(
    family: HypergeometricFamily,
    a: &[Jet],
    b: &[Jet],
    z: C,
    regularized: bool,
    derivative: HypergeometricDerivative,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    if is_real::<T>() {
        let terminating = a.iter().any(|x| crate::support::pole(x.v) && x.d == ZERO);
        if family == HypergeometricFamily::U && z.re < 0.0 && !terminating
            || family == HypergeometricFamily::Pfq
                && a.len() == 2
                && (b.is_empty() && z.re > 0.0 || b.len() == 1 && z.re > 1.0)
                && !terminating
        {
            return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
        }
    }
    work.check()?;
    let kind = match family {
        HypergeometricFamily::Pfq => Kind::pfq(a.len(), b.len()),
        HypergeometricFamily::U => Some(Kind::U),
    };
    if is_real::<T>()
        && let Some(kind) = kind
        && matches!(
            derivative,
            HypergeometricDerivative::None | HypergeometricDerivative::Argument
        )
    {
        if family == HypergeometricFamily::Pfq {
            kernel::validate(a, b, regularized, scioxide_core::argument!("denominator"))?;
        }
        let (numerator, denominator) = (real_parts(a), real_parts(b));
        if let Some(estimate) = crate::hypergeometric_family::hypergeometric_value::estimate(
            kind,
            &numerator[..a.len()],
            &denominator[..b.len()],
            z.re,
            regularized,
            derivative == HypergeometricDerivative::Argument,
            work,
        )? {
            return Ok(estimate);
        }
    }
    let z = if derivative == HypergeometricDerivative::Argument {
        Jet::variable(z)
    } else {
        Jet::constant(z)
    };
    match family {
        HypergeometricFamily::Pfq => kernel::pfq(a, b, z, regularized, work),
        HypergeometricFamily::U => kernel::hyperu(a[0], b[0], z, work),
    }
}

fn report<T: NumericScalar>(
    estimate: Result<Estimate, SciOxideError>,
    derivative: HypergeometricDerivative,
    work: &Work<'_, '_, '_>,
) -> Result<HypergeometricEvaluation<T>, SciOxideError> {
    crate::complex_support::report(
        estimate,
        derivative != HypergeometricDerivative::None,
        work,
        algorithm_component!("hypergeometric"),
    )
}

/// Computes the generalized hypergeometric function `ₚF_q(a; b; z)`.
///
/// ```text
/// ₚF_q(a; b; z) = Σₖ₌₀^∞ (a₁)ₖ ⋯ (aₚ)ₖ zᵏ / ((b₁)ₖ ⋯ (b_q)ₖ k!)
/// ```
///
/// `a` holds the `p` numerator parameters, `b` the `q` denominator parameters, and
/// `(x)ₖ = x(x + 1)⋯(x + k − 1)` is the Pochhammer symbol (DLMF 16.2.1). If some `aᵢ` is a
/// nonpositive integer, the series is a polynomial and every finite `z` is accepted. Otherwise
/// the series is summed for every finite `z` when `p ≤ q`, and for `|z| < 1` when `p = q + 1`.
/// The shapes `₀F₁`, `₁F₁`, `₂F₁` and `₂F₀` are continued beyond these domains as in
/// [`crate::hyp0f1`], [`crate::hyp1f1`], [`crate::hyp2f1`] and [`crate::hyp2f0`]. A
/// nonterminating series with `p > q + 1` diverges and is rejected, except `₂F₀`.
///
/// The series is summed by forward recurrence of its terms with compensated summation, and
/// stops when a ratio bound on the remaining terms falls below the rounding level. Each call
/// copies the parameters into `O(p + q)` storage. [`hyp_pfq_with_options`] also returns an
/// error estimate; [`crate::HypergeometricFunction`] stores the parameters for many arguments
/// and adds the regularized function and first derivatives.
///
/// # Accuracy
///
/// The shapes `₀F₁`, `₁F₁` and `₂F₁` with real arguments have the accuracy of their named
/// functions. Other shapes have no uniform ulp bound; their error estimates are checked against
/// independent high-precision references. See the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `NonFinite` if a parameter or `z` is not finite.
/// - `DomainError` (`Pole`) with argument `b` if some `bⱼ` is a nonpositive integer, unless a
///   numerator parameter is a nonpositive integer greater than `bⱼ`.
/// - `DomainError` (`OutsideDomain`) if `z` is outside the domain described above.
/// - `ConvergenceFailure` (`IterationLimit`) if the default iteration budget runs out.
/// - `AllocationFailure` if the parameter storage cannot be allocated.
/// - The errors of the named function for the shapes `₀F₁`, `₁F₁`, `₂F₁` and `₂F₀`.
///
/// # Examples
///
/// ```
/// use core::f64::consts::{LN_2, PI};
///
/// // ₃F₂(1, 1, 1; 2, 2; z) = Li₂(z)/z, and Li₂(1/2) = π²/12 − (ln 2)²/2
/// let f = scioxide_special::hyp_pfq(&[1.0_f64, 1.0, 1.0], &[2.0, 2.0], 0.5)?;
/// assert!((f - (PI * PI / 6.0 - LN_2 * LN_2)).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §16.2. <https://dlmf.nist.gov/16.2>
pub fn hyp_pfq<T: NumericScalar>(a: &[T], b: &[T], z: T) -> Result<T, SciOxideError> {
    let function = pfq_of(a, b, HypergeometricOptions::default())?;
    function
        .evaluate_for(
            z,
            HypergeometricDerivative::None,
            &mut function.workspace()?,
            None,
        )?
        .value()
}

/// Computes `ₚF_q(a; b; z)` with its error estimate and diagnostics.
///
/// The function, its domains and its value are those of [`hyp_pfq`]: the ordinary value of
/// the result equals the value of [`hyp_pfq`] with default options. `options` bounds the
/// iteration budget. The result carries no derivative; evaluate
/// [`crate::HypergeometricFunction`] with
/// [`HypergeometricDerivative::Argument`](crate::HypergeometricDerivative::Argument) for one.
/// See [detailed results](crate#detailed-results).
///
/// # Errors
///
/// The errors of [`hyp_pfq`]. When the budget runs out after an estimate exists, the result
/// carries an unsuccessful termination instead of `ConvergenceFailure`.
///
/// # Examples
///
/// ```
/// use core::f64::consts::{LN_2, PI};
///
/// // ₃F₂(1, 1, 1; 2, 2; 1/2) = 2 Li₂(1/2) = π²/6 − (ln 2)²
/// let a = [1.0_f64, 1.0, 1.0];
/// let result = scioxide_special::hyp_pfq_with_options(&a, &[2.0, 2.0], 0.5, Default::default())?;
/// let value = result.value().unwrap();
/// assert!((value.value() - (PI * PI / 6.0 - LN_2 * LN_2)).abs() < 1e-13);
/// assert!(value.absolute_error() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn hyp_pfq_with_options<T: NumericScalar>(
    a: &[T],
    b: &[T],
    z: T,
    options: HypergeometricOptions,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    pfq_of(a, b, options)?.evaluate(z, HypergeometricDerivative::None)
}

/// Computes `ₚF_q(a; b; z)` like [`hyp_pfq_with_options`] inside a solve scope.
///
/// The evaluation runs in one root or child epoch of `scope`: `options` bounds its local work,
/// ancestors receive nested counts, and the scope's cancellation and deadline are checked. See
/// [scoped evaluation](crate#scoped-evaluation).
///
/// # Errors
///
/// - The errors of [`hyp_pfq_with_options`].
/// - `ExecutionStopped` if the scope is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
///
/// // ₂F₁(1, 1; 2; 1/2) = 2 ln 2
/// let options = Default::default();
/// let context = ExecutionContext::sequential();
/// let result = scioxide_special::hyp_pfq_scoped(&[1.0_f64, 1.0], &[2.0], 0.5, options, context)?;
/// assert!((result.try_value()? - 2.0 * core::f64::consts::LN_2).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn hyp_pfq_scoped<'scope, 'cap, T: NumericScalar>(
    a: &[T],
    b: &[T],
    z: T,
    options: HypergeometricOptions,
    scope: impl Into<SolveScope<'scope, 'cap>>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    let function = pfq_of(a, b, options)?;
    function.evaluate_scoped(
        z,
        HypergeometricDerivative::None,
        &mut function.workspace()?,
        scope,
    )
}

/// Stores the parameters of a plain `ₚF_q` call, which names its denominator parameters `b`.
fn pfq_of<T: NumericScalar>(
    a: &[T],
    b: &[T],
    options: HypergeometricOptions,
) -> Result<HypergeometricFunction<T>, SciOxideError> {
    HypergeometricFunction::pfq_checked(a, b, false, options, scioxide_core::argument!("b"))
}

/// A value evaluation with the argument derivative of a separate evaluation.
///
/// Tracking the derivative changes the kernel's stopping decisions, so the value and its forms come
/// from the value-only evaluation the plain function runs, and only the derivative from the
/// differentiated evaluation.
#[derive(Clone, Copy)]
pub(crate) struct Differentiated<T> {
    value: crate::complex_support::ScaledForm,
    derivative: Option<DerivativeForm<T>>,
}

#[derive(Clone, Copy)]
enum DerivativeForm<T> {
    Value(Option<T>, crate::complex_support::ScaledForm),
    Direction(Option<T>, crate::complex_support::ScaledForm),
}

impl<T: NumericScalar> Publish<T> for Differentiated<T> {
    fn publish(self, value: Option<T>) -> Result<Forms<T>, SciOxideError> {
        let derivative = self
            .derivative
            .map(|form| match form {
                DerivativeForm::Value(value, form) => form.publish(value).map(|forms| forms.value),
                DerivativeForm::Direction(value, form) => {
                    form.publish(value).map(|forms| forms.derivative)
                }
            })
            .transpose()?
            .flatten();
        Ok(Forms {
            derivative,
            ..self.value.publish(value)?
        })
    }
}

/// Constant parameter jets of a named family.
fn constants<T: NumericScalar>(a: &[T], b: &[T]) -> Result<([Jet; 2], [Jet; 1]), SciOxideError> {
    let mut aa = [Jet::default(); 2];
    let mut bb = [Jet::default(); 1];
    for (out, &x) in aa.iter_mut().zip(a) {
        *out = Jet::constant(widen(x, "hypergeometric parameter x must be finite")?);
    }
    for (out, &x) in bb.iter_mut().zip(b) {
        *out = Jet::constant(widen(x, "hypergeometric parameter x must be finite")?);
    }
    Ok((aa, bb))
}

/// Constant parameter jets of a named family and the argument, with the denominator checked
/// for poles; `poles` names the denominator parameter.
fn checked<T: NumericScalar>(
    a: &[T],
    b: &[T],
    z: T,
    family: HypergeometricFamily,
    regularized: bool,
    poles: Argument,
) -> Result<([Jet; 2], [Jet; 1], C), SciOxideError> {
    let (aa, bb) = constants(a, b)?;
    let point = widen(z, "hypergeometric parameter z must be finite")?;
    if family == HypergeometricFamily::Pfq {
        kernel::validate(&aa[..a.len()], &bb[..b.len()], regularized, poles)?;
    }
    Ok((aa, bb, point))
}

/// The value-only evaluation a plain function returns; no result form is built.
fn plain<T: NumericScalar>(
    a: &[T],
    b: &[T],
    z: T,
    family: HypergeometricFamily,
    regularized: bool,
    poles: Argument,
) -> Result<T, SciOxideError> {
    let (aa, bb, point) = checked(a, b, z, family, regularized, poles)?;
    let mut work = Work::with_context(
        HypergeometricOptions::default().iteration_budget().get(),
        None,
    );
    let estimate = compute::<T>(
        family,
        &aa[..a.len()],
        &bb[..b.len()],
        point,
        regularized,
        HypergeometricDerivative::None,
        &mut work,
    );
    crate::complex_support::report_plain::<T>(
        estimate,
        &work,
        algorithm_component!("hypergeometric"),
    )?
    .value()
}

/// The detailed evaluation: the plain function's value-only evaluation with the argument derivative
/// computed through the contiguous relation without repeating its value.
#[allow(clippy::too_many_arguments)]
fn named<T: NumericScalar>(
    a: &[T],
    b: &[T],
    z: T,
    family: HypergeometricFamily,
    regularized: bool,
    poles: Argument,
    options: HypergeometricOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<T, Differentiated<T>>, SciOxideError> {
    let (aa, bb, point) = checked(a, b, z, family, regularized, poles)?;
    let (a, b) = (&aa[..a.len()], &bb[..b.len()]);
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let value = evaluate::<T>(
        family,
        a,
        b,
        point,
        regularized,
        HypergeometricDerivative::None,
        &mut work,
    )?;
    let defined = !(family == HypergeometricFamily::U && point == ZERO
        || family == HypergeometricFamily::Pfq
            && a.len() == 2
            && b.len() == 1
            && point == real(1.0));
    let derivative = if defined && value.termination().is_success() {
        let kind = match family {
            HypergeometricFamily::Pfq => Kind::pfq(a.len(), b.len()),
            HypergeometricFamily::U => Some(Kind::U),
        };
        use super::hypergeometric_value::DerivativeEstimate;
        let slope = if is_real::<T>()
            && let Some(kind) = kind
        {
            let (numerator, denominator) = (real_parts(a), real_parts(b));
            super::hypergeometric_value::derivative(
                kind,
                &numerator[..a.len()],
                &denominator[..b.len()],
                point.re,
                regularized,
                &mut work,
            )?
        } else {
            DerivativeEstimate::Unsupported
        };
        match slope {
            DerivativeEstimate::Unavailable => None,
            DerivativeEstimate::Estimated(slope) => {
                let derivative = report::<T>(Ok(slope), HypergeometricDerivative::None, &work)?;
                derivative
                    .termination()
                    .is_success()
                    .then(|| {
                        derivative
                            .form()
                            .copied()
                            .map(|form| DerivativeForm::Value(derivative.ordinary(), form))
                    })
                    .flatten()
            }
            DerivativeEstimate::Unsupported => {
                let derivative = evaluate::<T>(
                    family,
                    a,
                    b,
                    point,
                    regularized,
                    HypergeometricDerivative::Argument,
                    &mut work,
                )?;
                derivative
                    .termination()
                    .is_success()
                    .then(|| {
                        derivative
                            .form()
                            .copied()
                            .map(|form| DerivativeForm::Direction(derivative.ordinary(), form))
                    })
                    .flatten()
            }
        }
    } else {
        None
    };
    work.check()?;
    Ok(value
        .with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
        .map_form(|value| Differentiated { value, derivative }))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the confluent hypergeometric limit function `₀F₁(; b; z)`.

```text
₀F₁(; b; z) = Σₖ₌₀^∞ zᵏ / ((b)ₖ k!)
```

`(b)ₖ = b(b + 1)⋯(b + k − 1)` is the Pochhammer symbol (DLMF 16.2.1). The function is entire in `z`
and has no branch cut; `₀F₁(; b; 0) = 1`. `b` must not be a nonpositive integer.

Real arguments sum the series near zero. On the negative real axis outside that region they use the
connection with the Bessel function `J_(b−1)` (DLMF 10.16.9), with reflection for negative orders
and an exactly reduced phase. Complex arguments with `Re z < −8` or `|Im z| > 12` integrate the
differential equation; smaller arguments sum the series.

`b` may be real while `z` is complex.

# Examples

```
// ₀F₁(; 3/2; −z²/4) = sin(z)/z, here with z = 1
let f = scioxide_special::hyp0f1(1.5_f64, -0.25)?;
assert!((f - 1.0_f64.sin()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range, with absolute error bounds
next to zeros of the function. On the Bessel-connection path the error estimate is statistically
calibrated rather than derived. Complex arguments have no uniform ulp bound. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if `b` is a nonpositive integer.

# References

- NIST DLMF, §16.2 and §10.16. <https://dlmf.nist.gov/16>"###)]

    pub hyp0f1 [T: NumericScalar] (b: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [b] to T;
    options options: HypergeometricOptions;
    detailed hyp0f1_with_options -> SpecialFunctionResult<T>;
    value {
        plain(&[], &[b], z, HypergeometricFamily::Pfq, false, scioxide_core::argument!("b"))
    }
    report {
        named(
            &[],
            &[b],
            z,
            HypergeometricFamily::Pfq,
            false,
            scioxide_core::argument!("b"),
            options,
            None,
        )?
        .publish()
    }
    scoped pub hyp0f1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            named(
                &[],
                &[b],
                z,
                HypergeometricFamily::Pfq,
                false,
                scioxide_core::argument!("b"),
                options,
                Some(context),
            )
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Kummer's confluent hypergeometric function `M(a, b, z) = ₁F₁(a; b; z)`.

```text
₁F₁(a; b; z) = Σₖ₌₀^∞ (a)ₖ zᵏ / ((b)ₖ k!)
```

`(a)ₖ` is the Pochhammer symbol (DLMF 13.2.2). The function is entire in `z` and has no branch cut;
`₁F₁(a; b; 0) = 1`. If `a` is a nonpositive integer, the series is a polynomial of degree `−a`. `b`
must not be a nonpositive integer, unless `a` is a nonpositive integer with `a > b`, so that the
polynomial ends before the zero denominator.

Real arguments sum the series in two-term (double-double) arithmetic. Below `z = −1` they use
Kummer's transformation `₁F₁(a; b; z) = eᶻ ₁F₁(b − a; b; −z)` (DLMF 13.2.39), and for large `z` the
asymptotic expansion. When cancellation in the series exceeds what two-term arithmetic resolves, the
series is summed again in 4-, 8- and then 16-word arithmetic within the same iteration budget.

`a` and `b` may be real while `z` is complex.

# Examples

```
// ₁F₁(a; a; z) = eᶻ
let f = scioxide_special::hyp1f1(1.0_f64, 1.0, 0.5)?;
assert!((f - 0.5_f64.exp()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range, with absolute error bounds
next to zeros of the function. For `f64`, real arguments whose condition number is at most 1000 have
a relative error below 1e-12. Complex arguments have no uniform ulp bound; they are tested at all
phases up to `|z| = 10⁴`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if `b` is a nonpositive integer, unless `a` is a nonpositive integer with
  `a > b`.
- `NumericalInstability` (`LossOfSignificance`) if cancellation in a real series remains unresolved
  in 16-word arithmetic.

# References

- NIST DLMF, §13.2. <https://dlmf.nist.gov/13>"###)]

    pub hyp1f1 [T: NumericScalar] (a: impl crate::Parameter<T>, b: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [a, b] to T;
    options options: HypergeometricOptions;
    detailed hyp1f1_with_options -> SpecialFunctionResult<T>;
    value {
        plain(&[a], &[b], z, HypergeometricFamily::Pfq, false, scioxide_core::argument!("b"))
    }
    report {
        named(
            &[a],
            &[b],
            z,
            HypergeometricFamily::Pfq,
            false,
            scioxide_core::argument!("b"),
            options,
            None,
        )?
        .publish()
    }
    scoped pub hyp1f1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            named(
                &[a],
                &[b],
                z,
                HypergeometricFamily::Pfq,
                false,
                scioxide_core::argument!("b"),
                options,
                Some(context),
            )
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Gauss hypergeometric function `₂F₁(a, b; c; z)`.

```text
₂F₁(a, b; c; z) = Σₖ₌₀^∞ (a)ₖ (b)ₖ zᵏ / ((c)ₖ k!),   |z| < 1
```

`(a)ₖ` is the Pochhammer symbol (DLMF 15.2.1). For complex `z` the function is continued to the
plane cut along `[1, ∞)`; on the cut the sign of a zero imaginary part selects the side. For a real
type a nonterminating series is defined for `z < 1`, and at `z = 1` when `c − a − b > 0` (Gauss's
sum, `Γ(c)Γ(c − a − b) / (Γ(c − a)Γ(c − b))`). If `a` or `b` is a nonpositive integer, the series is
a polynomial and every finite `z` is accepted. `c` must not be a nonpositive integer, unless the
polynomial ends before the zero denominator; use [`crate::hyp2f1_regularized`] for that case.

Real arguments sum the series in two-term (double-double) arithmetic near zero, use Pfaff's
transformation `(1 − z)^(−a) ₂F₁(a, c − b; c; z/(z − 1))` on `[−9, −1/2)` (DLMF 15.8.1), and use
compensated connection formulas next to `z = 1` and below `z = −9`. The reciprocal connection below
`z = −9` takes the logarithmic limit for integer `b − a` (DLMF 15.8.8), and its cost does not grow
with `|z|`. Complex arguments sum the series at a point inside the disk of convergence and integrate
the hypergeometric differential equation from there to `z` in two-term arithmetic.

`a`, `b` and `c` may be real while `z` is complex.

# Examples

```
// ₂F₁(1, 1; 2; z) = −ln(1 − z)/z, so ₂F₁(1, 1; 2; 1/2) = 2 ln 2
let f = scioxide_special::hyp2f1(1.0_f64, 1.0, 2.0, 0.5)?;
assert!((f - 2.0 * core::f64::consts::LN_2).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range, with absolute error bounds
next to zeros of the function. Complex arguments have no uniform ulp bound; they are tested up to a
distance of `10⁻¹⁵` from `z = 1`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if `c` is a nonpositive integer, unless `a` or `b` is a nonpositive integer
  greater than `c`.
- `DomainError` (`OutsideDomain`) at `z = 1` if `c − a − b ≤ 0`.
- `DomainError` (`OutsideDomain`) for a real type if `z > 1` and the series does not terminate.

# References

- NIST DLMF, §15.2, §15.4 and §15.8. <https://dlmf.nist.gov/15>"###)]

    pub hyp2f1 [T: NumericScalar] (a: impl crate::Parameter<T>, b: impl crate::Parameter<T>, c: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [a, b, c] to T;
    options options: HypergeometricOptions;
    detailed hyp2f1_with_options -> SpecialFunctionResult<T>;
    value {
        plain(&[a, b], &[c], z, HypergeometricFamily::Pfq, false, scioxide_core::argument!("c"))
    }
    report {
        named(
            &[a, b],
            &[c],
            z,
            HypergeometricFamily::Pfq,
            false,
            scioxide_core::argument!("c"),
            options,
            None,
        )?
        .publish()
    }
    scoped pub hyp2f1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            named(
                &[a, b],
                &[c],
                z,
                HypergeometricFamily::Pfq,
                false,
                scioxide_core::argument!("c"),
                options,
                Some(context),
            )
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the hypergeometric function `₂F₀(a, b; ; z)` through Tricomi's function `U`.

```text
₂F₀(a, b; ; z) ~ Σₖ₌₀^∞ (a)ₖ (b)ₖ zᵏ / k!
₂F₀(a, b; ; z) = (−1/z)^a U(a, 1 + a − b, −1/z)
```

`(a)ₖ` is the Pochhammer symbol. Unless `a` or `b` is a nonpositive integer, the series diverges for
every `z ≠ 0` and is only asymptotic; the second line defines the function, with the principal
power. If `a` or `b` is a nonpositive integer, the series is a polynomial and every finite `z` is
accepted. `₂F₀(a, b; ; 0) = 1`. For a real type a nonterminating value requires `z < 0`. Negative
subnormal `z` sums the leading terms of the series.

`a` and `b` may be real while `z` is complex.

# Examples

```
// The series terminates: ₂F₀(−1, b; ; z) = 1 − bz
let f = scioxide_special::hyp2f0(-1.0_f64, 2.0, 0.25)?;
assert!((f - 0.5).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound. For real `z < 0` the error estimate accounts for the rounding of
`−1/z` and of `b` through first-order sensitivities. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for a real type if `z > 0` and the series does not terminate.
- `NumericalInstability` (`LossOfSignificance`) if `z` is a negative subnormal number and the
  parameters are too large for the leading terms of the series to give the value.

# References

- NIST DLMF, §13.6 and §16.2. <https://dlmf.nist.gov/13>"###)]

    pub hyp2f0 [T: NumericScalar] (a: impl crate::Parameter<T>, b: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [a, b] to T;
    options options: HypergeometricOptions;
    detailed hyp2f0_with_options -> SpecialFunctionResult<T>;
    value {
        plain(&[a, b], &[], z, HypergeometricFamily::Pfq, false, scioxide_core::argument!("b"))
    }
    report {
        named(
            &[a, b],
            &[],
            z,
            HypergeometricFamily::Pfq,
            false,
            scioxide_core::argument!("b"),
            options,
            None,
        )?
        .publish()
    }
    scoped pub hyp2f0_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            named(
                &[a, b],
                &[],
                z,
                HypergeometricFamily::Pfq,
                false,
                scioxide_core::argument!("b"),
                options,
                Some(context),
            )
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Tricomi's confluent hypergeometric function `U(a, b, z)`.

```text
U(a, b, z) = (1/Γ(a)) ∫₀^∞ e^(−zt) t^(a−1) (1 + t)^(b−a−1) dt,   Re a > 0, Re z > 0
```

(DLMF 13.4.4), continued analytically in `z` to the plane cut along `(−∞, 0]`, with
`U(a, b, z) ~ z^(−a)` for large `z`. If `a` is a nonpositive integer, `U` is a polynomial in `z`.
For a real type a nonterminating value requires `z ≥ 0`. At `z = 0`,
`U(a, b, 0) = Γ(1 − b) / Γ(a − b + 1)` for `Re b < 1`; for `Re b ≥ 1` the function is singular
there.

For real `z > 0`, large `z` sums the asymptotic expansion (DLMF 13.7.3). Otherwise Kummer's
transformation `U(a, b, z) = z^(1−b) U(1 + a − b, 2 − b, z)` (DLMF 13.2.40) makes `b ≥ 1`, backward
recurrence in `a` (DLMF 13.3.7) gives the minimal solution, and the Wronskian of `M` and `U` (DLMF
13.2.34) normalizes it. Below `z = 2`, Taylor steps of the differential equation continue `U` toward
the origin.

`a` and `b` may be real while `z` is complex.

# Examples

```
// U(a, a + 1, z) = z^(−a)
let u = scioxide_special::hyperu(1.0_f64, 2.0, 2.0)?;
assert!((u - 0.5).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range, with absolute error bounds
next to zeros of the function. Complex arguments have no uniform ulp bound. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) for a real type if `z < 0` and `a` is not a nonpositive integer.
- `DomainError` (`OutsideDomain`) at `z = 0` if `Re b ≥ 1`.

# References

- NIST DLMF, §13.2–§13.7. <https://dlmf.nist.gov/13>"###)]

    pub hyperu [T: NumericScalar] (a: impl crate::Parameter<T>, b: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [a, b] to T;
    options options: HypergeometricOptions;
    detailed hyperu_with_options -> SpecialFunctionResult<T>;
    value {
        plain(&[a], &[b], z, HypergeometricFamily::U, false, scioxide_core::argument!("b"))
    }
    report {
        named(&[a], &[b], z, HypergeometricFamily::U, false, scioxide_core::argument!("b"), options, None)?.publish()
    }
    scoped pub hyperu_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            named(
                &[a],
                &[b],
                z,
                HypergeometricFamily::U,
                false,
                scioxide_core::argument!("b"),
                options,
                Some(context),
            )
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the regularized Gauss hypergeometric function `₂F₁(a, b; c; z) / Γ(c)`.

```text
₂F₁(a, b; c; z) / Γ(c) = Σₖ₌₀^∞ (a)ₖ (b)ₖ zᵏ / (Γ(c + k) k!),   |z| < 1
```

`(a)ₖ` is the Pochhammer symbol (DLMF 15.2.2). The regularized function is entire in `c`: at
`c = 0, −1, −2, …` the coefficients use `1/Γ(c + k)`, which is zero for the leading terms, instead
of dividing two singular values. For complex `z` the function is continued to the plane cut along
`[1, ∞)`; on the cut the sign of a zero imaginary part selects the side. For a real type a
nonterminating series is defined for `z < 1`, and at `z = 1` when `c − a − b > 0` (Gauss's sum,
`Γ(c)Γ(c − a − b) / (Γ(c − a)Γ(c − b))`). If `a` or `b` is a nonpositive integer, the series is a
polynomial and every finite `z` is accepted.

Real arguments sum the series in two-term (double-double) arithmetic near zero, use Pfaff's
transformation `(1 − z)^(−a) ₂F₁(a, c − b; c; z/(z − 1))` on `[−9, −1/2)` (DLMF 15.8.1), and use
compensated connection formulas next to `z = 1` and below `z = −9`. The reciprocal connection below
`z = −9` takes the logarithmic limit for integer `b − a` (DLMF 15.8.8), and its cost does not grow
with `|z|`. Complex arguments sum the series at a point inside the disk of convergence and integrate
the hypergeometric differential equation from there to `z` in two-term arithmetic.

`a`, `b` and `c` may be real while `z` is complex.

# Examples

```
// At c = 0: ₂F₁(a, b; c; z)/Γ(c) = abz ₂F₁(a + 1, b + 1; 2; z), here z/(1 − z)²
let f = scioxide_special::hyp2f1_regularized(1.0_f64, 1.0, 0.0, 0.5)?;
assert!((f - 2.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; the error estimate is checked against independent high-precision
references. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) at `z = 1` if `c − a − b ≤ 0`.
- `DomainError` (`OutsideDomain`) for a real type if `z > 1` and the series does not terminate.

# References

- NIST DLMF, §15.2, §15.4 and §15.8. <https://dlmf.nist.gov/15>"###)]

    pub hyp2f1_regularized [T: NumericScalar] (a: impl crate::Parameter<T>, b: impl crate::Parameter<T>, c: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [a, b, c] to T;
    options options: HypergeometricOptions;
    detailed hyp2f1_regularized_with_options -> SpecialFunctionResult<T>;
    value {
        plain(&[a, b], &[c], z, HypergeometricFamily::Pfq, true, scioxide_core::argument!("c"))
    }
    report {
        named(
            &[a, b],
            &[c],
            z,
            HypergeometricFamily::Pfq,
            true,
            scioxide_core::argument!("c"),
            options,
            None,
        )?
        .publish()
    }
    scoped pub hyp2f1_regularized_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            named(
                &[a, b],
                &[c],
                z,
                HypergeometricFamily::Pfq,
                true,
                scioxide_core::argument!("c"),
                options,
                Some(context),
            )
        })?
        .publish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "hypergeometric_tests.rs"]
mod tests;
