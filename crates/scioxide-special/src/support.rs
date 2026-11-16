//! Private evaluation support shared by special-function families.
//!
//! Work accounting uses one exhaustion classification, one
//! counter and one scoped-provider path. Shared wide arithmetic and scalar
//! gamma primitives keep family dependencies directed.

use core::ops::{Add, Div, Mul, Neg, Sub};
use core::ops::{Deref, DerefMut};
use scioxide_core::{
    AlgorithmComponent, Argument, Complex, ConvergenceDiagnostics, DomainReason, NonFiniteOrigin,
    NumericScalar, NumericalIssue, RealScalar, ScaledValue, SciOxideError, SolveContext,
    algorithm_component,
};

pub(crate) mod constants;
mod gamma_coefficients;
pub(crate) mod gamma_core;
pub(crate) mod gamma_real;
pub(crate) mod gamma_series;
pub(crate) mod hurwitz;
pub(crate) mod polynomial;
mod quadrature;
pub(crate) mod reciprocal_gamma;
mod reciprocal_gamma_short;
pub(crate) mod wide;
pub(crate) use quadrature::GL32;

pub(crate) type C = Complex<f64>;
pub(crate) const ZERO: C = C::new(0.0, 0.0);
pub(crate) const ONE: C = C::new(1.0, 0.0);
pub(crate) type Z = ScaledValue<C>;

#[inline]
pub(crate) const fn real(value: f64) -> C {
    C::new(value, 0.0)
}

/// Whether a complex argument is a nonpositive integral pole.
pub(crate) fn pole(value: C) -> bool {
    value.im == 0.0 && value.re <= 0.0 && value.re == value.re.floor()
}

/// Evaluate in `f64` and project once when the destination is `f32`.
#[inline]
pub(crate) fn delegate_f32<S: RealScalar, W, R>(
    wide: impl FnOnce() -> Result<W, SciOxideError>,
    project: impl FnOnce(W) -> Result<R, SciOxideError>,
) -> Option<Result<R, SciOxideError>> {
    (S::MANTISSA_DIGITS <= f32::MANTISSA_DIGITS).then(|| wide().and_then(project))
}

/// Family wording for common domain and loss-of-significance failures.
#[derive(Clone, Copy)]
pub(crate) struct ErrorContext {
    domain_detail: &'static str,
    instability_detail: &'static str,
}

impl ErrorContext {
    const fn new(domain_detail: &'static str, instability_detail: &'static str) -> Self {
        Self {
            domain_detail,
            instability_detail,
        }
    }

    pub(crate) fn domain(self, argument: Argument, reason: DomainReason) -> SciOxideError {
        SciOxideError::domain_error(argument, reason, self.domain_detail)
    }

    pub(crate) fn outside(self, argument: Argument) -> SciOxideError {
        self.domain(argument, DomainReason::OutsideDomain)
    }

    pub(crate) fn unstable(self) -> SciOxideError {
        instability(self.instability_detail)
    }
}

pub(crate) fn instability(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::LossOfSignificance, detail)
}

/// Whether a numeric scalar has no imaginary component in its representation.
pub(crate) fn is_real<T: NumericScalar>() -> bool {
    T::IS_REAL
}

/// Widen a finite scalar to the `f64` complex kernel representation.
pub(crate) fn widen_numeric<T: NumericScalar>(
    value: T,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    require_finite_param(value, detail)?;
    Ok(C::new(
        value.real_component().try_to_f64_exact()?,
        value.imaginary_component().try_to_f64_exact()?,
    ))
}

/// Project the `f64` kernel representation once to the requested scalar.
pub(crate) fn narrow<T: NumericScalar>(z: C) -> Result<T, SciOxideError> {
    Ok(T::from_components(
        T::Real::try_from_f64_lossy(z.re)?,
        T::Real::try_from_f64_lossy(z.im)?,
    ))
}

pub(crate) const GAMMA_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented gamma or beta domain",
    "gamma or beta continuation lost representable accuracy",
);
pub(crate) const ELLIPTIC_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented elliptic domain or at a pole",
    "elliptic reduction cannot establish the requested precision",
);
pub(crate) const HYPERGEOMETRIC_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented hypergeometric domain",
    "no reliable estimate in the selected precision and work allowance",
);
pub(crate) const TRANSCENDENTAL_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented domain or at a singularity",
    "continuation could not establish a reliable finite estimate",
);
pub(crate) const ELLIPSOIDAL_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented triaxial harmonic domain",
    "the harmonic coefficients or evaluation are ill-conditioned",
);
pub(crate) const COMBINATORIAL_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented combinatorial domain",
    "combinatorial evaluation lost representable accuracy",
);
pub(crate) const PROBABILITY_ERRORS: ErrorContext = ErrorContext::new(
    "argument outside the documented probability-function domain",
    "probability evaluation lost representable accuracy",
);
pub(crate) const INTEGRAL_ERRORS: ErrorContext = ErrorContext::new(
    "argument outside the documented real integral domain",
    "special-integral evaluation lost representable accuracy",
);
pub(crate) const COULOMB_ERRORS: ErrorContext = ErrorContext::new(
    "outside the documented Coulomb domain",
    "Coulomb continuation lost representable accuracy",
);

#[inline]
pub(crate) fn norm(z: C) -> f64 {
    if z.im == 0.0 { z.re.abs() } else { z.norm() }
}

/// Cancellation-resistant `exp(z)-1`, including tiny imaginary components.
pub(crate) fn complex_expm1(z: C) -> C {
    if z.re > 700.0 {
        // Above this threshold subtraction of one is below scalar rounding;
        // Complex::exp also rescales components before range projection.
        return z.exp() - ONE;
    }
    let half_sine = (0.5 * z.im).sin();
    C::new(
        z.re.exp_m1() * z.im.cos() - 2.0 * half_sine * half_sine,
        z.re.exp() * z.im.sin(),
    )
}

/// An out-of-range discrete or bounded parameter, with static domain detail.
pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::OutOfRange,
        detail,
    )
}

/// A value and one analytic directional derivative; never finite differences.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Jet {
    pub v: C,
    pub d: C,
}
impl Jet {
    pub(crate) fn constant(v: C) -> Self {
        Self { v, d: ZERO }
    }
    pub(crate) fn variable(v: C) -> Self {
        Self { v, d: ONE }
    }
    pub(crate) fn exp(self) -> Self {
        let v = if self.v.im == 0.0 {
            real(self.v.re.exp())
        } else {
            self.v.exp()
        };
        if self.d == ZERO {
            Self::constant(v)
        } else {
            Self { v, d: self.d * v }
        }
    }
    pub(crate) fn ln(self) -> Self {
        let v = if self.v.im == 0.0 && self.v.re > 0.0 {
            real(self.v.re.ln())
        } else {
            self.v.ln()
        };
        if self.d == ZERO {
            Self::constant(v)
        } else {
            Self {
                v,
                d: self.d / self.v,
            }
        }
    }
    #[inline]
    pub(crate) fn norm(self) -> f64 {
        norm(self.v).max(norm(self.d))
    }
}
impl From<f64> for Jet {
    fn from(x: f64) -> Self {
        Self::constant(real(x))
    }
}
impl Add for Jet {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
}
impl Sub for Jet {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
}
impl Neg for Jet {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}
impl Mul for Jet {
    type Output = Self;
    #[inline]
    fn mul(self, b: Self) -> Self {
        if self.v.im == 0.0 && b.v.im == 0.0 && self.d.im == 0.0 && b.d.im == 0.0 {
            return Self {
                v: real(self.v.re * b.v.re),
                d: if self.d.re == 0.0 && b.d.re == 0.0 {
                    ZERO
                } else {
                    real(self.d.re * b.v.re + self.v.re * b.d.re)
                },
            };
        }
        let v = self.v * b.v;
        if self.d == ZERO && b.d == ZERO {
            Self::constant(v)
        } else {
            Self {
                v,
                d: self.d * b.v + self.v * b.d,
            }
        }
    }
}
impl Div for Jet {
    type Output = Self;
    #[inline]
    fn div(self, b: Self) -> Self {
        if self.v.im == 0.0 && b.v.im == 0.0 && self.d.im == 0.0 && b.d.im == 0.0 {
            let v = self.v.re / b.v.re;
            return Self {
                v: real(v),
                d: if self.d.re == 0.0 && b.d.re == 0.0 {
                    ZERO
                } else {
                    real((self.d.re - v * b.d.re) / b.v.re)
                },
            };
        }
        let v = self.v / b.v;
        if self.d == ZERO && b.d == ZERO {
            Self::constant(v)
        } else {
            Self {
                v,
                d: (self.d - v * b.d) / b.v,
            }
        }
    }
}

/// Kernel value paired with the error metadata required by its representation.
#[derive(Clone, Copy)]
pub(crate) struct Estimate<M> {
    data: M,
    reliability: scioxide_core::EstimateReliability,
}

impl<M> From<M> for Estimate<M> {
    fn from(data: M) -> Self {
        Self {
            data,
            reliability: scioxide_core::EstimateReliability::ModelBased,
        }
    }
}

impl<M> Estimate<M> {
    pub(crate) fn with_reliability(
        mut self,
        reliability: scioxide_core::EstimateReliability,
    ) -> Self {
        self.reliability = reliability;
        self
    }

    pub(crate) fn reliability(&self) -> scioxide_core::EstimateReliability {
        self.reliability
    }
}

impl<M> Deref for Estimate<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.data
    }
}

impl<M> DerefMut for Estimate<M> {
    fn deref_mut(&mut self) -> &mut M {
        &mut self.data
    }
}

#[derive(Clone, Copy)]
pub(crate) struct EllipticData {
    pub(crate) value: Z,
    pub(crate) relative: f64,
    pub(crate) absolute: f64,
    pub(crate) exact: bool,
}

pub(crate) type EllipticEstimate = Estimate<EllipticData>;

impl Estimate<EllipticData> {
    pub(crate) fn new(value: Z, relative: f64) -> Self {
        Self::from(EllipticData {
            value,
            relative,
            absolute: 0.0,
            exact: false,
        })
    }

    pub(crate) fn exact(value: C) -> Result<Self, SciOxideError> {
        Ok(Self::from(EllipticData {
            value: Z::from_value(value)?,
            relative: 0.0,
            absolute: 0.0,
            exact: true,
        }))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct JetData {
    pub(crate) derivative_error: Option<f64>,
    pub(crate) y: Jet,
    pub(crate) scale: Jet,
    pub(crate) error: f64,
    pub(crate) method: AlgorithmComponent,
}

pub(crate) type JetEstimate = Estimate<JetData>;

impl Estimate<JetData> {
    pub(crate) fn new(y: Jet, error: f64, method: AlgorithmComponent) -> Self {
        Self::from(JetData {
            derivative_error: None,
            y,
            scale: 0.0.into(),
            error,
            method,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TranscendentalData<V> {
    pub(crate) value: V,
    pub(crate) error: f64,
    pub(crate) derivative: Option<V>,
    pub(crate) derivative_error: f64,
    pub(crate) logarithm: Option<V>,
    /// Natural-log scale multiplying value and its absolute error.
    pub(crate) scale: f64,
    pub(crate) method: AlgorithmComponent,
}

pub(crate) type TranscendentalEstimate<V = C> = Estimate<TranscendentalData<V>>;

impl<V> Estimate<TranscendentalData<V>> {
    pub(crate) fn new(value: V, error: f64, method: AlgorithmComponent) -> Self {
        Self::from(TranscendentalData {
            value,
            error,
            derivative: None,
            derivative_error: 0.0,
            logarithm: None,
            scale: 0.0,
            method,
        })
    }
}

/// One bounded allowance, optionally charged to a parent solve context.
pub(crate) struct Work<'a, 's, 'c> {
    pub(crate) used: usize,
    pub(crate) remaining: usize,
    pub(crate) complete: bool,
    pub(crate) target: f64,
    pub(crate) family: AlgorithmComponent,
    context: Option<&'a mut SolveContext<'s, 'c>>,
    failure: Option<SciOxideError>,
    batched: bool,
    recorded: usize,
}

impl<'a, 's, 'c> Work<'a, 's, 'c> {
    #[inline]
    fn make(limit: usize, context: Option<&'a mut SolveContext<'s, 'c>>) -> Self {
        Self {
            used: 0,
            remaining: limit,
            complete: true,
            target: 0.0,
            family: algorithm_component!("special"),
            context,
            failure: None,
            batched: false,
            recorded: 0,
        }
    }

    /// Allowance for kernels that return a partial result on exhaustion.
    #[inline]
    pub(crate) fn new(limit: usize) -> Self {
        Self::make(limit, None)
    }

    /// Scoped allowance with deferred provider failures.
    #[inline]
    pub(crate) fn with_context(
        limit: usize,
        context: Option<&'a mut SolveContext<'s, 'c>>,
    ) -> Self {
        Self::make(limit, context)
    }

    /// Real leaf kernels batch at most sixteen already bounded work units.
    /// Timed scopes keep per-unit clock checks. The caller must capture the
    /// kernel result and call `check` before propagating it, including errors.
    pub(crate) fn with_batched_context(
        limit: usize,
        context: Option<&'a mut SolveContext<'s, 'c>>,
    ) -> Result<Self, SciOxideError> {
        let mut work = Self::make(limit, context);
        if let Some(context) = work.context.as_deref_mut() {
            context.execution().check()?;
            if context.execution().clock().is_none() {
                work.batched = true;
                if let Some(remaining) = context
                    .budget_report(scioxide_core::BudgetKind::Iteration)
                    .remaining()
                {
                    work.remaining = work.remaining.min(remaining);
                }
            }
        }
        Ok(work)
    }

    /// Elliptic allowance with its precision target and algorithm identity.
    #[inline]
    pub(crate) fn for_family(
        limit: usize,
        target: f64,
        context: Option<&'a mut SolveContext<'s, 'c>>,
        family: AlgorithmComponent,
    ) -> Self {
        let mut work = Self::make(limit, context);
        work.target = target;
        work.family = family;
        work
    }

    /// Default elliptic-family allowance.
    #[inline]
    pub(crate) fn elliptic(
        limit: usize,
        target: f64,
        context: Option<&'a mut SolveContext<'s, 'c>>,
    ) -> Self {
        Self::for_family(limit, target, context, algorithm_component!("elliptic"))
    }

    fn mark_exhausted(&mut self) {
        self.complete = false;
    }

    #[inline]
    pub(crate) const fn exhausted(&self) -> bool {
        !self.complete
    }

    /// Check cancellation and forward a deferred provider failure once.
    #[inline]
    pub(crate) fn check(&mut self) -> Result<(), SciOxideError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if let Some(context) = self.context.as_deref_mut() {
            if self.batched {
                if !context
                    .record_iterations(self.used - self.recorded)?
                    .is_granted()
                {
                    self.complete = false;
                    return Err(crate::result::exhausted_work(self.used));
                }
                self.recorded = self.used;
            }
            context.execution().check()?;
        }
        Ok(())
    }

    /// Counts one step, distinguishing exhaustion from execution failure.
    #[inline]
    pub(crate) fn poll_step(&mut self) -> Result<bool, SciOxideError> {
        if self.step() {
            Ok(true)
        } else {
            self.check()?;
            Ok(false)
        }
    }

    /// Error carrying the work counted so far.
    pub(crate) fn limit_error(&self) -> SciOxideError {
        crate::result::exhausted_work(self.used)
    }

    /// Charge one unit and report exhaustion immediately.
    #[inline]
    pub(crate) fn try_step(&mut self) -> Result<(), SciOxideError> {
        if self.batched {
            if self.step_batched() {
                return Ok(());
            }
            self.check()?;
            return Err(self.limit_error());
        }
        if self.context.is_none() && self.remaining != 0 {
            self.remaining -= 1;
            self.used += 1;
            return Ok(());
        }
        self.try_admit()
    }

    #[cold]
    #[inline(never)]
    fn try_admit(&mut self) -> Result<(), SciOxideError> {
        self.check()?;
        if self.remaining == 0
            || if let Some(context) = self.context.as_deref_mut() {
                !context.record_iteration()?.is_granted()
            } else {
                false
            }
        {
            self.mark_exhausted();
            return Err(self.limit_error());
        }
        self.remaining -= 1;
        self.used += 1;
        Ok(())
    }

    /// Charge one unit and return false when the allowance is exhausted.
    #[inline]
    pub(crate) fn step(&mut self) -> bool {
        if self.batched {
            return self.step_batched();
        }
        if self.failure.is_none() && self.context.is_none() {
            if self.remaining == 0 {
                self.mark_exhausted();
                return false;
            }
            self.remaining -= 1;
            self.used += 1;
            return true;
        }
        self.step_with_context()
    }

    #[inline]
    fn step_batched(&mut self) -> bool {
        if self.failure.is_some() {
            return false;
        }
        if self.used - self.recorded >= 16
            && let Err(error) = self.check()
        {
            self.failure = Some(error);
            self.mark_exhausted();
            return false;
        }
        if self.remaining == 0 {
            self.mark_exhausted();
            return false;
        }
        self.remaining -= 1;
        self.used += 1;
        true
    }

    #[cold]
    #[inline(never)]
    fn step_with_context(&mut self) -> bool {
        if let Err(error) = self.check() {
            self.failure = Some(error);
            self.mark_exhausted();
            return false;
        }
        if self.remaining == 0 {
            self.mark_exhausted();
            return false;
        }
        if let Some(context) = self.context.as_deref_mut() {
            match context.record_iteration() {
                Ok(admission) if admission.is_granted() => {}
                Ok(_) => {
                    self.mark_exhausted();
                    return false;
                }
                Err(error) => {
                    self.failure = Some(error);
                    self.mark_exhausted();
                    return false;
                }
            }
        }
        self.remaining -= 1;
        self.used += 1;
        true
    }

    /// Charge up to `units` and return false once the allowance or the scope refuses one.
    ///
    /// Fixed coefficient tables charge their length at once; a scope still counts each unit.
    pub(crate) fn admit(&mut self, units: usize) -> bool {
        if self.failure.is_none() && self.context.is_none() {
            let admitted = units.min(self.remaining);
            self.remaining -= admitted;
            self.used += admitted;
            if admitted != units {
                self.mark_exhausted();
            }
            return admitted == units;
        }
        (0..units).all(|_| self.step())
    }

    /// Reserve bounded helper work, preserving per-unit scoped accounting.
    pub(crate) fn steps(&mut self, units: usize) -> Result<(), SciOxideError> {
        if self.context.is_none() && self.remaining >= units {
            self.remaining -= units;
            self.used += units;
            return Ok(());
        }
        for _ in 0..units {
            self.try_step()?;
        }
        Ok(())
    }

    /// Charge work already attempted in a nested kernel.
    pub(crate) fn charge(&mut self, count: usize) -> Result<(), SciOxideError> {
        if count > self.remaining {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options.iteration_budget"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nested work exceeded its allowance",
            ));
        }
        if !self.batched
            && let Some(context) = self.context.as_deref_mut()
        {
            for _ in 0..count {
                if !context.record_iteration()?.is_granted() {
                    return Err(crate::result::exhausted(
                        ConvergenceDiagnostics::new().with_counts(context.counts()),
                    ));
                }
            }
        }
        self.used = self
            .used
            .checked_add(count)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("work.used")))?;
        self.remaining -= count;
        self.check()
    }
}

/// Round a real kernel result before deciding whether the destination overflows.
pub(crate) fn project_real<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    if value.is_infinite() {
        return Err(crate::result::projection_error(
            crate::ScalarRepresentation::Overflow,
        ));
    }
    let value = check_finite(value)?;
    S::try_from_f64_lossy(value).map_err(|error| match error.view() {
        scioxide_core::SciOxideErrorView::ScalarConversion {
            reason: scioxide_core::ScalarConversionReason::OutOfRange,
            ..
        } => crate::result::projection_error(crate::ScalarRepresentation::Overflow),
        _ => error,
    })
}

/// Classify infinite output as overflow and NaN output as non-finite output.
pub(crate) fn check_finite<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    finite(value, || {
        if value.is_infinite() {
            SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                "finite inputs produced an infinite result",
            )
        } else {
            SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "operation produced a non-finite result",
            )
        }
    })
}

/// Validate finite real or complex components with a static input description.
pub(crate) fn require_finite_param<T: NumericScalar>(
    value: T,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    finite(value, || {
        SciOxideError::non_finite(NonFiniteOrigin::Input, detail)
    })
    .map(|_| ())
}

/// One finite-component predicate; callers keep the origin and failure kind.
pub(crate) fn finite<T: NumericScalar>(
    value: T,
    error: impl FnOnce() -> SciOxideError,
) -> Result<T, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(error())
    }
}

pub(crate) fn check_positive<S: RealScalar>(
    value: S,
    param: scioxide_core::Argument,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    require_finite_param(value, "special-function argument must be finite")?;
    if value > S::ZERO {
        Ok(())
    } else {
        let reason = if crate::util::is_exact_zero(value) {
            scioxide_core::DomainReason::Pole
        } else {
            scioxide_core::DomainReason::OutsideDomain
        };
        Err(SciOxideError::domain_error(param, reason, detail))
    }
}

pub(crate) fn checked_probability<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    let value = check_finite(value)?;
    let tolerance = <S as RealScalar>::EPSILON * S::try_from_f64_lossy(8.0)?;

    if value >= S::ZERO && value <= S::ONE {
        Ok(value)
    } else if value < S::ZERO && value >= -tolerance {
        Ok(S::ZERO)
    } else if value > S::ONE && value <= S::ONE + tolerance {
        Ok(S::ONE)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Cancellation,
            "computed probability lies outside [0, 1]",
        ))
    }
}

pub(crate) fn admit_family_work(
    units: usize,
    budget: scioxide_core::Budget,
) -> Result<(), SciOxideError> {
    if units <= budget.get() {
        return Ok(());
    }
    Err(crate::result::exhausted_work(0))
}

pub(crate) fn interrupted() -> JetEstimate {
    JetEstimate::from(JetData {
        y: Jet::constant(real(0.0)),
        scale: Jet::constant(real(0.0)),
        error: 0.0,
        derivative_error: None,
        method: algorithm_component!("interrupted"),
    })
}
#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use scioxide_core::ConvergenceReason;

    #[test]
    fn real_projection_rounds_before_classifying_overflow() -> Result<(), SciOxideError> {
        let maximum = f64::from(f32::MAX);
        let spacing = maximum - f64::from(f32::MAX.next_after(0.0));
        for sign in [-1.0, 1.0] {
            assert_eq!(
                project_real::<f32>(sign * (maximum + spacing * 0.25))?,
                sign as f32 * f32::MAX
            );
            assert!(matches!(
                project_real::<f32>(sign * (maximum + spacing * 0.5)).map_err(|e| e.kind()),
                Err(scioxide_core::SciOxideErrorKind::NumericalInstability)
            ));
            assert_eq!(
                project_real::<f32>(sign * f64::from_bits(1))?.to_bits(),
                (sign as f32 * 0.0).to_bits()
            );
        }
        assert!(matches!(
            project_real::<f32>(f64::NAN).map_err(|e| e.kind()),
            Err(scioxide_core::SciOxideErrorKind::NonFinite)
        ));
        Ok(())
    }

    #[test]
    fn shared_elementary_helpers_match_independent_references() {
        use super::wide::{Wide, square_root};
        for row in include_str!("../tests/data/wide_elementary.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let mut fields = row.split_whitespace();
            let kind = fields.next().unwrap_or_else(|| panic!("missing operation"));
            let mut next = || {
                f64::from_bits(
                    u64::from_str_radix(fields.next().unwrap_or(""), 16)
                        .unwrap_or_else(|error| panic!("{error}")),
                )
            };
            let x = Wide {
                hi: next(),
                lo: next(),
            };
            let y = Wide {
                hi: next(),
                lo: next(),
            };
            let first = Wide {
                hi: next(),
                lo: next(),
            };
            let second = Wide {
                hi: next(),
                lo: next(),
            };
            if kind == "expm1" {
                let actual = complex_expm1(C::new(x.hi, y.hi));
                for (actual, reference) in [(actual.re, first), (actual.im, second)] {
                    assert!(
                        (actual - reference.value()).abs()
                            <= 16.0 * f64::EPSILON * reference.hi.abs() + f64::from_bits(1),
                        "{row}: {actual}"
                    );
                }
            } else {
                let (a, b) = if kind == "phase" {
                    x.sin_cos()
                } else {
                    (square_root(x), Wide::ZERO)
                };
                let bound = if kind == "phase" {
                    if x.hi.abs() > 1e14 {
                        16.0 * f64::EPSILON
                    } else {
                        16.0 * f64::EPSILON * f64::EPSILON * (1.0 + x.hi.abs())
                    }
                } else {
                    16.0 * f64::EPSILON * f64::EPSILON * first.hi.abs()
                };
                assert!(
                    (a - first).value().abs() <= bound + f64::from_bits(1),
                    "{row}: {a:?}"
                );
                assert!(
                    (b - second).value().abs() <= bound + f64::from_bits(1),
                    "{row}: {b:?}"
                );
            }
        }
    }

    #[test]
    fn shared_phase_propagates_admission_failure_and_preserves_low_phase() {
        use super::wide::{Wide, sin_cos_with_step};
        let mut used = 0;
        let failed = sin_cos_with_step(Wide::from(0.75), &mut || {
            if used == 2 {
                Err(())
            } else {
                used += 1;
                Ok(())
            }
        });
        assert!(failed.is_err());
        assert_eq!(used, 2);
        let angle = super::constants::FRAC_PI_2 + Wide::from(1e-20);
        let (sine, cosine) = angle.sin_cos();
        assert!((sine.value() - 1.0).abs() <= f64::EPSILON);
        assert!((cosine.value() + 1e-20).abs() <= 2.0 * f64::EPSILON * f64::EPSILON);
        assert_eq!(
            super::wide::square_root(Wide::from(-0.0)).hi.to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn work_admission_preserves_attempted_counts_and_exhaustion() {
        let mut counter = Work::new(2);
        assert!(counter.step());
        assert!(counter.step());
        assert!(!counter.step());
        assert_eq!(counter.used, 2);
        assert_eq!(counter.remaining, 0);
        assert!(!counter.complete);

        let mut deferred = Work::new(3);
        deferred.charge(2).unwrap_or_else(|error| panic!("{error}"));
        assert!(deferred.step());
        assert!(!deferred.step());
        assert_eq!(deferred.used, 3);

        let mut table = Work::new(5);
        assert!(table.admit(3));
        assert!(!table.admit(3));
        assert_eq!((table.used, table.remaining), (5, 0));
        assert!(table.exhausted());

        let mut immediate = Work::new(2);
        assert!(immediate.steps(2).is_ok());
        let Err(error) = immediate.try_step() else {
            panic!("expected exhaustion");
        };
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(immediate.used, 2);
        assert!(immediate.exhausted());
    }

    #[test]
    fn precision_delegation_runs_only_for_binary32() {
        let calls = Cell::new(0);
        let wide = || {
            calls.set(calls.get() + 1);
            Ok(1.5_f64)
        };
        assert!(delegate_f32::<f64, _, _>(wide, |v| Ok(v as f32)).is_none());
        assert_eq!(calls.get(), 0);
        let value = delegate_f32::<f32, _, _>(wide, |v| Ok(v as f32));
        assert!(matches!(value, Some(Ok(v)) if v.to_bits() == 1.5_f32.to_bits()));
        assert_eq!(calls.get(), 1);
    }
}
