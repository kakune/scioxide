//! Search for an interval on which a scalar function changes sign.

use crate::initialization::{Bracket, brackets_root};
use crate::traits::ScalarFunction;
use scioxide_core::checked_sum_of_products;
use scioxide_core::{Budget, Interval, RealScalar, SciOxideError};

/// Starting point or interval of [`find_bracket_with_options`].
#[derive(Debug, Clone, Copy)]
pub enum BracketSearchInput<S: RealScalar> {
    /// One finite point in the domain.
    Guess(S),
    /// Two finite points `a ≤ b` in the domain; `a = b` is allowed.
    Pair(S, S),
}

/// Direction in which [`find_bracket_with_options`] expands the interval.
///
/// `Increasing` and `Decreasing` declare that `f` is monotone; the search does
/// not check this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketSearchDirection {
    /// Expands the endpoint with the smaller `|f|`, or the other endpoint when
    /// the domain blocks it.
    Automatic,
    /// `f` is nondecreasing: expands upward while `f < 0` and downward while
    /// `f > 0`.
    Increasing,
    /// `f` is nonincreasing: expands upward while `f > 0` and downward while
    /// `f < 0`.
    Decreasing,
}

/// How [`find_bracket_with_options`] moves an endpoint outward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketExpansion {
    /// Moves the endpoint so that the width grows by the expansion factor `r`:
    /// `b ← b + (r − 1)(b − a)`. Suited to roots at an unknown distance.
    Additive,
    /// Multiplies or divides the endpoint by `r`, moving it away from the
    /// other endpoint without crossing zero. Suited to roots of unknown order
    /// of magnitude; requires a domain on one side of zero.
    Multiplicative,
}

/// Finds an interval `[a, b]` with `f(a) f(b) ≤ 0` around `guess` and returns
/// it as a [`Bracket`].
///
/// The search is [`find_bracket_with_options`] from
/// [`BracketSearchInput::Guess`] in the [`Automatic`](BracketSearchDirection::Automatic)
/// direction with the default [`BracketSearchOptions`]: it starts on
/// `[guess − h, guess + h]` with `h = max(|guess|, 1)` and repeatedly doubles
/// the width by moving the endpoint with the smaller `|f|` outward, at one
/// evaluation of `f` per step and at most 2,048 steps, anywhere in
/// `[−MAX, MAX]`. Pass the bracket's endpoints to [`find_root`](crate::find_root)
/// to compute the root. The call does not allocate.
///
/// Use [`find_bracket_with_options`] to set a domain, a direction for a
/// monotone function, multiplicative expansion or a start interval.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if `guess` is NaN or infinite.
/// - `ConvergenceFailure` (`IterationLimit`) if 2,048 expansions find no sign
///   change, or (`Stagnation`) if neither endpoint can move further.
/// - `NonFinite` (origin `Callback`) if `f` returns NaN or infinity; errors
///   returned by `f` are returned unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideError;
///
/// // Kepler's equation E − 0.5 sin E = 1 for the eccentric anomaly E.
/// let kepler = |e: f64| -> Result<f64, SciOxideError> { Ok(e - 0.5 * e.sin() - 1.0) };
/// let bracket = scioxide_rootfind::find_bracket(&kepler, 0.0)?;
/// let e = scioxide_rootfind::find_root(&kepler, bracket.lower(), bracket.upper())?;
/// assert!((e - 0.5 * e.sin() - 1.0).abs() < 1e-14);
/// # Ok::<(), SciOxideError>(())
/// ```
pub fn find_bracket<S, F>(f: F, guess: S) -> Result<Bracket<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    find_bracket_with_options(
        &crate::ScalarFn::new(f),
        BracketSearchInput::Guess(guess),
        BracketSearchDirection::Automatic,
        &BracketSearchOptions::default(),
    )
}

/// Finds an interval `[a, b]` with `f(a) f(b) ≤ 0` by expanding outward from
/// a starting point, and returns it as a [`Bracket`] for a bracketing solver.
///
/// The search stops at the first interval whose endpoint values have opposite
/// signs or include a zero. From one guess `x₀`, an `Automatic` additive search
/// starts on `[x₀ − h, x₀ + h]` with `h = max(|x₀|, 1)`. Other searches
/// evaluate `x₀` first and expand from it. From a pair, `a` is evaluated
/// first, and `f(a) = 0` ends the search. Each step moves one endpoint as set
/// by [`BracketSearchDirection`] and [`BracketExpansion`], and costs one
/// evaluation of `f`. A monotone search returns the interval between the last
/// two evaluated points. All evaluations stay inside
/// the domain of [`BracketSearchOptions::with_domain`].
///
/// # Errors
///
/// Option and seed errors are reported before `f` is called:
///
/// - `NonFinite` (origin `Input`) if a seed, a domain endpoint or the
///   expansion factor is NaN or infinite.
/// - `InvalidInput` if the seeds are unordered or outside the domain, the
///   domain is empty, the expansion factor is not greater than 1, or a
///   multiplicative search has a domain that contains zero.
///
/// During the search:
///
/// - `ConvergenceFailure` if no sign change is found within the expansion
///   limit or work limits, or the domain blocks every further step.
/// - `NonFinite` (origin `Callback`) if `f` returns NaN or infinity.
/// - Errors returned by `f` are returned unchanged.
///
/// # Examples
///
/// ```
/// use core::ops::Bound::{Excluded, Unbounded};
/// use scioxide_rootfind::{
///     BracketExpansion, BracketSearchDirection, BracketSearchInput, BracketSearchOptions, ScalarFn,
///     find_bracket_with_options,
/// };
///
/// // ln x − 2 is increasing on x > 0 and vanishes at x = e².
/// let options = BracketSearchOptions::default()
///     .with_expansion(BracketExpansion::Multiplicative)
///     .with_domain(Excluded(0.0), Unbounded);
/// let f = ScalarFn::new(|x: f64| Ok(x.ln() - 2.0));
/// let input = BracketSearchInput::Guess(1.0);
/// let direction = BracketSearchDirection::Increasing;
/// let bracket = find_bracket_with_options(&f, input, direction, &options)?;
/// assert!(bracket.lower() > 0.0 && bracket.interval().contains(2.0_f64.exp()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn find_bracket_with_options<S: RealScalar, F: ScalarFunction<S> + ?Sized>(
    f: &F,
    input: BracketSearchInput<S>,
    direction: BracketSearchDirection,
    options: &BracketSearchOptions<S>,
) -> Result<Bracket<S>, SciOxideError> {
    find_bracket_scoped(
        f,
        input,
        direction,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

/// Finds a bracket as [`find_bracket_with_options`] does, in an
/// [execution context](scioxide_core#execution-contexts) or as a nested search
/// inside a parent [`SolveContext`](scioxide_core::SolveContext).
/// [`find_bracket_with_options`] is its owned counterpart.
///
/// The work limits of [`BracketSearchOptions::with_budgets`] apply, and a
/// nested search also counts toward its parent's limits (see
/// [Solve accounting](scioxide_core#solve-accounting)).
///
/// # Errors
///
/// Returns the errors of [`find_bracket_with_options`], and `ExecutionStopped` on
/// cancellation or a deadline. A stopped search never returns a bracket.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_rootfind::{
///     BracketSearchDirection, BracketSearchInput, BracketSearchOptions, ScalarFn, find_bracket_scoped,
/// };
///
/// // 3 − x is decreasing and vanishes at x = 3.
/// let bracket = find_bracket_scoped(
///     &ScalarFn::new(|x: f64| Ok(3.0 - x)),
///     BracketSearchInput::Pair(0.0, 1.0),
///     BracketSearchDirection::Decreasing,
///     &BracketSearchOptions::default(),
///     ExecutionContext::sequential(),
/// )?;
/// assert!(bracket.interval().contains(3.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn find_bracket_scoped<'scope, 'capability, S: RealScalar, F: ScalarFunction<S> + ?Sized>(
    f: &F,
    input: BracketSearchInput<S>,
    direction: BracketSearchDirection,
    options: &BracketSearchOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<Bracket<S>, SciOxideError> {
    // Configuration rejection precedes both numerical callbacks and child work.
    let domain = options.domain_interval()?;
    let seeds = validate_seeds(input, domain)?;
    crate::context::run(
        scope.into(),
        scioxide_core::algorithm_id!("rootfind", "BracketSearch", 2),
        options.budgets(),
        |context| search(f, input, seeds, direction, options, domain, context),
    )
}

fn validate_seeds<S: RealScalar>(
    input: BracketSearchInput<S>,
    domain: Interval<S>,
) -> Result<(S, S), SciOxideError> {
    let seeds = match input {
        BracketSearchInput::Guess(x) => (x, x),
        BracketSearchInput::Pair(a, b) => (a, b),
    };
    if !seeds.0.is_finite() || !seeds.1.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "bracket seeds must be finite",
        ));
    }
    if seeds.0 > seeds.1 || !domain.contains(seeds.0) || !domain.contains(seeds.1) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("input"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "bracket seeds must be ordered and belong to the domain",
        ));
    }
    Ok(seeds)
}

fn search<S: RealScalar, F: ScalarFunction<S> + ?Sized>(
    f: &F,
    input: BracketSearchInput<S>,
    seeds: (S, S),
    direction: BracketSearchDirection,
    options: &BracketSearchOptions<S>,
    domain: Interval<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<Bracket<S>, SciOxideError> {
    let session = &crate::context::Session::new(context);
    // Bracket search additionally polls for a stop after every finite value, so
    // a cancelled or expired search never publishes a certificate.
    let evaluate = |point: S| -> Result<S, SciOxideError> {
        let value = crate::scalar::evaluate(
            session,
            scioxide_core::EvaluationRole::Function,
            point,
            |x, context| f.eval_scoped(x, context),
        )?;
        match session.execution()?.stop_reason()? {
            Some(reason) => Err(SciOxideError::execution_stopped(reason)),
            None => Ok(value),
        }
    };
    let clamp = |x: S| x.max(domain.lower()).min(domain.upper());
    let (mut a, mut b) = seeds;
    if matches!(input, BracketSearchInput::Guess(_))
        && direction == BracketSearchDirection::Automatic
        && options.expansion() == BracketExpansion::Additive
    {
        (a, b) = initial_bounds(a, a.abs().max(S::ONE));
        (a, b) = (clamp(a), clamp(b));
    }
    let mut fa = evaluate(a)?;
    let mut fb = if a == b || fa == S::ZERO {
        b = a;
        fa
    } else {
        evaluate(b)?
    };
    for _ in 0..options.max_expansions().get() {
        if brackets_root(fa, fb) {
            return certificate(a, b, fa, fb);
        }
        let prefer_upper = match direction {
            BracketSearchDirection::Automatic => fa.abs() >= fb.abs(),
            BracketSearchDirection::Increasing => fb < S::ZERO,
            BracketSearchDirection::Decreasing => fb > S::ZERO,
        };
        let mut expansion = None;
        for upper in [prefer_upper, !prefer_upper] {
            if upper != prefer_upper && direction != BracketSearchDirection::Automatic {
                break;
            }
            let candidate = clamp(match options.expansion() {
                BracketExpansion::Multiplicative => multiplicative_bound(
                    if upper { b } else { a },
                    options.expansion_factor(),
                    upper,
                ),
                BracketExpansion::Additive if a == b => {
                    let bounds = initial_bounds(a, a.abs().max(S::ONE));
                    if upper { bounds.1 } else { bounds.0 }
                }
                BracketExpansion::Additive => {
                    expanded_bound(a, b, options.expansion_factor() - S::ONE, upper)?
                }
            });
            if (upper && candidate > b) || (!upper && candidate < a) {
                expansion = Some((upper, candidate));
                break;
            }
        }
        let Some((upper, candidate)) = expansion else {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::Stagnation,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(session.accounting()?),
                "bracket cannot advance toward a root within the domain",
            ));
        };
        session.iteration()?;
        let value = evaluate(candidate)?;
        if upper {
            if direction != BracketSearchDirection::Automatic && brackets_root(fb, value) {
                a = b;
                fa = fb;
            }
            b = candidate;
            fb = value;
        } else {
            if direction != BracketSearchDirection::Automatic && brackets_root(value, fa) {
                b = a;
                fb = fa;
            }
            a = candidate;
            fa = value;
        }
    }
    if brackets_root(fa, fb) {
        return certificate(a, b, fa, fb);
    }
    Err(SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(session.accounting()?),
        "bracket expansion budget exhausted",
    ))
}

fn certificate<S: RealScalar>(a: S, b: S, fa: S, fb: S) -> Result<Bracket<S>, SciOxideError> {
    Bracket::from_evaluated(
        Interval::new(a, b)?,
        fa,
        fb,
        scioxide_core::argument!("input"),
    )
}

fn multiplicative_bound<S: RealScalar>(point: S, factor: S, upper: bool) -> S {
    let grow_magnitude = upper == (point > S::ZERO);
    let value = if grow_magnitude {
        point * factor
    } else {
        point / factor
    };
    if !value.is_finite() {
        if upper { S::MAX } else { -S::MAX }
    } else if value == S::ZERO {
        if point > S::ZERO {
            S::MIN_SUBNORMAL
        } else {
            -S::MIN_SUBNORMAL
        }
    } else {
        value
    }
}

fn initial_bounds<S: RealScalar>(center: S, radius: S) -> (S, S) {
    let maximum = S::MAX;
    let lower = if center < -maximum + radius {
        -maximum
    } else {
        center - radius
    };
    let upper = if center > maximum - radius {
        maximum
    } else {
        center + radius
    };
    (lower, upper)
}

/// Moves one endpoint outward; an endpoint that overflows becomes `±MAX`.
fn expanded_bound<S: RealScalar>(
    lower: S,
    upper: S,
    growth: S,
    expand_upper: bool,
) -> Result<S, SciOxideError> {
    let one_plus_growth = S::ONE + growth;
    let result = if expand_upper {
        checked_sum_of_products(&[-growth, one_plus_growth], &[lower, upper])
    } else {
        checked_sum_of_products(&[one_plus_growth, -growth], &[lower, upper])
    };
    match result {
        Ok(candidate) => Ok(candidate),
        Err(error) => match error.view() {
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            } => Ok(if expand_upper { S::MAX } else { -S::MAX }),
            _ => Err(error),
        },
    }
}

// ---------------------------------------------------------------------------
// Bracket options
// ---------------------------------------------------------------------------

/// Options of [`find_bracket_with_options`]; the search checks them before calling `f`.
///
/// The defaults are additive expansion with factor 2, at most 2,048 expansion
/// steps (enough to reach the largest finite `f32` and `f64` values), an
/// unbounded domain and no work limits.
#[derive(Debug, Clone, Copy)]
pub struct BracketSearchOptions<S: RealScalar> {
    /// Maximum number of expansion steps.
    max_expansions: Budget,
    /// Expansion factor `r > 1`.
    expansion_factor: S,
    expansion: crate::BracketExpansion,
    domain: (core::ops::Bound<S>, core::ops::Bound<S>),
    budgets: scioxide_core::SolveBudgets,
}

impl<S: RealScalar> BracketSearchOptions<S> {
    /// Sets the maximum number of expansion steps.
    #[must_use]
    pub fn with_max_expansions(mut self, max_expansions: Budget) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    /// Sets the expansion factor `r`; the search rejects `r ≤ 1`.
    #[must_use]
    pub const fn with_expansion_factor(mut self, expansion_factor: S) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// Selects additive or multiplicative expansion. Multiplicative expansion
    /// requires a domain that does not contain zero.
    #[must_use]
    pub const fn with_expansion(mut self, expansion: crate::BracketExpansion) -> Self {
        self.expansion = expansion;
        self
    }

    /// Returns the expansion mode; additive by default.
    pub const fn expansion(&self) -> crate::BracketExpansion {
        self.expansion
    }

    /// Restricts every evaluation point to the domain between `lower` and
    /// `upper`.
    ///
    /// An `Excluded` endpoint is replaced by the adjacent floating-point number
    /// inside the domain, and `Unbounded` by `±MAX`. Endpoints must be finite.
    /// The seeds must lie in the domain. A one-point domain can only find an
    /// exact zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::ops::Bound::{Excluded, Unbounded};
    /// use scioxide_rootfind::{
    ///     BracketSearchDirection, BracketSearchInput, BracketSearchOptions, ScalarFn,
    ///     find_bracket_with_options,
    /// };
    ///
    /// // ln x is defined only for x > 0.
    /// let options = BracketSearchOptions::default().with_domain(Excluded(0.0), Unbounded);
    /// let f = ScalarFn::new(|x: f64| Ok(x.ln() - 2.0));
    /// let input = BracketSearchInput::Guess(1.0);
    /// let direction = BracketSearchDirection::Increasing;
    /// let bracket = find_bracket_with_options(&f, input, direction, &options)?;
    /// assert!(bracket.lower() > 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[must_use]
    pub fn with_domain(mut self, lower: core::ops::Bound<S>, upper: core::ops::Bound<S>) -> Self {
        self.domain = (lower, upper);
        self
    }

    /// Returns the lower and upper domain limits.
    pub const fn domain(&self) -> (core::ops::Bound<S>, core::ops::Bound<S>) {
        self.domain
    }

    pub(crate) fn domain_interval(&self) -> Result<Interval<S>, SciOxideError> {
        validate_expansion_factor(self.expansion_factor)?;
        let domain = bracket_domain(self.domain.0, self.domain.1)?;
        if self.expansion == crate::BracketExpansion::Multiplicative
            && domain.lower() <= S::ZERO
            && domain.upper() >= S::ZERO
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("domain"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "multiplicative expansion requires a strictly signed domain",
            ));
        }
        Ok(domain)
    }

    /// Returns the maximum number of expansion steps.
    pub fn max_expansions(&self) -> Budget {
        self.max_expansions
    }

    /// Returns the expansion factor `r`.
    #[must_use]
    pub fn expansion_factor(&self) -> S {
        self.expansion_factor
    }

    /// Sets work limits for each search. Each expansion step counts as one
    /// iteration, so an iteration limit below [`Self::max_expansions`] stops
    /// the search earlier.
    #[must_use]
    pub const fn with_budgets(mut self, budgets: scioxide_core::SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Returns the work limits; unlimited by default.
    pub const fn budgets(&self) -> scioxide_core::SolveBudgets {
        self.budgets
    }
}

impl<S: RealScalar> Default for BracketSearchOptions<S> {
    fn default() -> Self {
        Self {
            max_expansions: Budget::ONE.saturating_add(2047),
            expansion_factor: S::TWO,
            expansion: crate::BracketExpansion::Additive,
            domain: (core::ops::Bound::Unbounded, core::ops::Bound::Unbounded),
            budgets: scioxide_core::SolveBudgets::unbounded(),
        }
    }
}

fn bracket_domain<S: RealScalar>(
    lower: core::ops::Bound<S>,
    upper: core::ops::Bound<S>,
) -> Result<Interval<S>, SciOxideError> {
    use core::ops::Bound::{Excluded, Included, Unbounded};
    for bound in [lower, upper] {
        if let Included(value) | Excluded(value) = bound
            && !value.is_finite()
        {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "bracket domain endpoints must be finite",
            ));
        }
    }
    let lower = match lower {
        Unbounded => -S::MAX,
        Included(value) => value,
        Excluded(value) => value.next_after(S::INFINITY),
    };
    let upper = match upper {
        Unbounded => S::MAX,
        Included(value) => value,
        Excluded(value) => value.next_after(S::NEG_INFINITY),
    };
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("domain"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "bracket domain contains no finite representable value",
        ));
    }
    Interval::new(lower, upper)
}

/// Validate `expansion_factor`.
fn validate_expansion_factor<S: RealScalar>(expansion_factor: S) -> Result<(), SciOxideError> {
    if !expansion_factor.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "expansion factor must be finite",
        ));
    }
    if expansion_factor <= S::ONE {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("expansion_factor"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "expansion factor must be greater than one",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bracket_options_default_f64() {
        let opts = BracketSearchOptions::<f64>::default();
        assert_eq!(opts.max_expansions().get(), 2048);
        assert!((opts.expansion_factor() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_bracket_options_default_f32() {
        let opts = BracketSearchOptions::<f32>::default();
        assert_eq!(opts.max_expansions().get(), 2048);
        assert!((opts.expansion_factor() - 2.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_bracket_search_options_reject_invalid_factor() {
        let options = BracketSearchOptions::default().with_max_expansions(Budget::ONE);
        let result = options.with_expansion_factor(1.0_f64).domain_interval();
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );

        let result = options.with_expansion_factor(f64::NAN).domain_interval();
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );

        let result = BracketSearchOptions::default()
            .with_expansion_factor(0.5_f64)
            .domain_interval();
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }

    // f(x) = x^2 - 2, root at sqrt(2) ≈ 1.414
    struct Sqrt2Fn;
    impl ScalarFunction<f64> for Sqrt2Fn {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(x.mul_add(x, -2.0))
        }
    }

    impl ScalarFunction<f32> for Sqrt2Fn {
        fn eval(&self, x: f32) -> Result<f32, SciOxideError> {
            Ok(x.mul_add(x, -2.0))
        }
    }

    // f(x) = x - 5, root at 5
    struct LinearFn;
    impl ScalarFunction<f64> for LinearFn {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(x - 5.0)
        }
    }

    // f(x) = x^2 + 1, no real root
    struct NoRealRootFn;
    impl ScalarFunction<f64> for NoRealRootFn {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(x.mul_add(x, 1.0))
        }
    }

    // f(x) = NaN for testing non-finite detection
    struct NanFn;
    impl ScalarFunction<f64> for NanFn {
        fn eval(&self, _x: f64) -> Result<f64, SciOxideError> {
            Ok(f64::NAN)
        }
    }

    #[test]
    fn expanded_bounds_saturate_only_on_overflow() {
        for expand_upper in [false, true] {
            let expected = if expand_upper { 4.0 } else { -1.0 };
            let candidate = expanded_bound(1.0_f64, 2.0, 2.0, expand_upper).unwrap();
            assert!((candidate - expected).abs() <= f64::EPSILON);

            let limit = expanded_bound(-f64::MAX, f64::MAX, 2.0, expand_upper).unwrap();
            assert_eq!(limit, if expand_upper { f64::MAX } else { -f64::MAX });

            let error = expanded_bound(f64::NAN, 2.0, 2.0, expand_upper).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        }
    }

    #[test]
    fn test_find_bracket_sqrt2_near_root() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &Sqrt2Fn,
            crate::BracketSearchInput::Guess(1.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_ok());
        let bracket = result.unwrap();
        let (a, b) = (bracket.lower(), bracket.upper());
        assert!(a < b);
        // The bracket should contain sqrt(2) ≈ 1.414
        let fa = Sqrt2Fn.eval(a).unwrap();
        let fb = Sqrt2Fn.eval(b).unwrap();
        assert!(fa * fb < 0.0);
    }

    #[test]
    fn test_find_bracket_sqrt2_far_from_root() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &Sqrt2Fn,
            crate::BracketSearchInput::Guess(10.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_ok());
        let bracket = result.unwrap();
        let (a, b) = (bracket.lower(), bracket.upper());
        let fa = Sqrt2Fn.eval(a).unwrap();
        let fb = Sqrt2Fn.eval(b).unwrap();
        assert!(fa * fb < 0.0);
    }

    #[test]
    fn test_find_bracket_linear() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &LinearFn,
            crate::BracketSearchInput::Guess(0.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_ok());
        let bracket = result.unwrap();
        let (a, b) = (bracket.lower(), bracket.upper());
        assert!(a < b);
        let fa = LinearFn.eval(a).unwrap();
        let fb = LinearFn.eval(b).unwrap();
        assert!(fa * fb < 0.0);
    }

    #[test]
    fn test_find_bracket_no_real_root() {
        let opts = BracketSearchOptions::default().with_max_expansions(scioxide_core::Budget::ONE);
        let result = find_bracket_with_options(
            &NoRealRootFn,
            crate::BracketSearchInput::Guess(0.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
    }

    #[test]
    fn test_find_bracket_nan_function() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &NanFn,
            crate::BracketSearchInput::Guess(0.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn test_find_bracket_max_expand_1() {
        let opts = BracketSearchOptions::default().with_max_expansions(scioxide_core::Budget::ONE);
        // With only 1 expansion step and starting far from root, may fail
        let result = find_bracket_with_options(
            &crate::ScalarFn::new(|x: f64| Ok(x - 100.0)),
            crate::BracketSearchInput::Guess(0.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        let error = result.unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        if let scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } =
            error.view()
        {
            assert_eq!(diagnostics.counts().own().iterations(), 1);
            assert_eq!(
                diagnostics
                    .counts()
                    .own()
                    .provider(scioxide_core::EvaluationRole::Function),
                3
            );
        }
    }

    #[test]
    fn test_find_bracket_enforces_expansion_factor() {
        let budget = scioxide_core::Budget::ONE;
        let options = BracketSearchOptions::default().with_max_expansions(budget);
        let conservative = options.with_expansion_factor(1.1_f64);
        let aggressive = options.with_expansion_factor(200_000_000.0_f64);

        assert!(
            find_bracket_with_options(
                &LinearFn,
                crate::BracketSearchInput::Guess(0.0),
                crate::BracketSearchDirection::Automatic,
                &conservative
            )
            .is_err()
        );
        assert!(
            find_bracket_with_options(
                &LinearFn,
                crate::BracketSearchInput::Guess(0.0),
                crate::BracketSearchDirection::Automatic,
                &aggressive
            )
            .is_ok()
        );
    }

    #[test]
    fn test_find_bracket_f32() {
        let opts = BracketSearchOptions::<f32>::default();
        let result = find_bracket_with_options(
            &Sqrt2Fn,
            crate::BracketSearchInput::Guess(1.0_f32),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_ok());
        let bracket = result.unwrap();
        let (a, b) = (bracket.lower(), bracket.upper());
        assert!(a < b);
    }

    #[test]
    fn test_find_bracket_at_zero() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &Sqrt2Fn,
            crate::BracketSearchInput::Guess(0.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_ok());
        let bracket = result.unwrap();
        let (a, b) = (bracket.lower(), bracket.upper());
        let fa = Sqrt2Fn.eval(a).unwrap();
        let fb = Sqrt2Fn.eval(b).unwrap();
        assert!(fa * fb < 0.0);
    }

    #[test]
    fn test_find_bracket_negative_start() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &Sqrt2Fn,
            crate::BracketSearchInput::Guess(-5.0),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert!(result.is_ok());
        let bracket = result.unwrap();
        let (a, b) = (bracket.lower(), bracket.upper());
        let fa = Sqrt2Fn.eval(a).unwrap();
        let fb = Sqrt2Fn.eval(b).unwrap();
        assert!(fa * fb < 0.0);
    }

    #[test]
    fn test_find_bracket_rejects_invalid_expansion_factor() {
        let options = BracketSearchOptions::default()
            .with_max_expansions(scioxide_core::Budget::ONE.saturating_add(9))
            .with_expansion_factor(1.0_f64);
        let calls = core::cell::Cell::new(0_usize);
        let counted = crate::ScalarFn::new(|x: f64| {
            calls.set(calls.get() + 1);
            Ok(x)
        });
        let result = find_bracket_with_options(
            &counted,
            crate::BracketSearchInput::Guess(1.0),
            crate::BracketSearchDirection::Automatic,
            &options,
        );
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
        assert_eq!(calls.get(), 0, "the factor is rejected before f is called");
    }

    #[test]
    fn test_find_bracket_rejects_non_finite_initial_guess() {
        let opts = BracketSearchOptions::default();
        let result = find_bracket_with_options(
            &LinearFn,
            crate::BracketSearchInput::Guess(f64::NAN),
            crate::BracketSearchDirection::Automatic,
            &opts,
        );
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }
}
