//! Affine arithmetic with real and complex coefficients.
mod math;

use crate::rectangle::Box2;
use crate::shell::{at_least, invalid, mul, within_limit};
use crate::{DecoratedInterval, Decoration, EnclosureStatus};
use alloc::{sync::Arc, vec::Vec};
use core::{
    mem::size_of,
    sync::atomic::{AtomicUsize, Ordering},
};
use math::evaluate;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};

/// Limits on the size of the affine forms of an [`AffineContext`].
///
/// - `max_symbols` (default 16) is the largest number of noise terms a form
///   keeps, counting the error terms that operations add. When a result has
///   more, the terms with the smallest coefficients are merged into the
///   residual interval of the form. The enclosure stays valid, but the
///   correlation those terms carried is lost, which
///   [`AffineReport::correlation_lost`] reports.
/// - `byte_limit` (default unlimited) bounds the scratch storage of each
///   [`AffineWorkspace`], excluding allocator overhead.
///
/// ```
/// use scioxide_validated::AffineOptions;
///
/// let options = AffineOptions::new()
///     .with_max_symbols(8)
///     .with_byte_limit(1 << 16);
/// assert_eq!(options.max_symbols(), 8);
/// assert_eq!(options.byte_limit(), 1 << 16);
/// assert_eq!(AffineOptions::default().max_symbols(), 16);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct AffineOptions {
    max_symbols: usize,
    byte_limit: usize,
}

impl Default for AffineOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl AffineOptions {
    /// Returns the defaults: 16 noise terms per form and no byte limit.
    pub const fn new() -> Self {
        Self {
            max_symbols: 16,
            byte_limit: usize::MAX,
        }
    }
    /// Sets the largest number of noise terms a form keeps.
    pub const fn with_max_symbols(mut self, max_symbols: usize) -> Self {
        self.max_symbols = max_symbols;
        self
    }
    /// Returns the largest number of noise terms a form keeps.
    pub const fn max_symbols(&self) -> usize {
        self.max_symbols
    }
    /// Sets the largest scratch storage of a workspace, in bytes.
    pub const fn with_byte_limit(mut self, bytes: usize) -> Self {
        self.byte_limit = bytes;
        self
    }
    /// Returns the largest scratch storage of a workspace, in bytes.
    pub const fn byte_limit(&self) -> usize {
        self.byte_limit
    }
}

#[derive(Debug)]
struct ContextData {
    identity: usize,
    namespaces: AtomicUsize,
    inputs: usize,
    options: AffineOptions,
}

/// Allocate a never-reused ordinal; exhaustion leaves the counter unchanged.
fn ordinal(counter: &AtomicUsize) -> Result<usize, SciOxideError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("affine_noise_identifier"))
        })
}
static CONTEXTS: AtomicUsize = AtomicUsize::new(0);

/// Set of noise symbols and limits shared by the affine forms that are combined
/// with each other.
///
/// An affine form represents a quantity as
///
/// ```text
/// x̂ = x₀ + x₁ε₁ + … + xₙεₙ + r,
/// ```
///
/// where the noise symbols `εᵢ` are unknown real numbers in `[−1, 1]`, the
/// center `x₀` and the coefficients `xᵢ` are real or complex, and `r` lies in
/// a residual interval (a rectangle for complex forms). Two forms that share a
/// symbol `εᵢ` are correlated through it, so `x̂ − x̂` is exactly zero.
///
/// A context declares how many input symbols there are and sets the
/// [`AffineOptions`]. Forms can be combined only if they come from the same
/// context or one of its clones; contexts created separately are incompatible,
/// even with equal options.
///
/// ```
/// use scioxide_validated::{AffineContext, AffineOptions};
///
/// let context = AffineContext::new(2, AffineOptions::new())?;
/// // ε₀ and ε₁ are the two input symbols.
/// assert!(context.symbol(1).is_ok() && context.symbol(2).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct AffineContext {
    data: Arc<ContextData>,
}
impl AffineContext {
    /// Creates a context with `input_symbols` input noise symbols.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `usize::MAX` contexts have been created in this
    ///   process.
    pub fn new(input_symbols: usize, options: AffineOptions) -> Result<Self, SciOxideError> {
        Ok(Self {
            data: Arc::new(ContextData {
                identity: ordinal(&CONTEXTS)?,
                namespaces: AtomicUsize::new(1),
                inputs: input_symbols,
                options,
            }),
        })
    }
    /// Returns the options.
    pub fn options(&self) -> AffineOptions {
        self.data.options
    }
    /// Returns input noise symbol number `index`. Every call with the same
    /// index returns the same symbol.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `index` is not less than the number of input symbols.
    pub fn symbol(&self, index: usize) -> Result<AffineSymbol, SciOxideError> {
        if index >= self.data.inputs {
            return Err(invalid(
                scioxide_core::argument!("index"),
                "input symbol is outside the plan",
            ));
        }
        Ok(AffineSymbol {
            context: self.data.identity,
            namespace: 0,
            index,
        })
    }
    /// Allocates a workspace for operations on forms with scalar type `T`.
    ///
    /// The workspace holds `2 · (2 · max_symbols + 2)` terms of type
    /// [`AffineTerm<T>`]. Each workspace draws the error symbols that its
    /// operations add from its own set, so results from different workspaces
    /// stay distinct.
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_validated::{AffineContext, AffineOptions};
    ///
    /// let context = AffineContext::new(1, AffineOptions::new().with_max_symbols(4))?;
    /// let _workspace = context.workspace::<f64>()?;
    /// let tight = AffineContext::new(1, AffineOptions::new().with_byte_limit(64))?;
    /// let error = tight.workspace::<f64>().err().unwrap();
    /// assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds [`AffineOptions::byte_limit`].
    /// - `SizeOverflow` if the size in bytes overflows `usize`, or if
    ///   `usize::MAX` workspaces have been created for this context.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn workspace<T: NumericScalar>(&self) -> Result<AffineWorkspace<T>, SciOxideError> {
        within_limit(self.workspace_bytes::<T>()?, self.data.options.byte_limit)?;
        Ok(AffineWorkspace {
            context: self.clone(),
            namespace: ordinal(&self.data.namespaces)?,
            selection: try_filled(self.workspace_slots()?, AffineTerm::default(), SCRATCH)?,
            next: 0,
            scratch: try_filled(self.workspace_slots()?, AffineTerm::default(), SCRATCH)?,
        })
    }
    /// Scratch payload bytes of a `T` workspace.
    pub(crate) fn workspace_bytes<T: NumericScalar>(&self) -> Result<usize, SciOxideError> {
        mul(
            self.workspace_slots()?,
            checked_mul(2, size_of::<AffineTerm<T>>(), SCRATCH)?,
        )
    }
    /// Scratch terms: two staged operands plus two component error symbols.
    fn workspace_slots(&self) -> Result<usize, SciOxideError> {
        checked_add(
            checked_mul(self.data.options.max_symbols, 2, SCRATCH)?,
            2,
            SCRATCH,
        )
    }
    fn max_symbols(&self) -> usize {
        self.data.options.max_symbols
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

const SCRATCH: scioxide_core::Quantity = scioxide_core::quantity!("affine.scratch");

/// Noise symbol `εᵢ` of an [`AffineContext`]: an input symbol or an error
/// symbol added by an operation.
///
/// Symbols are ordered by context, then by workspace (input symbols first),
/// then by index. A symbol is never reused, even after its context or
/// workspace is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AffineSymbol {
    context: usize,
    namespace: usize,
    index: usize,
}

/// Term `xᵢεᵢ` of an affine form: a coefficient and its noise symbol.
///
/// `Default` gives an unused slot with no symbol and a zero coefficient, for
/// filling output buffers.
#[derive(Debug, Clone, Copy)]
pub struct AffineTerm<T: NumericScalar> {
    symbol: Option<AffineSymbol>,
    coefficient: T,
}
impl<T: NumericScalar> Default for AffineTerm<T> {
    fn default() -> Self {
        Self {
            symbol: None,
            coefficient: T::ZERO,
        }
    }
}
impl<T: NumericScalar> AffineTerm<T> {
    /// Creates the term `coefficient · symbol`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `coefficient` is NaN or infinite.
    pub fn new(symbol: AffineSymbol, coefficient: T) -> Result<Self, SciOxideError> {
        if !coefficient.has_finite_components() {
            return Err(invalid(
                scioxide_core::argument!("coefficient"),
                "coefficient must be finite",
            ));
        }
        Ok(Self {
            symbol: Some(symbol),
            coefficient,
        })
    }
    /// Returns the noise symbol, or `None` for an unused slot.
    pub fn symbol(&self) -> Option<&AffineSymbol> {
        self.symbol.as_ref()
    }
    /// Returns the coefficient; zero for an unused slot.
    pub fn coefficient(&self) -> T {
        self.coefficient
    }
}

/// Why a result is only an interval enclosure, with no affine form.
///
/// A result without an affine form keeps its real and imaginary enclosures but
/// no noise terms, so it is uncorrelated with every other form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineFallback {
    /// An operand was already only an enclosure.
    EnclosureInput,
    /// The operand reaches a domain boundary, a branch cut or a point where the
    /// derivative is unbounded, so no linear approximation with a finite error
    /// bound exists.
    DomainOrBranch,
    /// A range, coefficient or error bound overflowed, in this operation or in
    /// an operand.
    Unbounded,
}

/// Enclosure and diagnostics of an affine form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineReport<S: RealScalar> {
    real: DecoratedInterval<S>,
    imaginary: DecoratedInterval<S>,
    status: EnclosureStatus,
    retained_symbols: usize,
    discarded_symbols: usize,
    correlation_lost: bool,
    fallback: Option<AffineFallback>,
}
impl<S: RealScalar> AffineReport<S> {
    /// Returns an enclosure of the real part of the represented value.
    pub const fn real(&self) -> DecoratedInterval<S> {
        self.real
    }
    /// Returns an enclosure of the imaginary part; exactly zero for real forms.
    pub const fn imaginary(&self) -> DecoratedInterval<S> {
        self.imaginary
    }
    /// Returns where the operands lie relative to the domain of the last
    /// operation; `Inconclusive` when that cannot be decided.
    pub const fn status(&self) -> EnclosureStatus {
        self.status
    }
    /// Returns the number of noise terms the form has.
    pub const fn retained_symbols(&self) -> usize {
        self.retained_symbols
    }
    /// Returns the number of noise terms that the last operation merged into
    /// the residual to stay within `max_symbols`.
    pub const fn discarded_symbols(&self) -> usize {
        self.discarded_symbols
    }
    /// Returns whether this form, or a form it was computed from, had terms
    /// merged into the residual or fell back to an enclosure. Once set, the
    /// flag stays set in every result computed from this form.
    pub const fn correlation_lost(&self) -> bool {
        self.correlation_lost
    }
    /// Returns why the value is only an enclosure, or `None` if it has an
    /// affine form.
    pub const fn fallback(&self) -> Option<AffineFallback> {
        self.fallback
    }
}

/// Center, residual and report of an affine form whose terms are stored in a
/// separate slice.
///
/// Use it with a term buffer to build an [`AffineView`] or [`AffineViewMut`];
/// see [`AffineForm::view`].
#[derive(Debug, Clone)]
pub struct AffineState<T: NumericScalar> {
    context: AffineContext,
    center: T,
    residual: Box2<T::Real>,
    report: AffineReport<T::Real>,
}
impl<T: NumericScalar> AffineState<T> {
    /// Returns the state of the form `0` with no terms.
    ///
    /// # Errors
    ///
    /// This function does not fail for `f32`, `f64` and their complex types.
    pub fn zero(context: &AffineContext) -> Result<Self, SciOxideError> {
        let zero = Box2::zero()?;
        Ok(Self {
            context: context.clone(),
            center: T::ZERO,
            residual: zero,
            report: AffineReport {
                real: zero.re,
                imaginary: zero.im,
                status: EnclosureStatus::Defined,
                retained_symbols: 0,
                discarded_symbols: 0,
                correlation_lost: false,
                fallback: None,
            },
        })
    }
}

/// Borrowed affine form: a state and the slice of its terms.
#[derive(Clone, Copy)]
pub struct AffineView<'a, T: NumericScalar> {
    state: &'a AffineState<T>,
    terms: &'a [AffineTerm<T>],
}
/// Output for the `_into` operations of [`AffineView`]: a mutable state and a
/// term buffer of at least `max_symbols` slots.
pub struct AffineViewMut<'a, T: NumericScalar> {
    state: &'a mut AffineState<T>,
    terms: &'a mut [AffineTerm<T>],
}
impl<'a, T: NumericScalar> AffineView<'a, T> {
    /// Checks a state and its terms and borrows them as a form.
    ///
    /// The check takes O(n) time for n terms: the symbols must belong to the
    /// state's context and be strictly increasing, the coefficients must be
    /// finite, and the enclosures in the state must contain the form.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `terms` is shorter than the number of terms in
    ///   the state.
    /// - `InvalidInput` if the state has more terms than `max_symbols`, or a
    ///   term has no symbol, a symbol of another context, a symbol out of
    ///   order or repeated, or a coefficient that is not finite, or if the
    ///   enclosures in the state do not contain the form, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    pub fn new(
        state: &'a AffineState<T>,
        terms: &'a [AffineTerm<T>],
    ) -> Result<Self, SciOxideError> {
        crate::environment::check()?;
        let view = Self { state, terms };
        view.validate(scioxide_core::argument!("terms"))?;
        Ok(view)
    }
    /// Validate this view as the `argument`.
    fn validate(&self, argument: scioxide_core::Argument) -> Result<(), SciOxideError> {
        let n = self.state.report.retained_symbols;
        at_least(n, self.terms.len())?;
        if n > self.state.context.max_symbols() {
            return Err(invalid(argument, "header exceeds the context symbol limit"));
        }
        for (i, term) in self.terms[..n].iter().enumerate() {
            let Some(symbol) = &term.symbol else {
                return Err(invalid(argument, "active coefficient has no symbol"));
            };
            if symbol.context != self.state.context.data.identity
                || !term.coefficient.has_finite_components()
                || (i > 0 && self.terms[i - 1].symbol.as_ref() >= Some(symbol))
            {
                return Err(invalid(
                    argument,
                    "invalid or duplicate coefficient identity",
                ));
            }
        }
        if self.state.report.fallback.is_none() {
            let mut radius = self.state.residual;
            for term in &self.terms[..n] {
                radius = radius.add(term.coefficient_box()?)?;
            }
            let enclosure = Box2::point(self.state.center)?.add(radius)?;
            if !self.state.report.real.contains_interval(&enclosure.re)
                || !self.state.report.imaginary.contains_interval(&enclosure.im)
            {
                return Err(invalid(
                    argument,
                    "borrowed coefficients do not match the certified header",
                ));
            }
        }
        Ok(())
    }
    /// Returns the enclosure and diagnostics.
    pub fn report(&self) -> AffineReport<T::Real> {
        self.state.report
    }
    /// Returns the center `x₀`, or `None` if the value is only an enclosure.
    pub fn center(&self) -> Option<T> {
        self.state
            .report
            .fallback
            .is_none()
            .then_some(self.state.center)
    }
    /// Returns the noise terms, sorted by symbol.
    pub fn terms(&self) -> &[AffineTerm<T>] {
        &self.terms[..self.state.report.retained_symbols]
    }
    /// Applies `operation` and writes the result to `output`; see
    /// [`AffineForm::unary`]. Does not allocate; see [`AffineForm::view`] for an
    /// example.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operand, the output or the workspace come from
    ///   different contexts, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - `DimensionMismatch` if the output has fewer than `max_symbols` term
    ///   slots.
    /// - `SizeOverflow` if the workspace has run out of error symbols.
    pub fn unary_into(
        &self,
        operation: AffineUnary,
        output: AffineViewMut<'_, T>,
        workspace: &mut AffineWorkspace<T>,
    ) -> Result<AffineReport<T::Real>, SciOxideError> {
        workspace.run(*self, None, Operation::Unary(operation), output)
    }
    /// Applies `operation` to `self` and `other` and writes the result to
    /// `output`; see [`AffineForm::binary`]. Does not allocate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` for [`AffineBinary::Atan2`] and [`AffineBinary::Hypot`]
    ///   on complex forms.
    /// - The errors of [`Self::unary_into`].
    pub fn binary_into(
        &self,
        other: Self,
        operation: AffineBinary,
        output: AffineViewMut<'_, T>,
        workspace: &mut AffineWorkspace<T>,
    ) -> Result<AffineReport<T::Real>, SciOxideError> {
        if !T::IS_REAL && matches!(operation, AffineBinary::Atan2 | AffineBinary::Hypot) {
            return Err(invalid(
                scioxide_core::argument!("operation"),
                "atan2 and hypot require real scalar forms",
            ));
        }
        workspace.run(*self, Some(other), Operation::Binary(operation), output)
    }
    /// Writes the real or imaginary part of the form to the real form
    /// `output`; see [`AffineForm::project`]. Takes O(n) time for n terms,
    /// needs no workspace and does not allocate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output` comes from another context, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - `DimensionMismatch` if the output has fewer than `max_symbols` term
    ///   slots.
    pub fn project_into(
        &self,
        projection: AffineProjection,
        output: AffineViewMut<'_, T::Real>,
    ) -> Result<AffineReport<T::Real>, SciOxideError> {
        crate::environment::check()?;
        if !self.state.context.same(&output.state.context) {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "affine contexts do not match",
            ));
        }
        at_least(self.state.context.max_symbols(), output.terms.len())?;
        let zero = Box2::<T::Real>::zero()?.im;
        let mut report = self.state.report;
        report.real = projection.interval(report.real, report.imaginary);
        report.imaginary = zero;
        let state = AffineState {
            context: self.state.context.clone(),
            center: projection.scalar(self.state.center),
            residual: Box2 {
                re: projection.interval(self.state.residual.re, self.state.residual.im),
                im: zero,
            },
            report,
        };
        let mut n = 0;
        for term in self.terms() {
            let coefficient = projection.scalar(term.coefficient);
            if coefficient != T::Real::ZERO {
                output.terms[n] = AffineTerm {
                    symbol: term.symbol,
                    coefficient,
                };
                n += 1;
            }
        }
        output.terms[n..].fill(AffineTerm::default());
        *output.state = state;
        output.state.report.retained_symbols = n;
        Ok(output.state.report)
    }
    /// Writes the form with at most `max_symbols` terms to `output`; see
    /// [`AffineForm::reduce`].
    ///
    /// The terms with the smallest coefficient magnitudes are merged into the
    /// residual; among equal magnitudes, the term with the larger symbol goes
    /// first. The result keeps the status and fallback reason of `self`. Later
    /// operations may again use up to the context's `max_symbols`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `max_symbols` exceeds the context's limit.
    /// - The errors of [`Self::unary_into`].
    pub fn reduce_into(
        &self,
        max_symbols: usize,
        output: AffineViewMut<'_, T>,
        workspace: &mut AffineWorkspace<T>,
    ) -> Result<AffineReport<T::Real>, SciOxideError> {
        workspace.run_limited(*self, None, Operation::Identity, output, max_symbols)
    }
}
impl<'a, T: NumericScalar> AffineViewMut<'a, T> {
    /// Pairs a state with a term buffer as an output.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `terms` has fewer than the context's
    ///   `max_symbols` slots. The buffer is not read or changed.
    pub fn new(
        state: &'a mut AffineState<T>,
        terms: &'a mut [AffineTerm<T>],
    ) -> Result<Self, SciOxideError> {
        at_least(state.context.max_symbols(), terms.len())?;
        Ok(Self { state, terms })
    }
}

/// Affine form `x̂ = x₀ + Σ xᵢεᵢ + r` with real or complex coefficients, owned.
///
/// Affine arithmetic keeps track of how a value depends on each input through
/// the shared noise symbols `εᵢ ∈ [−1, 1]` of an [`AffineContext`], so it
/// avoids the overestimation of interval arithmetic when variables repeat.
/// Linear operations are exact apart from rounding. A nonlinear operation is
/// replaced by a linear approximation at the center, and a rigorous bound on
/// the approximation and rounding errors is added as new noise terms. The
/// result's [`AffineReport`] encloses its real and imaginary parts.
///
/// The operations are [`Self::unary`] with an [`AffineUnary`],
/// [`Self::binary`] with an [`AffineBinary`], [`Self::project`] with an
/// [`AffineProjection`], and [`Self::reduce`]. Each takes O(n) time for n
/// terms and allocates the result. The `_into` forms on [`AffineView`] write
/// to storage you provide and do not allocate.
///
/// The modulus and argument of a complex form are real binary operations on
/// its parts, which keep the shared input symbols:
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_validated::{AffineBinary, AffineContext, AffineForm, AffineOptions, AffineProjection};
///
/// let context = AffineContext::new(1, AffineOptions::new())?;
/// let mut workspace = context.workspace::<f64>()?;
/// // z = (3 + 4i) + (1/8 − i/8)ε₀
/// let z = AffineForm::from_coefficients(
///     &context,
///     Complex::new(3.0, 4.0),
///     &[Complex::new(0.125, -0.125)],
/// )?;
/// let (re, im) = (
///     z.project(AffineProjection::Real)?,
///     z.project(AffineProjection::Imaginary)?,
/// );
/// let modulus = re.binary(&im, AffineBinary::Hypot, &mut workspace)?;
/// let argument = im.binary(&re, AffineBinary::Atan2, &mut workspace)?;
/// // At ε₀ = 0, |z| = 5 and arg z = atan2(4, 3).
/// assert!(modulus.report().real().contains(5.0));
/// assert!(argument.report().real().contains(4.0_f64.atan2(3.0)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// L. H. de Figueiredo and J. Stolfi, "Affine arithmetic: concepts and
/// applications", *Numerical Algorithms* 37, 2004, pp. 147–158.
#[derive(Debug, Clone)]
pub struct AffineForm<T: NumericScalar> {
    state: AffineState<T>,
    terms: Vec<AffineTerm<T>>,
}
impl<T: NumericScalar> AffineForm<T> {
    /// Returns the form `0`, with storage for `max_symbols` terms.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn zero(context: &AffineContext) -> Result<Self, SciOxideError> {
        Ok(Self {
            terms: try_filled(
                context.max_symbols(),
                AffineTerm::default(),
                scioxide_core::quantity!("affine.terms"),
            )?,
            state: AffineState::zero(context)?,
        })
    }
    /// Returns the exact constant `value`, with no noise terms.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `value` is NaN or infinite, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn constant(context: &AffineContext, value: T) -> Result<Self, SciOxideError> {
        Self::from_terms(context, value, &[])
    }
    /// Creates `center + Σ terms`, with the terms in any order.
    ///
    /// The terms are sorted by symbol, and zero coefficients and unused
    /// `Default` slots are dropped. If more than `max_symbols` terms remain,
    /// the smallest are merged into the residual as in [`Self::reduce`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `center` is NaN or infinite, a symbol belongs to
    ///   another context or appears twice, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn from_terms(
        context: &AffineContext,
        center: T,
        terms: &[AffineTerm<T>],
    ) -> Result<Self, SciOxideError> {
        crate::environment::check()?;
        if !center.has_finite_components() {
            return Err(invalid(
                scioxide_core::argument!("center"),
                "center must be finite",
            ));
        }
        let mut result = Self::zero(context)?;
        let mut residual = Box2::zero()?;
        let mut ordered = try_with_capacity(terms.len(), SCRATCH)?;
        // `AffineTerm::new` admits only finite coefficients, and the symbol-free
        // `Default` slot has a zero coefficient.
        for term in terms {
            let Some(symbol) = &term.symbol else {
                continue;
            };
            if symbol.context != context.data.identity {
                return Err(invalid(
                    scioxide_core::argument!("terms"),
                    "input symbol belongs to another context",
                ));
            }
            ordered.push(*term);
        }
        ordered.sort_unstable_by_key(|term| term.symbol);
        if ordered
            .windows(2)
            .any(|pair| pair[0].symbol == pair[1].symbol)
        {
            return Err(invalid(
                scioxide_core::argument!("terms"),
                "duplicate input symbol",
            ));
        }
        ordered.retain(|term| term.coefficient != T::ZERO);
        let mut selection = try_filled(ordered.len(), AffineTerm::default(), SCRATCH)?;
        let retained = compress(
            &mut ordered,
            &mut selection,
            context.max_symbols(),
            &mut residual,
        )?;
        let discarded = ordered.len() - retained;
        result.terms[..retained].copy_from_slice(&ordered[..retained]);
        result.state = make_state(
            context,
            center,
            residual,
            &result.terms[..retained],
            discarded,
            EnclosureStatus::Defined,
            Decoration::Com,
        )?;
        Ok(result)
    }
    /// Creates `center + Σᵢ coefficients[i] · εᵢ` over the first input symbols.
    ///
    /// ```
    /// use scioxide_validated::{AffineContext, AffineForm, AffineOptions};
    ///
    /// let context = AffineContext::new(2, AffineOptions::new())?;
    /// // x = 1 + ε₀/8 + ε₁/4 ranges over [1 − 3/8, 1 + 3/8].
    /// let x = AffineForm::from_coefficients(&context, 1.0_f64, &[0.125, 0.25])?;
    /// assert_eq!(x.view().terms().len(), 2);
    /// assert_eq!(
    ///     x.report().real().set().bounds(),
    ///     Some((Some(0.625), Some(1.375)))
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are more coefficients than input symbols, or a
    ///   coefficient is NaN or infinite.
    /// - The errors of [`Self::from_terms`].
    pub fn from_coefficients(
        context: &AffineContext,
        center: T,
        coefficients: &[T],
    ) -> Result<Self, SciOxideError> {
        if coefficients.len() > context.data.inputs {
            return Err(invalid(
                scioxide_core::argument!("coefficients"),
                "too many input coefficients",
            ));
        }
        let mut terms = try_with_capacity(
            coefficients.len(),
            scioxide_core::quantity!("affine.coefficients"),
        )?;
        for (i, &coefficient) in coefficients.iter().enumerate() {
            terms.push(AffineTerm::new(context.symbol(i)?, coefficient)?);
        }
        Self::from_terms(context, center, &terms)
    }
    /// Creates a form that ranges over an interval, or over a rectangle in the
    /// complex plane, as its noise symbols range over `[−1, 1]`.
    ///
    /// The real part uses `real_symbol`. The imaginary part needs its own
    /// `imaginary_symbol` unless `imaginary` is a single point. An empty part
    /// gives an enclosure with status `EmptyInput`, and an unbounded part gives
    /// an enclosure without an affine form.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a symbol belongs to another context, the two symbols
    ///   are equal, the imaginary part has positive width but no symbol, or a
    ///   real form has an imaginary part other than zero or empty, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    pub fn from_enclosure(
        context: &AffineContext,
        real: DecoratedInterval<T::Real>,
        imaginary: DecoratedInterval<T::Real>,
        real_symbol: AffineSymbol,
        imaginary_symbol: Option<AffineSymbol>,
    ) -> Result<Self, SciOxideError> {
        crate::environment::check()?;
        let bounds = if T::IS_REAL && imaginary.set().is_empty() {
            Box2::real(DecoratedInterval::empty())?
        } else {
            Box2 {
                re: real,
                im: imaginary,
            }
        };
        if T::IS_REAL && !bounds.im_is_zero() {
            return Err(invalid(
                scioxide_core::argument!("imaginary"),
                "real form has nonzero imaginary input",
            ));
        }
        if real_symbol.context != context.data.identity {
            return Err(invalid(
                scioxide_core::argument!("real_symbol"),
                "invalid rectangle input symbols",
            ));
        }
        if imaginary_symbol
            .as_ref()
            .is_some_and(|s| s.context != context.data.identity || *s == real_symbol)
        {
            return Err(invalid(
                scioxide_core::argument!("imaginary_symbol"),
                "invalid rectangle input symbols",
            ));
        }
        let Some((center, residual)) = bounds.midpoint::<T>()? else {
            let mut result = Self::zero(context)?;
            result.state = fallback_state(
                context,
                bounds,
                if bounds.empty() {
                    EnclosureStatus::EmptyInput
                } else {
                    EnclosureStatus::Defined
                },
                AffineFallback::EnclosureInput,
            )?;
            return Ok(result);
        };
        let radii = residual.radii()?;
        let real_term = AffineTerm::new(real_symbol, T::from_real(radii.0))?;
        let mut result = if radii.1 != T::Real::ZERO {
            let symbol = imaginary_symbol.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("imaginary_symbol"),
                    "imaginary input requires an independent symbol",
                )
            })?;
            let imag_term = AffineTerm::new(symbol, T::from_components(T::Real::ZERO, radii.1))?;
            Self::from_terms(context, center, &[real_term, imag_term])?
        } else {
            Self::from_terms(context, center, &[real_term])?
        };
        result.state.report.real = result
            .state
            .report
            .real
            .with_weaker(real.decoration().min(imaginary.decoration()))?;
        result.state.report.imaginary = result
            .state
            .report
            .imaginary
            .with_weaker(real.decoration().min(imaginary.decoration()))?;
        Ok(result)
    }
    /// Borrows the form, for the `_into` operations of [`AffineView`], which
    /// write to storage you provide and do not allocate.
    ///
    /// ```
    /// use scioxide_validated::{
    ///     AffineContext, AffineForm, AffineOptions, AffineState, AffineTerm, AffineUnary, AffineView,
    ///     AffineViewMut,
    /// };
    ///
    /// let context = AffineContext::new(1, AffineOptions::new().with_max_symbols(4))?;
    /// let mut workspace = context.workspace::<f64>()?;
    /// let x = AffineForm::from_coefficients(&context, 1.0_f64, &[0.125])?;
    /// // Output storage: a state and `max_symbols` term slots.
    /// let mut state = AffineState::zero(&context)?;
    /// let mut terms: [AffineTerm<f64>; 4] = Default::default();
    /// let output = AffineViewMut::new(&mut state, &mut terms)?;
    /// let report = x
    ///     .view()
    ///     .unary_into(AffineUnary::Exp, output, &mut workspace)?;
    /// assert!(report.real().contains(1.0_f64.exp()));
    /// assert_eq!(AffineView::new(&state, &terms)?.report(), report);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn view(&self) -> AffineView<'_, T> {
        AffineView {
            state: &self.state,
            terms: &self.terms,
        }
    }
    /// Borrows the form's storage as an output for the `_into` operations.
    pub fn view_mut(&mut self) -> AffineViewMut<'_, T> {
        AffineViewMut {
            state: &mut self.state,
            terms: &mut self.terms,
        }
    }
    /// Returns the enclosure and diagnostics.
    pub fn report(&self) -> AffineReport<T::Real> {
        self.state.report
    }
    /// Returns `operation` applied to the form.
    ///
    /// Arguments outside the domain of the operation are excluded, as for
    /// [`ElementaryEnclosure`](crate::ElementaryEnclosure), and reported by
    /// [`AffineReport::status`]. Near a domain boundary or branch cut the
    /// result is only an enclosure; see [`AffineFallback`].
    ///
    /// ```
    /// use scioxide_validated::{AffineContext, AffineForm, AffineOptions, AffineUnary};
    ///
    /// let context = AffineContext::new(1, AffineOptions::new())?;
    /// let mut workspace = context.workspace::<f64>()?;
    /// let x = AffineForm::from_coefficients(&context, 1.0_f64, &[0.125])?;
    /// let y = x.unary(AffineUnary::Exp, &mut workspace)?.report().real();
    /// assert!(y.contains(0.875_f64.exp()) && y.contains(1.125_f64.exp()));
    /// let cube = x
    ///     .unary(AffineUnary::Powi(3), &mut workspace)?
    ///     .report()
    ///     .real();
    /// assert!(cube.contains(0.875_f64.powi(3)) && cube.contains(1.125_f64.powi(3)));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result's storage cannot be reserved.
    /// - The errors of [`AffineView::unary_into`].
    pub fn unary(
        &self,
        operation: AffineUnary,
        workspace: &mut AffineWorkspace<T>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::zero(&self.state.context)?;
        self.view()
            .unary_into(operation, output.view_mut(), workspace)?;
        Ok(output)
    }
    /// Returns `operation` applied to `self` and `other`.
    ///
    /// Terms with the same symbol combine, so `x − x` is exactly zero. Domains
    /// are handled as in [`Self::unary`].
    ///
    /// ```
    /// use scioxide_validated::{AffineBinary, AffineContext, AffineForm, AffineOptions};
    ///
    /// let context = AffineContext::new(1, AffineOptions::new())?;
    /// let mut workspace = context.workspace::<f64>()?;
    /// let x = AffineForm::from_coefficients(&context, 1.0_f64, &[0.125])?;
    /// let zero = x.binary(&x, AffineBinary::Sub, &mut workspace)?;
    /// assert_eq!(
    ///     zero.report().real().set().bounds(),
    ///     Some((Some(0.0), Some(0.0)))
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result's storage cannot be reserved.
    /// - The errors of [`AffineView::binary_into`].
    pub fn binary(
        &self,
        other: &Self,
        operation: AffineBinary,
        workspace: &mut AffineWorkspace<T>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::zero(&self.state.context)?;
        self.view()
            .binary_into(other.view(), operation, output.view_mut(), workspace)?;
        Ok(output)
    }
    /// Returns the form with at most `max_symbols` terms, merging the smallest
    /// into the residual; see [`AffineView::reduce_into`].
    ///
    /// The enclosure stays valid, and [`AffineReport::discarded_symbols`] and
    /// [`AffineReport::correlation_lost`] report the merged terms.
    ///
    /// ```
    /// use scioxide_validated::{AffineContext, AffineForm, AffineOptions};
    ///
    /// let context = AffineContext::new(3, AffineOptions::new())?;
    /// let mut workspace = context.workspace::<f64>()?;
    /// let x = AffineForm::from_coefficients(&context, 0.0_f64, &[1.0, 0.5, 0.25])?;
    /// let reduced = x.reduce(1, &mut workspace)?.report();
    /// assert_eq!(
    ///     (reduced.retained_symbols(), reduced.discarded_symbols()),
    ///     (1, 2)
    /// );
    /// assert!(reduced.correlation_lost());
    /// assert!(reduced.real().contains_interval(&x.report().real()));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result's storage cannot be reserved.
    /// - The errors of [`AffineView::reduce_into`].
    pub fn reduce(
        &self,
        max_symbols: usize,
        workspace: &mut AffineWorkspace<T>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::zero(&self.state.context)?;
        self.view()
            .reduce_into(max_symbols, output.view_mut(), workspace)?;
        Ok(output)
    }
    /// Returns the real or imaginary part as a real form with the same symbols.
    ///
    /// The part's coefficients are copied, and terms whose coefficient in that
    /// part is zero are dropped. No rounding occurs and no correlation is lost.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_validated::{AffineContext, AffineForm, AffineOptions, AffineProjection};
    ///
    /// let context = AffineContext::new(0, AffineOptions::new())?;
    /// let z = AffineForm::constant(&context, Complex::new(3.0_f64, 4.0))?;
    /// let real = z.project(AffineProjection::Real)?;
    /// assert!(real.report().real().contains(3.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result's storage cannot be reserved.
    /// - The errors of [`AffineView::project_into`].
    pub fn project(
        &self,
        projection: AffineProjection,
    ) -> Result<AffineForm<T::Real>, SciOxideError> {
        let mut output = AffineForm::zero(&self.state.context)?;
        self.view().project_into(projection, output.view_mut())?;
        Ok(output)
    }
}

/// Part of a complex affine form selected by [`AffineForm::project`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineProjection {
    /// Select the real component.
    Real,
    /// Select the imaginary component; zero for a real form.
    Imaginary,
}
impl AffineProjection {
    fn scalar<T: NumericScalar>(self, value: T) -> T::Real {
        match self {
            Self::Real => value.real_component(),
            Self::Imaginary => value.imaginary_component(),
        }
    }
    fn interval<S: RealScalar>(
        self,
        real: DecoratedInterval<S>,
        imaginary: DecoratedInterval<S>,
    ) -> DecoratedInterval<S> {
        match self {
            Self::Real => real,
            Self::Imaginary => imaginary,
        }
    }
}

/// Unary operation of [`AffineForm::unary`].
///
/// Each function is defined on the real numbers for real forms and on the
/// complex plane for complex forms, with principal branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineUnary {
    /// Additive inverse.
    Neg,
    /// Complex conjugation; identity for real forms.
    Conj,
    /// Square `x²`, which is not the product of two independent copies of `x`.
    Square,
    /// Reciprocal on the nonzero domain.
    Reciprocal,
    /// Principal square root (nonnegative real domain for real forms).
    Sqrt,
    /// Real cube root or principal complex cube root.
    Cbrt,
    /// Natural exponential.
    Exp,
    /// Base-two exponential.
    Exp2,
    /// Base-ten exponential.
    Exp10,
    /// Exponential minus one.
    ExpM1,
    /// Principal natural logarithm.
    Ln,
    /// Base-two logarithm.
    Log2,
    /// Base-ten logarithm.
    Log10,
    /// Principal logarithm of one plus the argument.
    Ln1p,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Principal inverse sine.
    Asin,
    /// Principal inverse cosine.
    Acos,
    /// Principal inverse tangent.
    Atan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Principal inverse hyperbolic sine.
    Asinh,
    /// Principal inverse hyperbolic cosine.
    Acosh,
    /// Principal inverse hyperbolic tangent.
    Atanh,
    /// Integer power `xⁿ`, with `0⁰ = 1`, for any `i32` exponent.
    Powi(i32),
}

/// Binary operation of [`AffineForm::binary`]. `Atan2` and `Hypot` accept only
/// real forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineBinary {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division on the nonzero denominator domain.
    Div,
    /// Angle `atan2(left, right)` of the point `(right, left)`, in `(−π, π]`.
    Atan2,
    /// Euclidean norm `√(left² + right²)`.
    Hypot,
    /// Power `leftʳⁱᵍʰᵗ`, real or principal complex; an exponent that is an
    /// exact integer gives an integer power.
    Pow,
    /// Logarithm of `left` to base `right`.
    Log,
}
/// An evaluator operation; `Identity` copies and recompresses for `reduce`.
#[derive(Clone, Copy)]
enum Operation {
    Identity,
    Unary(AffineUnary),
    Binary(AffineBinary),
}

/// Scratch storage and a supply of error symbols for affine operations, from
/// [`AffineContext::workspace`].
///
/// Each workspace adds error symbols that no other workspace uses, and never
/// reuses one. Results stay valid after the workspace is dropped.
pub struct AffineWorkspace<T: NumericScalar> {
    context: AffineContext,
    namespace: usize,
    selection: Vec<AffineTerm<T>>,
    next: usize,
    scratch: Vec<AffineTerm<T>>,
}
impl<T: NumericScalar> AffineWorkspace<T> {
    /// Evaluate `op`, which reports failures.
    fn run(
        &mut self,
        a: AffineView<'_, T>,
        b: Option<AffineView<'_, T>>,
        op: Operation,
        output: AffineViewMut<'_, T>,
    ) -> Result<AffineReport<T::Real>, SciOxideError> {
        self.run_limited(a, b, op, output, self.context.max_symbols())
    }
    /// Evaluate `op` within `max_symbols`.
    /// `a` is its `self` operand and `b` its `other` operand.
    fn run_limited(
        &mut self,
        a: AffineView<'_, T>,
        b: Option<AffineView<'_, T>>,
        op: Operation,
        output: AffineViewMut<'_, T>,
        max_symbols: usize,
    ) -> Result<AffineReport<T::Real>, SciOxideError> {
        crate::environment::check()?;
        if max_symbols > self.context.max_symbols() {
            return Err(invalid(
                scioxide_core::argument!("max_symbols"),
                "requested reduction exceeds the context symbol limit",
            ));
        }
        if !a.state.context.same(&self.context)
            || !output.state.context.same(&self.context)
            || b.is_some_and(|v| !v.state.context.same(&self.context))
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "affine contexts do not match",
            ));
        }
        at_least(self.context.max_symbols(), output.terms.len())?;
        let mut candidate = evaluate(a, b, op, &mut self.scratch)?;
        let mut next = self.next;
        if candidate.fallback.is_none() {
            // Every new component error gets a fresh identity, so copies can
            // cancel it while independent evaluations cannot accidentally do so.
            let radii = candidate.error.radii()?;
            for coefficient in [
                T::from_real(radii.0),
                T::from_components(T::Real::ZERO, radii.1),
            ] {
                if coefficient != T::ZERO {
                    let index = next;
                    next = next.checked_add(1).ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!(
                            "affine_noise_identifier"
                        ))
                    })?;
                    self.scratch[candidate.len] = AffineTerm {
                        symbol: Some(AffineSymbol {
                            context: self.context.data.identity,
                            namespace: self.namespace,
                            index,
                        }),
                        coefficient,
                    };
                    // Insert at most two fresh errors into the canonical stream.
                    let position = self.scratch[..candidate.len]
                        .partition_point(|term| term.symbol < self.scratch[candidate.len].symbol);
                    self.scratch[position..=candidate.len].rotate_right(1);
                    candidate.len += 1;
                }
            }
        }
        let mut state = if let Some(reason) = candidate.fallback {
            fallback_state(&self.context, candidate.range, candidate.status, reason)?
        } else {
            // Stage compression inside scratch. The caller output remains intact
            // until all directed arithmetic and identity checks have succeeded.
            let mut residual = candidate.residual;
            let n = compress(
                &mut self.scratch[..candidate.len],
                &mut self.selection,
                max_symbols,
                &mut residual,
            )?;
            let discarded = candidate.len - n;
            make_state(
                &self.context,
                candidate.center,
                residual,
                &self.scratch[..n],
                discarded,
                candidate.status,
                candidate.decoration,
            )?
        };
        state.report.correlation_lost |=
            a.report().correlation_lost || b.is_some_and(|value| value.report().correlation_lost);
        let n = state.report.retained_symbols;
        for (out, term) in output.terms.iter_mut().zip(&self.scratch[..n]) {
            *out = *term;
        }
        // Clear unused caller slots as part of the commit.
        for out in &mut output.terms[n..] {
            *out = AffineTerm::default();
        }
        *output.state = state;
        self.next = next;
        Ok(output.state.report)
    }
}

fn magnitude<T: NumericScalar>(value: T) -> T::Real {
    value
        .real_component()
        .abs()
        .max(value.imaginary_component().abs())
}
impl<T: NumericScalar> AffineTerm<T> {
    fn coefficient_box(&self) -> Result<Box2<T::Real>, SciOxideError> {
        Box2::symmetric(
            self.coefficient.real_component().abs(),
            self.coefficient.imaginary_component().abs(),
        )
    }
}
/// Select the largest magnitudes, keeping the smallest symbol on ties.
/// Selection works in a separate buffer; compaction keeps the symbol order.
fn compress<T: NumericScalar>(
    terms: &mut [AffineTerm<T>],
    selection: &mut [AffineTerm<T>],
    max: usize,
    residual: &mut Box2<T::Real>,
) -> Result<usize, SciOxideError> {
    if terms.len() <= max {
        return Ok(terms.len());
    }
    let compare = |a: &AffineTerm<T>, b: &AffineTerm<T>| {
        let x = magnitude(a.coefficient);
        let y = magnitude(b.coefficient);
        (if x > y {
            core::cmp::Ordering::Less
        } else if x < y {
            core::cmp::Ordering::Greater
        } else {
            core::cmp::Ordering::Equal
        })
        .then_with(|| a.symbol.cmp(&b.symbol))
    };
    selection[..terms.len()].copy_from_slice(terms);
    let (_, cutoff, _) = selection[..terms.len()].select_nth_unstable_by(max, compare);
    let cutoff = *cutoff;
    let mut n = 0;
    for i in 0..terms.len() {
        let term = terms[i];
        if compare(&term, &cutoff).is_lt() {
            terms[n] = term;
            n += 1;
        } else {
            *residual = residual.add(term.coefficient_box()?)?;
        }
    }
    Ok(n)
}
fn make_state<T: NumericScalar>(
    context: &AffineContext,
    center: T,
    residual: Box2<T::Real>,
    terms: &[AffineTerm<T>],
    discarded: usize,
    status: EnclosureStatus,
    decoration: Decoration,
) -> Result<AffineState<T>, SciOxideError> {
    let mut radius = residual;
    if T::IS_REAL {
        for term in terms {
            radius.re = radius.re.add_admitted(&term.coefficient_box()?.re)?;
        }
    } else {
        for term in terms {
            radius = radius.add(term.coefficient_box()?)?;
        }
    }
    let range = Box2::point(center)?.add(radius)?;
    let range = range.decorate(if status == EnclosureStatus::Defined {
        decoration
    } else {
        Decoration::Trv
    })?;
    // A nonfinite accumulated radius cannot become bounded by adding a finite
    // center. Checking the complete enclosure also catches center/radius overflow.
    if !range.finite() {
        let mut state = fallback_state(context, range, status, AffineFallback::Unbounded)?;
        state.report.discarded_symbols = discarded;
        return Ok(state);
    }
    Ok(AffineState {
        context: context.clone(),
        center,
        residual,
        report: AffineReport {
            real: range.re,
            imaginary: range.im,
            status,
            retained_symbols: terms.len(),
            discarded_symbols: discarded,
            correlation_lost: !residual.is_zero(),
            fallback: None,
        },
    })
}
fn fallback_state<T: NumericScalar>(
    context: &AffineContext,
    range: Box2<T::Real>,
    status: EnclosureStatus,
    reason: AffineFallback,
) -> Result<AffineState<T>, SciOxideError> {
    let mut state = AffineState::zero(context)?;
    state.report = AffineReport {
        real: range.re,
        imaginary: range.im,
        status,
        retained_symbols: 0,
        discarded_symbols: 0,
        correlation_lost: true,
        fallback: Some(reason),
    };
    Ok(state)
}

#[cfg(test)]
mod tests;
