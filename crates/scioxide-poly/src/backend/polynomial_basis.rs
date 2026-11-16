//! Divided differences, barycentric weights, Newton evaluation and basis
//! conversion.
//!
//! Functions that return a `Vec` never change their input slices. The Newton
//! folds take the arithmetic as callbacks, so they work for real, complex,
//! exact or extended-range values. They cannot check those values themselves:
//! the callbacks are responsible for detecting non-finite values and overflow.
//! A callback error is returned at once, and a panic in a callback is not
//! caught.

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{AlgebraicScalar, NumericScalar, Scaled, ScaledValue};
use scioxide_core::{
    InvalidInputReason, NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError,
};

/// Nodes that have been checked to be finite, distinct and nonempty.
#[derive(Clone, Copy)]
pub(crate) struct ValidatedNodes<'a, C>(&'a [C]);

impl<'a, C: NumericScalar> ValidatedNodes<'a, C> {
    /// Reborrow the invariant of an immutable, privately constructed basis.
    pub(crate) fn prepared(nodes: &'a [C]) -> Self {
        Self(nodes)
    }
    pub(crate) fn new(nodes: &'a [C]) -> Result<Self, SciOxideError> {
        if nodes.is_empty() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("nodes"),
                InvalidInputReason::Empty,
                "at least one node is required",
            ));
        }
        if nodes.iter().any(|node| !node.has_finite_components()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "interpolation nodes must be finite",
            ));
        }
        // Ordered real data needs only adjacent comparisons. An unordered or
        // complex set uses the same validator's pairwise distinctness check.
        let ordered = C::IS_REAL
            && (nodes
                .windows(2)
                .all(|pair| pair[0].real_component() < pair[1].real_component())
                || nodes
                    .windows(2)
                    .all(|pair| pair[0].real_component() > pair[1].real_component()));
        if !ordered
            && nodes
                .iter()
                .enumerate()
                .any(|(index, node)| nodes[..index].contains(node))
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("nodes"),
                InvalidInputReason::Inconsistent,
                "nodes must be distinct",
            ));
        }
        Ok(Self(nodes))
    }
}

/// Computes the divided difference
/// `(left_numerator − right_numerator) / (left_denominator − right_denominator)`.
///
/// The operands must be finite and the denominators distinct. The quotient is
/// first formed in working precision. A finite, normal result (or an exact zero
/// numerator) needs no further check. Otherwise the operands are checked, and
/// if a difference or the quotient overflows or underflows, the computation is
/// repeated with a wider exponent range and only the quotient is rounded back.
/// The result is accurate to working precision but not correctly rounded.
/// Subnormal results are kept. The function does not allocate.
///
/// # Errors
///
/// - `NonFinite` if an operand is NaN or infinite, the denominators are
///   equal, or the quotient is not representable.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::linear_difference_ratio;
///
/// // (5 − 1) / (2 − 0) = 2.
/// assert!((linear_difference_ratio(5.0_f64, 1.0, 2.0, 0.0)? - 2.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn linear_difference_ratio<S: RealScalar>(
    left_numerator: S,
    right_numerator: S,
    left_denominator: S,
    right_denominator: S,
) -> Result<S, SciOxideError> {
    crate::admission::difference_quotient::<S, false>(
        left_numerator,
        right_numerator,
        left_denominator,
        right_denominator,
    )
    .map_or_else(
        || {
            checked_difference_ratio(
                left_numerator,
                right_numerator,
                left_denominator,
                right_denominator,
            )
        },
        Ok,
    )
}

/// Validated extended-range path of [`linear_difference_ratio`].
#[cold]
#[inline(never)]
fn checked_difference_ratio<S: RealScalar>(
    left_numerator: S,
    right_numerator: S,
    left_denominator: S,
    right_denominator: S,
) -> Result<S, SciOxideError> {
    crate::util::validate_slice(&[
        left_numerator,
        right_numerator,
        left_denominator,
        right_denominator,
    ])?;
    if left_denominator == right_denominator {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "linear basis denominator is zero",
        ));
    }
    extended_difference_ratio(
        ScaledValue::from_value(left_numerator)?,
        ScaledValue::from_value(right_numerator)?,
        left_denominator,
        right_denominator,
    )?
    .round()
    .map_err(|error| match error.view() {
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Overflow,
            ..
        } => SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "linear basis quotient exceeds the finite range",
        ),
        _ => error,
    })
}

fn extended_difference_ratio<C: NumericScalar>(
    a: ScaledValue<C>,
    b: ScaledValue<C>,
    x: C,
    y: C,
) -> Result<ScaledValue<C>, SciOxideError> {
    a.sub(b)?
        .div(ScaledValue::from_value(x)?.sub(ScaledValue::from_value(y)?)?)
}

pub(crate) enum DifferenceCoefficients<C: NumericScalar> {
    Native(Vec<C>),
    Extended(Vec<ScaledValue<C>>),
}

impl<C: NumericScalar> DifferenceCoefficients<C> {
    pub(crate) fn into_native(self) -> Result<Vec<C>, SciOxideError> {
        match self {
            Self::Native(values) => Ok(values),
            Self::Extended(values) => crate::util::try_collect(
                values.into_iter().map(|value| value.try_value()),
                scioxide_core::quantity!("newton_coefficients"),
            ),
        }
    }

    pub(crate) fn into_scaled(self) -> Result<Vec<ScaledValue<C>>, SciOxideError> {
        match self {
            Self::Extended(values) => Ok(values),
            Self::Native(values) => crate::util::try_collect(
                values.into_iter().map(ScaledValue::from_value),
                scioxide_core::quantity!("extended_newton_coefficients"),
            ),
        }
    }
}

/// One arithmetic policy for numeric interpolation, conversion and degree
/// classification. Real quotients are native while the crate's
/// difference-quotient range check accepts them.
/// On range loss the table is promoted to extended exponents before the failed
/// update is committed, and resumes there: preceding native orders need neither
/// copying nor replay. Complex tables are extended from the first update.
pub(crate) fn numeric_divided_differences<C: NumericScalar>(
    nodes: ValidatedNodes<'_, C>,
    coefficients: Vec<C>,
) -> Result<DifferenceCoefficients<C>, SciOxideError> {
    let nodes = nodes.0;
    if coefficients.len() != nodes.len() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(nodes.len()),
            scioxide_core::Dimensions::one(coefficients.len()),
        ));
    }
    let mut coefficients = coefficients;
    let native = if crate::admission::finite_gaps(nodes) {
        native_scalar_table::<C, true>(nodes, &mut coefficients)
    } else {
        native_scalar_table::<C, false>(nodes, &mut coefficients)
    };
    let Err(failed) = native else {
        return Ok(DifferenceCoefficients::Native(coefficients));
    };
    let mut coefficients = crate::util::try_collect(
        coefficients.into_iter().map(ScaledValue::from_value),
        scioxide_core::quantity!("extended_newton_coefficients"),
    )?;
    let resume = Resume {
        component: failed.component,
        order: failed.order,
        index: failed.index,
        previous: ScaledValue::from_value(failed.previous)?,
    };
    difference_table(
        &mut coefficients,
        1,
        1,
        Some(resume),
        |current, previous, index, other| {
            extended_difference_ratio(current, previous, nodes[index], nodes[other])
        },
    )
    .map_err(|(_, error)| error)?;
    Ok(DifferenceCoefficients::Extended(coefficients))
}

/// The native pass of [`numeric_divided_differences`], stopping at the first
/// entry that fails its range check.
#[inline]
fn native_scalar_table<C: NumericScalar, const FINITE_GAPS: bool>(
    nodes: &[C],
    coefficients: &mut [C],
) -> Result<(), Resume<C>> {
    difference_table(
        coefficients,
        1,
        1,
        None,
        |current, previous, index, other| {
            crate::admission::difference_quotient::<C, FINITE_GAPS>(
                current,
                previous,
                nodes[index],
                nodes[other],
            )
            .ok_or(())
        },
    )
    .map_err(|(resume, ())| resume)
}

/// The first uncommitted update of an interrupted Newton table: row `index` of
/// difference order `order` in `component`, with `previous` the order
/// `order - 1` entry of row `index - 1` that the update reads.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Resume<T> {
    component: usize,
    order: usize,
    index: usize,
    previous: T,
}

/// The one Newton divided-difference loop nest, shared by every table form.
///
/// `entries` holds node-major rows of `width` independent components. Each
/// component is an independent table: order `k` updates rows `k..len` in
/// ascending order, carrying the not-yet-updated entry of the row below, so
/// every update reads order `k - 1` values. `ratio(current, previous, i, i - k)`
/// returns the order-`k` entry of row `i` from its order-`k - 1` entry, the
/// order-`k - 1` entry of row `i - 1` and the node indices of the difference. Orders start at `first_order` (one for
/// a complete table), or at `resume`, the first uncommitted update of an
/// interrupted pass, so a promoted table neither copies nor replays committed
/// updates. A zero width has no updates. The forward traversal keeps the
/// single-component loop a first-order recurrence that compilers vectorize.
///
/// # Errors
/// Returns the failing update, uncommitted, as a resume point with the
/// callback error; every other entry holds its last committed order.
#[inline(always)]
fn difference_table<T: Copy, E>(
    entries: &mut [T],
    width: usize,
    first_order: usize,
    resume: Option<Resume<T>>,
    mut ratio: impl FnMut(T, T, usize, usize) -> Result<T, E>,
) -> Result<(), (Resume<T>, E)> {
    let Some(len) = entries.len().checked_div(width) else {
        return Ok(());
    };
    let (start_component, start_order) =
        resume.map_or((0, first_order), |resume| (resume.component, resume.order));
    for component in start_component..width {
        let orders = if component == start_component {
            start_order
        } else {
            first_order
        };
        for order in orders..len {
            let (first, mut previous) = match resume {
                Some(resume) if (resume.component, resume.order) == (component, order) => {
                    (resume.index, resume.previous)
                }
                _ => (order, entries[(order - 1) * width + component]),
            };
            let column = entries[first * width + component..]
                .iter_mut()
                .step_by(width);
            for (index, slot) in (first..len).zip(column) {
                let current = *slot;
                *slot = ratio(current, previous, index, index - order).map_err(|error| {
                    (
                        Resume {
                            component,
                            order,
                            index,
                            previous,
                        },
                        error,
                    )
                })?;
                previous = current;
            }
        }
    }
    Ok(())
}

/// Computes `coefficient · (left − right)` for finite operands.
///
/// A zero coefficient gives `+0` without forming the difference. If the
/// difference overflows, both operands are first divided by a common scale.
/// The product is rounded as `coefficient · (left − right)` in working
/// precision would be; it is not correctly rounded. Subnormal results,
/// including underflow to a signed zero, are returned as is. The function does
/// not allocate.
///
/// # Errors
///
/// - `NonFinite` if an operand is NaN or infinite (also when `coefficient` is
///   zero), or the result is not representable.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::linear_multiply_difference;
///
/// // 0.5 · (MAX − (−MAX)) = MAX, although the difference overflows.
/// assert_eq!(linear_multiply_difference(0.5_f64, f64::MAX, -f64::MAX)?, f64::MAX);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn linear_multiply_difference<S: RealScalar>(
    coefficient: S,
    left: S,
    right: S,
) -> Result<S, SciOxideError> {
    crate::util::validate_slice(&[coefficient, left, right])?;
    if coefficient == S::ZERO {
        return Ok(S::ZERO);
    }
    let direct_difference = left - right;
    if direct_difference.is_finite() {
        return crate::util::finite_intermediate(coefficient * direct_difference, INTERMEDIATE);
    }
    let scale = left.abs().max(right.abs()).max(S::ONE);
    let normalized = normalized_linear_difference(left, right, scale)?;
    let scaled_coefficient = coefficient * scale;
    if scaled_coefficient.is_finite() {
        crate::util::finite_intermediate(scaled_coefficient * normalized, INTERMEDIATE)
    } else {
        crate::util::finite_intermediate(
            crate::util::finite_intermediate(coefficient * normalized, INTERMEDIATE)? * scale,
            INTERMEDIATE,
        )
    }
}

/// Evaluates a Newton-form polynomial and its first derivative at `point` by
/// a backward recurrence (nested multiplication).
///
/// For coefficients `c₀, …, cₙ` and nodes `x₀, …, x₍ₙ₋₁₎` the polynomial is
/// `p(x) = c₀ + (x − x₀)(c₁ + (x − x₁)(c₂ + …))`. `terms` yields
/// `(cⱼ, xⱼ)` for `j < n` in ascending order, and `leading` is `cₙ`. An empty
/// iterator is the constant polynomial and makes no callback calls. `zero` is
/// the additive identity. `multiply_difference(a, x, y)` returns `a · (x − y)`
/// and `add(a, b)` returns `a + b`, both with any checks you need for `T`. The
/// fold makes two calls of each per term, O(n) work in all, and does not
/// allocate.
///
/// # Errors
///
/// Returns the first error from the iterator or a callback.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::newton_value_and_derivative_from_iter;
///
/// // p(x) = 1 + x(1 + (x − 1)) = 1 + x², so p(3) = 10 and p′(3) = 6.
/// let terms = [(1.0_f64, 0.0), (1.0, 1.0)].into_iter().map(Ok);
/// let (value, derivative) = newton_value_and_derivative_from_iter(
///     terms,
///     1.0,
///     3.0,
///     0.0,
///     |a, x, y| Ok(a * (x - y)),
///     |a, b| Ok(a + b),
/// )?;
/// assert!((value - 10.0).abs() < 1e-13 && (derivative - 6.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn newton_value_and_derivative_from_iter<T: Copy>(
    terms: impl DoubleEndedIterator<Item = Result<(T, T), SciOxideError>>,
    leading: T,
    point: T,
    zero: T,
    mut multiply_difference: impl FnMut(T, T, T) -> Result<T, SciOxideError>,
    mut add: impl FnMut(T, T) -> Result<T, SciOxideError>,
) -> Result<(T, T), SciOxideError> {
    // Horner on the Newton form, differentiated alongside:
    // p = c_k + (x - x_k) q and p' = q + (x - x_k) q', evaluated from the
    // leading coefficient downward.
    terms
        .rev()
        .try_fold((leading, zero), |(value, derivative), term| {
            let (coefficient, node) = term?;
            let next_derivative = add(multiply_difference(derivative, point, node)?, value)?;
            let next_value = add(multiply_difference(value, point, node)?, coefficient)?;
            Ok((next_value, next_derivative))
        })
}

/// Evaluates a Newton-form polynomial at `point`, without the derivative.
///
/// The arguments are those of [`newton_value_and_derivative_from_iter`]
/// without `zero`. The fold makes one call of each callback per term and does
/// not allocate.
///
/// # Errors
///
/// Returns the first error from the iterator or a callback.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::newton_value_from_iter;
///
/// // p(x) = 1 + x(1 + (x − 1)) = 1 + x², so p(3) = 10.
/// let terms = [(1.0_f64, 0.0), (1.0, 1.0)].into_iter().map(Ok);
/// let value = newton_value_from_iter(terms, 1.0, 3.0, |a, x, y| Ok(a * (x - y)), |a, b| Ok(a + b))?;
/// assert_eq!(value, 10.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn newton_value_from_iter<T: Copy>(
    terms: impl DoubleEndedIterator<Item = Result<(T, T), SciOxideError>>,
    leading: T,
    point: T,
    mut multiply_difference: impl FnMut(T, T, T) -> Result<T, SciOxideError>,
    mut add: impl FnMut(T, T) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    terms.rev().try_fold(leading, |value, term| {
        let (coefficient, node) = term?;
        add(multiply_difference(value, point, node)?, coefficient)
    })
}

/// Expands a Newton-form polynomial into power coefficients of a mapped
/// variable.
///
/// `nodes` holds `x_0..x_{n-1}` and `coefficients` yields `c_0..c_n` in
/// ascending Newton order, so `p = c_0 + (x - x_0)(c_1 + (x - x_1)(...))`.
/// `power` receives `n + 1` coefficients in ascending powers of the caller's
/// variable `u`, where `x = centre + scale * u`. Callbacks own the map:
/// `raise(q)` returns `scale * q`, and `shift(addend, q, node)` returns
/// `addend + (centre - node) * q`. The identity map (`raise = Ok`,
/// `shift = addend - node * q`) gives ordinary monomial coefficients.
/// Coefficients are expanded from the innermost term outward and updated from
/// the top down, so `power` is the only storage. The expansion makes O(n²)
/// callback calls and does not allocate. On error the contents of `power` are
/// unspecified.
///
/// # Errors
///
/// - `DimensionMismatch` unless `power.len() == nodes.len() + 1` and
///   `coefficients` yields exactly that many terms.
/// - The first error from the iterator or a callback.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::newton_to_power_into;
///
/// // p = 1 + (x − 1)(2 + 3(x − 2)) = 3x² − 7x + 5.
/// let mut power = [0.0_f64; 3];
/// let coefficients = [1.0, 2.0, 3.0].into_iter().map(Ok);
/// newton_to_power_into(&[1.0, 2.0], coefficients, &mut power, Ok, |a, q, x| Ok(a - x * q))?;
/// assert_eq!(power, [5.0, -7.0, 3.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn newton_to_power_into<N: Copy, T: Copy>(
    nodes: &[N],
    coefficients: impl DoubleEndedIterator<Item = Result<T, SciOxideError>>,
    power: &mut [T],
    mut raise: impl FnMut(T) -> Result<T, SciOxideError>,
    mut shift: impl FnMut(T, T, N) -> Result<T, SciOxideError>,
) -> Result<(), SciOxideError> {
    let terms = nodes.len() + 1;
    if power.len() != terms {
        return Err(SciOxideError::dimension_mismatch([terms], [power.len()]));
    }
    let mut count = 0;
    for coefficient in coefficients.rev() {
        let coefficient = coefficient?;
        if count == terms {
            return Err(SciOxideError::dimension_mismatch([terms], [count + 1]));
        }
        if count == 0 {
            power[0] = coefficient;
        } else {
            // Multiplying q(u) by (scale * u + centre - x_i) gives
            // new[k] = scale * q[k - 1] + (centre - x_i) * q[k].
            let degree = count - 1;
            let node = nodes[nodes.len() - count];
            power[degree + 1] = raise(power[degree])?;
            for k in (1..=degree).rev() {
                power[k] = shift(raise(power[k - 1])?, power[k], node)?;
            }
            power[0] = shift(coefficient, power[0], node)?;
        }
        count += 1;
    }
    if count == terms {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch([terms], [count]))
    }
}

/// Converts Bernstein coefficients on `[0, 1]` into power coefficients in
/// ascending order.
///
/// Uses `Σ_j b_j C(n,j) s^j (1-s)^(n-j) = Σ_r C(n,r) (Δ^r b)_0 s^r` with
/// in-place forward differences and the binomial recurrence
/// `C(n, r+1) = C(n, r) (n - r) / (r + 1)`. `differences` holds `b_0..b_n` on
/// entry and is overwritten; `power` receives the `n + 1` coefficients.
/// Callbacks own arithmetic: `sub(a, b) = a - b`, `scale(value, binomial)` and
/// `ratio(binomial, numerator, denominator) = binomial * numerator / denominator`;
/// `one` is the initial binomial. The conversion makes O(n²) callback calls and
/// does not allocate. On error the contents of both slices are unspecified.
///
/// # Errors
///
/// - `DimensionMismatch` if the slices differ in length or are empty.
/// - The first error from a callback.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::bernstein_to_power_into;
///
/// // b = [1, 2, 4] is 1 + 2s + s².
/// let mut differences = [1.0_f64, 2.0, 4.0];
/// let mut power = [0.0; 3];
/// bernstein_to_power_into(
///     &mut differences,
///     &mut power,
///     1.0_f64,
///     |a, b| Ok(a - b),
///     |v, c| Ok(v * c),
///     |c, n, d| Ok(c * n as f64 / d as f64),
/// )?;
/// assert_eq!(power, [1.0, 2.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn bernstein_to_power_into<T: Copy, B: Copy>(
    differences: &mut [T],
    power: &mut [T],
    one: B,
    mut sub: impl FnMut(T, T) -> Result<T, SciOxideError>,
    mut scale: impl FnMut(T, B) -> Result<T, SciOxideError>,
    mut ratio: impl FnMut(B, usize, usize) -> Result<B, SciOxideError>,
) -> Result<(), SciOxideError> {
    if differences.is_empty() || differences.len() != power.len() {
        return Err(SciOxideError::dimension_mismatch(
            [differences.len().max(1)],
            [power.len()],
        ));
    }
    let degree = differences.len() - 1;
    let mut binomial = one;
    for (order, slot) in power.iter_mut().enumerate() {
        *slot = scale(differences[0], binomial)?;
        for index in 0..degree - order {
            differences[index] = sub(differences[index + 1], differences[index])?;
        }
        if order < degree {
            binomial = ratio(binomial, degree - order, order + 1)?;
        }
    }
    Ok(())
}

/// Computes the Newton coefficients of the Hermite interpolant of values and
/// first derivatives, by divided differences on doubled nodes.
///
/// `nodes` are finite and distinct; `values` and `slopes` hold `dimension`
/// components per node. `coefficients` receives the Newton coefficients on the
/// doubled nodes `[x_0, x_0, x_1, x_1, ...]`, node-major with `2 n dimension`
/// entries. Each confluent pair replaces its `0/0` first difference by the
/// slope; every other entry uses [`linear_difference_ratio`] in the shared
/// table of [`real_divided_differences_into`]. This takes O(n² · dimension)
/// work and does not allocate. On error the contents of `coefficients` are
/// unspecified.
///
/// # Errors
///
/// - `InvalidInput` if `nodes` is empty or has duplicates.
/// - `NonFinite` if a node or sample is NaN or infinite, or a difference is
///   not representable.
/// - `DimensionMismatch` if a slice length does not match `nodes.len()` and
///   `dimension`.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::confluent_divided_differences_into;
///
/// // x² from values [0, 1] and slopes [0, 2] at nodes [0, 1].
/// let mut coefficients = [0.0_f64; 4];
/// let (nodes, values, slopes) = ([0.0, 1.0], [0.0, 1.0], [0.0, 2.0]);
/// confluent_divided_differences_into(&nodes, &values, &slopes, 1, &mut coefficients)?;
/// assert_eq!(coefficients, [0.0, 0.0, 1.0, 0.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn confluent_divided_differences_into<S: RealScalar>(
    nodes: &[S],
    values: &[S],
    slopes: &[S],
    dimension: usize,
    coefficients: &mut [S],
) -> Result<(), SciOxideError> {
    let nodes = ValidatedNodes::new(nodes)?.0;
    let samples = scioxide_core::backend::checked_mul(
        nodes.len(),
        dimension,
        scioxide_core::quantity!("hermite_samples"),
    )?;
    let expected = scioxide_core::backend::checked_mul(
        samples,
        2,
        scioxide_core::quantity!("hermite_coefficients"),
    )?;
    for (actual, wanted) in [
        (values.len(), samples),
        (slopes.len(), samples),
        (coefficients.len(), expected),
    ] {
        if actual != wanted {
            return Err(SciOxideError::dimension_mismatch([wanted], [actual]));
        }
    }
    crate::util::validate_slice(values)?;
    crate::util::validate_slice(slopes)?;
    if dimension == 0 {
        return Ok(());
    }
    // First order: a confluent pair's 0/0 difference is the supplied slope;
    // adjacent pairs use the ordinary quotient of their sample values.
    for (node, pair) in coefficients.chunks_exact_mut(2 * dimension).enumerate() {
        let row = node * dimension..(node + 1) * dimension;
        let (value, slope) = pair.split_at_mut(dimension);
        slope.copy_from_slice(&slopes[row.clone()]);
        if node == 0 {
            value.copy_from_slice(&values[row]);
        } else {
            for ((entry, &right), &left) in value
                .iter_mut()
                .zip(&values[row.clone()])
                .zip(&values[row.start - dimension..row.start])
            {
                *entry = linear_difference_ratio(right, left, nodes[node], nodes[node - 1])?;
            }
        }
    }
    difference_table(
        coefficients,
        dimension,
        2,
        None,
        |current, previous, index, other| {
            linear_difference_ratio(current, previous, nodes[index / 2], nodes[other / 2])
        },
    )
    .map_err(|(_, error)| error)
}

/// Replaces real samples by their Newton divided differences, in place.
///
/// `coefficients` holds `dimension` components per node, node by node. Each
/// component is an independent table over the same nodes, and all are updated
/// in one pass of O(n² · dimension) work. On return, entry `j` of a component
/// is the divided difference `f[x₀, …, xⱼ]`. When the nodes and samples are
/// known in advance to keep all quotients in range, no per-entry check is
/// made. Otherwise each quotient is checked, and one that overflows or
/// underflows is recomputed with a wider exponent range. Both paths give the
/// same values. The function does not allocate. On error the contents of
/// `coefficients` may have changed.
///
/// # Errors
///
/// - `InvalidInput` if `nodes` is empty or has duplicates.
/// - `NonFinite` if a node or sample is NaN or infinite, or a divided
///   difference overflows.
/// - `DimensionMismatch` if `coefficients.len() != nodes.len() · dimension`.
/// - `SizeOverflow` if that product overflows `usize`.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::real_divided_differences_into;
///
/// // 1 + x + x(x − 1) = 1 + x² through (0, 1), (1, 2), (2, 5).
/// let mut values = [1.0_f64, 2.0, 5.0];
/// real_divided_differences_into(&[0.0, 1.0, 2.0], 1, &mut values)?;
/// assert_eq!(values, [1.0, 1.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn real_divided_differences_into<S: RealScalar>(
    nodes: &[S],
    dimension: usize,
    coefficients: &mut [S],
) -> Result<(), SciOxideError> {
    let nodes = ValidatedNodes::new(nodes)?.0;
    let expected = scioxide_core::backend::checked_mul(
        nodes.len(),
        dimension,
        scioxide_core::quantity!("newton_coefficients"),
    )?;
    if coefficients.len() != expected {
        return Err(SciOxideError::dimension_mismatch(
            [expected],
            [coefficients.len()],
        ));
    }
    crate::util::validate_slice(coefficients)?;
    if crate::admission::difference_table(nodes, coefficients) {
        // A literal single-component width lets the shared loop nest compile
        // to a contiguous recurrence.
        if dimension == 1 {
            certified_real_table(nodes, 1, coefficients);
        } else {
            certified_real_table(nodes, dimension, coefficients);
        }
        return Ok(());
    }
    if crate::admission::finite_gaps(nodes) {
        checked_real_table::<S, true>(nodes, dimension, coefficients)
    } else {
        checked_real_table::<S, false>(nodes, dimension, coefficients)
    }
}

/// Unchecked native quotients of a table whose range was checked in advance.
#[inline(always)]
fn certified_real_table<S: RealScalar>(nodes: &[S], width: usize, coefficients: &mut [S]) {
    let result = difference_table(
        coefficients,
        width,
        1,
        None,
        |current, previous, index, other| {
            Ok::<_, core::convert::Infallible>((current - previous) / (nodes[index] - nodes[other]))
        },
    );
    if let Err((_, never)) = result {
        match never {}
    }
}

/// Per-entry range checks of [`real_divided_differences_into`] for other tables.
fn checked_real_table<S: RealScalar, const FINITE_GAPS: bool>(
    nodes: &[S],
    dimension: usize,
    coefficients: &mut [S],
) -> Result<(), SciOxideError> {
    difference_table(
        coefficients,
        dimension,
        1,
        None,
        |current, previous, index, other| {
            let (left, right) = (nodes[index], nodes[other]);
            crate::admission::difference_quotient::<S, FINITE_GAPS>(current, previous, left, right)
                .map_or_else(
                    || checked_difference_ratio(current, previous, left, right),
                    Ok,
                )
        },
    )
    .map_err(|(_, error)| error)
}

/// Computes the barycentric weights of distinct real or complex nodes.
///
/// The weights are `wⱼ = 1 / Πₖ≠ⱼ (xⱼ − xₖ)`, scaled so that the largest
/// `|wⱼ|` is one, in node order; a single node has weight one. The products
/// are formed with a wide exponent range, so differences that overflow and
/// gaps that underflow do not lose the weights. Signs and complex phases are
/// kept, and `+0` and `−0` count as the same node. This takes O(n²) work and
/// allocates two vectors of length `n`.
///
/// # Errors
///
/// - `InvalidInput` if `nodes` is empty or has duplicates.
/// - `NonFinite` if a node is NaN or infinite.
/// - `NumericalInstability` (`Underflow`) if a scaled weight rounds to zero;
///   subnormal weights are returned.
/// - `AllocationFailure` if the vectors cannot be allocated.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::barycentric_weights;
///
/// // For nodes 0, 1, 2 the weights 1/2, −1, 1/2 are scaled to a maximum of one.
/// let weights = barycentric_weights(&[0.0_f64, 1.0, 2.0])?;
/// assert!(weights.iter().zip([0.5_f64, -1.0, 0.5]).all(|(a, b)| (a - b).abs() < 1e-13));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
/// *SIAM Rev.* 46(3), 2004, 501–517.
pub fn barycentric_weights<C: NumericScalar>(nodes: &[C]) -> Result<Vec<C>, SciOxideError> {
    weights_for_nodes(ValidatedNodes::new(nodes)?)
}

pub(crate) fn weights_for_nodes<C: NumericScalar>(
    nodes: ValidatedNodes<'_, C>,
) -> Result<Vec<C>, SciOxideError> {
    let nodes = nodes.0;
    if nodes.len() == 1 {
        return try_filled(1, C::ONE, scioxide_core::quantity!("barycentric_weights"));
    }

    let mut inverse_products = try_filled(
        nodes.len(),
        ScaledValue::ZERO,
        scioxide_core::quantity!("barycentric_inverse_products"),
    )?;
    let mut weights = try_filled(
        nodes.len(),
        C::ZERO,
        scioxide_core::quantity!("barycentric_weights"),
    )?;
    fill_weights(nodes, &mut weights, &mut inverse_products)?;
    Ok(weights)
}

/// Computes the barycentric weights of [`barycentric_weights`] into `output`.
///
/// `output` and `scratch` must each have one entry per node. The computation
/// takes O(n²) work and does not allocate. Invalid nodes or lengths leave
/// `output` unchanged, but a numerical failure can leave it partly written.
/// The contents of `scratch` are unspecified afterward.
///
/// # Errors
///
/// - The errors of [`barycentric_weights`] other than `AllocationFailure`.
/// - `DimensionMismatch` if `output` or `scratch` does not have one entry per
///   node.
///
/// ```
/// use scioxide_core::ScaledValue;
/// use scioxide_poly::backend::polynomial_basis::barycentric_weights_into;
///
/// let mut weights = [0.0_f64; 3];
/// let mut scratch = [ScaledValue::ZERO; 3];
/// barycentric_weights_into(&[-1.0, 0.0, 1.0], &mut weights, &mut scratch)?;
/// assert_eq!(weights, [0.5, -1.0, 0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn barycentric_weights_into<C: NumericScalar>(
    nodes: &[C],
    output: &mut [C],
    scratch: &mut [ScaledValue<C>],
) -> Result<(), SciOxideError> {
    let nodes = ValidatedNodes::new(nodes)?.0;
    if output.len() != nodes.len() || scratch.len() != nodes.len() {
        return Err(SciOxideError::dimension_mismatch(
            [nodes.len(), nodes.len()],
            [output.len(), scratch.len()],
        ));
    }
    fill_weights(nodes, output, scratch)
}

fn fill_weights<C: NumericScalar>(
    nodes: &[C],
    weights: &mut [C],
    inverse_products: &mut [ScaledValue<C>],
) -> Result<(), SciOxideError> {
    // Most bounded real node sets need no extended exponents. Retain native
    // inverse products directly and restart the complete construction on range
    // loss. Validation above guarantees that a zero product is underflow.
    let mut maximum_native = C::Real::ZERO;
    let mut native = C::IS_REAL;
    for (left, &node) in nodes.iter().enumerate() {
        if !native {
            break;
        }
        let mut product = C::ONE;
        for (right, &other) in nodes.iter().enumerate() {
            if left != right {
                product = product * (node - other);
                if !product.has_finite_components() || product.abs() < C::Real::MIN_POSITIVE {
                    native = false;
                    break;
                }
            }
        }
        if !native {
            break;
        }
        let inverse = C::ONE / product;
        if !inverse.has_finite_components() {
            native = false;
            break;
        }
        weights[left] = inverse;
        maximum_native = maximum_native.max(inverse.abs());
    }
    if native {
        for weight in weights.iter_mut() {
            *weight = *weight / C::from_real(maximum_native);
            if *weight == C::ZERO {
                return Err(instability(
                    NumericalIssue::Underflow,
                    "relative barycentric weight is not representable",
                ));
            }
        }
        return Ok(());
    }
    // Keep products native while normal; extended exponents preserve range.
    let mut maximum = Scaled::ZERO;
    for (left, &node) in nodes.iter().enumerate() {
        let mut product = C::ONE;
        let mut extended = ScaledValue::from_value(C::ONE)?;
        for (right, &other) in nodes.iter().enumerate() {
            if left == right {
                continue;
            }
            let difference = node - other;
            let candidate = product * difference;
            if candidate.has_finite_components() && candidate.abs() >= C::Real::MIN_POSITIVE {
                product = candidate;
            } else {
                // Flush the ordinary product before range is lost. Binary
                // exponents preserve subnormal gaps and overflowed differences
                // without logarithms or an approximate coordinate remapping.
                let difference = if difference.has_finite_components() {
                    ScaledValue::from_value(difference)?
                } else {
                    ScaledValue::from_value(node)?.sub(ScaledValue::from_value(other)?)?
                };
                extended = extended
                    .mul(ScaledValue::from_value(product)?)?
                    .mul(difference)?;
                product = C::ONE;
            }
        }
        let inverse = ScaledValue::from_value(C::ONE)?
            .div(extended.mul(ScaledValue::from_value(product)?)?)?;
        let magnitude = inverse.magnitude()?;
        if magnitude.abs_cmp(maximum).is_gt() {
            maximum = magnitude;
        }
        inverse_products[left] = inverse;
    }
    let normalization = Scaled::from_value(C::Real::ONE)?.div(maximum)?;
    for (slot, inverse) in weights.iter_mut().zip(inverse_products.iter()) {
        let weight = inverse.mul_real(normalization)?.round()?;
        if weight == C::ZERO {
            return Err(instability(
                NumericalIssue::Underflow,
                "relative barycentric weight is not representable",
            ));
        }
        *slot = weight;
    }
    Ok(())
}

/// Computes `(left − right) / scale` for finite operands and a nonzero scale.
///
/// When `left − right` is finite, the result is that difference divided by
/// `scale`. When the difference overflows, each operand is divided by `scale`
/// first, which costs one more rounding but stays finite. The function does
/// not allocate.
///
/// # Errors
///
/// - `NumericalInstability` (`Breakdown`) if `scale` is zero.
/// - `NonFinite` if an operand is NaN or infinite, or the quotient is not
///   representable.
///
/// ```
/// use scioxide_poly::backend::polynomial_basis::normalized_linear_difference;
///
/// assert_eq!(normalized_linear_difference(3.0_f64, 1.0, 4.0)?, 0.5);
/// assert_eq!(normalized_linear_difference(f64::MAX, -f64::MAX, f64::MAX)?, 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn normalized_linear_difference<S: RealScalar>(
    left: S,
    right: S,
    scale: S,
) -> Result<S, SciOxideError> {
    if scale == S::ZERO {
        return Err(instability(
            NumericalIssue::Breakdown,
            "normalization scale is zero",
        ));
    }
    let direct = left - right;
    let result = if direct.is_finite() {
        direct / scale
    } else {
        left / scale - right / scale
    };
    crate::util::finite_intermediate(result, INTERMEDIATE)
}

/// Detail of a non-finite linear-kernel intermediate.
const INTERMEDIATE: &str = "intermediate arithmetic produced a non-finite value";

fn instability(issue: NumericalIssue, detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(issue, detail)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn strided_real_differences_share_scalar_arithmetic_and_range_recovery() {
        let nodes = [0.0_f64, 0.5, 1.0];
        let mut values = [1.0, 0.0, 2.0, 0.25, 5.0, 1.0];
        real_divided_differences_into(&nodes, 2, &mut values).unwrap();
        assert_eq!(values, [1.0, 0.0, 2.0, 0.5, 4.0, 1.0]);
        let mut wide = [-f64::MAX, f64::MAX];
        real_divided_differences_into(&[-f64::MAX, f64::MAX], 1, &mut wide).unwrap();
        assert_eq!(wide[1], 1.0);
        let mut tiny = [0.0, f64::from_bits(1)];
        real_divided_differences_into(&[0.0, 1.0], 1, &mut tiny).unwrap();
        assert_eq!(tiny[1], f64::from_bits(1));
        assert!(real_divided_differences_into(&[0.0, 1.0], 2, &mut [0.0]).is_err());
        assert!(real_divided_differences_into(&[0.0, 0.0], 1, &mut [0.0, 1.0]).is_err());
        assert!(real_divided_differences_into(&[0.0, 1.0], 1, &mut [0.0, f64::NAN]).is_err());
        // Nonadjacent gaps of unordered nodes can be much smaller than every
        // adjacent gap; those tables must retain checked range recovery.
        let mut unordered = [0.0, 0.0, 1e150, 0.0, 0.0];
        assert!(
            real_divided_differences_into(&[0.0, 1.0, 1e-150, 2.0, 2e-150], 1, &mut unordered)
                .is_err()
        );
    }

    #[test]
    fn divided_differences_and_newton_fold_recover_degree_two() {
        let nodes = [0.0_f64, 1.0, 2.0];
        let mut coefficients = [1.0_f64, 2.0, 5.0];
        real_divided_differences_into(&nodes, 1, &mut coefficients).unwrap();
        assert_eq!(coefficients, [1.0, 1.0, 1.0]);
        let (value, derivative) = newton_value_and_derivative_from_iter(
            coefficients[..2]
                .iter()
                .copied()
                .zip(nodes[..2].iter().copied())
                .map(Ok),
            coefficients[2],
            3.0,
            0.0,
            |value, point, node| Ok(value * (point - node)),
            |left, right| Ok(left + right),
        )
        .unwrap();
        assert_eq!(value, 10.0);
        assert_eq!(derivative, 6.0);
    }

    #[test]
    fn barycentric_weights_are_invariant_to_extreme_common_scaling() {
        let ordinary = barycentric_weights(&[-2.0_f64, -0.5, 1.0, 3.0]).unwrap();
        let scaled = barycentric_weights(&[-2.0e200_f64, -0.5e200, 1.0e200, 3.0e200]).unwrap();
        ordinary
            .iter()
            .zip(scaled)
            .for_each(|(left, right)| assert!((left - right).abs() <= 8.0 * f64::EPSILON));
    }

    #[test]
    fn normalized_linear_difference_rejects_zero_scale_and_recovers_range() {
        assert_eq!(
            normalized_linear_difference(3.0_f64, 1.0, 4.0).unwrap(),
            0.5
        );
        assert_eq!(
            normalized_linear_difference(f64::MAX, -f64::MAX, f64::MAX).unwrap(),
            2.0
        );
        let zero = normalized_linear_difference(1.0_f64, 0.0, 0.0).unwrap_err();
        assert!(matches!(
            zero.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Breakdown,
                ..
            }
        ));
        let overflow = normalized_linear_difference(f64::MAX, -f64::MAX, 0.25).unwrap_err();
        assert_eq!(overflow.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn newton_to_power_maps_and_checks_shape() {
        // p = 1 + (x - 1)(2 + 3(x - 2)) = 3x² - 7x + 5.
        let mut power = [0.0_f64; 3];
        newton_to_power_into(
            &[1.0, 2.0],
            [1.0, 2.0, 3.0].into_iter().map(Ok),
            &mut power,
            Ok,
            |addend, q, node| Ok(addend - node * q),
        )
        .unwrap();
        assert_eq!(power, [5.0, -7.0, 3.0]);
        // Same polynomial in u with x = 2 + 4u: 48u² + 20u + 3.
        newton_to_power_into(
            &[1.0, 2.0],
            [1.0, 2.0, 3.0].into_iter().map(Ok),
            &mut power,
            |q| Ok(4.0 * q),
            |addend, q, node| Ok(addend + (2.0 - node) * q),
        )
        .unwrap();
        assert_eq!(power, [3.0, 20.0, 48.0]);
        let mut constant = [0.0_f64];
        newton_to_power_into::<f64, f64>(
            &[],
            [7.0].into_iter().map(Ok),
            &mut constant,
            Ok,
            |a, _, _| Ok(a),
        )
        .unwrap();
        assert_eq!(constant, [7.0]);
        for (coefficients, storage) in [
            (vec![1.0, 2.0], 3),
            (vec![1.0, 2.0, 3.0, 4.0], 3),
            (vec![1.0, 2.0, 3.0], 2),
        ] {
            let mut out = vec![0.0_f64; storage];
            let error = newton_to_power_into(
                &[1.0, 2.0],
                coefficients.into_iter().map(Ok),
                &mut out,
                Ok,
                |a, q, x| Ok(a - x * q),
            )
            .unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
        }
        let failure = newton_to_power_into(
            &[1.0_f64],
            [1.0, 2.0].into_iter().map(Ok),
            &mut [0.0; 2],
            |_| {
                Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Intermediate,
                    "raise",
                ))
            },
            |a, q, x| Ok(a - x * q),
        );
        assert_eq!(
            failure.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn bernstein_to_power_matches_expanded_basis() {
        let ratio = |c: f64, n: usize, d: usize| Ok(c * n as f64 / d as f64);
        // Cubic Bernstein [1, 0, 0, 0] is (1 - s)³ = 1 - 3s + 3s² - s³.
        let mut differences = [1.0_f64, 0.0, 0.0, 0.0];
        let mut power = [0.0; 4];
        bernstein_to_power_into(
            &mut differences,
            &mut power,
            1.0,
            |a, b| Ok(a - b),
            |v, c| Ok(v * c),
            ratio,
        )
        .unwrap();
        assert_eq!(power, [1.0, -3.0, 3.0, -1.0]);
        let mut single = [2.5_f64];
        let mut out = [0.0];
        bernstein_to_power_into(
            &mut single,
            &mut out,
            1.0,
            |a, b| Ok(a - b),
            |v, c| Ok(v * c),
            ratio,
        )
        .unwrap();
        assert_eq!(out, [2.5]);
        for (len, storage) in [(0, 0), (2, 3)] {
            let mut differences = vec![0.0_f64; len];
            let mut power = vec![0.0_f64; storage];
            let error = bernstein_to_power_into(
                &mut differences,
                &mut power,
                1.0,
                |a, b| Ok(a - b),
                |v, c| Ok(v * c),
                ratio,
            )
            .unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
        }
    }

    #[test]
    fn confluent_differences_reproduce_cubic_hermite_data() {
        // p = x³ - x on nodes [-1, 0, 2], two interleaved components (p, 2p).
        let nodes = [-1.0_f64, 0.0, 2.0];
        let p = |x: f64| x * x * x - x;
        let dp = |x: f64| 3.0 * x * x - 1.0;
        let values: Vec<f64> = nodes.iter().flat_map(|&x| [p(x), 2.0 * p(x)]).collect();
        let slopes: Vec<f64> = nodes.iter().flat_map(|&x| [dp(x), 2.0 * dp(x)]).collect();
        let mut coefficients = [0.0; 12];
        confluent_divided_differences_into(&nodes, &values, &slopes, 2, &mut coefficients).unwrap();
        let doubled = [-1.0, -1.0, 0.0, 0.0, 2.0];
        for component in 0..2 {
            let leading = coefficients[10 + component];
            let terms = (0..5).map(|k| Ok((coefficients[2 * k + component], doubled[k])));
            for x in [-1.5, -0.25, 0.75, 2.0] {
                let (value, slope) = newton_value_and_derivative_from_iter(
                    terms.clone(),
                    leading,
                    x,
                    0.0,
                    |a, x, y| Ok(a * (x - y)),
                    |a, b| Ok(a + b),
                )
                .unwrap();
                let scale = (component + 1) as f64;
                assert!((value - scale * p(x)).abs() <= 1e-13);
                assert!((slope - scale * dp(x)).abs() <= 1e-13);
            }
        }
        let shape = confluent_divided_differences_into(&nodes, &values, &slopes, 2, &mut [0.0; 6]);
        assert_eq!(
            shape.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        let duplicate =
            confluent_divided_differences_into(&[0.0, 0.0], &[0.0; 2], &[0.0; 2], 1, &mut [0.0; 4]);
        assert_eq!(
            duplicate.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
        let nonfinite =
            confluent_divided_differences_into(&[0.0], &[0.0], &[f64::NAN], 1, &mut [0.0; 2]);
        assert_eq!(
            nonfinite.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
        let empty = confluent_divided_differences_into::<f64>(&[], &[], &[], 1, &mut []);
        assert_eq!(
            empty.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }
}
