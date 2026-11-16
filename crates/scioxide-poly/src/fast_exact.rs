//! Exact products, composition, Taylor shifts and product trees over
//! user-defined rings and fields.
use crate::util::{invalid, minimum_len, try_grow};
use crate::{ExactCoefficient, ExactField, ExactPolynomial};
use alloc::vec::Vec;
use scioxide_core::backend::checked_add;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{Capacity, ExecutionContext, SciOxideError};

#[cfg(test)]
mod tests;

const KARATSUBA_CROSSOVER: usize = 32;
const SCRATCH: scioxide_core::Quantity = scioxide_core::quantity!("coefficient_scratch");

// Bounds follow peak live storage, not the sum of storage across sibling calls.
// A balanced product stores two sums and their product (4m - 1 slots), with
// m <= ceil(n/3); a blocked product stores at most 2m - 1 slots. Both recurse
// on products of at most 2m - 1 coefficients. The geometric sum is O(n).
fn product_scratch(n: usize) -> Result<usize, SciOxideError> {
    if n < 2 * KARATSUBA_CROSSOVER - 1 {
        return Ok(0);
    }
    let m = n.div_ceil(3);
    let local = scioxide_core::backend::checked_mul(m, 4, SCRATCH)? - 1;
    checked_add(local, product_scratch(2 * m - 1)?, SCRATCH)
}

fn power_scratch(n: usize) -> Result<usize, SciOxideError> {
    if n < 3 {
        return Ok(0);
    }
    let half = n.div_ceil(2);
    // Odd powers retain the half power and its square during multiplication.
    let product = checked_add(n - 1, product_scratch(n)?, SCRATCH)?;
    checked_add(half, power_scratch(half)?.max(product), SCRATCH)
}

fn composition_scratch(n: usize) -> Result<usize, SciOxideError> {
    if n == 1 {
        return Ok(0);
    }
    // For outer length >= 3, q^floor(len(p)/2) has at most (2n+1)/3
    // coefficients. Outer length 2 retains n+2 slots and its power is a copy.
    let power = n - (n - 1) / 3;
    let local = checked_add(n, power.max(2), SCRATCH)?;
    let child = composition_scratch(n.div_ceil(2))?
        .max(power_scratch(power)?)
        .max(product_scratch(n)?);
    checked_add(local, child, SCRATCH)
}

/// Reports the number of coefficients an [`ExactPolynomialWorkspace`] needs.
///
/// The count does not include memory that individual coefficients allocate
/// themselves. There is no memory limit.
///
/// ```
/// use scioxide_poly::ExactPolynomialWorkspace;
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
///
/// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
/// assert!(requirements.coefficient_len() >= 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactPolynomialWorkspaceRequirements {
    coefficients: usize,
}

impl ExactPolynomialWorkspaceRequirements {
    /// Returns the number of coefficients for the temporary output and the
    /// Karatsuba and composition stack.
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomialWorkspace;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
    /// assert!(requirements.coefficient_len() >= 8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn coefficient_len(self) -> usize {
        self.coefficients
    }
}

/// Holds the coefficient scratch of [`exact_multiply_into`],
/// [`exact_compose_into`] and [`exact_taylor_shift_into`].
///
/// Create one for results of up to `capacity` coefficients with
/// [`Self::workspace`], or size an empty workspace with [`Self::try_reserve`].
/// A workspace reserved for a larger capacity serves smaller results. A call
/// with a workspace that is too small returns `DimensionMismatch`. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_poly::ExactPolynomialWorkspace;
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
///
/// let workspace = ExactPolynomialWorkspace::<Q>::workspace(8)?;
/// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct ExactPolynomialWorkspace<C: ExactCoefficient> {
    data: Vec<C>,
}

impl<C: ExactCoefficient> Default for ExactPolynomialWorkspace<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: ExactCoefficient> ExactPolynomialWorkspace<C> {
    /// Returns an empty workspace without allocating.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::ExactPolynomialWorkspace;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
    /// let ws = ExactPolynomialWorkspace::<Q>::new();
    /// assert_eq!(ws.check(&requirements)?, Capacity::NeedsGrowth);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self { data: Vec::new() }
    }
    /// Returns the requirements of results of up to `capacity` coefficients:
    /// the result itself and the Karatsuba and composition stack, at most about
    /// seven times `capacity` coefficients.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `capacity` is zero.
    /// - `SizeOverflow` if the length does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomialWorkspace;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
    /// assert!(requirements.coefficient_len() >= 8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        capacity: usize,
    ) -> Result<ExactPolynomialWorkspaceRequirements, SciOxideError> {
        if capacity == 0 {
            return Err(invalid(
                scioxide_core::argument!("capacity"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "exact capacity must be positive",
            ));
        }
        Ok(ExactPolynomialWorkspaceRequirements {
            coefficients: required_len(capacity)?,
        })
    }
    /// Allocates a zero-filled workspace for results of up to `capacity`
    /// coefficients.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be allocated.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::ExactPolynomialWorkspace;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// let ws = ExactPolynomialWorkspace::<Q>::workspace(8)?;
    /// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(4)?;
    /// assert_eq!(ws.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(capacity: usize) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(capacity)?)?;
        Ok(workspace)
    }
    /// Reports whether the workspace meets `requirements`, without allocating.
    ///
    /// # Errors
    ///
    /// None; the `Result` matches the `check` method of other SciOxide workspaces.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::ExactPolynomialWorkspace;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
    /// let ws = ExactPolynomialWorkspace::<Q>::workspace(8)?;
    /// assert_eq!(ws.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn check(
        &self,
        requirements: &ExactPolynomialWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(if self.data.len() >= requirements.coefficients {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace until it meets `requirements`. A workspace that is
    /// already large enough is left as it is.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the workspace cannot grow. Existing
    ///   coefficients are kept.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::ExactPolynomialWorkspace;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// let requirements = ExactPolynomialWorkspace::<Q>::workspace_requirements(8)?;
    /// let mut ws = ExactPolynomialWorkspace::<Q>::new();
    /// ws.try_reserve(&requirements)?;
    /// assert_eq!(ws.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &ExactPolynomialWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        try_grow(
            &mut self.data,
            requirements.coefficients,
            C::zero(),
            scioxide_core::quantity!("coefficient_scratch"),
        )
    }
}

/// Coefficients of the result and stack of a call with `len` result coefficients.
fn required_len(len: usize) -> Result<usize, SciOxideError> {
    checked_add(len, composition_scratch(len)?, SCRATCH)
}
/// Checks that `output` has exactly `expected` entries and that `workspace`
/// holds the result and stack of that many coefficients.
fn admit<C: ExactCoefficient>(
    expected: usize,
    output: &[C],
    workspace: &ExactPolynomialWorkspace<C>,
) -> Result<(), SciOxideError> {
    if output.len() != expected {
        return Err(SciOxideError::dimension_mismatch(expected, output.len()));
    }
    minimum_len(required_len(expected)?, workspace.data.len())
}
/// Computes the exact product `a·b` of two polynomials with coefficients in a
/// user-defined ring and writes it to `output`.
///
/// Coefficient slices are in ascending order, `[c₀, c₁, …, cₙ]` for
/// `c₀ + c₁x + … + cₙxⁿ`, and all arithmetic is exact in the
/// [`ExactCoefficient`] ring. `output` must have `a.len() + b.len() − 1`
/// entries. Products use the direct sum when the shorter operand has fewer
/// than 32 coefficients and Karatsuba's method otherwise; an unbalanced product
/// is split into blocks of the shorter length. The Karatsuba cost is
/// `O(n^log₂3) ≈ O(n^1.585)` ring operations. `workspace` must meet
/// [`ExactPolynomialWorkspace::workspace_requirements`] for a capacity of
/// `output.len()`; a workspace for a larger capacity serves too. The call does
/// not allocate, apart from allocations inside your coefficient type, and
/// `output` is unchanged on error.
///
/// # Errors
///
/// - `InvalidInput` if `a` or `b` is empty.
/// - `DimensionMismatch` if `output` has the wrong length or `workspace` is
///   too small.
/// - `ExecutionStopped` on cancellation or deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::{ExactPolynomialWorkspace, exact_multiply_into};
/// mod ring {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/integer.rs"
///     ));
/// # }
/// # use ring::Integer as Z;
///
/// // (1 + 2x)(1 − x) = 1 + x − 2x² over the integers.
/// let mut ws = ExactPolynomialWorkspace::<Z>::workspace(8)?;
/// let mut product = [Z(0); 3];
/// let cx = ExecutionContext::sequential();
/// exact_multiply_into(&[Z(1), Z(2)], &[Z(1), Z(-1)], &mut product, &mut ws, cx)?;
/// assert_eq!(product, [Z(1), Z(1), Z(-2)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Karatsuba and Yu. Ofman, "Multiplication of multidigit numbers on
///   automata", *Soviet Physics Doklady* 7, 1963, pp. 595–596.
/// - J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
///   Cambridge University Press, 2013, chapter 8.
pub fn exact_multiply_into<C: ExactCoefficient>(
    a: &[C],
    b: &[C],
    output: &mut [C],
    workspace: &mut ExactPolynomialWorkspace<C>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    crate::fast::nonempty(a, scioxide_core::argument!("a"))?;
    crate::fast::nonempty(b, scioxide_core::argument!("b"))?;
    let len = checked_add(
        a.len(),
        b.len(),
        scioxide_core::quantity!("product_coefficients"),
    )? - 1;
    admit(len, output, workspace)?;
    let (result, scratch) = workspace.data.split_at_mut(output.len());
    context.check()?;
    karatsuba(a, b, result, scratch, context)?;
    context.check()?;
    output.swap_with_slice(result);
    Ok(())
}
/// Computes the exact product by Karatsuba's method, checking for cancellation.
fn karatsuba<C: ExactCoefficient>(
    a: &[C],
    b: &[C],
    out: &mut [C],
    scratch: &mut [C],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    out.fill(C::zero());
    let short = a.len().min(b.len());
    let long = a.len().max(b.len());
    let m = long.div_ceil(2);
    if short < KARATSUBA_CROSSOVER {
        context.check()?;
        for (i, x) in a.iter().enumerate() {
            context.check()?;
            for (j, y) in b.iter().enumerate() {
                out[i + j] = out[i + j].clone() + x.clone() * y.clone();
            }
        }
        return Ok(());
    }
    context.check()?;
    if short <= m {
        let longer = if a.len() >= b.len() { a } else { b };
        let (block, rest) = scratch.split_at_mut(2 * short - 1);
        for (i, part) in longer.chunks(short).enumerate() {
            let len = part.len() + short - 1;
            let (left, right) = if a.len() >= b.len() {
                (part, b)
            } else {
                (a, part)
            };
            karatsuba(left, right, &mut block[..len], rest, context)?;
            for (x, y) in out[i * short..].iter_mut().zip(&block[..len]) {
                *x = x.clone() + y.clone();
            }
        }
        return Ok(());
    }
    let (a0, a1) = a.split_at(m);
    let (b0, b1) = b.split_at(m);
    // `z0 = a0 b0`, `z2 = a1 b1`, `z1 = (a0 + a1)(b0 + b1) - z0 - z2`, giving
    // `a b = z0 + z1 x^m + z2 x^{2m}`. z0 and z2 occupy disjoint output
    // ranges; finish subtracting both from z1 before adding the overlapping z1.
    let low_len = 2 * m - 1;
    karatsuba(a0, b0, &mut out[..low_len], scratch, context)?;
    karatsuba(a1, b1, &mut out[2 * m..], scratch, context)?;
    let (sa, rest) = scratch.split_at_mut(m);
    let (sb, rest) = rest.split_at_mut(m);
    let (z1, rest) = rest.split_at_mut(2 * m - 1);
    for (i, v) in sa.iter_mut().enumerate() {
        *v = a0[i].clone() + a1.get(i).cloned().unwrap_or_else(C::zero);
    }
    for (i, v) in sb.iter_mut().enumerate() {
        *v = b0[i].clone() + b1.get(i).cloned().unwrap_or_else(C::zero);
    }
    karatsuba(sa, sb, z1, rest, context)?;
    for (i, v) in z1.iter_mut().enumerate() {
        *v = v.clone()
            - out[..low_len].get(i).cloned().unwrap_or_else(C::zero)
            - out[2 * m..].get(i).cloned().unwrap_or_else(C::zero);
    }
    for (x, y) in out[m..].iter_mut().zip(z1) {
        *x = x.clone() + y.clone();
    }
    Ok(())
}
/// Returns the exact product of two nonempty coefficient slices in a new vector.
pub(crate) fn multiply<C: ExactCoefficient>(a: &[C], b: &[C]) -> Result<Vec<C>, SciOxideError> {
    let n = checked_add(
        a.len(),
        b.len(),
        scioxide_core::quantity!("product_coefficients"),
    )? - 1;
    let mut output = try_filled(
        n,
        C::zero(),
        scioxide_core::quantity!("product_coefficients"),
    )?;
    // Same base-case test as `karatsuba`: when it would not recurse, no scratch is
    // needed and the plan allocation is skipped.
    if a.len().min(b.len()) < KARATSUBA_CROSSOVER {
        karatsuba(
            a,
            b,
            &mut output,
            &mut [],
            ExecutionContext::sequential_native(),
        )?;
    } else {
        let mut scratch = try_filled(product_scratch(n)?, C::zero(), SCRATCH)?;
        karatsuba(
            a,
            b,
            &mut output,
            &mut scratch,
            ExecutionContext::sequential_native(),
        )?;
    }
    Ok(output)
}
impl<C: ExactCoefficient> ExactPolynomial<C> {
    /// Returns the Taylor shift `self(x + shift)`, computed exactly.
    ///
    /// The shift is the composition with `x + shift`, so no factorials appear
    /// and the method works over any ring, including fields of positive
    /// characteristic. See [`exact_compose_into`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the result does not fit in
    ///   memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomial;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// // p(x) = (1 + x)², so p(1 + 1) = 9.
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.taylor_shift(Q(1, 1))?;
    /// assert_eq!(result.evaluate(Q(1, 1)), Q(9, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn taylor_shift(&self, shift: C) -> Result<Self, SciOxideError> {
        let mut q = try_filled(2, C::one(), scioxide_core::quantity!("shift_coefficients"))?;
        q[0] = shift;
        self.compose(&Self::new(q)?)
    }
    /// Returns the composition `self(inner(x))`, computed exactly.
    ///
    /// The method is described on [`exact_compose_into`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the result does not fit in
    ///   memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomial;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    ///
    /// // p(x) = (1 + x)², so p(p(1)) = p(4) = 25.
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.compose(&p)?;
    /// assert_eq!(result.evaluate(Q(1, 1)), Q(25, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compose(&self, inner: &Self) -> Result<Self, SciOxideError> {
        compose(self.coefficients(), inner)
    }
}
/// Returns `a(q(x))` in a new polynomial.
fn compose<C: ExactCoefficient>(
    a: &[C],
    q: &ExactPolynomial<C>,
) -> Result<ExactPolynomial<C>, SciOxideError> {
    let n = crate::composition::length(
        a.len(),
        q.coefficients().len(),
        scioxide_core::argument!("self"),
        scioxide_core::argument!("inner"),
    )?;
    let mut work = ExactPolynomialWorkspace::workspace(n)?;
    let mut result = try_filled(
        n,
        C::zero(),
        scioxide_core::quantity!("composition_coefficients"),
    )?;
    exact_compose_into(
        a,
        q.coefficients(),
        &mut result,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    ExactPolynomial::new(result)
}
/// Computes the composition `p(q(x))` exactly and writes it to `output`.
///
/// `output` must have `(p.len() − 1)(q.len() − 1) + 1` entries, and
/// `workspace` must meet [`ExactPolynomialWorkspace::workspace_requirements`]
/// for a capacity of `output.len()`. The composition splits `p = low + xᵐ·high`
/// with `m = ⌊p.len()/2⌋`, evaluates `low(q) + qᵐ·high(q)` recursively, and
/// forms `qᵐ` by repeated squaring, with products as in
/// [`exact_multiply_into`]. `output` is unchanged on error.
///
/// # Errors
///
/// - `InvalidInput` if `p` or `q` is empty.
/// - `DimensionMismatch` if `output` has the wrong length or `workspace` is
///   too small.
/// - `ExecutionStopped` on cancellation or deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::{ExactPolynomialWorkspace, exact_compose_into};
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
///
/// // p(x) = (1 + x)², q(x) = 2x, so p(q(x)) = 1 + 4x + 4x².
/// let mut ws = ExactPolynomialWorkspace::<Q>::workspace(3)?;
/// let mut out = [Q(0, 1); 3];
/// let (p, q) = ([Q(1, 1), Q(2, 1), Q(1, 1)], [Q(0, 1), Q(2, 1)]);
/// exact_compose_into(&p, &q, &mut out, &mut ws, ExecutionContext::sequential())?;
/// assert_eq!(out, [Q(1, 1), Q(4, 1), Q(4, 1)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn exact_compose_into<C: ExactCoefficient>(
    p: &[C],
    q: &[C],
    output: &mut [C],
    workspace: &mut ExactPolynomialWorkspace<C>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let expected = crate::composition::length(
        p.len(),
        q.len(),
        scioxide_core::argument!("p"),
        scioxide_core::argument!("q"),
    )?;
    admit(expected, output, workspace)?;
    let (result, scratch) = workspace.data.split_at_mut(output.len());
    compose_recursive(p, q, result, scratch, context)?;
    context.check()?;
    output.clone_from_slice(result);
    Ok(())
}
/// Computes the Taylor shift `p(x + shift)` exactly and writes it to `output`.
///
/// The shift is the composition with `x + shift` of [`exact_compose_into`],
/// so no factorials appear and the method works over any ring, including
/// fields of positive characteristic. `output` must have `p.len()` entries.
///
/// # Errors
///
/// The errors of [`exact_compose_into`].
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::{ExactPolynomialWorkspace, exact_taylor_shift_into};
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
///
/// // (x + 1 + 1)² = 4 + 4x + x².
/// let mut ws = ExactPolynomialWorkspace::<Q>::workspace(3)?;
/// let mut out = [Q(0, 1); 3];
/// let p = [Q(1, 1), Q(2, 1), Q(1, 1)];
/// let cx = ExecutionContext::sequential();
/// exact_taylor_shift_into(&p, Q(1, 1), &mut out, &mut ws, cx)?;
/// assert_eq!(out, [Q(4, 1), Q(4, 1), Q(1, 1)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn exact_taylor_shift_into<C: ExactCoefficient>(
    p: &[C],
    shift: C,
    output: &mut [C],
    workspace: &mut ExactPolynomialWorkspace<C>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    exact_compose_into(p, &[shift, C::one()], output, workspace, context)
}
/// Computes `p(q(x))` recursively, taking temporaries from `stack`.
fn compose_recursive<C: ExactCoefficient>(
    p: &[C],
    q: &[C],
    out: &mut [C],
    stack: &mut [C],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    if q.len() == 1 {
        out[0] = p
            .iter()
            .rev()
            .fold(C::zero(), |v, c| v * q[0].clone() + c.clone());
        return Ok(());
    }
    if p.len() == 1 {
        out[0] = p[0].clone();
        return Ok(());
    }
    // Split `p = low + x^m high`, so `p(q) = low(q) + q^m high(q)`. Both halves
    // recurse on the same `q`; the power is formed by repeated squaring. `low(q)`
    // is shorter than the full result and is added onto the zero-filled product.
    // Scratch is taken from `stack` in call order and released on return.
    let m = p.len() / 2;
    let length = |a, b| {
        crate::composition::length(
            a,
            b,
            scioxide_core::argument!("p"),
            scioxide_core::argument!("q"),
        )
    };
    let (low, stack) = stack.split_at_mut(length(m, q.len())?);
    let (high, stack) = stack.split_at_mut(length(p.len() - m, q.len())?);
    let (power, stack) = stack.split_at_mut(length(m + 1, q.len())?);
    compose_recursive(&p[..m], q, low, stack, context)?;
    compose_recursive(&p[m..], q, high, stack, context)?;
    exact_power(q, m, power, stack, context)?;
    karatsuba(high, power, out, stack, context)?;
    for (x, y) in out.iter_mut().zip(low) {
        *x = x.clone() + y.clone();
    }
    Ok(())
}
/// Computes `qⁿ` by repeated squaring.
fn exact_power<C: ExactCoefficient>(
    q: &[C],
    n: usize,
    out: &mut [C],
    stack: &mut [C],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if n == 0 {
        out[0] = C::one();
        return Ok(());
    }
    if q.len() == 1 && n > 1 {
        // Keep the same recurrence without retaining polynomial temporaries.
        exact_power(q, n / 2, out, &mut [], context)?;
        out[0] = out[0].clone() * out[0].clone();
        if !n.is_multiple_of(2) {
            out[0] = out[0].clone() * q[0].clone();
        }
        return Ok(());
    }
    if n == 1 {
        out.clone_from_slice(q);
        return Ok(());
    }
    // `crate::composition::length(k + 1, len(q))` is the length of `q^k`; square `q^{n/2}` and
    // multiply by one extra `q` when `n` is odd.
    let half_len = crate::composition::length(
        n / 2 + 1,
        q.len(),
        scioxide_core::argument!("p"),
        scioxide_core::argument!("q"),
    )?;
    let (half, stack) = stack.split_at_mut(half_len);
    exact_power(q, n / 2, half, stack, context)?;
    if n.is_multiple_of(2) {
        karatsuba(half, half, out, stack, context)?;
    } else {
        let (square, stack) = stack.split_at_mut(2 * half_len - 1);
        karatsuba(half, half, square, stack, context)?;
        karatsuba(square, q, out, stack, context)?;
    }
    Ok(())
}

/// Returns `polynomialᵉ` by the repeated squaring of the composition kernel.
pub(crate) fn power<C: ExactCoefficient>(
    polynomial: &ExactPolynomial<C>,
    exponent: usize,
) -> Result<ExactPolynomial<C>, SciOxideError> {
    if exponent == 0 {
        return ExactPolynomial::constant(C::one());
    }
    if exponent == 1 || polynomial.is_zero() {
        return polynomial.try_copy();
    }
    let n = polynomial
        .degree()
        .checked_mul(exponent)
        .and_then(|degree| degree.checked_add(1))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("power_coefficients"))
        })?;
    let mut out = try_filled(n, C::zero(), scioxide_core::quantity!("power_coefficients"))?;
    let mut stack = try_filled(power_scratch(n)?, C::zero(), SCRATCH)?;
    exact_power(
        polynomial.coefficients(),
        exponent,
        &mut out,
        &mut stack,
        ExecutionContext::sequential_native(),
    )?;
    ExactPolynomial::new(out)
}

// Count the multiplications performed by the same schoolbook, blocked and
// balanced kernels. Saturation makes an unrepresentable estimate conservative.
fn product_cost(a: usize, b: usize) -> usize {
    let short = a.min(b);
    let long = a.max(b);
    if short < KARATSUBA_CROSSOVER {
        return a.saturating_mul(b);
    }
    let m = long.div_ceil(2);
    if short <= m {
        return (long / short)
            .saturating_mul(product_cost(short, short))
            .saturating_add(product_cost(long % short, short));
    }
    let low = product_cost(m, m);
    if a - m == m && b - m == m {
        low.saturating_mul(3)
    } else {
        low.saturating_mul(2)
            .saturating_add(product_cost(a - m, b - m))
    }
}

fn use_newton(a: usize, b: usize) -> bool {
    let q = a - b + 1;
    if q < KARATSUBA_CROSSOVER || b < KARATSUBA_CROSSOVER {
        return false;
    }
    let mut cost = product_cost(q, q).saturating_add(product_cost(q, b));
    let mut precision = 1;
    while precision < q {
        let next = precision.saturating_mul(2).min(q);
        cost = cost.saturating_add(product_cost(next, precision).saturating_mul(2));
        precision = next;
    }
    cost < q.saturating_mul(b - 1)
}

/// Holds the storage of one tree query: Karatsuba scratch and the buffers of
/// the Newton iteration. The buffers grow fallibly and are reused by every
/// node and precision.
struct TreeWork<C: ExactCoefficient> {
    scratch_len: usize,
    scratch: Vec<C>,
    inverse: Vec<C>,
    reversed: Vec<C>,
    error: Vec<C>,
    correction: Vec<C>,
}

impl<C: ExactCoefficient> TreeWork<C> {
    fn new(capacity: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            scratch_len: product_scratch(capacity)?,
            scratch: Vec::new(),
            inverse: Vec::new(),
            reversed: Vec::new(),
            error: Vec::new(),
            correction: Vec::new(),
        })
    }

    fn prepare_multiplication(&mut self) -> Result<(), SciOxideError> {
        try_grow(&mut self.scratch, self.scratch_len, C::zero(), SCRATCH)
    }

    fn multiply(
        &mut self,
        a: &ExactPolynomial<C>,
        b: &ExactPolynomial<C>,
    ) -> Result<ExactPolynomial<C>, SciOxideError> {
        if a.is_zero() || b.is_zero() {
            return ExactPolynomial::constant(C::zero());
        }
        let quantity = scioxide_core::quantity!("product_coefficients");
        let n = checked_add(a.coefficients().len(), b.coefficients().len(), quantity)? - 1;
        let mut out = try_filled(n, C::zero(), quantity)?;
        if a.coefficients().len().min(b.coefficients().len()) < KARATSUBA_CROSSOVER {
            karatsuba(
                a.coefficients(),
                b.coefficients(),
                &mut out,
                &mut [],
                ExecutionContext::sequential_native(),
            )?;
        } else {
            self.prepare_multiplication()?;
            karatsuba(
                a.coefficients(),
                b.coefficients(),
                &mut out,
                &mut self.scratch,
                ExecutionContext::sequential_native(),
            )?;
        }
        ExactPolynomial::new(out)
    }
}

fn remainder_capacity(n: usize) -> Result<usize, SciOxideError> {
    checked_add(n, n, scioxide_core::quantity!("product_coefficients")).map(|n| n - 1)
}

/// Evaluates and interpolates polynomials at a fixed set of points
/// `x₀, …, xₙ₋₁` with exact arithmetic in a user-defined field.
///
/// The tree stores the subproducts `∏ (x − xᵢ)` over balanced halves of the
/// points, built bottom-up with the products of [`exact_multiply_into`].
/// [`Self::evaluate`] reduces a polynomial modulo the subproducts from the root
/// down (a remainder tree), and [`Self::interpolate`] combines Lagrange terms
/// from the leaves up. Each remainder uses Newton iteration for the reciprocal
/// of the reversed divisor when that needs fewer coefficient products than
/// classical division, and classical division otherwise. With fast products
/// both queries cost `O(M(n) log n)` field operations, where `M(n)` is the
/// cost of a product of length `n`. Besides the tree, a query stores
/// `O(m + n)` coefficients for a polynomial of length `m`, and returns newly
/// allocated results.
///
/// # Examples
///
/// ```
/// use scioxide_poly::{ExactPolynomial, ExactPolynomialProductTree};
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
///
/// // p(x) = (1 + x)² at x = 0, 1, 2 is 1, 4, 9; interpolation recovers p.
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// let tree = ExactPolynomialProductTree::new(&[Q(0, 1), Q(1, 1), Q(2, 1)])?;
/// let values = tree.evaluate(&p)?;
/// assert_eq!(values, [Q(1, 1), Q(4, 1), Q(9, 1)]);
/// assert_eq!(tree.interpolate(&values)?, p);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
/// Cambridge University Press, 2013, chapters 9 (Newton iteration) and 10
/// (fast multipoint evaluation and interpolation).
#[derive(Clone, Debug)]
pub struct ExactPolynomialProductTree<C: ExactField> {
    points: Vec<C>,
    nodes: Vec<ExactPolynomial<C>>,
    base: usize,
}
impl<C: ExactField> ExactPolynomialProductTree<C> {
    /// Builds the subproduct tree of `points`, keeping their order.
    ///
    /// The points need not be distinct for [`Self::evaluate`];
    /// [`Self::interpolate`] requires distinct points.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the tree does not fit in
    ///   memory.
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomialProductTree;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// let tree = ExactPolynomialProductTree::new(&[Q(0, 1), Q(1, 1), Q(2, 1)])?;
    /// assert_eq!(tree.points().len(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(points: &[C]) -> Result<Self, SciOxideError> {
        let base = scioxide_core::backend::checked_next_power_of_two(
            points.len(),
            scioxide_core::quantity!("tree_leaves"),
        )?;
        let count =
            scioxide_core::backend::checked_mul(base, 2, scioxide_core::quantity!("tree_nodes"))?;
        let capacity = checked_add(
            points.len(),
            1,
            scioxide_core::quantity!("product_coefficients"),
        )?;
        let mut work = TreeWork::new(capacity)?;
        // Binary heap layout, padded with
        // constant-one leaves. No Leja reordering: exact arithmetic has no conditioning
        // to protect, so points keep their input order.
        let mut nodes = try_with_capacity(count, scioxide_core::quantity!("tree_nodes"))?;
        for _ in 0..count {
            nodes.push(ExactPolynomial::constant(C::one())?);
        }
        for (i, x) in points.iter().enumerate() {
            let mut c = try_filled(2, C::one(), scioxide_core::quantity!("node_coefficients"))?;
            c[0] = -x.clone();
            nodes[base + i] = ExactPolynomial::new(c)?;
        }
        for i in (1..base).rev() {
            let (left, right) = (&nodes[2 * i], &nodes[2 * i + 1]);
            nodes[i] = if left.degree() == 0 {
                right.try_copy()?
            } else if right.degree() == 0 {
                left.try_copy()?
            } else {
                work.multiply(left, right)?
            };
        }
        let mut owned = try_with_capacity(points.len(), scioxide_core::quantity!("points"))?;
        owned.extend_from_slice(points);
        Ok(Self {
            points: owned,
            nodes,
            base,
        })
    }
    /// Returns the product `∏ᵢ (x − xᵢ)` over all points; the constant one for
    /// no points.
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomialProductTree;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// let tree = ExactPolynomialProductTree::new(&[Q(0, 1), Q(1, 1), Q(2, 1)])?;
    ///
    /// // x(x − 1)(x − 2) = 2x − 3x² + x³.
    /// let want = [Q(0, 1), Q(2, 1), Q(-3, 1), Q(1, 1)];
    /// assert_eq!(tree.product().coefficients(), &want);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn product(&self) -> &ExactPolynomial<C> {
        &self.nodes[1]
    }
    /// Returns the points in the order they were given.
    ///
    /// ```
    /// use scioxide_poly::ExactPolynomialProductTree;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// let tree = ExactPolynomialProductTree::new(&[Q(0, 1), Q(1, 1), Q(2, 1)])?;
    /// assert_eq!(tree.points(), &[Q(0, 1), Q(1, 1), Q(2, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn points(&self) -> &[C] {
        &self.points
    }
    /// Returns `p(xᵢ)` at every point, in point order, computed with the
    /// remainder tree. An empty tree returns an empty vector.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the remainders do not fit in
    ///   memory.
    ///
    /// ```
    /// use scioxide_poly::{ExactPolynomial, ExactPolynomialProductTree};
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// // p(x) = (1 + x)².
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let tree = ExactPolynomialProductTree::new(&[Q(0, 1), Q(1, 1), Q(2, 1)])?;
    /// assert_eq!(tree.evaluate(&p)?, [Q(1, 1), Q(4, 1), Q(9, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, polynomial: &ExactPolynomial<C>) -> Result<Vec<C>, SciOxideError> {
        if self.points.is_empty() {
            return Ok(Vec::new());
        }
        let mut work = TreeWork::new(remainder_capacity(polynomial.coefficients().len())?)?;
        self.evaluate_with_work(polynomial, &mut work)
    }

    fn evaluate_with_work(
        &self,
        polynomial: &ExactPolynomial<C>,
        work: &mut TreeWork<C>,
    ) -> Result<Vec<C>, SciOxideError> {
        let quantity = scioxide_core::quantity!("tree_remainder_frontier");
        let mut current = try_with_capacity(self.points.len(), quantity)?;
        let mut next = try_with_capacity(self.points.len(), quantity)?;
        current.push(fast_remainder(polynomial, self.product(), work)?);
        let mut first = 1;
        let mut span = self.base;
        // Real leaves form a prefix, so live nodes at each level do too.
        while span > 1 {
            let child_span = span / 2;
            for (offset, parent) in current.drain(..).enumerate() {
                let child = 2 * (first + offset);
                next.push(fast_remainder(&parent, &self.nodes[child], work)?);
                if (2 * offset + 1) * child_span < self.points.len() {
                    next.push(fast_remainder(&parent, &self.nodes[child + 1], work)?);
                }
            }
            core::mem::swap(&mut current, &mut next);
            first *= 2;
            span = child_span;
        }
        let mut result = try_with_capacity(self.points.len(), scioxide_core::quantity!("points"))?;
        result.extend(current.into_iter().map(|p| p.coefficients()[0].clone()));
        Ok(result)
    }

    /// Returns the polynomial of degree below `n` that takes `values[i]` at the
    /// `i`-th point.
    ///
    /// The weights `1/M′(xᵢ)`, with `M = ∏ (x − xᵢ)`, come from evaluating `M′`
    /// on the remainder tree; the Lagrange terms are then combined up the tree.
    /// No points and no values give the zero polynomial.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `values.len()` differs from the number of points, or
    ///   two points are equal.
    /// - `SizeOverflow` or `AllocationFailure` if the result does not fit in
    ///   memory.
    ///
    /// ```
    /// use scioxide_poly::{ExactPolynomial, ExactPolynomialProductTree};
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// // The values 1, 4, 9 at x = 0, 1, 2 come from (1 + x)².
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let tree = ExactPolynomialProductTree::new(&[Q(0, 1), Q(1, 1), Q(2, 1)])?;
    /// let result = tree.interpolate(&[Q(1, 1), Q(4, 1), Q(9, 1)])?;
    /// assert_eq!(result, p);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn interpolate(&self, values: &[C]) -> Result<ExactPolynomial<C>, SciOxideError> {
        if values.len() != self.points.len() {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "exact interpolation value count",
            ));
        }
        if values.is_empty() {
            return ExactPolynomial::constant(C::zero());
        }
        // In exact arithmetic `M'(x_i) = prod_{k != i} (x_i - x_k)` can be evaluated
        // from the expanded derivative without cancellation concerns; a zero weight
        // is exactly a repeated point.
        let mut work = TreeWork::new(remainder_capacity(values.len())?)?;
        let weights = self.evaluate_with_work(&self.product().derivative()?, &mut work)?;
        if weights.iter().any(C::is_zero) {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "exact interpolation requires distinct points",
            ));
        }
        let mut partial =
            try_with_capacity(self.nodes.len(), scioxide_core::quantity!("tree_nodes"))?;
        for _ in 0..self.nodes.len() {
            partial.push(ExactPolynomial::constant(C::zero())?);
        }
        for (i, (y, w)) in values.iter().zip(weights).enumerate() {
            partial[self.base + i] = ExactPolynomial::constant(y.clone() / w)?;
        }
        for i in (1..self.base).rev() {
            // Linear-combination tree: `c_left * M_right + c_right * M_left` assembles the
            // Lagrange sum over the node's points (see the floating-point tree).
            partial[i] = if self.nodes[2 * i].degree() == 0 {
                partial[2 * i + 1].try_copy()?
            } else if self.nodes[2 * i + 1].degree() == 0 {
                partial[2 * i].try_copy()?
            } else {
                work.multiply(&partial[2 * i], &self.nodes[2 * i + 1])?
                    .try_add(&work.multiply(&partial[2 * i + 1], &self.nodes[2 * i])?)?
            };
        }
        Ok(partial.swap_remove(1))
    }
}
/// Returns the remainder of `a` modulo the monic tree node `b`.
fn fast_remainder<C: ExactField>(
    a: &ExactPolynomial<C>,
    b: &ExactPolynomial<C>,
    work: &mut TreeWork<C>,
) -> Result<ExactPolynomial<C>, SciOxideError> {
    if b.degree() == 0 {
        return ExactPolynomial::constant(C::zero());
    }
    if a.degree() < b.degree() {
        return a.try_copy();
    }
    let q = a.degree() - b.degree() + 1;
    if !use_newton(a.coefficients().len(), b.coefficients().len()) {
        return a.remainder(b);
    }
    // Newton inversion of `rev(b)` modulo `x^q` (precision doubling). The seed 1 is
    // exact only because every tree node is monic; this helper relies on that.
    let quantity = scioxide_core::quantity!("reciprocal_coefficients");
    let product_len = remainder_capacity(q)?;
    work.prepare_multiplication()?;
    try_grow(&mut work.inverse, q, C::zero(), quantity)?;
    try_grow(&mut work.reversed, q, C::zero(), quantity)?;
    try_grow(&mut work.error, product_len, C::zero(), quantity)?;
    try_grow(
        &mut work.correction,
        product_len.max(a.coefficients().len()),
        C::zero(),
        quantity,
    )?;
    let TreeWork {
        scratch,
        inverse,
        reversed,
        error,
        correction,
        ..
    } = work;
    reversed[..q].fill(C::zero());
    for (x, y) in reversed[..q].iter_mut().zip(b.coefficients().iter().rev()) {
        *x = y.clone();
    }
    inverse[0] = C::one();
    let mut precision = 1;
    while precision < q {
        let n = precision.saturating_mul(2).min(q);
        let len = n + precision - 1;
        karatsuba(
            &reversed[..n],
            &inverse[..precision],
            &mut error[..len],
            scratch,
            ExecutionContext::sequential_native(),
        )?;
        for x in &mut error[..n] {
            *x = -x.clone();
        }
        error[0] = error[0].clone() + C::one() + C::one();
        karatsuba(
            &inverse[..precision],
            &error[..n],
            &mut correction[..len],
            scratch,
            ExecutionContext::sequential_native(),
        )?;
        inverse[..n].clone_from_slice(&correction[..n]);
        precision = n;
    }
    // `rev(quotient) = rev(a) * rev(b)^{-1} mod x^q` uses only the top `q`
    // coefficients of `a`; the remainder is `a - quotient * b`, of which just the
    // low `deg(b)` coefficients can be nonzero.
    for (x, y) in reversed[..q].iter_mut().zip(a.coefficients().iter().rev()) {
        *x = y.clone();
    }
    karatsuba(
        &reversed[..q],
        &inverse[..q],
        &mut error[..product_len],
        scratch,
        ExecutionContext::sequential_native(),
    )?;
    error[..q].reverse();
    karatsuba(
        &error[..q],
        b.coefficients(),
        &mut correction[..a.coefficients().len()],
        scratch,
        ExecutionContext::sequential_native(),
    )?;
    let mut remainder = try_with_capacity(
        b.degree(),
        scioxide_core::quantity!("remainder_coefficients"),
    )?;
    remainder.extend(
        a.coefficients()
            .iter()
            .zip(correction.iter())
            .take(b.degree())
            .map(|(x, y)| x.clone() - y.clone()),
    );
    ExactPolynomial::new(remainder)
}
