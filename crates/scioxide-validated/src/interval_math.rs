//! Shared outward rectangle arithmetic for linear and nonlinear certificates.
use crate::{DecoratedInterval as D, rectangle::Box2};
use crate::{IntervalBox, shell::invalid};
use core::cmp::Ordering;
use scioxide_core::{AlgebraicScalar, RangeAccumulator};
use scioxide_core::{Argument, Complex, NumericScalar, RealScalar, SciOxideError};
type Result<T> = core::result::Result<T, SciOxideError>;

/// Containment of both Cartesian components.
pub(crate) fn contains<R: RealScalar>(a: IntervalBox<R>, b: IntervalBox<R>) -> bool {
    a.re.contains_interval(&b.re) && a.im.contains_interval(&b.im)
}

/// Empty components mark output slots a provider has not written.
pub(crate) fn sentinel<R: RealScalar>() -> IntervalBox<R> {
    Complex::new(
        scioxide_core::ExtendedInterval::empty(),
        scioxide_core::ExtendedInterval::empty(),
    )
}

/// Finite midpoint of a box derived from the named argument.
pub(crate) fn midpoint<S: NumericScalar>(
    value: IntervalBox<S::Real>,
    argument: Argument,
) -> Result<S> {
    unpack(value)
        .center()?
        .ok_or_else(|| invalid(argument, "expected bounded interval"))
}

pub(super) fn unpack<R: RealScalar>(value: IntervalBox<R>) -> Box2<R> {
    Box2 {
        re: value.re.into(),
        im: value.im.into(),
    }
}
pub(super) fn pack<R: RealScalar>(value: Box2<R>) -> IntervalBox<R> {
    Complex::new(value.re.set(), value.im.set())
}
pub(super) fn zero<R: RealScalar>() -> Result<IntervalBox<R>> {
    Ok(pack(Box2::zero()?))
}
/// Validate the boxes of `argument`.
pub(super) fn validate_boxes<S: NumericScalar>(
    values: &[IntervalBox<S::Real>],
    bounded: bool,
    argument: Argument,
) -> Result<()> {
    values.iter().try_for_each(|&v| {
        let v = unpack(v);
        if v.empty() || (bounded && !v.finite()) || (S::IS_REAL && !v.im_is_zero()) {
            Err(invalid( argument, "interval components must be nonempty, bounded on input, and real for a real scalar"))
        } else { Ok(()) }
    })
}
/// Finite endpoints of a value derived from `argument`.
pub(super) fn bounded<R: RealScalar>(value: D<R>, argument: Argument) -> Result<(R, R)> {
    value
        .bounds()
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .ok_or_else(|| invalid(argument, "expected bounded interval"))
}
pub(super) fn upper<R: RealScalar>(value: D<R>) -> R {
    value.bounds().map_or(R::INFINITY, |(_, hi)| hi)
}
/// Upward-rounded sum.
pub(super) fn add_up<R: RealScalar>(a: R, b: R) -> R {
    crate::rounding::endpoint(a, b, crate::rounding::Operation::Add, true)
}
/// Upward-rounded product.
pub(super) fn mul_up<R: RealScalar>(a: R, b: R) -> R {
    crate::rounding::endpoint(a, b, crate::rounding::Operation::Multiply, true)
}
/// One streamed entry of I-RJ and its contribution to the real-block row norm.
pub(crate) fn defect_entry<R: RealScalar>(
    identity: IntervalBox<R>,
    product: IntervalBox<R>,
    row: &mut R,
) -> Result<Box2<R>> {
    let e = unpack(identity).sub(unpack(product))?;
    let (re, im) = e.radii()?;
    *row = add_up(*row, add_up(re, im));
    Ok(e)
}
pub(super) fn scale<R: RealScalar>(mut x: Box2<R>, mut exponent: i32) -> Result<Box2<R>> {
    // Bounded factors avoid constructing an overflowing power of two. Each
    // multiplication encloses subnormal rounding, including destructive underflow.
    while exponent != 0 {
        let step = exponent.clamp(-64, 64);
        x = x.mul(Box2::real(D::point(R::ONE.scalbn(step))?)?)?;
        exponent -= step;
    }
    Ok(x)
}

/// Exact endpoint sums.
#[derive(Debug, Clone)]
struct BoundSum<R: RealScalar> {
    lo: RangeAccumulator<R>,
    hi: RangeAccumulator<R>,

    argument: Argument,
}
impl<R: RealScalar> BoundSum<R> {
    fn new(argument: Argument) -> Self {
        Self {
            lo: RangeAccumulator::new(),
            hi: RangeAccumulator::new(),

            argument,
        }
    }
    fn add(&mut self, x: D<R>) -> Result<()> {
        let (lo, hi) = bounded(x, self.argument)?;
        self.lo.add(lo)?;
        self.hi.add(hi)
    }
    fn product(&mut self, a: D<R>, b: D<R>, negative: bool) -> Result<()> {
        let (al, ah) = bounded(a, self.argument)?;
        let (al, ah) = if negative { (-ah, -al) } else { (al, ah) };
        let (bl, bh) = bounded(b, self.argument)?;
        let (low, high) = crate::rounding::product_corners(al, ah, bl, bh);
        self.lo.add_product(low.0, low.1)?;
        self.hi.add_product(high.0, high.1)
    }
    fn finish(self) -> Result<D<R>> {
        fn endpoint<R: RealScalar>(mut sum: RangeAccumulator<R>, up: bool) -> Result<R> {
            let x = match sum.finish() {
                Ok(x) => x,
                Err(error) => match error.view() {
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    } => {
                        return Ok(match (sum.sign(), up) {
                            (Ordering::Less, false) => -R::INFINITY,
                            (Ordering::Less, true) => -R::MAX,
                            (_, false) => R::MAX,
                            (_, true) => R::INFINITY,
                        });
                    }
                    _ => return Err(error),
                },
            };
            sum.add(-x)?;
            Ok(match (sum.sign(), up) {
                (Ordering::Greater, true) => x.next_after(R::INFINITY),
                (Ordering::Less, false) => x.next_after(-R::INFINITY),
                _ => x,
            })
        }
        D::result(
            endpoint(self.lo, false)?,
            endpoint(self.hi, true)?,
            crate::Decoration::Com,
        )
    }
}
/// Endpoint sums are exact until final outward rounding, including products
/// outside scalar range which subsequently cancel.
#[derive(Debug, Clone)]
pub(super) struct BoxSum<R: RealScalar> {
    re: BoundSum<R>,
    im: BoundSum<R>,
}
impl<R: RealScalar> BoxSum<R> {
    /// Start a sum of values derived from `argument`.
    pub(crate) fn new(addend: Box2<R>, argument: Argument) -> Result<Self> {
        let mut s = Self {
            re: BoundSum::new(argument),
            im: BoundSum::new(argument),
        };
        s.re.add(addend.re)?;
        s.im.add(addend.im)?;
        Ok(s)
    }
    pub(crate) fn product(&mut self, a: Box2<R>, b: Box2<R>, negative: bool) -> Result<()> {
        self.re.product(a.re, b.re, negative)?;
        self.re.product(a.im, b.im, !negative)?;
        self.im.product(a.re, b.im, negative)?;
        self.im.product(a.im, b.re, negative)
    }
    /// Add or remove corresponding endpoint contributions from an exact sum.
    /// Removing a summand differs from adding its interval negation.
    pub(crate) fn update(&mut self, value: Box2<R>, remove: bool) -> Result<()> {
        for (sum, value) in [(&mut self.re, value.re), (&mut self.im, value.im)] {
            let (lo, hi) = bounded(value, sum.argument)?;
            sum.lo.add(if remove { -lo } else { lo })?;
            sum.hi.add(if remove { -hi } else { hi })?;
        }
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<Box2<R>> {
        Ok(Box2 {
            re: self.re.finish()?,
            im: self.im.finish()?,
        })
    }
}

/// Divide by a triangular pivot of the factor `self`.
pub(super) fn divide_point<S: NumericScalar>(
    numerator: Box2<S::Real>,
    denominator: S,
) -> Result<Box2<S::Real>> {
    if S::IS_REAL {
        return numerator.div(Box2::point(denominator)?, false);
    }
    let magnitude = denominator
        .real_component()
        .abs()
        .max(denominator.imaginary_component().abs());
    if magnitude == S::Real::ZERO || !magnitude.is_finite() {
        return Err(invalid(
            scioxide_core::argument!("self"),
            "invalid triangular pivot",
        ));
    }
    let exponent = scioxide_core::backend::standard::exponent(magnitude)? - 1;
    // Scaling prevents underflow/overflow in the complex squared modulus.
    scale(numerator, -exponent)?.div(scale(Box2::point(denominator)?, -exponent)?, true)
}
