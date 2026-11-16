//! Component interval algebra and second-order Taylor proofs. No scalar libm
//! approximation is treated as a certificate.
use super::{
    AffineBinary as B, AffineFallback, AffineTerm, AffineUnary as U, AffineView, Operation,
};
use crate::{DecoratedInterval as D, Decoration, EnclosureStatus as Status};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use crate::elementary::integer_exponent;
use crate::rectangle::{Box2, constant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Unary(U),
    Binary(B),
}
fn same_box<S: RealScalar>(a: Box2<S>, b: Box2<S>) -> bool {
    a.re == b.re && a.im == b.im
}
fn same_other<S: RealScalar>(a: Option<Box2<S>>, b: Option<Box2<S>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => same_box(a, b),
        _ => false,
    }
}
fn canonical<S: RealScalar>(x: Box2<S>) -> Box2<S> {
    Box2 {
        re: D::from(x.re.set()),
        im: D::from(x.im.set()),
    }
}

/// Operation-local certified values. The exact linear/product routes need no cache.
trait Values<S: RealScalar> {
    #[cfg(test)]
    fn record_evaluation(&mut self) {}
    fn kernel(
        &mut self,
        x: Box2<S>,
        other: Option<Box2<S>>,
        operation: Kernel,
        complex: bool,
    ) -> Result<(Box2<S>, Status), SciOxideError>;
    fn classified(
        &mut self,
        x: Box2<S>,
        op: U,
        complex: bool,
    ) -> Result<(Box2<S>, Status), SciOxideError> {
        self.kernel(x, None, Kernel::Unary(op), complex)
    }
    fn binary(
        &mut self,
        x: Box2<S>,
        y: Box2<S>,
        op: B,
        complex: bool,
    ) -> Result<(Box2<S>, Status), SciOxideError> {
        self.kernel(x, Some(y), Kernel::Binary(op), complex)
    }
    fn real_binary(&mut self, x: D<S>, y: D<S>, op: B) -> Result<D<S>, SciOxideError> {
        Ok(self.binary(Box2::real(x)?, Box2::real(y)?, op, false)?.0.re)
    }
    fn real(&mut self, x: D<S>, op: U) -> Result<D<S>, SciOxideError> {
        Ok(self.value(Box2::real(x)?, op, false)?.re)
    }
    fn value(&mut self, x: Box2<S>, op: U, complex: bool) -> Result<Box2<S>, SciOxideError> {
        Ok(self.classified(x, op, complex)?.0)
    }
}

struct DirectValues;
impl<S: RealScalar> Values<S> for DirectValues {
    fn kernel(
        &mut self,
        x: Box2<S>,
        other: Option<Box2<S>>,
        operation: Kernel,
        complex: bool,
    ) -> Result<(Box2<S>, Status), SciOxideError> {
        classified_kernel(x, other, operation, complex, self)
    }
}

#[derive(Clone, Copy)]
struct CertifiedValue<S: RealScalar> {
    argument: Box2<S>,
    other: Option<Box2<S>>,
    operation: Kernel,
    complex: bool,
    result: (Box2<S>, Status),
}
/// The finite expression graphs use fewer than 32 distinct certified nodes, including
/// range, center, supporting rectangle and derivative auxiliaries. No data is
/// kept between public calls and no scratch allocation is needed.
struct CertifiedValues<S: RealScalar> {
    entries: [Option<CertifiedValue<S>>; 32],
    len: usize,
    #[cfg(test)]
    evaluations: usize,
}
impl<S: RealScalar> CertifiedValues<S> {
    fn new() -> Self {
        Self {
            entries: [None; 32],
            len: 0,
            #[cfg(test)]
            evaluations: 0,
        }
    }
}
impl<S: RealScalar> Values<S> for CertifiedValues<S> {
    #[cfg(test)]
    fn record_evaluation(&mut self) {
        self.evaluations += 1;
    }
    fn kernel(
        &mut self,
        x: Box2<S>,
        other: Option<Box2<S>>,
        operation: Kernel,
        complex: bool,
    ) -> Result<(Box2<S>, Status), SciOxideError> {
        // Decorations do not change numerical arguments. Compute with the strongest
        // admissible decorations and weaken each use independently.
        let decoration = x
            .decoration()
            .min(other.map_or(Decoration::Com, Box2::decoration));
        let x = canonical(x);
        let other = other.map(canonical);
        for entry in self.entries[..self.len].iter().flatten() {
            if entry.operation == operation
                && entry.complex == complex
                && same_box(entry.argument, x)
                && same_other(entry.other, other)
            {
                return Ok((entry.result.0.decorate(decoration)?, entry.result.1));
            }
        }
        let result = classified_kernel(x, other, operation, complex, self)?;
        let entry = self.entries.get_mut(self.len).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("affine.certified_values"))
        })?;
        *entry = Some(CertifiedValue {
            argument: x,
            other,
            operation,
            complex,
            result,
        });
        self.len += 1;
        Ok((result.0.decorate(decoration)?, result.1))
    }
}
fn classified_kernel<S: RealScalar>(
    x: Box2<S>,
    other: Option<Box2<S>>,
    operation: Kernel,
    complex: bool,
    values: &mut impl Values<S>,
) -> Result<(Box2<S>, Status), SciOxideError> {
    #[cfg(test)]
    values.record_evaluation();
    match operation {
        Kernel::Unary(op) => classified_value(x, op, complex, values),
        Kernel::Binary(op) => binary_range(
            x,
            other.ok_or_else(|| crate::shell::breakdown("missing binary operand"))?,
            op,
            complex,
            values,
        ),
    }
}

fn classified_value<S: RealScalar>(
    x: Box2<S>,
    op: U,
    complex: bool,
    values: &mut impl Values<S>,
) -> Result<(Box2<S>, Status), SciOxideError> {
    if x.empty() {
        return Ok((x, Status::EmptyInput));
    }
    if !complex {
        let (range, status) = real_value(x.re, op)?;
        return Ok((Box2::real(range)?, status));
    }
    let result = complex_value(x, op, values)?;
    let status = if result.empty() {
        Status::Undefined
    } else if touches_cut(x, op)
        || (op == U::Tan && values.value(x, U::Cos, true)?.contains_zero())
        || (op == U::Tanh && values.value(x, U::Cosh, true)?.contains_zero())
    {
        Status::Inconclusive
    } else {
        Status::Defined
    };
    Ok((result, status))
}

// A sign selected over a rectangle includes both sides on an axis. This avoids
// incorrectly interpreting a stored zero bit as a one-sided set restriction.
fn signed<S: RealScalar>(absolute: D<S>, sign: D<S>) -> Result<D<S>, SciOxideError> {
    let Some((lo, hi)) = sign.bounds() else {
        return Ok(D::empty());
    };
    if lo > S::ZERO {
        Ok(absolute)
    } else if hi < S::ZERO {
        absolute.neg_admitted()
    } else {
        Ok(absolute.hull(&absolute.neg_admitted()?))
    }
}

/// Keep both sides of a complex branch cut, irrespective of stored zero signs.
fn argument<S: RealScalar>(x: Box2<S>, values: &mut impl Values<S>) -> Result<D<S>, SciOxideError> {
    let result = values.real_binary(x.im, x.re, B::Atan2)?;
    if x.im.contains(S::ZERO) && x.re.bounds().is_some_and(|(lo, _)| lo < S::ZERO) {
        Ok(result.hull(&result.neg_admitted()?))
    } else {
        Ok(result)
    }
}

fn complex_value<S: RealScalar>(
    x: Box2<S>,
    op: U,
    values: &mut impl Values<S>,
) -> Result<Box2<S>, SciOxideError> {
    let one = Box2::one()?;
    match op {
        U::Neg => x.neg(),
        U::Conj => x.conj(),
        U::Square => x.square(),
        U::Reciprocal => x.reciprocal(true),
        U::Powi(n) => x.powi(n, true),
        U::Exp => {
            let radius = values.real(x.re, U::Exp)?;
            Ok(Box2 {
                re: radius.mul_admitted(&values.real(x.im, U::Cos)?)?,
                im: radius.mul_admitted(&values.real(x.im, U::Sin)?)?,
            })
        }
        U::Exp2 | U::Exp10 => {
            let base = values.real(constant(if op == U::Exp2 { 2.0 } else { 10.0 })?, U::Ln)?;
            values.value(x.mul(Box2::real(base)?)?, U::Exp, true)
        }
        U::ExpM1 => values.value(x, U::Exp, true)?.sub(one),
        U::Ln => Ok(Box2 {
            re: {
                let radius = values.real_binary(x.re, x.im, B::Hypot)?;
                values.real(radius, U::Ln)?
            },
            im: argument(x, values)?,
        }),
        U::Ln1p => values.value(x.add(one)?, U::Ln, true),
        U::Log2 | U::Log10 => {
            let divisor = values.real(constant(if op == U::Log2 { 2.0 } else { 10.0 })?, U::Ln)?;
            values
                .value(x, U::Ln, true)?
                .div(Box2::real(divisor)?, true)
        }
        U::Sqrt => {
            let radius = values.real_binary(x.re, x.im, B::Hypot)?;
            let re = values.real(
                radius.add_admitted(&x.re)?.mul_admitted(&constant(0.5)?)?,
                U::Sqrt,
            )?;
            let im = values.real(
                radius.sub_admitted(&x.re)?.mul_admitted(&constant(0.5)?)?,
                U::Sqrt,
            )?;
            Ok(Box2 {
                re,
                im: signed(im, x.im)?,
            })
        }
        U::Cbrt => {
            if x.is_zero() {
                return Box2::zero();
            }
            let magnitude = values.real_binary(x.re, x.im, B::Hypot)?;
            let radius = values.real(magnitude, U::Cbrt)?;
            let phase = argument(x, values)?.div_admitted(&constant(3.0)?)?;
            Ok(Box2 {
                re: radius.mul_admitted(&values.real(phase, U::Cos)?)?,
                im: radius.mul_admitted(&values.real(phase, U::Sin)?)?,
            })
        }
        U::Sin => Ok(Box2 {
            re: values
                .real(x.re, U::Sin)?
                .mul_admitted(&values.real(x.im, U::Cosh)?)?,
            im: values
                .real(x.re, U::Cos)?
                .mul_admitted(&values.real(x.im, U::Sinh)?)?,
        }),
        U::Cos => Ok(Box2 {
            re: values
                .real(x.re, U::Cos)?
                .mul_admitted(&values.real(x.im, U::Cosh)?)?,
            im: values
                .real(x.re, U::Sin)?
                .mul_admitted(&values.real(x.im, U::Sinh)?)?
                .neg_admitted()?,
        }),
        U::Sinh => Ok(Box2 {
            re: values
                .real(x.re, U::Sinh)?
                .mul_admitted(&values.real(x.im, U::Cos)?)?,
            im: values
                .real(x.re, U::Cosh)?
                .mul_admitted(&values.real(x.im, U::Sin)?)?,
        }),
        U::Cosh => Ok(Box2 {
            re: values
                .real(x.re, U::Cosh)?
                .mul_admitted(&values.real(x.im, U::Cos)?)?,
            im: values
                .real(x.re, U::Sinh)?
                .mul_admitted(&values.real(x.im, U::Sin)?)?,
        }),
        U::Tan => {
            let (re, im) = tangent(x.re, x.im, values)?;
            Ok(Box2 { re, im })
        }
        // tanh(x + iy) = −i tan(i(x + iy)) swaps the components of tan(y + ix).
        U::Tanh => {
            let (im, re) = tangent(x.im, x.re, values)?;
            Ok(Box2 { re, im })
        }
        U::Asin | U::Acos => {
            // Real/imaginary decomposition using distances to +/-1; unlike a
            // naive log/sqrt identity this respects both principal cut sides.
            let plus = values.real_binary(x.re.add_admitted(&constant(1.0)?)?, x.im, B::Hypot)?;
            let minus = values.real_binary(x.re.sub_admitted(&constant(1.0)?)?, x.im, B::Hypot)?;
            let a = plus.add_admitted(&minus)?.mul_admitted(&constant(0.5)?)?;
            let a = D::from(a.set().intersection(
                &D::from(scioxide_core::ExtendedInterval::new(Some(S::ONE), None)?).set(),
            ));
            let b = x.re.div_admitted(&a)?;
            let b = D::from(b.set().intersection(&D::new(-S::ONE, S::ONE)?.set()));
            let imaginary = signed(values.real(a, U::Acosh)?, x.im)?;
            if op == U::Asin {
                Ok(Box2 {
                    re: values.real(b, U::Asin)?,
                    im: imaginary,
                })
            } else {
                Ok(Box2 {
                    re: values.real(b, U::Acos)?,
                    im: imaginary.neg_admitted()?,
                })
            }
        }
        U::Asinh => values.value(x.times_i()?, U::Asin, true)?.times_i()?.neg(),
        U::Acosh => {
            let acos = values.value(x, U::Acos, true)?;
            Ok(Box2 {
                re: acos.im.abs_admitted()?,
                im: signed(acos.re, x.im)?,
            })
        }
        U::Atanh => {
            let plus = values.real_binary(x.re.add_admitted(&constant(1.0)?)?, x.im, B::Hypot)?;
            let plus = values.real(plus, U::Ln)?;
            let minus = values.real_binary(x.re.sub_admitted(&constant(1.0)?)?, x.im, B::Hypot)?;
            let minus = values.real(minus, U::Ln)?;
            let denominator = constant(1.0)?
                .sub_admitted(&x.re.square_admitted()?)?
                .sub_admitted(&x.im.square_admitted()?)?;
            let angle = values
                .real_binary(x.im.mul_admitted(&constant(2.0)?)?, denominator, B::Atan2)?
                .mul_admitted(&constant(0.5)?)?;
            Ok(Box2 {
                re: plus.sub_admitted(&minus)?.mul_admitted(&constant(0.5)?)?,
                im: signed(angle.abs_admitted()?, x.im)?,
            })
        }
        U::Atan => values.value(x.times_i()?, U::Atanh, true)?.times_i()?.neg(),
    }
}

/// Enclose the components of `tan(p + iq)`.
///
/// The textbook quotient `(sin 2p + i sinh 2q)/(cos 2p + cosh 2q)` has the denominator
/// `2(cos²p + sinh²q)`; dividing both by `2 cosh²q` gives
///
/// ```text
/// tan(p + iq) = (sin p cos p / cosh²q + i tanh q) / (cos²p / cosh²q + tanh²q),
/// ```
///
/// whose factors are bounded for every `q` and whose denominator sums nonnegative
/// terms, vanishing only at the poles `q = 0`, `cos p = 0`.
fn tangent<S: RealScalar>(
    p: D<S>,
    q: D<S>,
    values: &mut impl Values<S>,
) -> Result<(D<S>, D<S>), SciOxideError> {
    let cos = values.real(p, U::Cos)?;
    let cosh = values.real(q, U::Cosh)?.square_admitted()?;
    let tanh = values.real(q, U::Tanh)?;
    let denominator = cos
        .square_admitted()?
        .div_admitted(&cosh)?
        .add_admitted(&tanh.square_admitted()?)?;
    Ok((
        values
            .real(p, U::Sin)?
            .mul_admitted(&cos)?
            .div_admitted(&cosh)?
            .div_admitted(&denominator)?,
        tanh.div_admitted(&denominator)?,
    ))
}

fn real_value<S: RealScalar>(x: D<S>, op: U) -> Result<(D<S>, Status), SciOxideError> {
    let zero = S::ZERO;
    let result = match op {
        U::Conj | U::Neg | U::Square | U::Reciprocal | U::Sqrt => {
            let value = match op {
                U::Neg => x.neg_admitted()?,
                U::Square => x.square_admitted()?,
                U::Reciprocal => x.reciprocal_admitted()?,
                U::Sqrt => x.sqrt_admitted()?,
                _ => x,
            };
            let status = if x.set().is_empty() {
                Status::EmptyInput
            } else if value.set().is_empty() {
                Status::Undefined
            } else if (op == U::Reciprocal && x.contains(zero))
                || (op == U::Sqrt && x.bounds().is_some_and(|(a, _)| a < zero))
            {
                Status::PartiallyDefined
            } else {
                Status::Defined
            };
            return Ok((value, status));
        }
        U::Exp => x.exp_admitted()?,
        U::Exp2 => x.exp2_admitted()?,
        U::Exp10 => x.exp10_admitted()?,
        U::ExpM1 => x.exp_m1_admitted()?,
        U::Ln => x.ln_admitted()?,
        U::Log2 => x.log2_admitted()?,
        U::Log10 => x.log10_admitted()?,
        U::Ln1p => x.ln_1p_admitted()?,
        U::Sin => x.sin_admitted()?,
        U::Cos => x.cos_admitted()?,
        U::Tan => x.tan_admitted()?,
        U::Asin => x.asin_admitted()?,
        U::Acos => x.acos_admitted()?,
        U::Atan => x.atan_admitted()?,
        U::Sinh => x.sinh_admitted()?,
        U::Cosh => x.cosh_admitted()?,
        U::Tanh => x.tanh_admitted()?,
        U::Asinh => x.asinh_admitted()?,
        U::Acosh => x.acosh_admitted()?,
        U::Atanh => x.atanh_admitted()?,
        U::Cbrt => x.cbrt_admitted()?,
        U::Powi(n) => x.powi_admitted(n)?,
    };
    Ok((result.hull(), result.status()))
}
// Return f', f'' on the whole supporting rectangle. Holomorphic formulas are
// used only after branch admission. The real cube-root derivative is identical
// on each component of R\{0}, using the real root supplied by `value`.
fn derivatives<S: RealScalar>(
    x: Box2<S>,
    op: U,
    complex: bool,
    values: &mut impl Values<S>,
) -> Result<(Box2<S>, Box2<S>), SciOxideError> {
    let one = Box2::one()?;
    let zero = Box2::zero()?;
    match op {
        U::Conj => Ok((one, zero)),
        U::Neg => Ok((one.neg()?, zero)),
        U::Square => Ok((x.scale(2.0)?, Box2::constant(2.0)?)),
        U::Reciprocal => {
            let r = values.value(x, U::Reciprocal, complex)?;
            let square = r.square()?;
            Ok((square.neg()?, square.mul(r)?.scale(2.0)?))
        }
        U::Exp | U::ExpM1 => {
            let r = values.value(x, U::Exp, complex)?;
            Ok((r, r))
        }
        U::Exp2 | U::Exp10 => {
            let l =
                Box2::real(values.real(constant(if op == U::Exp2 { 2.0 } else { 10.0 })?, U::Ln)?)?;
            let r = values.value(x, op, complex)?;
            Ok((r.mul(l)?, r.mul(l.square()?)?))
        }
        U::Ln | U::Log2 | U::Log10 | U::Ln1p => {
            let base = if op == U::Ln1p { x.add(one)? } else { x };
            let r = base.reciprocal(complex)?;
            let factor = if matches!(op, U::Log2 | U::Log10) {
                Box2::real(values.real(constant(if op == U::Log2 { 2.0 } else { 10.0 })?, U::Ln)?)?
            } else {
                one
            };
            Ok((
                r.div(factor, complex)?,
                r.square()?.neg()?.div(factor, complex)?,
            ))
        }
        U::Sqrt => {
            let r = values.value(x, U::Sqrt, complex)?;
            Ok((
                r.reciprocal(complex)?.scale(0.5)?,
                x.mul(r)?.reciprocal(complex)?.scale(-0.25)?,
            ))
        }
        U::Cbrt => {
            let r = values.value(x, U::Cbrt, complex)?;
            let first = r.square()?.scale(3.0)?.reciprocal(complex)?;
            Ok((
                first,
                first
                    .div(x, complex)?
                    .scale(-2.0)?
                    .div(Box2::constant(3.0)?, complex)?,
            ))
        }
        U::Sin => Ok((
            values.value(x, U::Cos, complex)?,
            values.value(x, U::Sin, complex)?.neg()?,
        )),
        U::Cos => Ok((
            values.value(x, U::Sin, complex)?.neg()?,
            values.value(x, U::Cos, complex)?.neg()?,
        )),
        U::Sinh => Ok((
            values.value(x, U::Cosh, complex)?,
            values.value(x, U::Sinh, complex)?,
        )),
        U::Cosh => Ok((
            values.value(x, U::Sinh, complex)?,
            values.value(x, U::Cosh, complex)?,
        )),
        // tan' = 1 + tan² and tanh' = 1 − tanh² stay bounded where cos and cosh
        // overflow; tan'' = 2 tan tan' and tanh'' = −2 tanh tanh'.
        U::Tan | U::Tanh => {
            let r = values.value(x, op, complex)?;
            let first = if op == U::Tan {
                one.add(r.square()?)?
            } else {
                one.sub(r.square()?)?
            };
            Ok((
                first,
                first.mul(r)?.scale(if op == U::Tan { 2.0 } else { -2.0 })?,
            ))
        }
        U::Asin | U::Acos => {
            let d = one.sub(x.square()?)?;
            let r = values.value(d, U::Sqrt, complex)?;
            let first = r.reciprocal(complex)?;
            let second = x.div(d.mul(r)?, complex)?;
            if op == U::Asin {
                Ok((first, second))
            } else {
                Ok((first.neg()?, second.neg()?))
            }
        }
        U::Atan => {
            let d = one.add(x.square()?)?;
            Ok((
                d.reciprocal(complex)?,
                x.scale(-2.0)?.div(d.square()?, complex)?,
            ))
        }
        U::Asinh => {
            let d = one.add(x.square()?)?;
            let r = values.value(d, U::Sqrt, complex)?;
            Ok((r.reciprocal(complex)?, x.neg()?.div(d.mul(r)?, complex)?))
        }
        U::Acosh => {
            // Product of principal roots, not sqrt(z*z-1), fixes the sheet.
            let r = values
                .value(x.sub(one)?, U::Sqrt, complex)?
                .mul(values.value(x.add(one)?, U::Sqrt, complex)?)?;
            Ok((
                r.reciprocal(complex)?,
                x.neg()?.div(x.square()?.sub(one)?.mul(r)?, complex)?,
            ))
        }
        U::Atanh => {
            let d = one.sub(x.square()?)?;
            Ok((
                d.reciprocal(complex)?,
                x.scale(2.0)?.div(d.square()?, complex)?,
            ))
        }
        U::Powi(n) => {
            if n == 0 {
                return Ok((zero, zero));
            }
            if n == 1 {
                return Ok((one, zero));
            }
            // Avoid n-1/n-2 integer overflow, including i32::MIN. Away from
            // zero use f/x; at zero the nonnegative polynomial path is exact.
            let factor = Box2::real(constant(f64::from(n))?)?;
            if n >= 2 {
                let first = values.value(x, U::Powi(n - 1), complex)?.mul(factor)?;
                let second = values
                    .value(x, U::Powi(n - 2), complex)?
                    .mul(factor)?
                    .mul(Box2::real(constant(f64::from(n) - 1.0)?)?)?;
                Ok((first, second))
            } else {
                let first = values.value(x, op, complex)?.div(x, complex)?.mul(factor)?;
                let second = first
                    .div(x, complex)?
                    .mul(Box2::real(constant(f64::from(n) - 1.0)?)?)?;
                Ok((first, second))
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Jet<S: RealScalar> {
    v: Box2<S>,
    x: Box2<S>,
    y: Box2<S>,
    xx: Box2<S>,
    xy: Box2<S>,
    yy: Box2<S>,
}
impl<S: RealScalar> Jet<S> {
    fn input(v: Box2<S>, second: bool) -> Result<Self, SciOxideError> {
        let zero = Box2::zero()?;
        let one = Box2::one()?;
        Ok(Self {
            v,
            x: if second { zero } else { one },
            y: if second { one } else { zero },
            xx: zero,
            xy: zero,
            yy: zero,
        })
    }
    fn add(self, b: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            v: self.v.add(b.v)?,
            x: self.x.add(b.x)?,
            y: self.y.add(b.y)?,
            xx: self.xx.add(b.xx)?,
            xy: self.xy.add(b.xy)?,
            yy: self.yy.add(b.yy)?,
        })
    }
    fn neg(self) -> Result<Self, SciOxideError> {
        Ok(Self {
            v: self.v.neg()?,
            x: self.x.neg()?,
            y: self.y.neg()?,
            xx: self.xx.neg()?,
            xy: self.xy.neg()?,
            yy: self.yy.neg()?,
        })
    }
    fn mul(self, b: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            v: self.v.mul(b.v)?,
            x: self.x.mul(b.v)?.add(self.v.mul(b.x)?)?,
            y: self.y.mul(b.v)?.add(self.v.mul(b.y)?)?,
            xx: self
                .xx
                .mul(b.v)?
                .add(self.x.mul(b.x)?.scale(2.0)?)?
                .add(self.v.mul(b.xx)?)?,
            xy: self
                .xy
                .mul(b.v)?
                .add(self.x.mul(b.y)?)?
                .add(self.y.mul(b.x)?)?
                .add(self.v.mul(b.xy)?)?,
            yy: self
                .yy
                .mul(b.v)?
                .add(self.y.mul(b.y)?.scale(2.0)?)?
                .add(self.v.mul(b.yy)?)?,
        })
    }
    fn unary(
        self,
        op: U,
        complex: bool,
        values: &mut impl Values<S>,
    ) -> Result<Self, SciOxideError> {
        let (first, second) = derivatives(self.v, op, complex, values)?;
        Ok(Self {
            v: values.value(self.v, op, complex)?,
            x: first.mul(self.x)?,
            y: first.mul(self.y)?,
            xx: second.mul(self.x.square()?)?.add(first.mul(self.xx)?)?,
            xy: second.mul(self.x.mul(self.y)?)?.add(first.mul(self.xy)?)?,
            yy: second.mul(self.y.square()?)?.add(first.mul(self.yy)?)?,
        })
    }
}

fn binary_jet<S: RealScalar>(
    a: Box2<S>,
    b: Box2<S>,
    op: B,
    complex: bool,
    values: &mut impl Values<S>,
) -> Result<Jet<S>, SciOxideError> {
    let x = Jet::input(a, false)?;
    let y = Jet::input(b, true)?;
    match op {
        B::Add => x.add(y),
        B::Sub => x.add(y.neg()?),
        B::Mul => x.mul(y),
        B::Div => x.mul(y.unary(U::Reciprocal, complex, values)?),
        B::Pow => x
            .unary(U::Ln, complex, values)?
            .mul(y)?
            .unary(U::Exp, complex, values),
        B::Log => x
            .unary(U::Ln, complex, values)?
            .mul(
                y.unary(U::Ln, complex, values)?
                    .unary(U::Reciprocal, complex, values)?,
            ),
        B::Hypot => x
            .unary(U::Square, false, values)?
            .add(y.unary(U::Square, false, values)?)?
            .unary(U::Sqrt, false, values),
        B::Atan2 => {
            // atan2(a,b): gradient (b,-a)/(a*a+b*b), including correct quadrants.
            let d = a.square()?.add(b.square()?)?;
            let d2 = d.square()?;
            Ok(Jet {
                v: values.binary(a, b, B::Atan2, false)?.0,
                x: b.div(d, false)?,
                y: a.neg()?.div(d, false)?,
                xx: a.mul(b)?.scale(-2.0)?.div(d2, false)?,
                xy: a.square()?.sub(b.square()?)?.div(d2, false)?,
                yy: a.mul(b)?.scale(2.0)?.div(d2, false)?,
            })
        }
    }
}

fn touches_cut<S: RealScalar>(x: Box2<S>, op: U) -> bool {
    let zero = S::ZERO;
    let one = S::ONE;
    let Some((rl, rh)) = x.re.bounds() else {
        return true;
    };
    let Some((il, ih)) = x.im.bounds() else {
        return true;
    };
    match op {
        U::Ln | U::Log2 | U::Log10 | U::Sqrt | U::Cbrt => il <= zero && ih >= zero && rl <= zero,
        U::Ln1p => il <= zero && ih >= zero && rl <= -one,
        U::Asin | U::Acos | U::Atanh => il <= zero && ih >= zero && (rl <= -one || rh >= one),
        U::Acosh => il <= zero && ih >= zero && rl <= one,
        U::Asinh | U::Atan => rl <= zero && rh >= zero && (il <= -one || ih >= one),
        U::Reciprocal => x.contains_zero(),
        U::Powi(n) => n < 0 && x.contains_zero(),
        _ => false,
    }
}
fn smooth<S: RealScalar>(
    x: Box2<S>,
    op: U,
    complex: bool,
    values: &mut impl Values<S>,
) -> Result<bool, SciOxideError> {
    let singular_derivative = !complex
        && match op {
            U::Sqrt | U::Cbrt => x.re.contains(S::ZERO),
            U::Asin | U::Acos => x.re.contains(-S::ONE) || x.re.contains(S::ONE),
            U::Acosh => x.re.contains(S::ONE),
            _ => false,
        };
    Ok(x.finite()
        && values.classified(x, op, complex)?.1 == Status::Defined
        && !singular_derivative)
}

/// Binary range for the numerical operator `op`.
fn binary_range<S: RealScalar>(
    a: Box2<S>,
    b: Box2<S>,
    op: B,
    complex: bool,
    values: &mut impl Values<S>,
) -> Result<(Box2<S>, Status), SciOxideError> {
    if a.empty() || b.empty() {
        return Ok((
            Box2 {
                re: D::empty(),
                im: if complex { D::empty() } else { constant(0.0)? },
            },
            Status::EmptyInput,
        ));
    }
    let (r, status) = if !complex {
        match op {
            B::Add => (a.re.add_admitted(&b.re)?, Status::Defined),
            B::Sub => (a.re.sub_admitted(&b.re)?, Status::Defined),
            B::Mul => (a.re.mul_admitted(&b.re)?, Status::Defined),
            B::Div => {
                let r = a.re.div_admitted(&b.re)?;
                (
                    r,
                    if r.set().is_empty() {
                        Status::Undefined
                    } else if b.re.contains(S::ZERO) {
                        Status::PartiallyDefined
                    } else {
                        Status::Defined
                    },
                )
            }
            B::Atan2 => {
                let r = a.re.atan2_admitted(&b.re)?;
                (r.hull(), r.status())
            }
            B::Hypot => {
                let r = a.re.hypot_admitted(&b.re)?;
                (r.hull(), r.status())
            }
            B::Pow => {
                let r = a.re.powf_admitted(&b.re)?;
                (r.hull(), r.status())
            }
            B::Log => {
                let r = a.re.log_admitted(&b.re)?;
                (r.hull(), r.status())
            }
        }
    } else {
        let range = match op {
            B::Add => a.add(b)?,
            B::Sub => a.sub(b)?,
            B::Mul => a.mul(b)?,
            B::Div => a.div(b, true)?,
            B::Pow => {
                if b.is_zero() {
                    Box2::one()?
                } else if a.is_zero() {
                    // Continuous extension at zero for Re(exponent)>0;
                    // 0^0=1. Other zero-base tuples are outside the domain.
                    let has_positive = b.re.bounds().is_some_and(|(_, hi)| hi > S::ZERO);
                    let has_zero = b.contains_zero();
                    let result = if has_positive && has_zero {
                        Box2::real(D::new(S::ZERO, S::ONE)?)?
                    } else if has_positive {
                        Box2::zero()?
                    } else if has_zero {
                        Box2::one()?
                    } else {
                        Box2 {
                            re: D::empty(),
                            im: D::empty(),
                        }
                    };
                    let status = if result.empty() {
                        Status::Undefined
                    } else if b.re.bounds().is_some_and(|(lo, _)| lo > S::ZERO) {
                        Status::Defined
                    } else {
                        Status::Inconclusive
                    };
                    return Ok((result, status));
                } else if b.im_is_zero()
                    && let Some((lo, hi)) = b.re.bounds()
                    && lo == hi
                    && let Some(n) = integer_exponent(lo.into_f64())
                {
                    a.powi(n, true)?
                } else {
                    let logarithm = values.value(a, U::Ln, true)?;
                    values.value(logarithm.mul(b)?, U::Exp, true)?
                }
            }
            B::Log => values
                .value(a, U::Ln, true)?
                .div(values.value(b, U::Ln, true)?, true)?,
            B::Atan2 | B::Hypot => {
                return Err(crate::shell::breakdown(
                    "complex atan2 and hypot are rejected before evaluation",
                ));
            }
        };
        let uncertain = match op {
            B::Div => b.contains_zero(),
            B::Pow => !b.is_zero() && touches_cut(a, U::Ln),
            B::Log => {
                touches_cut(a, U::Ln)
                    || touches_cut(b, U::Ln)
                    || values.value(b, U::Ln, true)?.contains_zero()
            }
            _ => false,
        };
        let status = if range.empty() {
            Status::Undefined
        } else if uncertain {
            Status::Inconclusive
        } else {
            Status::Defined
        };
        return Ok((range, status));
    };
    Ok((Box2::real(r)?, status))
}

pub(super) struct Candidate<T: NumericScalar> {
    pub center: T,
    pub residual: Box2<T::Real>,
    pub error: Box2<T::Real>,
    pub len: usize,
    pub range: Box2<T::Real>,
    pub status: Status,
    pub decoration: Decoration,
    pub fallback: Option<AffineFallback>,
}
impl<T: NumericScalar> Candidate<T> {
    fn fallback(
        range: Box2<T::Real>,
        status: Status,
        reason: AffineFallback,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            center: T::ZERO,
            residual: Box2::zero()?,
            error: Box2::zero()?,
            len: 0,
            range,
            status,
            decoration: range.decoration(),
            fallback: Some(reason),
        })
    }
}
fn range<T: NumericScalar>(x: AffineView<'_, T>) -> Box2<T::Real> {
    Box2 {
        re: x.state.report.real,
        im: x.state.report.imaginary,
    }
}
fn radius<T: NumericScalar>(x: AffineView<'_, T>) -> Result<Box2<T::Real>, SciOxideError> {
    let mut result = x.state.residual;
    for term in x.terms() {
        result = result.add(term.coefficient_box()?)?;
    }
    Ok(result)
}
type Coefficient<T> = (T, Box2<<T as NumericScalar>::Real>);

/// An exact point needs neither midpoint arithmetic nor a residual subtraction.
fn midpoint<T: NumericScalar>(
    value: Box2<T::Real>,
) -> Result<Option<Coefficient<T>>, SciOxideError> {
    if let (Some((rl, rh)), Some((il, ih))) = (value.re.bounds(), value.im.bounds())
        && rl == rh
        && il == ih
        && rl.is_finite()
        && il.is_finite()
    {
        return Ok(Some((T::from_components(rl, il), Box2::zero()?)));
    }
    value.midpoint::<T>()
}

/// Exact sign/conjugation maps avoid interval arithmetic on unchanged coefficients.
#[derive(Clone, Copy)]
enum CoefficientMap<S: RealScalar> {
    Linear { negative: bool, conjugate: bool },
    Scale(Box2<S>),
}
impl<S: RealScalar> CoefficientMap<S> {
    fn coefficient<T: NumericScalar<Real = S>>(
        self,
        value: T,
    ) -> Result<Option<(T, Box2<S>)>, SciOxideError> {
        if let Self::Linear {
            negative,
            conjugate,
        } = self
        {
            let value = if conjugate { value.conj() } else { value };
            return Ok(Some((if negative { -value } else { value }, Box2::zero()?)));
        }
        midpoint(self.apply(value)?)
    }
    fn apply<T: NumericScalar<Real = S>>(self, value: T) -> Result<Box2<S>, SciOxideError> {
        match self {
            Self::Linear {
                negative,
                conjugate,
            } => {
                let value = if conjugate { value.conj() } else { value };
                Box2::point(if negative { -value } else { value })
            }
            Self::Scale(factor) => Box2::point(value)?.mul(factor),
        }
    }
}

/// Merge symbol-sorted streams, keeping the existing two-stage rounding for shared
/// coefficients. Each source coefficient is visited once; zero sums are omitted.
fn merge_coefficients<T: NumericScalar>(
    a: AffineView<'_, T>,
    b: Option<AffineView<'_, T>>,
    alpha: CoefficientMap<T::Real>,
    beta: CoefficientMap<T::Real>,
    terms: &mut [AffineTerm<T>],
    error: &mut Box2<T::Real>,
) -> Result<Option<usize>, SciOxideError> {
    let mut left = a.terms().iter().peekable();
    let mut right = b
        .map_or(&[][..], |v| &v.terms[..v.state.report.retained_symbols])
        .iter()
        .peekable();
    let mut len = 0;
    while left.peek().is_some() || right.peek().is_some() {
        let key = match (left.peek(), right.peek()) {
            (Some(l), Some(r)) => l.symbol.min(r.symbol),
            (Some(l), None) => l.symbol,
            (None, Some(r)) => r.symbol,
            (None, None) => break,
        };
        let mut coefficient = T::ZERO;
        if let Some(term) = left.next_if(|term| term.symbol == key) {
            let Some((value, rounding)) = alpha.coefficient(term.coefficient)? else {
                return Ok(None);
            };
            coefficient = value;
            if !rounding.is_zero() {
                *error = error.add(rounding)?;
            }
        }
        if let Some(term) = right.next_if(|term| term.symbol == key) {
            let mapped = if coefficient == T::ZERO {
                beta.coefficient(term.coefficient)?
            } else {
                midpoint(
                    beta.apply(term.coefficient)?
                        .add(Box2::point(coefficient)?)?,
                )?
            };
            let Some((value, rounding)) = mapped else {
                return Ok(None);
            };
            coefficient = value;
            if !rounding.is_zero() {
                *error = error.add(rounding)?;
            }
        }
        if coefficient != T::ZERO {
            terms[len] = AffineTerm {
                symbol: key,
                coefficient,
            };
            len += 1;
        }
    }
    Ok(Some(len))
}

/// Compute the affine candidate for `op`, propagating enclosure failures.
pub(super) fn evaluate<T: NumericScalar>(
    a: AffineView<'_, T>,
    b: Option<AffineView<'_, T>>,
    op: Operation,
    terms: &mut [AffineTerm<T>],
) -> Result<Candidate<T>, SciOxideError> {
    match op {
        Operation::Identity
        | Operation::Unary(U::Neg | U::Conj | U::Square)
        | Operation::Binary(B::Add | B::Sub | B::Mul) => {
            evaluate_using(a, b, op, terms, &mut DirectValues)
        }
        _ => evaluate_analytic(a, b, op, terms),
    }
}

// Keep the fixed analytic value storage out of exact-operation stack frames.
#[inline(never)]
fn evaluate_analytic<T: NumericScalar>(
    a: AffineView<'_, T>,
    b: Option<AffineView<'_, T>>,
    op: Operation,
    terms: &mut [AffineTerm<T>],
) -> Result<Candidate<T>, SciOxideError> {
    evaluate_using(a, b, op, terms, &mut CertifiedValues::new())
}

fn evaluate_using<T: NumericScalar>(
    a: AffineView<'_, T>,
    b: Option<AffineView<'_, T>>,
    op: Operation,
    terms: &mut [AffineTerm<T>],
    values: &mut impl Values<T::Real>,
) -> Result<Candidate<T>, SciOxideError> {
    let complex = !T::IS_REAL;
    let ar = range(a);
    if matches!(op, Operation::Binary(B::Pow))
        && let Some(b) = b
        && b.report().fallback.is_none()
        && b.terms().is_empty()
        && b.state.residual.is_zero()
        && b.state.center.imaginary_component() == T::Real::ZERO
    {
        let n = b.state.center.real_component().into_f64();
        if let Some(n) = integer_exponent(n) {
            let mut candidate = evaluate(a, None, Operation::Unary(U::Powi(n)), terms)?;
            candidate.decoration = candidate.decoration.min(range(b).decoration());
            candidate.range = candidate.range.decorate(candidate.decoration)?;
            return Ok(candidate);
        }
    }
    // A copy keeps its operand's enclosure, status and fallback reason unchanged.
    if matches!(op, Operation::Identity)
        && let Some(reason) = a.report().fallback
    {
        return Candidate::fallback(ar, a.report().status, reason);
    }
    let (output_range, status) = match op {
        Operation::Identity => (ar, a.report().status),
        Operation::Unary(u) => values.classified(ar, u, complex)?,
        Operation::Binary(binary) => values.binary(
            ar,
            range(b.ok_or_else(|| crate::shell::breakdown("missing binary operand"))?),
            binary,
            complex,
        )?,
    };
    let decoration = ar
        .decoration()
        .min(b.map_or(Decoration::Com, |b| range(b).decoration()));
    let output_range = output_range.decorate(if status == Status::Defined {
        decoration
    } else {
        Decoration::Trv
    })?;
    // Operations on enclosures report the enclosure input, or a propagated overflow.
    if a.report().fallback.is_some() || b.is_some_and(|b| b.report().fallback.is_some()) {
        let reason = if a.report().fallback == Some(AffineFallback::Unbounded)
            || b.is_some_and(|b| b.report().fallback == Some(AffineFallback::Unbounded))
        {
            AffineFallback::Unbounded
        } else {
            AffineFallback::EnclosureInput
        };
        return Candidate::fallback(output_range, status, reason);
    }
    if output_range.empty() {
        return Candidate::fallback(output_range, status, AffineFallback::DomainOrBranch);
    }

    // Exact linear transformations retain anonymous residuals as anonymous.
    if matches!(
        op,
        Operation::Identity
            | Operation::Unary(U::Neg | U::Conj)
            | Operation::Binary(B::Add | B::Sub)
    ) {
        let mut candidate = Candidate {
            center: T::ZERO,
            residual: Box2::zero()?,
            error: Box2::zero()?,
            len: 0,
            range: output_range,
            status,
            decoration,
            fallback: None,
        };
        let mut ac = Box2::point(a.state.center)?;
        let mut residual = a.state.residual;
        let conjugate = matches!(op, Operation::Unary(U::Conj));
        let negative = matches!(op, Operation::Unary(U::Neg));
        if conjugate {
            ac = ac.conj()?;
            residual = residual.conj()?;
        }
        if negative {
            ac = ac.neg()?;
            residual = residual.neg()?;
        }
        if let Some(b) = b {
            let sign = if matches!(op, Operation::Binary(B::Sub)) {
                -1.0
            } else {
                1.0
            };
            ac = ac.add(Box2::point(b.state.center)?.scale(sign)?)?;
            residual = residual.add(b.state.residual.scale(sign)?)?;
        }
        let Some((center, error)) = midpoint(ac)? else {
            return Candidate::fallback(output_range, status, AffineFallback::Unbounded);
        };
        candidate.center = center;
        candidate.error = error;
        candidate.residual = residual;
        let alpha = CoefficientMap::Linear {
            negative,
            conjugate,
        };
        let beta = CoefficientMap::Linear {
            negative: matches!(op, Operation::Binary(B::Sub)),
            conjugate: false,
        };
        let Some(len) = merge_coefficients(a, b, alpha, beta, terms, &mut candidate.error)? else {
            return Candidate::fallback(output_range, status, AffineFallback::Unbounded);
        };
        candidate.len = len;
        return Ok(candidate);
    }
    // Product uses exact shared-symbol quadratic structure, including diagonal
    // recentering epsilon_i^2 = 1/2 + [-1/2,1/2].
    if matches!(op, Operation::Binary(B::Mul) | Operation::Unary(U::Square)) {
        return product(a, b.unwrap_or(a), terms, output_range, status, decoration);
    }
    let ac = Box2::point(a.state.center)?;
    let da = radius(a)?;
    let supporting_a = ar.hull(ac);
    let (center_jet, whole) = match op {
        Operation::Identity => {
            return Err(crate::shell::breakdown(
                "identity is an exact linear operation",
            ));
        }
        Operation::Unary(u) => {
            if !smooth(supporting_a, u, complex, values)? {
                return Candidate::fallback(output_range, status, AffineFallback::DomainOrBranch);
            }
            (
                Jet::input(ac, false)?.unary(u, complex, values)?,
                Jet::input(supporting_a, false)?.unary(u, complex, values)?,
            )
        }
        Operation::Binary(binary) => {
            let b = b.ok_or_else(|| crate::shell::breakdown("missing operand"))?;
            let bc = Box2::point(b.state.center)?;
            let supporting_b = range(b).hull(bc);
            let admitted = match binary {
                B::Div => !supporting_b.contains_zero(),
                B::Pow => smooth(supporting_a, U::Ln, complex, values)?,
                B::Log => {
                    smooth(supporting_a, U::Ln, complex, values)?
                        && smooth(supporting_b, U::Ln, complex, values)?
                        && !values.value(supporting_b, U::Ln, complex)?.contains_zero()
                }
                B::Atan2 => {
                    !(Box2 {
                        re: supporting_b.re,
                        im: supporting_a.re,
                    }
                    .contains_zero()
                        || (supporting_a.re.contains(T::Real::ZERO)
                            && supporting_b
                                .re
                                .bounds()
                                .is_some_and(|(lo, _)| lo < T::Real::ZERO)))
                }
                B::Hypot => !Box2 {
                    re: supporting_a.re,
                    im: supporting_b.re,
                }
                .contains_zero(),
                _ => true,
            };
            if !admitted {
                return Candidate::fallback(output_range, status, AffineFallback::DomainOrBranch);
            }
            (
                binary_jet(ac, bc, binary, complex, values)?,
                binary_jet(supporting_a, supporting_b, binary, complex, values)?,
            )
        }
    };
    let Some((center, mut error)) = center_jet.v.midpoint::<T>()? else {
        return Candidate::fallback(output_range, status, AffineFallback::Unbounded);
    };
    if center_jet.x.empty() || center_jet.y.empty() {
        return Candidate::fallback(output_range, status, AffineFallback::Unbounded);
    }
    let (alpha, alpha_error) = slope(center_jet.x)?;
    let (beta, beta_error) = slope(center_jet.y)?;
    let db = if let Some(b) = b {
        radius(b)?
    } else {
        Box2::zero()?
    };
    // The linearization remainder f(x, y) − center − α·Δa − β·Δb over the deviations
    // Δa ∈ da, Δb ∈ db has two rigorous enclosures: the second-order Taylor bound, and
    // the image bound output_range − center − α·da − β·db, because f(x, y) lies in
    // output_range for every value of the operands. Separately enclosed derivatives and
    // deviation powers underflow or saturate at extreme magnitudes, where only the
    // image bound stays tight.
    let taylor = error
        .add(alpha_error.mul(da)?)?
        .add(beta_error.mul(db)?)?
        .add(whole.xx.mul(da.square()?)?.scale(0.5)?)?
        .add(whole.xy.mul(da.mul(db)?)?)?
        .add(whole.yy.mul(db.square()?)?.scale(0.5)?)?;
    let mut image = output_range
        .sub(Box2::point(center)?)?
        .sub(alpha.mul(da)?)?;
    if b.is_some() {
        image = image.sub(beta.mul(db)?)?;
    }
    error = taylor.intersection(image).ok_or_else(|| {
        crate::shell::breakdown("disjoint enclosures of the affine linearization remainder")
    })?;
    error = error.add(alpha.mul(a.state.residual)?)?;
    if let Some(b) = b {
        error = error.add(beta.mul(b.state.residual)?)?;
    }
    let Some(len) = merge_coefficients(
        a,
        b,
        CoefficientMap::Scale(alpha),
        CoefficientMap::Scale(beta),
        terms,
        &mut error,
    )?
    else {
        return Candidate::fallback(output_range, status, AffineFallback::Unbounded);
    };
    error = error.symmetric_hull()?;
    if !error.finite() {
        return Candidate::fallback(output_range, status, AffineFallback::Unbounded);
    }
    Ok(Candidate {
        center,
        residual: Box2::zero()?,
        error,
        len,
        range: output_range,
        status,
        decoration,
        fallback: None,
    })
}

/// Choose a slope for the nonempty derivative enclosure at the center and return it
/// with the enclosure of the derivative minus the slope.
///
/// Each component takes the midpoint of its enclosure, or zero where the enclosure
/// contains zero or is unbounded: an underflowed or overflowed derivative carries no
/// certified sign or magnitude, and the image bound then encloses the remainder.
fn slope<S: RealScalar>(derivative: Box2<S>) -> Result<(Box2<S>, Box2<S>), SciOxideError> {
    let part = |x: D<S>| match x.bounds() {
        Some((lo, hi)) if lo.is_finite() && hi.is_finite() && !x.contains(S::ZERO) => {
            scioxide_core::checked_midpoint(lo, hi)
        }
        _ => Ok(S::ZERO),
    };
    let value = Box2 {
        re: D::point(part(derivative.re)?)?,
        im: D::point(part(derivative.im)?)?,
    };
    Ok((value, derivative.sub(value)?))
}

/// Enclose the product while keeping correlations between shared symbols.
fn product<T: NumericScalar>(
    a: AffineView<'_, T>,
    b: AffineView<'_, T>,
    terms: &mut [AffineTerm<T>],
    range: Box2<T::Real>,
    status: Status,
    decoration: Decoration,
) -> Result<Candidate<T>, SciOxideError> {
    let ac = Box2::point(a.state.center)?;
    let bc = Box2::point(b.state.center)?;
    let mut center = ac.mul(bc)?;
    let mut error = Box2::zero()?;
    let Some(len) = merge_coefficients(
        a,
        Some(b),
        CoefficientMap::Scale(bc),
        CoefficientMap::Scale(ac),
        terms,
        &mut error,
    )?
    else {
        return Candidate::fallback(range, status, AffineFallback::Unbounded);
    };
    let mut la = Box2::zero()?;
    let mut lb = Box2::zero()?;
    // Positive component sums bound the off-diagonal products. Keeping them as
    // intervals permits safe subtraction of the shared diagonal lower bounds.
    let mut ra = Box2::zero()?;
    let mut rb = Box2::zero()?;
    for term in a.terms() {
        la = la.add(term.coefficient_box()?)?;
        ra = ra.add(Box2::point(T::from_components(
            term.coefficient.real_component().abs(),
            term.coefficient.imaginary_component().abs(),
        ))?)?;
    }
    for term in b.terms() {
        lb = lb.add(term.coefficient_box()?)?;
        rb = rb.add(Box2::point(T::from_components(
            term.coefficient.real_component().abs(),
            term.coefficient.imaginary_component().abs(),
        ))?)?;
    }
    let mut quadratic = Box2 {
        re: ra
            .re
            .mul_admitted(&rb.re)?
            .add_admitted(&ra.im.mul_admitted(&rb.im)?)?,
        im: ra
            .re
            .mul_admitted(&rb.im)?
            .add_admitted(&ra.im.mul_admitted(&rb.re)?)?,
    };
    let mut left = a.terms().iter().peekable();
    let mut right = b.terms().iter().peekable();
    while let (Some(l), Some(r)) = (left.peek(), right.peek()) {
        match l.symbol.cmp(&r.symbol) {
            core::cmp::Ordering::Less => {
                left.next();
            }
            core::cmp::Ordering::Greater => {
                right.next();
            }
            core::cmp::Ordering::Equal => {
                let x = Box2::point(l.coefficient)?;
                let y = Box2::point(r.coefficient)?;
                let diagonal = x.mul(y)?;
                let ax = Box2::point(T::from_components(
                    l.coefficient.real_component().abs(),
                    l.coefficient.imaginary_component().abs(),
                ))?;
                let ay = Box2::point(T::from_components(
                    r.coefficient.real_component().abs(),
                    r.coefficient.imaginary_component().abs(),
                ))?;
                let removed = Box2 {
                    re: ax
                        .re
                        .mul_admitted(&ay.re)?
                        .add_admitted(&ax.im.mul_admitted(&ay.im)?)?,
                    im: ax
                        .re
                        .mul_admitted(&ay.im)?
                        .add_admitted(&ax.im.mul_admitted(&ay.re)?)?,
                };
                quadratic = quadratic.sub(removed)?;
                center = center.add(diagonal.scale(0.5)?)?;
                error = error.add(diagonal.scale(0.5)?.symmetric_hull()?)?;
                left.next();
                right.next();
            }
        }
    }
    if !quadratic.finite() {
        return Candidate::fallback(range, status, AffineFallback::Unbounded);
    }
    let upper = |x: D<T::Real>| {
        x.bounds().map_or(T::Real::INFINITY, |(_, hi)| {
            if hi > T::Real::ZERO {
                hi
            } else {
                T::Real::ZERO
            }
        })
    };
    error = error.add(Box2::symmetric(upper(quadratic.re), upper(quadratic.im))?)?;
    error = error
        .add(ac.add(la)?.mul(b.state.residual)?)?
        .add(bc.add(lb)?.mul(a.state.residual)?)?
        .add(a.state.residual.mul(b.state.residual)?)?;
    let Some((center, rounding)) = midpoint(center)? else {
        return Candidate::fallback(range, status, AffineFallback::Unbounded);
    };
    error = error.add(rounding)?.symmetric_hull()?;
    if !error.finite() {
        return Candidate::fallback(range, status, AffineFallback::Unbounded);
    }
    Ok(Candidate {
        center,
        residual: Box2::zero()?,
        error,
        len,
        range,
        status,
        decoration,
        fallback: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AffineContext, AffineForm, AffineOptions};
    use scioxide_core::Complex;

    fn evaluation_count<T: NumericScalar>(
        center: T,
        coefficient: T,
        operation: Operation,
    ) -> usize {
        let context = AffineContext::new(1, AffineOptions::new()).unwrap();
        let x = AffineForm::from_coefficients(&context, center, &[coefficient]).unwrap();
        let y = AffineForm::from_coefficients(
            &context,
            T::from_real(T::Real::try_from_f64_lossy(2.5).unwrap()),
            &[coefficient],
        )
        .unwrap();
        let mut terms = [AffineTerm::default(); 34];
        let mut values = CertifiedValues::new();
        let other = matches!(operation, Operation::Binary(_)).then_some(y.view());
        let result = evaluate_using(x.view(), other, operation, &mut terms, &mut values).unwrap();
        assert_eq!(result.fallback, None);
        // Admission, value and derivatives must visit each certified node only once.
        assert_eq!(values.evaluations, values.len);
        for (index, entry) in values.entries[..values.len].iter().flatten().enumerate() {
            assert!(
                values.entries[..index]
                    .iter()
                    .flatten()
                    .all(|previous| previous.operation != entry.operation
                        || previous.complex != entry.complex
                        || previous.argument.re != entry.argument.re
                        || previous.argument.im != entry.argument.im
                        || !same_other(previous.other, entry.other))
            );
        }
        values.evaluations
    }

    #[test]
    fn complex_components_share_identical_elementary_arguments() {
        let mut values = CertifiedValues::<f64>::new();
        let zero = Box2::zero().unwrap();
        let (root, status) = values.classified(zero, U::Sqrt, true).unwrap();
        assert!(root.is_zero());
        assert_eq!(status, Status::Inconclusive);
        // One norm, one real sqrt, and one complex sqrt, despite equal components.
        assert_eq!(values.evaluations, 3);
        let _ = values.classified(zero, U::Sqrt, true).unwrap();
        assert_eq!(values.evaluations, 3);
    }

    #[test]
    fn certified_values_are_shared_by_classification_admission_and_jets() {
        assert_eq!(
            evaluation_count(1.0_f64, 0.125, Operation::Unary(U::Exp)),
            2
        );
        assert_eq!(evaluation_count(1.0_f64, 0.0, Operation::Unary(U::Exp)), 1);
        assert_eq!(
            evaluation_count(1.0_f64, 0.125, Operation::Unary(U::Sin)),
            4
        );
        assert_eq!(
            evaluation_count(1.0_f64, 0.125, Operation::Unary(U::ExpM1)),
            4
        );
        assert_eq!(
            evaluation_count(1.0_f64, 0.125, Operation::Unary(U::Powi(-3))),
            2
        );
        assert_eq!(
            evaluation_count(1.0_f64, 0.125, Operation::Binary(B::Atan2)),
            2
        );
        assert_eq!(
            evaluation_count(1.0_f64, 0.0, Operation::Binary(B::Atan2)),
            1
        );
        for operation in [B::Pow, B::Log] {
            assert!(evaluation_count(1.0_f64, 0.125, Operation::Binary(operation)) > 0);
            assert!(
                evaluation_count(
                    Complex::new(1.0_f64, 0.25),
                    Complex::new(0.125, 0.0625),
                    Operation::Binary(operation)
                ) > 0
            );
        }
        assert_eq!(
            evaluation_count(
                Complex::new(1.0_f64, 0.25),
                Complex::new(0.125, 0.0625),
                Operation::Unary(U::Exp)
            ),
            8
        );
        assert_eq!(
            evaluation_count(
                Complex::new(1.0_f64, 0.25),
                Complex::new(0.125, 0.0625),
                Operation::Unary(U::Tan)
            ),
            14
        );
    }
}
