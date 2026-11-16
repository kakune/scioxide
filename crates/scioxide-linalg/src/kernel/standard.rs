//! Dense adapters for shared Standard arithmetic.

pub(crate) use scioxide_core::backend::standard::{
    dot, exponent, finish_substitution, fused_update, magnitude, ordinary_quotient, product_scales,
    restore, scale, substitute_row, unit_then_normalized,
};

use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};

/// Largest component magnitude `max(|re|, |im|)` of a view, restricted to
/// `triangle` when given; zero for an empty view. NaN components are ignored,
/// as by a sequential `max` fold. A maximum does not depend on reduction
/// order, so the lane-parallel walk over contiguous lines returns exactly
/// the value of an element-by-element scan.
pub(crate) fn component_maximum<S: NumericScalar>(
    view: crate::MatrixView<'_, S>,
    triangle: Option<scioxide_core::Triangle>,
) -> S::Real {
    lines(view, triangle)
        .map(|line| line_extent(line).maximum)
        .fold(S::Real::ZERO, |left, right| left.max(right))
}

/// Smallest nonzero component magnitude `max(|re|, |im|)` of a view, or zero when every
/// entry is zero.
pub(crate) fn nonzero_minimum<S: NumericScalar>(view: crate::MatrixView<'_, S>) -> S::Real {
    lines(view, None)
        .flat_map(|line| (0..line.len).map(move |index| line.data[index * line.stride]))
        .map(magnitude)
        .filter(|&value| value != S::Real::ZERO)
        .reduce(|left, right| left.min(right))
        .unwrap_or(S::Real::ZERO)
}

/// Returns whether folding `alpha` into an operand `A` before multiplying by `B` can break
/// the Standard bound in the unit frame.
///
/// A nonzero `alpha · a` that rounds below the normal range loses up to half a subnormal
/// spacing, and a factor `|b| ≥ 1` carries that loss into the product beyond the absolute
/// term `k u_sub` of the bound. `minimum` returns the smallest nonzero component magnitude of
/// `A` (zero when `A` is zero) and is called only when the answer depends on it; `other` is
/// the largest component magnitude of `B`. A real or imaginary `alpha` whose magnitude is a
/// power of two of at least one folds exactly and never needs a frame.
pub(crate) fn folding_underflows<S: NumericScalar>(
    alpha: S,
    minimum: impl FnOnce() -> S::Real,
    other: S::Real,
) -> Result<bool, SciOxideError> {
    if alpha == S::ZERO || other < S::Real::ONE {
        return Ok(false);
    }
    let alpha_magnitude = magnitude(alpha);
    let alpha_e = exponent(alpha_magnitude)?;
    let axis_aligned =
        alpha.real_component() == S::Real::ZERO || alpha.imaginary_component() == S::Real::ZERO;
    if axis_aligned
        && alpha_magnitude >= S::Real::ONE
        && alpha_magnitude == S::Real::ONE.scalbn(alpha_e - 1)
    {
        return Ok(false);
    }
    let minimum = minimum();
    // `|alpha| |a| ≥ 2^(alpha_e + minimum_e - 2)`, which is normal unless the test holds.
    Ok(minimum != S::Real::ZERO
        && alpha_e + exponent(minimum)? - 1 < exponent(S::Real::MIN_POSITIVE)?)
}

/// [`component_maximum`] of a whole view that must be finite.
///
/// # Errors
/// Returns `NumericalInstability(Overflow)`, as
/// [`crate::checked::finite_candidate`] does, when any component is not finite.
pub(crate) fn finite_component_maximum<S: NumericScalar>(
    view: crate::MatrixView<'_, S>,
) -> Result<S::Real, SciOxideError> {
    finite(lines(view, None).map(line_extent))
}

/// [`component_maximum`] of contiguous values.
pub(crate) fn slice_maximum<S: NumericScalar>(values: &[S]) -> S::Real {
    line_extent(Line {
        data: values,
        len: values.len(),
        stride: 1,
    })
    .maximum
}

/// [`finite_component_maximum`] of contiguous values.
///
/// # Errors
/// Returns `NumericalInstability(Overflow)` when any component is not finite.
pub(crate) fn finite_slice_maximum<S: NumericScalar>(
    values: &[S],
) -> Result<S::Real, SciOxideError> {
    finite(core::iter::once(line_extent(Line {
        data: values,
        len: values.len(),
        stride: 1,
    })))
}

fn finite<R: RealScalar>(extents: impl Iterator<Item = Extent<R>>) -> Result<R, SciOxideError> {
    let extent = extents.fold(
        Extent {
            maximum: R::ZERO,
            finite: true,
        },
        |left, right| Extent {
            maximum: left.maximum.max(right.maximum),
            finite: left.finite && right.finite,
        },
    );
    if extent.finite && extent.maximum.is_finite() {
        Ok(extent.maximum)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "finite operands produced a non-finite result",
        ))
    }
}

/// One run of equally spaced view entries.
#[derive(Clone, Copy)]
struct Line<'a, S> {
    data: &'a [S],
    len: usize,
    stride: usize,
}

/// Largest finite-or-infinite magnitude of a line and whether no component is NaN.
#[derive(Clone, Copy)]
struct Extent<R> {
    maximum: R,
    finite: bool,
}

/// Rows of a view, or its columns when only those are contiguous, clipped to `triangle`.
fn lines<S: NumericScalar>(
    view: crate::MatrixView<'_, S>,
    triangle: Option<scioxide_core::Triangle>,
) -> impl Iterator<Item = Line<'_, S>> {
    use scioxide_core::Triangle;
    let crate::MatrixView {
        data,
        rows,
        cols,
        row_stride,
        col_stride,
        ..
    } = view;
    // Lines run along the unit stride; a transposed walk mirrors the triangle.
    let by_columns = row_stride == 1 && col_stride != 1;
    let (count, length, line_stride, step) = if by_columns {
        (cols, rows, col_stride, row_stride)
    } else {
        (rows, cols, row_stride, col_stride)
    };
    let triangle = triangle.map(|triangle| match (triangle, by_columns) {
        (Triangle::Lower, true) => Triangle::Upper,
        (Triangle::Upper, true) => Triangle::Lower,
        (triangle, false) => triangle,
    });
    (0..if length == 0 { 0 } else { count }).filter_map(move |line| {
        let range = match triangle {
            None => 0..length,
            Some(Triangle::Lower) => 0..(line + 1).min(length),
            Some(Triangle::Upper) => line.min(length)..length,
        };
        (!range.is_empty()).then(|| Line {
            data: &data[line * line_stride + range.start * step..],
            len: range.len(),
            stride: step,
        })
    })
}

/// Lane-parallel magnitude maximum and NaN screen of one line.
// Keep the scan available in the consuming codegen unit. A conditional store
// here can become eight scalar branches when unrelated monomorphizations move
// this function. Assign every lane unconditionally so the comparison is a
// value selection, including when a NaN must leave the previous maximum intact.
#[inline]
fn line_extent<S: NumericScalar>(line: Line<'_, S>) -> Extent<S::Real> {
    const LANES: usize = 8;
    let mut maximum = [S::Real::ZERO; LANES];
    let mut nan = [false; LANES];
    let mut visit = |lane: usize, value: S| {
        let real = value.real_component().abs();
        let (magnitude, screened) = if S::IS_REAL {
            (real, real.is_nan())
        } else {
            let imaginary = value.imaginary_component().abs();
            (
                if imaginary > real { imaginary } else { real },
                real.is_nan() || imaginary.is_nan(),
            )
        };
        nan[lane] |= screened;
        let previous = maximum[lane];
        maximum[lane] = if magnitude > previous {
            magnitude
        } else {
            previous
        };
    };
    if line.stride == 1 {
        let values = &line.data[..line.len];
        let mut chunks = values.chunks_exact(LANES);
        chunks.by_ref().for_each(|chunk| {
            chunk
                .iter()
                .enumerate()
                .for_each(|(lane, &value)| visit(lane, value));
        });
        chunks
            .remainder()
            .iter()
            .enumerate()
            .for_each(|(lane, &value)| visit(lane, value));
    } else {
        line.data
            .iter()
            .step_by(line.stride)
            .take(line.len)
            .enumerate()
            .for_each(|(index, &value)| visit(index % LANES, value));
    }
    Extent {
        maximum: maximum
            .into_iter()
            .fold(S::Real::ZERO, |left, right| left.max(right)),
        finite: !nan.into_iter().any(|nan| nan),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Publication {
    Direct,
    Checked,
}

/// Validated coefficients and ranges for one affine vector pass.
pub(crate) struct Affine<S: NumericScalar> {
    alpha: S,
    beta: S,
    x: i32,
    y: i32,
    restore: i32,
    pub(crate) publication: Publication,
}

impl<S: NumericScalar> Affine<S> {
    pub(crate) fn is_unscaled(&self) -> bool {
        self.x == 0 && self.y == 0 && self.restore == 0 && self.publication == Publication::Direct
    }

    pub(crate) fn new(alpha: S, x: S::Real, beta: S, y: S::Real) -> Result<Self, SciOxideError> {
        let ae = exponent(magnitude(alpha))?;
        let xe = exponent(x)?;
        let be = exponent(magnitude(beta))?;
        let ye = exponent(y)?;
        let e = match (
            alpha == S::ZERO || x == S::Real::ZERO,
            beta == S::ZERO || y == S::Real::ZERO,
        ) {
            (true, true) => 0,
            (false, true) => ae + xe + 3,
            (true, false) => be + ye + 3,
            (false, false) => (ae + xe).max(be + ye) + 3,
        }
        .max(0);
        // The bound is nonnegative, so only overflow-prone ranges need a frame;
        // two once-rounded terms lose at most the destination subnormal grid.
        if e < exponent(S::Real::MAX)? {
            Ok(Self {
                alpha,
                beta,
                x: 0,
                y: 0,
                restore: 0,
                publication: Publication::Direct,
            })
        } else {
            Ok(Self {
                alpha: scale(scale(alpha, -ae), ae + xe - e),
                beta: scale(scale(beta, -be), be + ye - e),
                x: xe,
                y: ye,
                restore: e,
                publication: Publication::Checked,
            })
        }
    }

    #[inline]
    pub(crate) fn apply(&self, x: S, y: S) -> S {
        let x = scale(x, -self.x);
        // A pure scaling has no addend whose rounding fusion could avoid.
        let value = if self.beta == S::ZERO {
            self.alpha * x
        } else {
            fused_update(self.alpha, x, self.beta * scale(y, -self.y))
        };
        scale(value, self.restore)
    }
}

/// Shared binary frame for `alpha * sum(a*x) + beta*y`.
///
/// The unit frame evaluates `(alpha * a) * x`; `a_minimum` is the smallest nonzero component
/// magnitude of the `a` operands (zero when there is none), which decides whether that fold
/// can lose bits below the normal range (see [`folding_underflows`]).
pub(crate) struct AffineProduct<S: NumericScalar> {
    a: i32,
    x: i32,
    restore: i32,
    coefficient: S,
    alpha: S,
    beta: S,
    beta_exponent: i32,
    y: i32,
    scaled: bool,
}
impl<S: NumericScalar> AffineProduct<S> {
    pub(crate) const fn is_unscaled(&self) -> bool {
        !self.scaled
    }

    pub(crate) fn new(
        alpha: S,
        (a, a_minimum): (S::Real, S::Real),
        x: S::Real,
        terms: usize,
        beta: S,
        y: S::Real,
    ) -> Result<Self, SciOxideError> {
        let (ae, xe, alpha_e, beta_e, ye) = (
            exponent(a)?,
            exponent(x)?,
            exponent(magnitude(alpha))?,
            exponent(magnitude(beta))?,
            exponent(y)?,
        );
        let growth = usize::BITS as i32 - terms.max(1).leading_zeros() as i32 + 4;
        let bound = (ae + xe + alpha_e + growth).max(ye + beta_e + 3).max(0);
        let scaled = bound >= exponent(S::Real::MAX)?
            || ae + alpha_e + 2 >= exponent(S::Real::MAX)?
            || ae + xe + alpha_e
                < exponent(S::Real::MIN_POSITIVE)? + S::Real::MANTISSA_DIGITS as i32
            || folding_underflows(alpha, || a_minimum, x)?;
        let (a, x, restore, coefficient) = if scaled {
            (
                ae,
                xe,
                bound,
                scale(scale(alpha, -alpha_e), ae + xe + alpha_e - bound),
            )
        } else {
            (0, 0, 0, alpha)
        };
        Ok(Self {
            a,
            x,
            restore,
            coefficient,
            alpha,
            beta,
            beta_exponent: beta_e,
            y: ye,
            scaled,
        })
    }
    pub(crate) fn apply(
        &self,
        prior: S,
        products: impl Iterator<Item = (S, S)> + Clone,
    ) -> Result<S, SciOxideError> {
        // As in unit_then_normalized, try actual unscaled products when
        // global maxima predict overflow: large factors may only meet tiny
        // partners. Scale alpha after that dot to avoid an overflowing triple.
        if self.scaled && self.a + self.x > 0 {
            let prior = if self.beta == S::ZERO {
                S::ZERO
            } else {
                self.beta * prior
            };
            let ordinary = self.alpha * dot(products.clone()) + prior;
            if ordinary.has_finite_components() {
                return Ok(ordinary);
            }
        }
        let mut sum = if self.beta == S::ZERO {
            S::ZERO
        } else if self.scaled {
            scale(
                scale(self.beta, -self.beta_exponent) * scale(prior, -self.y),
                self.y + self.beta_exponent - self.restore,
            )
        } else {
            self.beta * prior
        };
        if self.coefficient != S::ZERO {
            for (a, x) in products {
                sum = self.coefficient * scale(a, -self.a) * scale(x, -self.x) + sum;
            }
        }
        restore(sum, self.restore)
    }
}

/// Euclidean norm of entries validated on the way in.
pub(crate) fn norm<S: NumericScalar>(
    values: impl Iterator<Item = S>,
) -> Result<S::Real, SciOxideError> {
    try_norm(values.map(|value| {
        crate::checked::validate_input(value, "vector entry")?;
        Ok(value)
    }))
}

pub(crate) fn norm_finite<S: NumericScalar>(
    values: crate::FiniteVectorView<'_, S>,
) -> Result<S::Real, SciOxideError> {
    norm_workspace(values.iter())
}

/// Finite solver storage needs neither input validation nor a maximum scan.
pub(crate) fn norm_workspace<S: NumericScalar>(
    values: impl Iterator<Item = S>,
) -> Result<S::Real, SciOxideError> {
    values.fold(Blue::new()?, Blue::add).finish()
}

/// [`norm_workspace`] of fallible entries, stopping at the first error.
/// Frobenius residuals interleave execution checks with the entries they measure.
pub(crate) fn try_norm<S: NumericScalar>(
    mut values: impl Iterator<Item = Result<S, SciOxideError>>,
) -> Result<S::Real, SciOxideError> {
    values
        .try_fold(Blue::new()?, |sum, value| value.map(|value| sum.add(value)))?
        .finish()
}

/// Blue's three-range scaled sum of squares. Thresholds are binary powers;
/// there is no division, exponent decoding or error path in the element loop.
#[derive(Clone, Copy)]
pub(crate) struct Blue<R: RealScalar> {
    small: R,
    large: R,
    up: R,
    down: R,
    small_exponent: i32,
    large_exponent: i32,
    tiny: R,
    medium: R,
    big: R,
}

impl<R: RealScalar> Blue<R> {
    pub(crate) fn new() -> Result<Self, SciOxideError> {
        let high = exponent(R::MAX)?;
        let low = exponent(R::MIN_POSITIVE)? - 1;
        let digits = R::MANTISSA_DIGITS as i32;
        let small_exponent = (low - digits).div_euclid(2);
        let large_exponent = (high + digits).div_euclid(2);
        Ok(Self {
            small: R::ONE.scalbn(low.div_euclid(2)),
            large: R::ONE.scalbn((high - digits).div_euclid(2)),
            up: R::ONE.scalbn(-small_exponent),
            down: R::ONE.scalbn(-large_exponent),
            small_exponent,
            large_exponent,
            tiny: R::ZERO,
            medium: R::ZERO,
            big: R::ZERO,
        })
    }

    /// Accumulate the real, then the imaginary component.
    #[inline]
    pub(crate) fn add<S: NumericScalar<Real = R>>(self, value: S) -> Self {
        let sum = self.component(value.real_component().abs());
        if S::IS_REAL {
            sum
        } else {
            sum.component(value.imaginary_component().abs())
        }
    }

    #[inline]
    fn component(self, component: R) -> Self {
        if component == R::ZERO {
            self
        } else if component > self.large {
            let scaled = component * self.down;
            Self {
                big: scaled * scaled + self.big,
                ..self
            }
        } else if component < self.small {
            let scaled = component * self.up;
            Self {
                tiny: scaled * scaled + self.tiny,
                ..self
            }
        } else {
            Self {
                medium: component * component + self.medium,
                ..self
            }
        }
    }

    pub(crate) fn finish(self) -> Result<R, SciOxideError> {
        let result = if self.big != R::ZERO {
            (self.big + (self.medium * self.down) * self.down)
                .sqrt()
                .scalbn(self.large_exponent)
        } else if self.medium != R::ZERO {
            // Combining square roots avoids underflow when the tiny bucket is
            // restored before squaring.
            let normal_norm = self.medium.sqrt();
            if self.tiny == R::ZERO {
                normal_norm
            } else {
                let small_norm = self.tiny.sqrt().scalbn(self.small_exponent);
                let ratio = small_norm / normal_norm;
                normal_norm * (ratio * ratio + R::ONE).sqrt()
            }
        } else {
            self.tiny.sqrt().scalbn(self.small_exponent)
        };
        crate::checked::finite_candidate(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_norm_ranges<R: RealScalar>() -> Result<(), SciOxideError> {
        let three = R::try_from_f64_exact(3.)?;
        let four = R::try_from_f64_exact(4.)?;
        let five = R::try_from_f64_exact(5.)?;
        for scale in [
            R::MIN_POSITIVE.scalbn(1 - R::MANTISSA_DIGITS as i32),
            R::MIN_POSITIVE,
            R::ONE,
            R::MAX / R::try_from_f64_exact(16.)?,
        ] {
            let norm = norm_workspace(
                [R::ZERO, three * scale, -R::ZERO, four * scale, R::ZERO].into_iter(),
            )?;
            assert!((norm / scale - five).abs() <= R::try_from_f64_exact(32.)? * R::EPSILON);
            let complex = norm_workspace(
                [
                    scioxide_core::Complex::new(three * scale, four * scale),
                    scioxide_core::Complex::new(R::ZERO, -R::ZERO),
                ]
                .into_iter(),
            )?;
            assert!((complex / scale - five).abs() <= R::try_from_f64_exact(32.)? * R::EPSILON);
        }
        let zero = norm_workspace([-R::ZERO, R::ZERO].into_iter())?;
        assert_eq!(zero, R::ZERO);
        assert!(!zero.is_sign_negative());
        assert!(norm_workspace([R::MAX, R::MAX].into_iter()).is_err());
        Ok(())
    }

    #[test]
    fn sparse_real_and_complex_norms_preserve_subnormal_and_large_ranges()
    -> Result<(), SciOxideError> {
        sparse_norm_ranges::<f32>()?;
        sparse_norm_ranges::<f64>()
    }

    fn reference<S: NumericScalar>(
        view: crate::MatrixView<'_, S>,
        triangle: Option<scioxide_core::Triangle>,
    ) -> S::Real {
        (0..view.rows())
            .flat_map(|row| (0..view.cols()).map(move |col| (row, col)))
            .filter(|&(row, col)| {
                triangle.is_none_or(|triangle| crate::kernel::triangle_contains(triangle, row, col))
            })
            .fold(S::Real::ZERO, |bound, (row, col)| {
                bound.max(magnitude(view.get_element(row, col)))
            })
    }

    #[test]
    fn range_scans_match_entrywise_folds_for_every_layout_and_triangle() -> Result<(), SciOxideError>
    {
        use scioxide_core::{Complex, Triangle};
        let real: alloc::vec::Vec<f64> = (0..23 * 19)
            .map(|index| {
                ((index * 37 % 101) as f64 - 50.0) * if index % 7 == 0 { 1e3 } else { 1.0 }
            })
            .collect();
        let complex: alloc::vec::Vec<Complex<f64>> = real
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex::new(value * 0.5, (index % 13) as f64 * 40.0))
            .collect();
        let base = crate::MatrixView::from_slice(23, 19, &real)?;
        let complex = crate::MatrixView::from_slice(23, 19, &complex)?;
        for view in [
            base,
            base.transpose(),
            base.block(2, 3, 17, 9)?,
            base.block(2, 3, 17, 9)?.transpose(),
            base.block(0, 0, 0, 5)?,
        ] {
            for triangle in [None, Some(Triangle::Lower), Some(Triangle::Upper)] {
                assert_eq!(component_maximum(view, triangle), reference(view, triangle));
            }
            assert_eq!(finite_component_maximum(view)?, reference(view, None));
        }
        for view in [complex, complex.transpose(), complex.block(1, 4, 9, 15)?] {
            for triangle in [None, Some(Triangle::Lower), Some(Triangle::Upper)] {
                assert_eq!(component_maximum(view, triangle), reference(view, triangle));
            }
        }
        assert_eq!(finite_slice_maximum(&real)?, reference(base, None));
        Ok(())
    }

    #[test]
    fn finite_scans_reject_every_non_finite_component() -> Result<(), SciOxideError> {
        use scioxide_core::Complex;
        for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut values = [1.0_f64; 21];
            values[17] = poison;
            assert!(finite_slice_maximum(&values).is_err());
            let view = crate::MatrixView::from_slice(3, 7, &values)?;
            assert!(finite_component_maximum(view).is_err());
            assert!(finite_component_maximum(view.transpose()).is_err());
            let mut complex = [Complex::new(1.0, 2.0); 9];
            complex[4] = Complex::new(3.0, poison);
            assert!(finite_slice_maximum(&complex).is_err());
        }
        // NaN never becomes the maximum of an unchecked scan.
        let values = [1.0, f64::NAN, 3.0, -4.0];
        assert_eq!(
            component_maximum(crate::MatrixView::from_slice(2, 2, &values)?, None),
            4.0
        );
        Ok(())
    }

    #[test]
    fn maximum_scans_ignore_nan_in_every_lane_and_tail() -> Result<(), SciOxideError> {
        for len in [0, 1, 7, 8, 9, 15, 16, 17, 23] {
            let values: alloc::vec::Vec<f64> = (1..=len).map(|i| -(i as f64)).collect();
            assert_eq!(slice_maximum(&values), len as f64);
            for index in 0..len {
                let mut poisoned = values.clone();
                poisoned[index] = f64::NAN;
                let expected = if index + 1 == len { len - 1 } else { len } as f64;
                assert_eq!(slice_maximum(&poisoned), expected);
                assert!(finite_slice_maximum(&poisoned).is_err());
                let single: alloc::vec::Vec<f32> = poisoned.iter().map(|&x| x as f32).collect();
                assert_eq!(slice_maximum(&single), expected as f32);
                assert!(finite_slice_maximum(&single).is_err());

                // Padding must not enter a strided scan, even when it is larger
                // than every logical value. This also exercises the scalar tail.
                let strided: alloc::vec::Vec<f64> = poisoned
                    .iter()
                    .flat_map(|&value| [value, f64::INFINITY])
                    .collect();
                let view =
                    crate::MatrixView::from_slice_with_strides(1, len, &strided, 2 * len, 2)?;
                assert_eq!(component_maximum(view, None), expected);
                assert!(finite_component_maximum(view).is_err());
            }
        }
        assert_eq!(slice_maximum(&[-0.0_f64, f64::NAN]).to_bits(), 0);
        assert_eq!(slice_maximum(&[-0.0_f32, f32::NAN]).to_bits(), 0);
        Ok(())
    }

    #[test]
    fn exponent_preserves_finite_boundaries_and_rejects_non_finite_bounds() {
        for (value, expected) in [
            (0.0_f64, 0),
            (-0.0, 0),
            (f64::from_bits(1), -1073),
            (f64::MIN_POSITIVE, -1021),
            (0.5, 0),
            (1.0, 1),
            (2.0, 2),
            (f64::MAX, 1024),
        ] {
            assert_eq!(exponent(value).unwrap(), expected);
        }
        for (value, expected) in [
            (0.0_f32, 0),
            (-0.0, 0),
            (f32::from_bits(1), -148),
            (f32::MIN_POSITIVE, -125),
            (0.5, 0),
            (1.0, 1),
            (2.0, 2),
            (f32::MAX, 128),
        ] {
            assert_eq!(exponent(value).unwrap(), expected);
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                exponent(value).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ));
            assert!(product_scales(value, 1.0, 1).is_err());
            assert!(product_scales(1.0, value, 1).is_err());
            assert!(Affine::new(1.0_f64, value, 1.0, 1.0).is_err());
            assert!(Affine::new(1.0_f64, 1.0, 1.0, value).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                exponent(value).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ));
        }
    }
}
