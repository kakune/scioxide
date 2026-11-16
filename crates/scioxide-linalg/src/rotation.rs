//! Plane rotations on validated workspace values.
//!
//! `G = [c s; -conj(s) c]`, with real `c`. Generation chooses nonnegative
//! `c` with `G [f; g] = [r; 0]`, the phase of `r` equal to that of `f` when
//! `f != 0`, and otherwise `r = |g|`. Zero `g` gives the identity. These are
//! the conventions of Bindel, Demmel, Kahan and Marques (2002).

use crate::{MatrixViewMut, kernel::standard};
use core::ops::Range;
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Rotation<S: NumericScalar> {
    pub(crate) c: S::Real,
    pub(crate) s: S,
}

/// Unsquared-range fallback for exceptionally small or large rotation lanes.
/// In the ordinary interval a native sum of squares cannot overflow and its
/// leading term cannot underflow, as in the real generation below.
#[inline]
pub(crate) fn hypot<R: RealScalar>(x: R, y: R) -> R {
    let maximum = x.abs().max(y.abs());
    if maximum >= R::MIN_POSITIVE.sqrt() && maximum <= R::MAX.sqrt() * R::HALF {
        (x * x + y * y).sqrt()
    } else {
        x.hypot(y)
    }
}

/// Cancellation-free Jacobi coefficients for a diagonal difference and twice
/// the signed real cross term. Callers choose the orientation and complex phase.
/// Inputs are finite and normalized; zero cross gives the identity, including
/// when both inputs vanish. Equal diagonals select the positive delta branch.
pub(crate) fn jacobi<R: RealScalar>(delta: R, cross: R) -> (R, R) {
    if cross == R::ZERO {
        return (R::ONE, R::ZERO);
    }
    let sign = if delta < R::ZERO { -R::ONE } else { R::ONE };
    let t = sign * cross / (delta.abs() + delta.hypot(cross));
    let c = R::ONE / R::ONE.hypot(t);
    (c, t * c)
}

impl<S: NumericScalar> Rotation<S> {
    pub(crate) fn generate(f: S, g: S) -> Result<(Self, S), SciOxideError> {
        crate::checked::finite_candidate(f)?;
        crate::checked::finite_candidate(g)?;
        if g == S::ZERO {
            return Ok((
                Self {
                    c: S::Real::ONE,
                    s: S::ZERO,
                },
                f,
            ));
        }
        if S::IS_REAL {
            let maximum = f.abs().max(g.abs());
            // In this interval the sum of squares cannot overflow and its
            // leading term cannot underflow. A tiny second square may round
            // away only below the ordinary relative rounding allowance.
            // Extreme scales retain the binary-normalized path below.
            if maximum >= S::Real::MIN_POSITIVE.sqrt()
                && maximum <= S::Real::MAX.sqrt() * S::Real::HALF
            {
                let f = f.real_component();
                let g = g.real_component();
                let norm = (f * f + g * g).sqrt();
                let sign = if f == S::Real::ZERO { S::Real::ONE } else { f };
                let r = norm.copysign(sign);
                return Ok((
                    Self {
                        c: f.abs() / norm,
                        s: S::from_real(g / r),
                    },
                    S::from_real(r),
                ));
            }
        }
        let fe = standard::exponent(standard::magnitude(f))?;
        let ge = standard::exponent(standard::magnitude(g))?;
        let e = if f == S::ZERO { ge } else { fe.max(ge) };
        let fs = standard::scale(f, -e);
        let gs = standard::scale(g, -e);
        let fa = fs.abs();
        let ga = gs.abs();
        let norm = fa.hypot(ga);
        // Determine f's phase at its own scale: f may be subnormal relative
        // to g without its phase being lost.
        let phase = if f == S::ZERO {
            S::ONE
        } else {
            let local = standard::scale(f, -fe);
            divide_real(local, local.abs())
        };
        let rotation = Self {
            c: fa / norm,
            s: phase * divide_real(gs.conj(), norm),
        };
        let r = standard::restore(phase * S::from_real(norm), e)?;
        Ok((rotation, r))
    }

    pub(crate) fn conjugate(self) -> Self {
        Self {
            c: self.c,
            s: self.s.conj(),
        }
    }
    /// Pure two-lane operation; the workspace caller checks publication.
    pub(crate) fn pair(self, x: S, y: S) -> Result<(S, S), SciOxideError> {
        let c = S::from_real(self.c);
        let maximum = standard::magnitude(x).max(standard::magnitude(y));
        let e = if maximum > S::Real::MAX * (S::Real::HALF * S::Real::HALF * S::Real::HALF) {
            standard::exponent(maximum)?
        } else {
            0
        };
        let a = standard::scale(x, -e);
        let b = standard::scale(y, -e);
        Ok((
            standard::scale(c * a + self.s * b, e),
            standard::scale(c * b - self.s.conj() * a, e),
        ))
    }

    /// Apply to two rows; transpose the view for columns and conjugate G
    /// for right application of G^H. All mutations are workspace mutations.
    pub(crate) fn apply(
        self,
        mut matrix: MatrixViewMut<'_, S>,
        first: usize,
        second: usize,
        columns: Range<usize>,
    ) -> Result<(), SciOxideError> {
        for column in columns {
            let (x, y) = self.pair(
                matrix.get_element(first, column),
                matrix.get_element(second, column),
            )?;
            crate::checked::finite_candidate(x)?;
            crate::checked::finite_candidate(y)?;
            matrix.set_element(first, column, x);
            matrix.set_element(second, column, y);
        }
        Ok(())
    }

    /// Plain `[x; y] := G [x; y]` on rows `first < second` of row-major
    /// `data` (row stride `stride`) over `columns`.
    ///
    /// Sweep kernels call this on normalized workspace values, where no
    /// intermediate can overflow; callers check finiteness once at the end.
    pub(crate) fn rotate_rows(
        self,
        data: &mut [S],
        stride: usize,
        first: usize,
        second: usize,
        columns: Range<usize>,
    ) -> Result<(), SciOxideError> {
        debug_assert!(first < second);
        let (upper, lower) = crate::workspace::split_mut(data, second * stride)?;
        let x = &mut upper[first * stride + columns.start..first * stride + columns.end];
        let y = &mut lower[columns];
        let (c, s) = (self.c, self.s);
        x.iter_mut().zip(y).for_each(|(x, y)| {
            let (a, b) = (*x, *y);
            *x = scale_real(a, c) + s * b;
            *y = scale_real(b, c) - s.conj() * a;
        });

        Ok(())
    }

    /// Plain `[x y] := [x y] Gᵀ` on columns `first`, `second` of row-major
    /// `data` (row stride `stride`) over `rows`; conjugate G for `G^H`.
    /// The arithmetic contract is that of [`Self::rotate_rows`].
    pub(crate) fn rotate_columns(
        self,
        data: &mut [S],
        stride: usize,
        first: usize,
        second: usize,
        rows: Range<usize>,
    ) {
        let (c, s) = (self.c, self.s);
        for row in rows {
            let base = row * stride;
            let (a, b) = (data[base + first], data[base + second]);
            data[base + first] = scale_real(a, c) + s * b;
            data[base + second] = scale_real(b, c) - s.conj() * a;
        }
    }
}

/// Multiply by a real factor componentwise.
pub(crate) fn scale_real<S: NumericScalar>(value: S, factor: S::Real) -> S {
    S::from_components(
        value.real_component() * factor,
        value.imaginary_component() * factor,
    )
}

pub(crate) fn divide_real<S: NumericScalar>(value: S, divisor: S::Real) -> S {
    S::from_components(
        value.real_component() / divisor,
        value.imaginary_component() / divisor,
    )
}

/// Apply a backward sweep of adjacent real rotations to orthonormal rows.
/// Each column tile stays hot across the sweep. The caller owns the bounded
/// workspace and checks finiteness before writing the completed vectors.
pub(crate) fn apply_adjacent_rows<S: NumericScalar>(
    data: &mut [S],
    stride: usize,
    cosines: &[S::Real],
    sines: &[S::Real],
    first: usize,
) -> Result<(), SciOxideError> {
    for column in (0..stride).step_by(32) {
        let end = (column + 32).min(stride);
        for (index, (&c, &s)) in cosines.iter().zip(sines).enumerate().rev() {
            let row = first + index;
            let (upper, lower) = crate::workspace::split_mut(data, (row + 1) * stride)?;
            let x = &mut upper[row * stride + column..row * stride + end];
            let y = &mut lower[column..end];
            for (x, y) in x.iter_mut().zip(y) {
                let (a, b) = (*x, *y);
                *x = scale_real(a, c) + scale_real(b, s);
                *y = scale_real(b, c) - scale_real(a, s);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    #[test]
    fn rotation_norm_preserves_small_and_extreme_lanes() {
        fn check<R: RealScalar>() {
            let lower = R::MIN_POSITIVE.sqrt();
            let upper = R::MAX.sqrt() * R::HALF;
            for x in [
                R::MIN_SUBNORMAL,
                R::MIN_POSITIVE,
                lower.next_after(R::ZERO),
                lower,
                lower.next_after(R::MAX),
                upper.next_after(R::ZERO),
                upper,
                upper.next_after(R::MAX),
                R::MAX * R::HALF,
            ] {
                let y = x * R::HALF;
                let expected = x.hypot(y);
                let actual = hypot(x, y);
                assert!(actual > R::ZERO && actual.is_finite());
                assert!((actual / expected - R::ONE).abs() <= R::EPSILON * R::TWO * R::TWO);
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn real_rotation_crosses_safe_squaring_boundaries() {
        fn check<R: RealScalar>() {
            let lower = R::MIN_POSITIVE.sqrt();
            let upper = R::MAX.sqrt() * R::HALF;
            let bound = R::try_from_usize_exact(16).unwrap() * R::EPSILON;
            for boundary in [lower, upper] {
                for scale in [
                    boundary.next_after(R::ZERO),
                    boundary,
                    boundary.next_after(R::MAX),
                ] {
                    for sign in [-R::ONE, R::ONE] {
                        let (rotation, r) =
                            Rotation::generate(sign * scale, scale * R::HALF).unwrap();
                        assert!(rotation.c >= R::ZERO);
                        assert!(
                            (rotation.c * rotation.c + rotation.s * rotation.s - R::ONE).abs()
                                <= bound
                        );
                        assert!((-rotation.s * sign + rotation.c * R::HALF).abs() <= bound);
                        assert!(
                            (rotation.c * sign + rotation.s * R::HALF - r / scale).abs() <= bound
                        );
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    fn jacobi_cases<R: RealScalar>() {
        for scale in [
            R::MIN_POSITIVE * R::TWO,
            R::ONE,
            R::MAX / R::try_from_f64_exact(16.0).unwrap(),
        ] {
            for (delta, cross) in [
                (0., 0.),
                (2., 0.),
                (0., 2.),
                (0., -2.),
                (3., 4.),
                (-3., 4.),
                (3., -4.),
                (-3., -4.),
            ] {
                let delta = R::try_from_f64_exact(delta).unwrap();
                let cross = R::try_from_f64_exact(cross).unwrap();
                let (c, s) = jacobi(delta * scale, cross * scale);
                let bound = R::try_from_f64_exact(32.0).unwrap() * R::EPSILON;
                assert!((c * c + s * s - R::ONE).abs() <= bound);
                // Off-diagonal of the independently transformed symmetric
                // 2x2 Gram matrix, evaluated in its original unit frame.
                assert!((-c * s * delta + (c * c - s * s) * cross * R::HALF).abs() <= bound);
                if cross == R::ZERO {
                    assert_eq!((c, s), (R::ONE, R::ZERO));
                }
            }
        }
    }

    #[test]
    fn jacobi_coefficients_preserve_signs_and_annihilate_cross_terms() {
        jacobi_cases::<f32>();
        jacobi_cases::<f64>();
    }

    #[test]
    fn scaled_annihilation_and_signs() {
        for scale in [f64::from_bits(1), 1e-300, 1.0, 1e300] {
            for (a, b) in [(3., 4.), (-3., 4.), (3., -4.), (0., -4.), (-3., 0.)] {
                let (g, r) = Rotation::generate(a * scale, b * scale).unwrap();
                let (x, y) = g.pair(a, b).unwrap();
                assert!(g.c >= 0.);
                assert!((g.c * g.c + g.s * g.s - 1.).abs() < 1e-14);
                assert!(y.abs() < 1e-14);
                assert!(
                    (x - r / scale).abs()
                        <= if scale < f64::MIN_POSITIVE {
                            1.0
                        } else {
                            1e-14
                        }
                );
                if a != 0. {
                    assert_eq!(r.is_sign_negative(), a.is_sign_negative());
                }
            }
        }
    }
    #[test]
    fn complex_phase_and_extreme_spread() {
        for (f, g) in [
            (Complex::new(3., -4.), Complex::new(-5., 2.)),
            (Complex::new(1e-300, 1e-300), Complex::new(1e300, -1e300)),
            (Complex::new(0., 0.), Complex::new(0., -2.)),
        ] {
            let (rotation, r) = Rotation::generate(f, g).unwrap();
            let scale = f.norm().max(g.norm());
            let (x, y) = rotation.pair(f / scale, g / scale).unwrap();
            assert!((x - r / scale).norm() < 1e-14);
            assert!(y.norm() < 1e-14);
            assert!((rotation.c * rotation.c + rotation.s.norm_sqr() - 1.).abs() < 1e-14);
        }
    }
    #[test]
    fn plain_row_and_column_rotations_match_the_checked_kernel() -> Result<(), SciOxideError> {
        let n = 5;
        let data: alloc::vec::Vec<Complex<f64>> = (0..n * n)
            .map(|i| Complex::new((i % 7) as f64 * 0.25 - 0.5, (i % 3) as f64 * 0.5 - 0.25))
            .collect();
        let (g, _) =
            Rotation::generate(Complex::new(0.75, -0.5), Complex::new(-1.5, 0.25)).unwrap();
        for (columns, rows) in [(1..4, 0..n), (0..n, 2..5)] {
            let mut plain = data.clone();
            let mut checked = data.clone();
            g.rotate_rows(&mut plain, n, 1, 3, columns.clone())?;
            g.apply(
                MatrixViewMut::from_slice(n, n, &mut checked).unwrap(),
                1,
                3,
                columns,
            )
            .unwrap();
            g.rotate_columns(&mut plain, n, 0, 4, rows.clone());
            g.apply(
                MatrixViewMut::from_slice(n, n, &mut checked)
                    .unwrap()
                    .transpose(),
                0,
                4,
                rows,
            )
            .unwrap();
            plain.iter().zip(&checked).for_each(|(a, b)| {
                assert!((*a - *b).norm() < 1e-15, "{a:?} {b:?}");
            });
        }
        Ok(())
    }

    #[test]
    fn batched_backward_rotations_match_checked_application_across_tiles()
    -> Result<(), SciOxideError> {
        for n in [2, 31, 32, 33, 65] {
            let mut matrix = crate::Matrix::<Complex<f64>>::identity(n).unwrap();
            let mut reference = matrix.clone();
            let cosines: alloc::vec::Vec<_> = (0..n - 1)
                .map(|i| if i % 3 == 0 { 1.0 } else { 0.6 })
                .collect();
            let sines: alloc::vec::Vec<_> = (0..n - 1)
                .map(|i| if i % 3 == 0 { 0.0 } else { -0.8 })
                .collect();
            // Starting late also exercises flushing only completed rotations.
            for first in [0, n / 2, n - 1] {
                apply_adjacent_rows(
                    matrix.as_mut_slice(),
                    n,
                    &cosines[first..],
                    &sines[first..],
                    first,
                )?;
                for i in (first..n - 1).rev() {
                    Rotation {
                        c: cosines[i],
                        s: Complex::new(sines[i], 0.0),
                    }
                    .apply(reference.view_mut(), i, i + 1, 0..n)
                    .unwrap();
                }
                for (actual, expected) in matrix.as_slice().iter().zip(reference.as_slice()) {
                    assert!((*actual - *expected).norm() <= 8.0 * n as f64 * f64::EPSILON);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn binary32_scales_and_nonfinite_candidates() {
        for scale in [f32::from_bits(1), 1e-30, 1., 1e30] {
            let (g, r) = Rotation::generate(-3. * scale, 4. * scale).unwrap();
            let (x, y) = g.pair(-3., 4.).unwrap();
            assert!((x + 5.).abs() < 1e-5 && y.abs() < 1e-5);
            assert!((r / scale + 5.).abs() < 1e-5);
        }
        assert!(Rotation::generate(f64::INFINITY, 1.).is_err());
        assert!(Rotation::generate(f64::MAX, f64::MAX).is_err());
    }
}
