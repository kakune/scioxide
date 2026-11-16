//! Exact geometric predicates: orientation, in-circle and in-sphere tests.

mod exact;
mod filter;
pub(crate) use filter::in_range as fixed_in_range;
pub(crate) mod nd;

pub(crate) use exact::magnitude as determinant_magnitude;

use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{Budget, InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError};

/// Sign of a geometric determinant, computed exactly for the floating-point
/// inputs as given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[must_use]
pub enum PredicateSign {
    /// The determinant is strictly negative.
    Negative,
    /// The determinant is exactly zero. No tolerance or perturbation is
    /// applied.
    Zero,
    /// The determinant is strictly positive.
    Positive,
}

impl From<Ordering> for PredicateSign {
    fn from(value: Ordering) -> Self {
        match value {
            Ordering::Less => Self::Negative,
            Ordering::Equal => Self::Zero,
            Ordering::Greater => Self::Positive,
        }
    }
}

/// Exact orientation test for a simplex in any dimension `d`.
///
/// [`Self::orient`] returns the sign of `det(v₁ − v₀, …, v_d − v₀)` for vertices
/// `v₀, …, v_d`. This agrees with [`orient2d`] in two dimensions and has the
/// opposite sign of [`orient3d`] in three. A floating-point filter decides the
/// sign when it can certify it, for determinants of order up to ten; otherwise
/// the determinant is evaluated exactly. The exact evaluation expands the
/// determinant into its `(d + 1)!` Leibniz terms, so [`Self::new`] requires
/// `max_terms ≥ (d + 1)!` once, at construction. The workspace holds
/// `O(d² + 2^min(d+1, 10))` scratch values and [`Self::orient`] does not
/// allocate.
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_spatial::{PredicateSign, SimplexPredicateWorkspace};
/// let mut workspace = SimplexPredicateWorkspace::new(2, Budget::try_from(6)?)?;
/// let vertices: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
/// assert_eq!(workspace.orient(&vertices)?, PredicateSign::Positive);
/// assert!(workspace.orient(&vertices[..2]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct SimplexPredicateWorkspace {
    exact: nd::ExactWorkspace,
    rows: Vec<f64>,
    dimension: usize,
}

impl SimplexPredicateWorkspace {
    /// Creates a workspace for simplices in `dimension` dimensions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero or `max_terms < (dimension + 1)!`.
    /// - `SizeOverflow` or `AllocationFailure` if the scratch cannot be
    ///   reserved.
    pub fn new(dimension: usize, max_terms: Budget) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("dimension"),
                InvalidInputReason::OutOfRange,
                "simplex dimension must be positive",
            ));
        }
        let count = dimension.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("simplex_predicate_order"))
        })?;
        let terms = (1..=count).try_fold(1usize, |n, factor| {
            checked_mul(
                n,
                factor,
                scioxide_core::quantity!("simplex_predicate_terms"),
            )
        })?;
        if terms > max_terms.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("max_terms"),
                InvalidInputReason::OutOfRange,
                "exact orientation exceeds the caller term budget",
            ));
        }
        let entries = checked_mul(
            count,
            dimension,
            scioxide_core::quantity!("simplex_predicate_coordinates"),
        )?;
        Ok(Self {
            exact: nd::ExactWorkspace::new(count)?,
            rows: try_filled(
                entries,
                0.0,
                scioxide_core::quantity!("simplex_predicate_coordinates"),
            )?,
            dimension,
        })
    }

    /// Returns the orientation of the simplex with the given `d + 1` vertices.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if there are not `d + 1` vertices of length `d`.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    pub fn orient<S: RealScalar>(
        &mut self,
        vertices: &[&[S]],
    ) -> Result<PredicateSign, SciOxideError> {
        let dimension = self.dimension;
        if vertices.len() != dimension + 1 {
            return Err(SciOxideError::dimension_mismatch(
                [dimension + 1],
                [vertices.len()],
            ));
        }
        if let Some(vertex) = vertices.iter().find(|v| v.len() != dimension) {
            return Err(SciOxideError::dimension_mismatch(
                [dimension],
                [vertex.len()],
            ));
        }
        for (destination, &coordinate) in self
            .rows
            .iter_mut()
            .zip(vertices.iter().flat_map(|v| v.iter()))
        {
            if !coordinate.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "coordinates must be finite",
                ));
            }
            *destination = coordinate.try_to_f64_exact()?;
        }
        let sign = self
            .exact
            .sign(&self.rows, dimension, false, &mut |_| Ok(()))?;
        // Moving the last column of ones to the first row's origin changes
        // the determinant sign in odd coordinate dimensions.
        Ok(match (sign, dimension % 2) {
            (0, _) => PredicateSign::Zero,
            (1, 0) | (-1, 1) => PredicateSign::Positive,
            _ => PredicateSign::Negative,
        })
    }

    /// Returns the dimension `d` of the workspace.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Returns the orientation of three points in the plane.
///
/// The result is the sign of
///
/// ```text
/// | aₓ  a_y  1 |
/// | bₓ  b_y  1 |
/// | cₓ  c_y  1 |
/// ```
///
/// It is positive when `a, b, c` are in counterclockwise order, negative when
/// clockwise, and zero when they are collinear or two of them coincide. The
/// sign is exact for every finite `f32` or `f64` input, including subnormal
/// numbers. A floating-point filter decides most cases; the rest are
/// evaluated in exact arithmetic on the stack.
///
/// # Errors
///
/// - `NonFinite` if a coordinate is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_spatial::{PredicateSign, orient2d};
/// let a = [0.0_f64, 0.0];
/// let b = [1.0, 0.0];
/// let c = [0.0, 1.0];
/// assert_eq!(orient2d(&a, &b, &c)?, PredicateSign::Positive);
/// assert!(orient2d(&a, &b, &[f64::NAN, 1.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn orient2d<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
) -> Result<PredicateSign, SciOxideError> {
    let points = [a, b, c];
    if filter::eligible(points)
        && let Some(sign) = filter::orient2d(a, b, c)
    {
        return Ok(sign);
    }
    exact::checked_fixed(points, false)
}

/// Returns the orientation of four points in space.
///
/// The result is the sign of the determinant with rows `[x, y, z, 1]` for
/// `a, b, c, d`. It is positive when `d` lies below the plane through `a, b, c`,
/// where below means that `a, b, c` appear counterclockwise when viewed from
/// above. It is zero when the points are coplanar. Swapping two arguments
/// reverses the sign. The sign is exact for every finite `f32` or `f64` input.
///
/// # Errors
///
/// - `NonFinite` if a coordinate is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_spatial::{PredicateSign, orient3d};
/// let a = [0.0_f64; 3];
/// let b = [1.0, 0.0, 0.0];
/// let c = [0.0, 1.0, 0.0];
/// let d = [0.0, 0.0, 1.0];
/// assert_eq!(orient3d(&a, &b, &c, &d)?, PredicateSign::Negative);
/// assert!(orient3d(&a, &b, &c, &[f64::INFINITY; 3]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn orient3d<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
) -> Result<PredicateSign, SciOxideError> {
    let points = [a, b, c, d];
    if filter::eligible(points)
        && let Some(sign) = filter::orient3d(a, b, c, d)
    {
        return Ok(sign);
    }
    exact::checked_fixed(points, false)
}

/// Returns whether `d` lies inside the circle through `a, b, c`.
///
/// The result is the sign of the determinant with rows `[x, y, x² + y², 1]`
/// for `a, b, c, d`. For counterclockwise `a, b, c` it is positive when `d` is
/// inside the circle, negative outside, and zero on it; clockwise `a, b, c`
/// reverse the sign. If `a, b, c` are collinear there is no circle, and the
/// sign does not describe inside or outside. The sign is exact for every
/// finite `f32` or `f64` input; no center or radius is computed.
///
/// # Errors
///
/// - `NonFinite` if a coordinate is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_spatial::{PredicateSign, incircle};
/// let a = [0.0_f64, 0.0];
/// let b = [1.0, 0.0];
/// let c = [0.0, 1.0];
/// assert_eq!(
///     incircle(&a, &b, &c, &[0.25, 0.25])?,
///     PredicateSign::Positive
/// );
/// assert!(incircle(&a, &b, &c, &[f64::NAN, 0.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn incircle<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
    d: &[S; 2],
) -> Result<PredicateSign, SciOxideError> {
    let points = [a, b, c, d];
    if filter::eligible(points)
        && let Some(sign) = filter::incircle(a, b, c, d)
    {
        return Ok(sign);
    }
    exact::checked_fixed(points, true)
}

/// Returns whether `e` lies inside the sphere through `a, b, c, d`.
///
/// The result is the sign of the determinant with rows
/// `[x, y, z, x² + y² + z², 1]` for `a, b, c, d, e`. When
/// `orient3d(a, b, c, d)` is positive, it is positive when `e` is inside the
/// sphere, negative outside, and zero on it; the opposite orientation reverses
/// the sign. If `a, b, c, d` are coplanar there is no sphere, and the sign does
/// not describe inside or outside. The sign is exact for every finite `f32` or
/// `f64` input.
///
/// # Errors
///
/// - `NonFinite` if a coordinate is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_spatial::{PredicateSign, insphere};
/// let a = [0.0_f64; 3];
/// let b = [1.0, 0.0, 0.0];
/// let c = [0.0, 1.0, 0.0];
/// let d = [0.0, 0.0, 1.0];
/// // This tetrahedron has negative orient3d, so an interior query is negative.
/// assert_eq!(
///     insphere(&a, &b, &c, &d, &[0.25; 3])?,
///     PredicateSign::Negative
/// );
/// assert!(insphere(&a, &b, &c, &d, &[f64::NAN; 3]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn insphere<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
    e: &[S; 3],
) -> Result<PredicateSign, SciOxideError> {
    let points = [a, b, c, d, e];
    if filter::eligible(points)
        && let Some(sign) = filter::insphere(a, b, c, d, e)
    {
        return Ok(sign);
    }
    exact::checked_fixed(points, true)
}

/// Filter only; the caller validates finiteness and any claimed envelope.
#[inline]
pub(crate) fn orient2d_filter<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
    envelope: bool,
) -> Option<PredicateSign> {
    let points = [a, b, c];
    if envelope || filter::eligible(points) {
        filter::orient2d(a, b, c)
    } else {
        None
    }
}

/// Filter only; the caller validates finiteness and any claimed envelope.
#[inline]
pub(crate) fn orient3d_filter<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
    envelope: bool,
) -> Option<PredicateSign> {
    let points = [a, b, c, d];
    if envelope || filter::eligible(points) {
        filter::orient3d(a, b, c, d)
    } else {
        None
    }
}

/// Filter only; the caller validates finiteness and any claimed envelope.
#[inline]
pub(crate) fn incircle_filter<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
    d: &[S; 2],
    envelope: bool,
) -> Option<PredicateSign> {
    let points = [a, b, c, d];
    if envelope || filter::eligible(points) {
        filter::incircle(a, b, c, d)
    } else {
        None
    }
}

/// Filter only; the caller validates finiteness and any claimed envelope.
#[inline]
pub(crate) fn insphere_filter<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
    e: &[S; 3],
    envelope: bool,
) -> Option<PredicateSign> {
    let points = [a, b, c, d, e];
    if envelope || filter::eligible(points) {
        filter::insphere(a, b, c, d, e)
    } else {
        None
    }
}

/// Exact sign for already validated finite coordinates.
/// The caller establishes finiteness and, when `envelope` is true, the filter
/// exponent envelope for every coordinate. Otherwise the envelope is checked.
#[inline]
pub(crate) fn orient2d_unchecked<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
    envelope: bool,
) -> PredicateSign {
    let points = [a, b, c];
    if (envelope || filter::eligible(points))
        && let Some(sign) = filter::orient2d(a, b, c)
    {
        return sign;
    }
    exact::fixed(points, false)
}

/// Exact sign for already validated finite coordinates.
/// The caller establishes finiteness and any claimed exponent envelope.
#[inline]
pub(crate) fn orient3d_unchecked<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
    envelope: bool,
) -> PredicateSign {
    let points = [a, b, c, d];
    if (envelope || filter::eligible(points))
        && let Some(sign) = filter::orient3d(a, b, c, d)
    {
        return sign;
    }
    exact::fixed(points, false)
}

/// Exact sign for already validated finite coordinates.
/// The caller establishes finiteness and any claimed exponent envelope.
#[inline]
pub(crate) fn incircle_unchecked<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
    d: &[S; 2],
    envelope: bool,
) -> PredicateSign {
    let points = [a, b, c, d];
    if (envelope || filter::eligible(points))
        && let Some(sign) = filter::incircle(a, b, c, d)
    {
        return sign;
    }
    exact::fixed(points, true)
}

/// Exact sign for already validated finite coordinates.
/// The caller establishes finiteness and any claimed exponent envelope.
#[inline]
pub(crate) fn insphere_unchecked<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
    e: &[S; 3],
    envelope: bool,
) -> PredicateSign {
    let points = [a, b, c, d, e];
    if (envelope || filter::eligible(points))
        && let Some(sign) = filter::insphere(a, b, c, d, e)
    {
        return sign;
    }
    exact::fixed(points, true)
}

// Leibniz term bounds of the fixed 2-D and 3-D exact predicates.
pub(crate) fn fixed_upper(dimension: usize, lifted: bool) -> usize {
    match (dimension, lifted) {
        (2, false) => 6,
        (3, false) => 24,
        (2, true) => 48,
        _ => 360,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_entries_preserve_exact_signs_at_envelope_boundaries() {
        let min = f64::from_bits((1023 - 100) << 52);
        let max = f64::from_bits((1023 + 100) << 52);
        for x in [0.0, -0.0, min, -min, max, -max] {
            assert!(filter::in_range(x));
        }
        for x in [min.next_down(), max.next_up(), f64::from_bits(1), f64::MAX] {
            assert!(!filter::in_range(x));
            assert!(!filter::in_range(-x));
        }
        for scale in [
            0.0,
            min.next_down(),
            min,
            min.next_up(),
            1.0,
            max.next_down(),
            max,
            max.next_up(),
            f64::from_bits(1),
            f64::MAX,
        ] {
            let points = [
                [scale, -0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, scale],
                [-scale, 0.0, 0.0],
                [0.0; 3],
            ];
            for repeated in [false, true] {
                let mut p = points;
                if repeated {
                    p[1] = p[0];
                }
                let q = p.map(|p| [p[0], p[1]]);
                let actual = [
                    orient2d_unchecked(&q[0], &q[1], &q[2], false),
                    orient3d_unchecked(&p[0], &p[1], &p[2], &p[3], false),
                    incircle_unchecked(&q[0], &q[1], &q[2], &q[3], false),
                    insphere_unchecked(&p[0], &p[1], &p[2], &p[3], &p[4], false),
                ];
                let checked = [
                    orient2d(&q[0], &q[1], &q[2]).unwrap(),
                    orient3d(&p[0], &p[1], &p[2], &p[3]).unwrap(),
                    incircle(&q[0], &q[1], &q[2], &q[3]).unwrap(),
                    insphere(&p[0], &p[1], &p[2], &p[3], &p[4]).unwrap(),
                ];
                let expected = [
                    exact::sign(&p, 2, 3, false),
                    exact::sign(&p, 3, 4, false),
                    exact::sign(&p, 2, 4, true),
                    exact::sign(&p, 3, 5, true),
                ];
                if filter::in_range(scale) {
                    assert_eq!(
                        [
                            orient2d_unchecked(&q[0], &q[1], &q[2], true),
                            orient3d_unchecked(&p[0], &p[1], &p[2], &p[3], true),
                            incircle_unchecked(&q[0], &q[1], &q[2], &q[3], true),
                            insphere_unchecked(&p[0], &p[1], &p[2], &p[3], &p[4], true),
                        ],
                        expected
                    );
                }
                assert_eq!(actual, expected, "scale={scale}, repeated={repeated}");
                assert_eq!(checked, expected, "scale={scale}, repeated={repeated}");
            }
        }
    }

    #[test]
    fn general_simplex_workspace_classifies_orientation_and_degeneracy() {
        let mut workspace =
            SimplexPredicateWorkspace::new(4, Budget::try_from(120).unwrap()).unwrap();
        let origin = [0.0f64; 4];
        let e1 = [1.0, 0.0, 0.0, 0.0];
        let e2 = [0.0, 1.0, 0.0, 0.0];
        let e3 = [0.0, 0.0, 1.0, 0.0];
        let e4 = [0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            workspace.orient(&[&origin, &e1, &e2, &e3, &e4]).unwrap(),
            PredicateSign::Positive
        );
        assert_eq!(
            workspace.orient(&[&origin, &e2, &e1, &e3, &e4]).unwrap(),
            PredicateSign::Negative
        );
        assert_eq!(
            workspace.orient(&[&origin, &e1, &e1, &e3, &e4]).unwrap(),
            PredicateSign::Zero
        );
        assert!(SimplexPredicateWorkspace::new(4, Budget::try_from(119).unwrap()).is_err());
        assert!(workspace.orient(&[&origin, &e1]).is_err());
    }

    #[test]
    fn cached_primitive_envelopes_preserve_f32_geometry() {
        for scale in [2_f32.powi(-100), 1., 2_f32.powi(100)] {
            let p = [
                [scale, 0., 0.],
                [0., scale, 0.],
                [0., 0., scale],
                [-scale, 0., 0.],
                [0.; 3],
            ];
            let q = p.map(|p| [p[0], p[1]]);
            assert_eq!(
                [
                    orient2d_unchecked(&q[0], &q[1], &q[2], true),
                    orient3d_unchecked(&p[0], &p[1], &p[2], &p[3], true),
                    incircle_unchecked(&q[0], &q[1], &q[2], &q[3], true),
                    insphere_unchecked(&p[0], &p[1], &p[2], &p[3], &p[4], true),
                ],
                [
                    PredicateSign::Positive,
                    PredicateSign::Positive,
                    PredicateSign::Negative,
                    PredicateSign::Positive
                ]
            );
        }
    }

    #[test]
    fn filters_certify_ordinary_inputs_and_defer_degeneracy_and_extremes() {
        let classify = |points: &[[f64; 3]; 5], dimension, count, lifted| {
            if !points[..count]
                .iter()
                .flatten()
                .all(|&x| filter::in_range(x))
            {
                return None;
            }
            let q = points.map(|p| [p[0], p[1]]);
            match (dimension, lifted) {
                (2, false) => filter::orient2d(&q[0], &q[1], &q[2]),
                (3, false) => filter::orient3d(&points[0], &points[1], &points[2], &points[3]),
                (2, true) => filter::incircle(&q[0], &q[1], &q[2], &q[3]),
                _ => filter::insphere(&points[0], &points[1], &points[2], &points[3], &points[4]),
            }
        };
        for (dimension, count, lifted, points) in [
            (
                2,
                3,
                false,
                [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0.; 3], [0.; 3]],
            ),
            (
                3,
                4,
                false,
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.; 3], [0.; 3]],
            ),
            (
                2,
                4,
                true,
                [[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0.; 3], [0.; 3]],
            ),
            (
                3,
                5,
                true,
                [
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],
                    [-1., 0., 0.],
                    [0.; 3],
                ],
            ),
        ] {
            assert_eq!(
                classify(&points, dimension, count, lifted),
                Some(exact::sign(&points, dimension, count, lifted))
            );
            let mut repeated = points;
            repeated[1] = repeated[0];
            assert_eq!(classify(&repeated, dimension, count, lifted), None);
            assert_eq!(
                exact::sign(&repeated, dimension, count, lifted),
                PredicateSign::Zero
            );
            let extreme = points.map(|p| p.map(|x| x * f64::MAX));
            assert_eq!(classify(&extreme, dimension, count, lifted), None);
        }
    }
}
