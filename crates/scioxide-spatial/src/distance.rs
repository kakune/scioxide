//! Point sets, distance metrics, distance matrices and the Hausdorff distance.
//!
//! Batch functions evaluate pairs in blocks of 256. A block on contiguous
//! rows runs without per-pair checks and is checked once at its end; a block
//! that fails the check is evaluated again pair by pair with full checks. The
//! operation order, and so every result bit, is the same for all layouts.
//!
//! ```
//! use scioxide_spatial::{PointSet, pairwise, directed_hausdorff};
//! let sites = [[0.0_f64, 0.0], [3.0, 4.0]];
//! let points = PointSet::from_points(&sites)?;
//! assert_eq!(pairwise(&points, &points)?.as_slice(), [0.0, 5.0, 5.0, 0.0]);
//! assert!(directed_hausdorff(&points, &points)?.distance.abs() < 1e-12);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```

use crate::validation::out_of_range as invalid;
use crate::workspace::require;
use crate::{
    DistanceOptions, DistanceWorkspace, DistanceWorkspaceRequest, SpatialExecutionContext,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{Argument, DomainReason, NdView, NonFiniteOrigin, RealScalar, SciOxideError};
use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, Matrix, MatrixView};

mod blocks;
#[cfg(test)]
mod tests;

// `argument` lies outside the metric's domain.
pub(crate) fn domain(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::domain_error(argument, DomainReason::Undefined, detail)
}
// Reject an overflowed intermediate.
#[inline]
pub(crate) fn finite(value: f64) -> Result<f64, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "unrepresentable distance arithmetic",
        ))
    }
}

/// Borrows finite coordinates from an array of points, flat rows or a rank-two [`NdView`].
///
/// Any strides are allowed, including zero strides that repeat a row. The
/// coordinates are checked once, at construction, so the functions that take
/// a point set do not check them again. Cloning copies the borrow and inline strides;
/// layout arrays need not outlive construction.
///
/// ```
/// use scioxide_core::{NdView, StridedLayout};
/// use scioxide_spatial::PointSet;
///
/// // Two points in the plane, with a padding coordinate between rows.
/// let data = [0.0_f64, 0.0, 99.0, 3.0, 4.0];
/// let points = PointSet::new(NdView::new(&data, StridedLayout::new(&[2, 2], &[3, 1])?)?)?;
/// assert_eq!((points.len(), points.dimension()), (2, 2));
/// assert_eq!(points.coordinate(1, 1)?, 4.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PointSet<'data, S: RealScalar> {
    data: &'data [S],
    len: usize,
    dimension: usize,
    strides: [usize; 2],
}
impl<'data, S: RealScalar> PointSet<'data, S> {
    /// Creates a point set from a view whose rows are points.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the view does not have rank two.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    pub fn new(view: NdView<'data, '_, S>) -> Result<Self, SciOxideError> {
        if view.layout().shape().rank() != 2 {
            return Err(invalid(
                scioxide_core::argument!("view"),
                "points require a rank-two layout",
            ));
        }
        // Packed transform batches should pay one linear validation scan, not
        // rank-two offset calculation for every scalar. Ignore trailing storage;
        // padding and broadcast layouts still follow their logical coordinates.
        let layout = view.layout();
        let packed = layout.strides()[1] == 1 && layout.strides()[0] == layout.shape().extents()[1];
        let non_finite = if packed {
            view.storage()[..layout.shape().len()]
                .iter()
                .any(|x| !x.is_finite())
        } else {
            view.iter().any(|x| !x.is_finite())
        };
        if non_finite {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "coordinates must be finite",
            ));
        }
        Ok(Self {
            data: view.storage(),
            len: layout.shape().extents()[0],
            dimension: layout.shape().extents()[1],
            strides: [layout.strides()[0], layout.strides()[1]],
        })
    }
    /// Borrows an array of points without allocating; validates all coordinates in `O(n D)`.
    ///
    /// # Errors
    /// Returns `NonFinite` for NaN or infinite coordinates. Zero dimensions are accepted.
    ///
    /// ```
    /// let points = scioxide_spatial::PointSet::from_points(&[[0.0_f64, 1.0], [2.0, 3.0]])?;
    /// assert!((points.coordinate(1, 0)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_points<const D: usize>(points: &'data [[S; D]]) -> Result<Self, SciOxideError> {
        Self::validate_coordinates(points.as_flattened())?;
        Ok(Self::from_validated(points.as_flattened(), points.len(), D))
    }
    /// Borrows row-major coordinates with `dimension` coordinates per point in `O(n d)` time.
    /// Allocates no storage.
    ///
    /// # Errors
    /// Returns `InvalidInput` for zero dimension, `DimensionMismatch` for incomplete rows,
    /// or `NonFinite` for NaN or infinite coordinates, in that order.
    pub fn from_rows(coordinates: &'data [S], dimension: usize) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "dimension must be positive",
            ));
        }
        let remainder = coordinates.len() % dimension;
        if remainder != 0 {
            return Err(SciOxideError::dimension_mismatch([dimension], [remainder]));
        }
        Self::validate_coordinates(coordinates)?;
        Ok(Self::from_validated(
            coordinates,
            coordinates.len() / dimension,
            dimension,
        ))
    }
    fn validate_coordinates(coordinates: &[S]) -> Result<(), SciOxideError> {
        if coordinates.iter().any(|x| !x.is_finite()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "coordinates must be finite",
            ));
        }
        Ok(())
    }
    pub(crate) fn from_validated(data: &'data [S], len: usize, dimension: usize) -> Self {
        Self {
            data,
            len,
            dimension,
            strides: [dimension, 1],
        }
    }
    /// Returns the number of points. Points of dimension zero count too.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Returns whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the number of coordinates per point.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns coordinate `axis` of point `point`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `point` or `axis` is out of range.
    pub fn coordinate(&self, point: usize, axis: usize) -> Result<S, SciOxideError> {
        for (index, bound) in [(point, self.len), (axis, self.dimension)] {
            if index >= bound {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("layout_coordinate"),
                    index,
                    bound,
                ));
            }
        }
        Ok(self.data[point * self.strides[0] + axis * self.strides[1]])
    }
    pub(crate) fn contiguous(&self) -> bool {
        self.strides[1] == 1
    }
    // Only called with a proved in-range row. Empty coordinate layouts have
    // no storage addresses, even when their unused strides are very large.
    // This view adapter must disappear inside fixed-width transform loops.
    #[inline(always)]
    pub(crate) fn row<const CONTIGUOUS: bool>(&self, point: usize) -> Row<'_, S> {
        let dimension = self.dimension();
        let strides = self.strides;
        let data = if dimension == 0 {
            &[]
        } else {
            let start = point * strides[0];
            let span = if CONTIGUOUS {
                dimension
            } else {
                (dimension - 1) * strides[1] + 1
            };
            &self.data[start..start + span]
        };
        Row {
            data,
            dimension,
            stride: if CONTIGUOUS { 1 } else { strides[1] },
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Row<'a, S> {
    data: &'a [S],
    dimension: usize,
    stride: usize,
}
impl<S: RealScalar> Row<'_, S> {
    // Native-scalar access for geometry kernels after dimension validation.
    #[inline]
    pub(crate) fn value(self, axis: usize) -> S {
        self.data[axis * self.stride]
    }
    #[inline]
    pub(crate) fn coordinate(self, axis: usize) -> f64 {
        self.value(axis).into_f64()
    }
    #[inline]
    fn pair(self, other: Self, axis: usize) -> (f64, f64) {
        (
            self.data[axis * self.stride].into_f64(),
            other.data[axis * other.stride].into_f64(),
        )
    }
}

// Select a concrete metric before entering a bulk loop. The literal named
// metrics let the compiler specialize their arithmetic, without an indirect
// function call or a parameter match for each pair.
macro_rules! dispatch_metric {
    ($prepared:expr, $kernel:ident, $body:expr) => {{
        macro_rules! named_kernel {
            ($metric:expr) => {{
                let $kernel = |left: Row<'_, S>, right: Row<'_, S>| {
                    finish(named($metric, left.dimension, |axis| {
                        left.pair(right, axis)
                    })?)
                };
                $body
            }};
        }
        match &$prepared.parameters {
            Parameters::Named(metric) => match *metric {
                DistanceMetric::BrayCurtis => named_kernel!(DistanceMetric::BrayCurtis),
                DistanceMetric::Canberra => named_kernel!(DistanceMetric::Canberra),
                DistanceMetric::Chebyshev => named_kernel!(DistanceMetric::Chebyshev),
                DistanceMetric::CityBlock => named_kernel!(DistanceMetric::CityBlock),
                DistanceMetric::Correlation => named_kernel!(DistanceMetric::Correlation),
                DistanceMetric::Cosine => named_kernel!(DistanceMetric::Cosine),
                DistanceMetric::Euclidean => named_kernel!(DistanceMetric::Euclidean),
                DistanceMetric::SquaredEuclidean => named_kernel!(DistanceMetric::SquaredEuclidean),
                DistanceMetric::Minkowski(p) => named_kernel!(DistanceMetric::Minkowski(p)),
                DistanceMetric::JensenShannon => named_kernel!(DistanceMetric::JensenShannon),
                DistanceMetric::Dice => named_kernel!(DistanceMetric::Dice),
                DistanceMetric::Hamming => named_kernel!(DistanceMetric::Hamming),
                DistanceMetric::Jaccard => named_kernel!(DistanceMetric::Jaccard),
                DistanceMetric::RogersTanimoto => named_kernel!(DistanceMetric::RogersTanimoto),
                DistanceMetric::RussellRao => named_kernel!(DistanceMetric::RussellRao),
                DistanceMetric::SokalSneath => named_kernel!(DistanceMetric::SokalSneath),
                DistanceMetric::Yule => named_kernel!(DistanceMetric::Yule),
            },
            Parameters::Standardized(scales) => {
                let $kernel = |left: Row<'_, S>, right: Row<'_, S>| {
                    finish(norm2(left.dimension, |axis| {
                        let (a, b) = left.pair(right, axis);
                        (a - b) / scales[axis]
                    })?)
                };
                $body
            }
            Parameters::Mahalanobis(factor) => {
                let $kernel =
                    |left: Row<'_, S>, right: Row<'_, S>| finish(mahalanobis(left, right, factor)?);
                $body
            }
        }
    }};
}

// A batch kernel consumes a contiguous output segment, not one fallible pair.
// Keep the checked scalar algorithms authoritative for general metrics/layouts
// and for replaying exceptional blocks.
macro_rules! dispatch_blocks {
    ($prepared:expr, $left:expr, $right:expr, $kernel:ident, $body:expr) => {{
        macro_rules! fast {
            ($metric:expr) => {{
                let $kernel = |a: Row<'_, S>, points: &PointSet<'_, S>, first, output: &mut [S]| {
                    blocks::fill($metric, a, points, first, output)
                };
                $body
            }};
        }
        macro_rules! checked {
            () => {
                dispatch_metric!($prepared, pair, {
                    let $kernel =
                        |a: Row<'_, S>, points: &PointSet<'_, S>, first, output: &mut [S]| {
                            for (column, value) in (first..).zip(output) {
                                *value = pair(a, points.row::<false>(column))?;
                            }
                            Ok(())
                        };
                    $body
                })
            };
        }
        if $left.contiguous() && $right.contiguous() {
            match &$prepared.parameters {
                Parameters::Named(DistanceMetric::Euclidean | DistanceMetric::Minkowski(2.0)) => {
                    fast!(DistanceMetric::Euclidean)
                }
                Parameters::Named(DistanceMetric::SquaredEuclidean) => {
                    fast!(DistanceMetric::SquaredEuclidean)
                }
                Parameters::Named(DistanceMetric::CityBlock) => fast!(DistanceMetric::CityBlock),
                Parameters::Named(DistanceMetric::Chebyshev) => fast!(DistanceMetric::Chebyshev),
                _ => checked!(),
            }
        } else {
            checked!()
        }
    }};
}

#[inline]
fn finish<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(finite(value)?).map_err(|_| {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "distance exceeds output scalar range",
        )
    })
}

// Try the unit frame first. A subnormal square sum cannot retain the relative
// accuracy of tiny norms, so it too is recomputed in the scaled frame.
#[inline]
fn norm2(n: usize, value: impl Fn(usize) -> f64) -> Result<f64, SciOxideError> {
    let mut sum = 0.0;
    for axis in 0..n {
        let x = value(axis);
        sum += x * x;
    }
    if sum.is_finite() && sum >= f64::MIN_POSITIVE {
        Ok(RealScalar::sqrt(sum))
    } else {
        norm(n, 2.0, |axis| Ok(value(axis)))
    }
}

#[inline]
fn mahalanobis<S: RealScalar>(
    left: Row<'_, S>,
    right: Row<'_, S>,
    factor: &CholeskyDecomposition<f64>,
) -> Result<f64, SciOxideError> {
    let n = left.dimension;
    let lower = factor.l_matrix().as_slice();
    // Preparation uses unpivoted Cholesky: VI = L L^T, d = ||L^T(a-b)||.
    norm2(n, |column| {
        let mut sum = 0.0;
        for row in column..n {
            let (a, b) = left.pair(right, row);
            sum += lower[row * n + column] * (a - b);
        }
        sum
    })
}

/// Distance and dissimilarity functions between two vectors `u` and `v`.
///
/// `Dice`, `Jaccard`, `RogersTanimoto`, `RussellRao`, `SokalSneath` and `Yule`
/// require every coordinate to be 0 or 1. `Hamming` accepts real coordinates.
/// Some functions are undefined for particular inputs, as noted;
/// evaluating them there returns `DomainError`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Bray–Curtis dissimilarity `Σ|uᵢ − vᵢ| / Σ|uᵢ + vᵢ|`.
    BrayCurtis,
    /// Canberra distance `Σ |uᵢ − vᵢ| / (|uᵢ| + |vᵢ|)`, where a term `0/0` counts
    /// as zero.
    Canberra,
    /// Chebyshev distance `maxᵢ |uᵢ − vᵢ|`; zero for vectors of dimension zero.
    Chebyshev,
    /// City-block (Manhattan) distance `Σ|uᵢ − vᵢ|`.
    CityBlock,
    /// Correlation distance: one minus the cosine similarity of `u − ū` and
    /// `v − v̄`, where `ū` and `v̄` are the means. Undefined for constant vectors.
    Correlation,
    /// Cosine distance `1 − u·v / (|u| |v|)`. Undefined for zero vectors.
    Cosine,
    /// Euclidean distance `|u − v|`.
    Euclidean,
    /// Squared Euclidean distance `|u − v|²`.
    SquaredEuclidean,
    /// Minkowski distance `(Σ|uᵢ − vᵢ|ᵖ)^(1/p)` for a finite exponent `p > 0`.
    /// For `p < 1` it is a dissimilarity, not a metric. When the sum would
    /// overflow, the root is computed through logarithms, so a representable
    /// result is never rejected; the error of that path grows with the
    /// exponent of the result.
    Minkowski(f64),
    /// Square root of the Jensen–Shannon divergence, with natural logarithms.
    /// Nonnegative inputs are normalized independently; both totals must be
    /// positive. Normalized masses retain extended exponents through the
    /// square root so tiny representable distances survive normalization.
    JensenShannon,
    /// Dice dissimilarity of boolean vectors. Undefined when both vectors are
    /// all zero.
    Dice,
    /// Hamming distance: the fraction of coordinates that differ. Undefined
    /// for dimension zero.
    Hamming,
    /// Jaccard distance of boolean vectors; zero when both are all zero.
    Jaccard,
    /// Rogers–Tanimoto dissimilarity of boolean vectors.
    RogersTanimoto,
    /// Russell–Rao dissimilarity of boolean vectors.
    RussellRao,
    /// Sokal–Sneath dissimilarity of boolean vectors. Undefined when both
    /// vectors are all zero.
    SokalSneath,
    /// Yule dissimilarity of boolean vectors; zero when no coordinates differ.
    Yule,
}

#[derive(Debug)]
enum Parameters {
    Named(DistanceMetric),
    Standardized(Vec<f64>),
    Mahalanobis(CholeskyDecomposition<f64>),
}

/// A distance function with checked parameters, ready to evaluate.
///
/// Create one from a [`DistanceMetric`] with [`Self::new`], or as a
/// standardized Euclidean or Mahalanobis distance. Creation may allocate;
/// evaluation does not. Arithmetic uses `f64` intermediates for both `f32` and
/// `f64` input. Euclidean-type norms first sum unscaled squares and repeat the
/// sum with rescaling if it overflows or underflows. A difference, transformed
/// component or result that is not representable is an error, not an
/// infinity.
#[derive(Debug)]
pub struct PreparedDistance {
    parameters: Parameters,
}
impl PreparedDistance {
    // Auto planning estimate per pair: one pass over the coordinates, a
    // triangular solve for Mahalanobis, and logarithms for Jensen-Shannon.
    fn pair_work(&self, dimension: usize) -> core::num::NonZeroUsize {
        let d = dimension.max(1);
        crate::batch::work(match &self.parameters {
            Parameters::Mahalanobis(_) => d.saturating_mul(d),
            Parameters::Named(DistanceMetric::JensenShannon) => d.saturating_mul(16),
            _ => d,
        })
    }
    // Point `dimension` must match the prepared parameters.
    pub(crate) fn validate_dimension(&self, dimension: usize) -> Result<(), SciOxideError> {
        let expected = match &self.parameters {
            Parameters::Named(_) => dimension,
            Parameters::Standardized(scales) => scales.len(),
            Parameters::Mahalanobis(factor) => factor.order(),
        };
        if dimension != expected {
            return Err(SciOxideError::dimension_mismatch([expected], [dimension]));
        }
        Ok(())
    }
    /// Prepares one of the [`DistanceMetric`] functions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a Minkowski exponent is not finite and positive.
    pub fn new(metric: DistanceMetric) -> Result<Self, SciOxideError> {
        if let DistanceMetric::Minkowski(p) = metric
            && (!p.is_finite() || p <= 0.0)
        {
            return Err(invalid(
                scioxide_core::argument!("metric"),
                "Minkowski exponent must be positive and finite",
            ));
        }
        Ok(Self {
            parameters: Parameters::Named(metric),
        })
    }
    /// Prepares the standardized Euclidean distance `(Σ (uᵢ − vᵢ)² / σᵢ²)^(1/2)`
    /// with the given variances `σᵢ²`. The variances are not estimated from
    /// data.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a variance is not finite and positive.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn standardized<S: RealScalar>(variances: &[S]) -> Result<Self, SciOxideError> {
        let mut scales = try_filled(
            variances.len(),
            0.0,
            scioxide_core::quantity!("variance_scales"),
        )?;
        for (out, &variance) in scales.iter_mut().zip(variances) {
            let v = variance.into_f64();
            if !v.is_finite() || v <= 0.0 {
                return Err(invalid(
                    scioxide_core::argument!("variances"),
                    "variances must be positive and finite",
                ));
            }
            *out = scioxide_core::RealScalar::sqrt(v);
        }
        Ok(Self {
            parameters: Parameters::Standardized(scales),
        })
    }
    /// Prepares the Mahalanobis distance `((u − v)ᵀ V⁻¹ (u − v))^(1/2)` from the
    /// inverse covariance matrix `V⁻¹`.
    ///
    /// `V⁻¹` must be exactly symmetric and positive definite. It is factored
    /// once as `V⁻¹ = L Lᵀ` by Cholesky, and each distance is `|Lᵀ (u − v)|`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the matrix is not square.
    /// - `DomainError` if it is not positive definite.
    /// - Errors of the Cholesky factorization, for example for a matrix that
    ///   is not symmetric, and `AllocationFailure`.
    pub fn mahalanobis<'a, S: RealScalar>(
        inverse_covariance: impl Into<MatrixView<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        let inverse_covariance = inverse_covariance.into();
        const INVERSE_COVARIANCE: Argument = scioxide_core::argument!("inverse_covariance");
        let n = inverse_covariance.rows();
        if inverse_covariance.cols() != n {
            return Err(invalid(
                INVERSE_COVARIANCE,
                "inverse covariance must be square",
            ));
        }
        let mut values = try_filled(
            n.checked_mul(n).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("covariance"))
            })?,
            0.0,
            scioxide_core::quantity!("covariance"),
        )?;
        for (i, value) in values.iter_mut().enumerate() {
            *value = inverse_covariance.get(i / n, i % n)?.into_f64();
        }
        let matrix = Matrix::from_vec(n, n, values)?;
        // A strictly positive diagonal of L is the definiteness certificate.
        // A zero pivot would make the form only semidefinite, so distinct
        // points could be at distance zero.
        let factor = CholeskyDecomposition::factor(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            CholeskyPivoting::None,
        )?;
        if (0..n).any(|i| factor.l_matrix().as_slice()[i * n + i] <= 0.0) {
            return Err(domain(
                INVERSE_COVARIANCE,
                "inverse covariance must be positive definite",
            ));
        }
        Ok(Self {
            parameters: Parameters::Mahalanobis(factor),
        })
    }
    /// Returns the distance between two vectors.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the vectors differ in length or do not match
    ///   the dimension of a standardized or Mahalanobis distance.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `DomainError` if the distance is undefined for these vectors.
    /// - `NumericalInstability` if the result is not representable.
    pub fn distance<S: RealScalar>(&self, left: &[S], right: &[S]) -> Result<S, SciOxideError> {
        if left.len() != right.len() {
            return Err(SciOxideError::dimension_mismatch(
                [left.len()],
                [right.len()],
            ));
        }
        self.validate_dimension(left.len())?;
        if left.iter().chain(right).any(|value| !value.is_finite()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "coordinates must be finite",
            ));
        }
        let left = Row {
            data: left,
            dimension: left.len(),
            stride: 1,
        };
        let right = Row {
            data: right,
            dimension: right.len(),
            stride: 1,
        };
        dispatch_metric!(self, kernel, kernel(left, right))
    }
    /// Returns the distance between point `i` of `left` and point `j` of
    /// `right`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `i` or `j` is out of range.
    /// - As [`Self::distance`] otherwise.
    pub fn between<S: RealScalar>(
        &self,
        left: &PointSet<'_, S>,
        i: usize,
        right: &PointSet<'_, S>,
        j: usize,
    ) -> Result<S, SciOxideError> {
        if i >= left.len() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("left_point"),
                i,
                left.len(),
            ));
        }
        if j >= right.len() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("right_point"),
                j,
                right.len(),
            ));
        }
        if left.dimension() != right.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                [left.dimension()],
                [right.dimension()],
            ));
        }
        self.validate_dimension(left.dimension())?;
        dispatch_metric!(
            self,
            kernel,
            kernel(left.row::<false>(i), right.row::<false>(j))
        )
    }
}

// Scaled p-norm.
pub(crate) fn norm(
    n: usize,
    p: f64,
    value: impl Fn(usize) -> Result<f64, SciOxideError>,
) -> Result<f64, SciOxideError> {
    if p == 1.0 {
        let mut sum = 0.0;
        for i in 0..n {
            sum += finite(value(i)?)?.abs();
        }
        return finite(sum);
    }
    // Two-pass scaled p-norm: dividing by max |x_i| keeps every |x_i|^p in
    // [0, 1]. For p < 1, tiny ratios use logarithms before exponentiation.
    // The scaled sum
    // lies in [1, n]. For p < 1 its root `sum^(1/p)` can still overflow while
    // the norm is representable; then logarithms are combined before
    // exponentiating, so the result overflows only if the true norm does.
    // `value` is re-evaluated in the second pass rather than buffered, which
    // keeps the kernel allocation-free. The 1-norm needs no powers.
    let mut scale: f64 = 0.0;
    for i in 0..n {
        scale = scale.max(finite(value(i)?)?.abs());
    }
    if scale == 0.0 {
        return Ok(scale);
    }
    let mut sum = 0.0;
    for i in 0..n {
        let magnitude = value(i)?.abs();
        let scaled = magnitude / scale;
        sum += if p == 2.0 {
            scaled * scaled
        } else {
            if p < 1.0 && scaled < f64::MIN_POSITIVE && magnitude != 0.0 {
                RealScalar::exp(p * (RealScalar::ln(magnitude) - RealScalar::ln(scale)))
            } else {
                RealScalar::powf(scaled, p)
            }
        };
    }
    if p == 2.0 {
        return finite(scale * scioxide_core::RealScalar::sqrt(sum));
    }
    let root = RealScalar::powf(sum, 1.0 / p);
    if root.is_finite() {
        return finite(scale * root);
    }
    finite(RealScalar::exp(
        RealScalar::ln(scale) + RealScalar::ln(sum) / p,
    ))
}

// A zero denominator depends on both operands; it is attributed to `left`.
#[inline]
fn ratio(a: f64, b: f64) -> Result<f64, SciOxideError> {
    if b == 0.0 {
        Err(domain(
            scioxide_core::argument!("left"),
            "distance denominator is zero",
        ))
    } else {
        finite(a / b)
    }
}
// Named metric kernel, whose operands are named
// `left` and `right`.
#[inline(always)]
fn named(
    metric: DistanceMetric,
    n: usize,
    pair: impl Fn(usize) -> (f64, f64) + Copy,
) -> Result<f64, SciOxideError> {
    use DistanceMetric::*;
    const LEFT: Argument = scioxide_core::argument!("left");
    const RIGHT: Argument = scioxide_core::argument!("right");
    match metric {
        CityBlock => {
            let mut sum = 0.0;
            for axis in 0..n {
                let (a, b) = pair(axis);
                sum += (a - b).abs();
            }
            finite(sum)
        }
        Chebyshev => {
            let mut maximum: f64 = 0.0;
            for axis in 0..n {
                let (a, b) = pair(axis);
                maximum = maximum.max((a - b).abs());
            }
            finite(maximum)
        }
        Euclidean | SquaredEuclidean | Minkowski(_) => {
            let p = match metric {
                Minkowski(p) => p,
                _ => 2.0,
            };
            let d = if p == 2.0 {
                norm2(n, |i| {
                    let (a, b) = pair(i);
                    a - b
                })?
            } else {
                norm(n, p, |i| {
                    let (a, b) = pair(i);
                    Ok(a - b)
                })?
            };
            if metric == SquaredEuclidean {
                finite(d * d)
            } else {
                Ok(d)
            }
        }
        // Cosine similarity is invariant under positive scaling of either
        // vector. Dividing each by its own max |x| keeps the dot products in
        // range for any finite input. Correlation centers the scaled
        // vectors. Rounding can push |cos| slightly past 1, hence the clamp.
        Cosine | Correlation => {
            let mut sa: f64 = 0.0;
            let mut sb: f64 = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                sa = sa.max(a.abs());
                sb = sb.max(b.abs());
            }
            if sa == 0.0 || sb == 0.0 {
                let argument = if sa == 0.0 { LEFT } else { RIGHT };
                return Err(domain(argument, "zero norm in cosine/correlation"));
            }
            let mut ma = 0.0;
            let mut mb = 0.0;
            if metric == Correlation {
                for i in 0..n {
                    let (a, b) = pair(i);
                    ma += a / sa;
                    mb += b / sb;
                }
                ma /= n as f64;
                mb /= n as f64;
            }
            let mut aa = 0.0;
            let mut bb = 0.0;
            let mut ab = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                let a = a / sa - ma;
                let b = b / sb - mb;
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
            Ok((1.0
                - ratio(
                    ab,
                    scioxide_core::RealScalar::sqrt(aa) * scioxide_core::RealScalar::sqrt(bb),
                )?)
            .clamp(0.0, 2.0))
        }
        // Both are ratios and scale-invariant. Bray-Curtis shares one global
        // scale between its sums. Each Canberra term is its own ratio, so it
        // gets a local scale, which keeps a + b from overflowing term by
        // term. A 0/0 Canberra term contributes zero by definition.
        Canberra => {
            let mut sum = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                let local = a.abs().max(b.abs());
                if local != 0.0 {
                    sum += (a / local - b / local).abs() / (a.abs() / local + b.abs() / local);
                }
            }
            Ok(sum)
        }
        BrayCurtis => {
            let mut scale: f64 = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                scale = scale.max(a.abs()).max(b.abs());
            }
            if scale == 0.0 {
                return Err(domain(LEFT, "Bray-Curtis denominator is zero"));
            }
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                numerator += (a / scale - b / scale).abs();
                denominator += (a / scale + b / scale).abs();
            }
            ratio(numerator, denominator)
        }
        JensenShannon => {
            let mut sa: f64 = 0.0;
            let mut sb: f64 = 0.0;
            let mut raw_a = 0.0;
            let mut raw_b = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                if a < 0.0 || b < 0.0 {
                    let argument = if a < 0.0 { LEFT } else { RIGHT };
                    return Err(domain(argument, "probabilities must be nonnegative"));
                }
                sa = sa.max(a);
                sb = sb.max(b);
                raw_a += a;
                raw_b += b;
            }
            if sa == 0.0 || sb == 0.0 {
                let argument = if sa == 0.0 { LEFT } else { RIGHT };
                return Err(domain(argument, "probability mass must be positive"));
            }
            // Inputs are normalized to probability vectors. The raw sum is
            // used when finite. If it overflows, each vector is first scaled
            // by its maximum, and the sum of the scaled values is used.
            let mut ta = 0.0;
            let mut tb = 0.0;
            for i in 0..n {
                let (a, b) = pair(i);
                ta += a / sa;
                tb += b / sb;
            }
            // With t = |a-b|/(a+b), divergence is (a+b)*h(t)/4, where
            // h(t) = (1+t)ln(1+t) + (1-t)ln(1-t). Sum its coordinate
            // square roots as a norm, retaining extended normalized masses.
            let scaled = scioxide_core::Scaled::from_value;
            let total_a = if raw_a.is_finite() {
                scaled(raw_a)?
            } else {
                scaled(sa)?.mul(scaled(ta)?)?
            };
            let total_b = if raw_b.is_finite() {
                scaled(raw_b)?
            } else {
                scaled(sb)?.mul(scaled(tb)?)?
            };
            norm(n, 2.0, |i| {
                let (a, b) = pair(i);
                let a = scaled(a)?.div(total_a)?;
                let b = scaled(b)?.div(total_b)?;
                let mass = a.add(b)?;
                if mass.is_zero() {
                    return Ok(0.0);
                }
                let t = a.sub(b)?.div(mass)?.round()?.abs();

                // Positive even series avoids cancellation as distributions
                // coincide: h(t) = sum t^(2k)/(k*(2k-1)).
                let h = if t < 0.125 {
                    let square = t * t;
                    let mut power = square;
                    let mut sum = 0.0;
                    for k in 1..=10 {
                        sum += power / f64::from(k * (2 * k - 1));
                        power *= square;
                    }
                    sum
                } else if t == 1.0 {
                    // One side is zero. The limit (1-t) ln(1-t) -> 0 would
                    // otherwise evaluate as 0 * -inf = NaN.
                    2.0 * RealScalar::ln(2.0)
                } else {
                    (1.0 + t) * RealScalar::ln_1p(t) + (1.0 - t) * RealScalar::ln_1p(-t)
                };
                mass.sqrt()?
                    .mul(scaled(0.5 * scioxide_core::RealScalar::sqrt(h))?)?
                    .round()
            })
        }
        Dice => discrete(DiscreteMetric::Dice, n, pair),
        Hamming => discrete(DiscreteMetric::Hamming, n, pair),
        Jaccard => discrete(DiscreteMetric::Jaccard, n, pair),
        RogersTanimoto => discrete(DiscreteMetric::RogersTanimoto, n, pair),
        RussellRao => discrete(DiscreteMetric::RussellRao, n, pair),
        SokalSneath => discrete(DiscreteMetric::SokalSneath, n, pair),
        Yule => discrete(DiscreteMetric::Yule, n, pair),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiscreteMetric {
    Dice,
    Hamming,
    Jaccard,
    RogersTanimoto,
    RussellRao,
    SokalSneath,
    Yule,
}
fn discrete(
    metric: DiscreteMetric,
    n: usize,
    pair: impl Fn(usize) -> (f64, f64),
) -> Result<f64, SciOxideError> {
    use DiscreteMetric::*;
    const LEFT: Argument = scioxide_core::argument!("left");
    const RIGHT: Argument = scioxide_core::argument!("right");
    let (mut tt, mut tf, mut ft, mut ff) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        let (a, b) = pair(i);
        // Hamming accepts any real coordinates and reuses `tt`/`tf`
        // as equal/unequal counters. The other metrics need 0/1
        // values and a full 2x2 contingency table.
        if metric == Hamming {
            if a == b {
                tt += 1;
            } else {
                tf += 1;
            }
            continue;
        }
        if (a != 0.0 && a != 1.0) || (b != 0.0 && b != 1.0) {
            let argument = if a != 0.0 && a != 1.0 { LEFT } else { RIGHT };
            return Err(domain(argument, "Boolean distances require zero or one"));
        }
        match (a != 0.0, b != 0.0) {
            (true, true) => tt += 1,
            (true, false) => tf += 1,
            (false, true) => ft += 1,
            (false, false) => ff += 1,
        }
    }
    let (t, u, v, f) = (tt as f64, tf as f64, ft as f64, ff as f64);
    let mismatch = u + v;
    match metric {
        Dice => ratio(mismatch, 2.0 * t + mismatch),
        Hamming => ratio(mismatch, n as f64),
        Jaccard => {
            if t + mismatch == 0.0 {
                Ok(0.0)
            } else {
                ratio(mismatch, t + mismatch)
            }
        }
        RogersTanimoto => ratio(2.0 * mismatch, t + f + 2.0 * mismatch),
        RussellRao => ratio(n as f64 - t, n as f64),
        SokalSneath => ratio(2.0 * mismatch, t + 2.0 * mismatch),
        Yule => {
            if u * v == 0.0 {
                Ok(0.0)
            } else {
                ratio(2.0 * u * v, t * f + u * v)
            }
        }
    }
}

/// Returns `n (n − 1) / 2`, the length of the condensed distance vector of `n`
/// points.
///
/// # Errors
///
/// - `SizeOverflow` if the length exceeds `usize`.
pub fn condensed_len(n: usize) -> Result<usize, SciOxideError> {
    // n(n-1)/2 with the halving applied to whichever factor is even first.
    // The only overflowing multiplication is then the one whose true result
    // overflows.
    let (a, b) = if n.is_multiple_of(2) {
        (n / 2, n.saturating_sub(1))
    } else {
        (n, (n - 1) / 2)
    };
    a.checked_mul(b).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("condensed_distances"))
    })
}
/// Returns the position of the pair `(i, j)` in the condensed distance vector
/// of `n` points.
///
/// The condensed vector lists the pairs `i < j` row by row:
/// `(0, 1), (0, 2), …, (0, n − 1), (1, 2), …`.
///
/// # Errors
///
/// - `InvalidInput` unless `i < j < n`.
/// - `SizeOverflow` if the position exceeds `usize`.
pub fn condensed_index(n: usize, i: usize, j: usize) -> Result<usize, SciOxideError> {
    if i >= j || j >= n {
        let argument = if i >= j {
            scioxide_core::argument!("i")
        } else {
            scioxide_core::argument!("j")
        };
        return Err(invalid(argument, "condensed index requires i < j < n"));
    }
    // Rows 0..i occupy everything except the trailing triangle of the last
    // n - i points, and row i then starts at column i + 1.
    Ok(condensed_len(n)? - condensed_len(n - i)? + (j - i - 1))
}

pub(crate) const DISTANCE_BLOCK: usize = 256;
const DISTANCE_ROW_TILE: usize = 32;
const DISTANCE_POLL_INTERVAL: usize = 256;

// Budgets count pairs or conversion entries, independently of row lengths.
// A block reserves its work before execution, including scalar replay.
struct DistancePoll<'a> {
    context: scioxide_core::ExecutionContext<'a>,
    remaining: usize,
}

impl<'a> DistancePoll<'a> {
    fn new(context: scioxide_core::ExecutionContext<'a>) -> Self {
        Self {
            context,
            remaining: 0,
        }
    }

    #[inline]
    fn before(&mut self, work: usize) -> Result<(), SciOxideError> {
        debug_assert!(work <= DISTANCE_POLL_INTERVAL);
        if self.remaining < work {
            self.context.check()?;
            self.remaining = DISTANCE_POLL_INTERVAL;
        }
        self.remaining -= work;
        Ok(())
    }
}

fn fill_pairwise<S: RealScalar, const CONTIGUOUS: bool>(
    left: &PointSet<'_, S>,
    right: &PointSet<'_, S>,
    offset: usize,
    staging: &mut [S],
    kernel: impl Fn(Row<'_, S>, &PointSet<'_, S>, usize, &mut [S]) -> Result<(), SciOxideError>,
    poll: &mut DistancePoll<'_>,
) -> Result<(), SciOxideError> {
    if staging.is_empty() {
        return Ok(());
    }
    let columns = right.len();
    let end = offset + staging.len();
    let first_row = offset / columns;
    let last_row = (end - 1) / columns + 1;
    for row_block in (first_row..last_row).step_by(DISTANCE_ROW_TILE) {
        for column_block in (0..columns).step_by(DISTANCE_BLOCK) {
            for row in row_block..last_row.min(row_block.saturating_add(DISTANCE_ROW_TILE)) {
                let base = row * columns;
                let first = column_block.max(offset.saturating_sub(base));
                let last = columns
                    .min(column_block.saturating_add(DISTANCE_BLOCK))
                    .min(end.saturating_sub(base));
                let a = left.row::<CONTIGUOUS>(row);
                if first < last {
                    poll.before(last - first)?;
                    kernel(
                        a,
                        right,
                        first,
                        &mut staging[base + first - offset..base + last - offset],
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn fill_condensed<S: RealScalar, const CONTIGUOUS: bool>(
    points: &PointSet<'_, S>,
    offset: usize,
    staging: &mut [S],
    kernel: impl Fn(Row<'_, S>, &PointSet<'_, S>, usize, &mut [S]) -> Result<(), SciOxideError>,
    poll: &mut DistancePoll<'_>,
) -> Result<(), SciOxideError> {
    if staging.is_empty() {
        return Ok(());
    }
    let n = points.len();
    let (mut row, mut column) = condensed_position(n, condensed_len(n)?, offset)?;
    let mut remaining = staging;
    while !remaining.is_empty() {
        let len = (n - column).min(remaining.len());
        let (values, tail) = remaining.split_at_mut(len);
        let a = points.row::<CONTIGUOUS>(row);
        for (block, values) in values.chunks_mut(DISTANCE_BLOCK).enumerate() {
            poll.before(values.len())?;
            kernel(a, points, column + block * DISTANCE_BLOCK, values)?;
        }
        remaining = tail;
        row += 1;
        column = row + 1;
    }
    Ok(())
}

// Row and column of an in-range condensed index (hence n >= 2).
fn condensed_position(
    n: usize,
    total: usize,
    index: usize,
) -> Result<(usize, usize), SciOxideError> {
    let row = condensed_row(n, total, index)?;
    let base = total - condensed_len(n - row)?;
    Ok((row, row + 1 + index - base))
}
fn condensed_row(n: usize, total: usize, index: usize) -> Result<usize, SciOxideError> {
    let (mut low, mut high) = (0, n - 1);
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if total - condensed_len(n - middle)? <= index {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(low)
}
impl PreparedDistance {
    /// Computes the distances between every point of `left` and every point of
    /// `right` and writes them to `output`.
    ///
    /// `output[i · m + j]` is the distance between `left` point `i` and `right`
    /// point `j`, where `m = right.len()`. `output` needs at least `n m` entries for
    /// `n = left.len()`; later entries are not changed. The workspace must meet a
    /// [`DistanceWorkspaceRequest::pairwise`] request. See the
    /// [crate documentation](crate#execution) for parallel execution and
    /// cancellation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the point sets differ in dimension, or `output`
    ///   or `workspace` is too short.
    /// - `ResourceLimit` if the workspace exceeds its byte limit.
    /// - As [`PreparedDistance::distance`] for each pair.
    /// - `ExecutionStopped` on cancellation or deadline.
    #[allow(clippy::too_many_arguments)]
    pub fn pairwise_into<'scope, 'capability, S: RealScalar>(
        &self,
        left: &PointSet<'_, S>,
        right: &PointSet<'_, S>,
        options: DistanceOptions,
        output: &mut [S],
        workspace: &mut DistanceWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        if left.dimension() != right.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                [left.dimension()],
                [right.dimension()],
            ));
        }
        self.validate_dimension(left.dimension())?;
        let scratch = workspace.prepare(DistanceWorkspaceRequest::pairwise(
            left.len(),
            right.len(),
            options,
        ))?;
        let len = scratch.len();
        require(output.len(), len)?;
        let context = context.into();
        let execution = context.execution();
        dispatch_blocks!(self, left, right, kernel, {
            crate::batch::dispatch_chunks(
                scratch,
                options.execution_policy(),
                execution,
                self.pair_work(left.dimension()),
                |offset, values| {
                    let mut poll = DistancePoll::new(execution);
                    if left.contiguous() && right.contiguous() {
                        fill_pairwise::<S, true>(left, right, offset, values, kernel, &mut poll)
                    } else {
                        fill_pairwise::<S, false>(left, right, offset, values, kernel, &mut poll)
                    }
                },
            )
        })?;
        execution.check()?;
        output[..len].copy_from_slice(scratch);
        Ok(())
    }

    /// Returns the distances between every point of `left` and every point of
    /// `right`, laid out as in [`Self::pairwise_into`].
    ///
    /// Runs on the calling thread and allocates the result and a workspace.
    ///
    /// # Errors
    ///
    /// As [`Self::pairwise_into`], and `AllocationFailure`.
    pub fn pairwise<S: RealScalar>(
        &self,
        left: &PointSet<'_, S>,
        right: &PointSet<'_, S>,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = DistanceOptions::new();
        let request = DistanceWorkspaceRequest::pairwise(left.len(), right.len(), options);
        let requirements = DistanceWorkspace::<S>::workspace_requirements(request)?;
        let mut workspace = DistanceWorkspace::workspace(request)?;
        let mut out = try_filled(
            requirements.scalar_len(),
            S::ZERO,
            scioxide_core::quantity!("pairwise_output"),
        )?;
        self.pairwise_into(
            left,
            right,
            options,
            &mut out,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Matrix::from_vec(left.len(), right.len(), out)
    }

    /// Computes the distances between all pairs of distinct points and writes the
    /// condensed distance vector to `output`.
    ///
    /// The pairs `i < j` appear in the order of [`condensed_index`]. `output` needs
    /// at least [`condensed_len`] entries; later entries are not changed. The
    /// workspace must meet a [`DistanceWorkspaceRequest::condensed`] request.
    ///
    /// # Errors
    ///
    /// As [`Self::pairwise_into`].
    pub fn condensed_into<'scope, 'capability, S: RealScalar>(
        &self,
        points: &PointSet<'_, S>,
        options: DistanceOptions,
        output: &mut [S],
        workspace: &mut DistanceWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.validate_dimension(points.dimension())?;
        let scratch =
            workspace.prepare(DistanceWorkspaceRequest::condensed(points.len(), options))?;
        let len = scratch.len();
        require(output.len(), len)?;
        let context = context.into();
        let execution = context.execution();
        dispatch_blocks!(
            self,
            points,
            points,
            kernel,
            crate::batch::dispatch_chunks(
                scratch,
                options.execution_policy(),
                execution,
                self.pair_work(points.dimension()),
                |offset, values| {
                    let mut poll = DistancePoll::new(execution);
                    if points.contiguous() {
                        fill_condensed::<S, true>(points, offset, values, kernel, &mut poll)
                    } else {
                        fill_condensed::<S, false>(points, offset, values, kernel, &mut poll)
                    }
                }
            )
        )?;
        execution.check()?;
        output[..len].copy_from_slice(scratch);
        Ok(())
    }
}
/// Checks that `values` is a condensed distance vector of `n` points: its
/// length is `n (n − 1) / 2` and its entries are finite and nonnegative.
///
/// The triangle inequality is not checked.
///
/// # Errors
///
/// - `InvalidInput` if the length is wrong.
/// - `DomainError` if an entry is negative or not finite.
/// - `SizeOverflow` if the expected length exceeds `usize`.
pub fn validate_condensed<S: RealScalar>(n: usize, values: &[S]) -> Result<(), SciOxideError> {
    validate_condensed_polled(
        n,
        values,
        &mut DistancePoll::new(scioxide_core::ExecutionContext::sequential_native()),
    )
}

fn validate_condensed_polled<S: RealScalar>(
    n: usize,
    values: &[S],
    poll: &mut DistancePoll<'_>,
) -> Result<(), SciOxideError> {
    const VALUES: Argument = scioxide_core::argument!("values");
    if values.len() != condensed_len(n)? {
        return Err(invalid(VALUES, "condensed length mismatch"));
    }
    for block in values.chunks(DISTANCE_POLL_INTERVAL) {
        poll.before(block.len())?;
        if block.iter().any(|v| !v.is_finite() || *v < S::ZERO) {
            return Err(domain(VALUES, "distances must be finite and nonnegative"));
        }
    }
    Ok(())
}
/// Checks that `matrix` is a distance matrix: square, finite, nonnegative,
/// symmetric and zero on the diagonal, up to the absolute `tolerance`.
///
/// The triangle inequality is not checked.
///
/// # Errors
///
/// - `DimensionMismatch` if the matrix is not square.
/// - `NonFinite` if an entry is NaN or infinite.
/// - `DomainError` if the matrix is not a distance matrix within `tolerance`.
pub fn validate_distance_matrix<'a, S: RealScalar>(
    matrix: impl Into<MatrixView<'a, S>>,
    tolerance: S,
) -> Result<(), SciOxideError> {
    let matrix = matrix.into();
    let tolerance = scioxide_core::Tolerance::new(tolerance, S::ZERO)?.absolute();
    validate_matrix(
        matrix,
        tolerance,
        &mut DistancePoll::new(scioxide_core::ExecutionContext::sequential_native()),
    )
}
fn validate_matrix<S: RealScalar>(
    matrix: MatrixView<'_, S>,
    tolerance: S,
    poll: &mut DistancePoll<'_>,
) -> Result<(), SciOxideError> {
    let n = matrix.rows();
    if matrix.cols() != n {
        return Err(SciOxideError::dimension_mismatch(
            [n, n],
            [n, matrix.cols()],
        ));
    }
    for i in 0..n {
        for first in (0..n).step_by(DISTANCE_POLL_INTERVAL) {
            let end = first.saturating_add(DISTANCE_POLL_INTERVAL).min(n);
            poll.before(end - first)?;
            for j in first..end {
                if !matrix.get(i, j)?.is_finite() {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Input,
                        "distance matrix must be finite",
                    ));
                }
            }
        }
    }
    for i in 0..n {
        for first in (0..n).step_by(DISTANCE_POLL_INTERVAL) {
            let end = first.saturating_add(DISTANCE_POLL_INTERVAL).min(n);
            poll.before(end - first)?;
            for j in first..end {
                let v = matrix.get(i, j)?;
                if v < S::ZERO
                    || (i == j && v > tolerance)
                    || !scioxide_core::backend::absolute_difference_le(
                        v,
                        matrix.get(j, i)?,
                        tolerance,
                    )
                {
                    return Err(domain(
                        scioxide_core::argument!("matrix"),
                        "distance matrix must be nonnegative, symmetric, and zero-diagonal",
                    ));
                }
            }
        }
    }
    Ok(())
}
/// Expands a condensed distance vector of `n` points into the full `n × n`
/// matrix, row-major, and writes it to `output`.
///
/// The workspace must meet a [`DistanceWorkspaceRequest::square`] request.
///
/// # Errors
///
/// - As [`validate_condensed`].
/// - `DimensionMismatch` if `output` or `workspace` is too short.
/// - `ResourceLimit` if the workspace exceeds its byte limit.
/// - `ExecutionStopped` on cancellation or deadline.
pub fn condensed_to_square_into<'scope, 'capability, S: RealScalar>(
    n: usize,
    values: &[S],
    options: DistanceOptions,
    output: &mut [S],
    workspace: &mut DistanceWorkspace<S>,
    context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    let context = context.into();
    let execution = context.execution();
    execution.check()?;
    // Size the destination before the O(n^2) validation scan.
    let scratch = workspace.prepare(DistanceWorkspaceRequest::square(n, options))?;
    require(output.len(), scratch.len())?;
    validate_condensed_polled(n, values, &mut DistancePoll::new(execution))?;
    crate::batch::dispatch_chunks(
        scratch,
        options.execution_policy(),
        execution,
        crate::batch::work(1),
        |offset, output| fill_square(n, values, offset, output, &mut DistancePoll::new(execution)),
    )?;
    execution.check()?;
    output[..scratch.len()].copy_from_slice(scratch);
    Ok(())
}
/// Checks a distance matrix as [`validate_distance_matrix`] does and writes
/// its upper triangle to `output` as a condensed distance vector.
///
/// The workspace must meet a [`DistanceWorkspaceRequest::condensed`] request.
///
/// # Errors
///
/// - As [`validate_distance_matrix`].
/// - `DimensionMismatch` if `output` or `workspace` is too short.
/// - `ResourceLimit` if the workspace exceeds its byte limit.
/// - `ExecutionStopped` on cancellation or deadline.
#[allow(clippy::too_many_arguments)]
pub fn square_to_condensed_into<'a, 'scope, 'capability, S: RealScalar>(
    matrix: impl Into<MatrixView<'a, S>>,
    tolerance: S,
    options: DistanceOptions,
    output: &mut [S],
    workspace: &mut DistanceWorkspace<S>,
    context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    let matrix = matrix.into();
    let tolerance = scioxide_core::Tolerance::new(tolerance, S::ZERO)?.absolute();
    let context = context.into();
    let execution = context.execution();
    execution.check()?;
    // Size the destination before the O(n^2) validation scan; squareness is
    // checked first so the condensed length is meaningful.
    if matrix.rows() != matrix.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [matrix.rows(), matrix.rows()],
            [matrix.rows(), matrix.cols()],
        ));
    }
    let scratch = workspace.prepare(DistanceWorkspaceRequest::condensed(matrix.rows(), options))?;
    require(output.len(), scratch.len())?;
    validate_matrix(matrix, tolerance, &mut DistancePoll::new(execution))?;
    let total = scratch.len();
    crate::batch::dispatch_chunks(
        scratch,
        options.execution_policy(),
        execution,
        crate::batch::work(1),
        |offset, values| {
            fill_upper(
                matrix,
                total,
                offset,
                values,
                &mut DistancePoll::new(execution),
            )
        },
    )?;
    execution.check()?;
    output[..scratch.len()].copy_from_slice(scratch);
    Ok(())
}

fn fill_square<S: RealScalar>(
    n: usize,
    values: &[S],
    offset: usize,
    output: &mut [S],
    poll: &mut DistancePoll<'_>,
) -> Result<(), SciOxideError> {
    for (block, output) in output.chunks_mut(DISTANCE_POLL_INTERVAL).enumerate() {
        poll.before(output.len())?;
        let first = offset + block * DISTANCE_POLL_INTERVAL;
        for (index, value) in (first..).zip(output) {
            let (i, j) = (index / n, index % n);
            *value = if i == j {
                S::ZERO
            } else {
                values[condensed_index(n, i.min(j), i.max(j))?]
            };
        }
    }
    Ok(())
}

fn fill_upper<S: RealScalar>(
    matrix: MatrixView<'_, S>,
    total: usize,
    offset: usize,
    values: &mut [S],
    poll: &mut DistancePoll<'_>,
) -> Result<(), SciOxideError> {
    if values.is_empty() {
        return Ok(());
    }
    let n = matrix.rows();
    let (mut row, mut column) = condensed_position(n, total, offset)?;
    for block in values.chunks_mut(DISTANCE_POLL_INTERVAL) {
        poll.before(block.len())?;
        for value in block {
            *value = matrix.get(row, column)?;
            column += 1;
            if column == n {
                row += 1;
                column = row + 1;
            }
        }
    }
    Ok(())
}
/// Result of [`directed_hausdorff`]: the distance and a pair of points that
/// attains it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HausdorffResult<S: RealScalar> {
    /// The directed Hausdorff distance.
    pub distance: S,
    /// The lowest-index source point whose nearest target is farthest.
    pub source: usize,
    /// The lowest-index target point nearest to `source`.
    pub target: usize,
}
/// Computes the directed Hausdorff distance from `source` to `target`.
///
/// ```text
/// h(A, B) = max_{a ∈ A} min_{b ∈ B} |a − b|
/// ```
///
/// with the Euclidean norm. Both sets are visited in a random order with a
/// fixed seed, and the inner loop stops as soon as a distance falls below the
/// current maximum (Taha and Hanbury). The worst-case cost is `O(m n d)` for
/// `m` source and `n` target points of dimension `d`, and the expected cost
/// is much lower. The result, including the witness pair, does not depend on
/// the random order. Because of the early stop, a pair that would overflow is
/// reported only if it is evaluated.
///
/// # Errors
///
/// - `DomainError` if either set is empty.
/// - `DimensionMismatch` if the sets differ in dimension.
/// - `NumericalInstability` if a distance is not representable.
/// - `AllocationFailure` if the `O(m + n)` index storage cannot be reserved.
///
/// ```
/// use scioxide_spatial::{PointSet, directed_hausdorff};
///
/// let (a, b) = ([0.0_f64, 0.0, 1.0, 0.0], [0.0_f64, 3.0]);
/// let source = PointSet::from_rows(&a, 2)?;
/// let target = PointSet::from_rows(&b, 2)?;
/// let result = directed_hausdorff(&source, &target)?;
/// // The farthest source point is (1, 0), at distance √10 from (0, 3).
/// assert!((result.distance - 10.0_f64.sqrt()).abs() < 1e-15);
/// assert_eq!((result.source, result.target), (1, 0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. A. Taha and A. Hanbury, "An efficient algorithm for calculating the
/// exact Hausdorff distance", *IEEE Trans. Pattern Anal. Mach. Intell.*
/// 37(11), 2015.
pub fn directed_hausdorff<S: RealScalar>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
) -> Result<HausdorffResult<S>, SciOxideError> {
    if source.is_empty() || target.is_empty() {
        let argument = if source.is_empty() {
            scioxide_core::argument!("source")
        } else {
            scioxide_core::argument!("target")
        };
        return Err(domain(argument, "Hausdorff requires nonempty sets"));
    }
    if source.dimension() != target.dimension() {
        return Err(SciOxideError::dimension_mismatch(
            [source.dimension()],
            [target.dimension()],
        ));
    }
    let sources = shuffled_indices(source.len(), 0x243f_6a88_85a3_08d3)?;
    let targets = shuffled_indices(target.len(), 0x1319_8a2e_0370_7344)?;
    if source.contiguous() && target.contiguous() {
        blocks::hausdorff::<S, true>(source, target, &sources, &targets)
    } else {
        blocks::hausdorff::<S, false>(source, target, &sources, &targets)
    }
}

// Private fixed-seed xorshift64* stream for Fisher–Yates traversal order only.
// Modulo bias is immaterial: the algorithm requires a deterministic permutation,
// not statistically uniform random samples. Every index occurs exactly once.
fn shuffled_indices(n: usize, mut state: u64) -> Result<Vec<usize>, SciOxideError> {
    let mut indices = try_filled(n, 0, scioxide_core::quantity!("hausdorff_indices"))?;
    for (index, value) in indices.iter_mut().enumerate() {
        *value = index;
    }
    for index in (1..n).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let word = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let other = (u128::from(word) % (index as u128 + 1)) as usize;
        indices.swap(index, other);
    }
    Ok(indices)
}

/// Computes the Euclidean distance between two finite coordinate vectors.
/// Uses [`PreparedDistance::distance`] with the Euclidean metric, in linear time without allocation.
///
/// # Errors
/// As [`PreparedDistance::distance`].
pub fn distance<S: RealScalar>(left: &[S], right: &[S]) -> Result<S, SciOxideError> {
    PreparedDistance::new(DistanceMetric::Euclidean)?.distance(left, right)
}
/// Computes the Euclidean distance matrix, with entry `(i, j)` between rows `i` and `j`.
/// Allocates the row-major matrix and equally sized staging storage; cost is `O(n m d)`
/// for `n` left points, `m` right points and dimension `d`.
///
/// # Errors
/// As [`PreparedDistance::pairwise`].
///
/// ```
/// let points = scioxide_spatial::PointSet::from_points(&[[0.0_f64, 0.0], [3.0, 4.0]])?;
/// let matrix = scioxide_spatial::pairwise(&points, &points)?;
/// assert!((matrix.view().get(0, 1)? - 5.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn pairwise<S: RealScalar>(
    left: &PointSet<'_, S>,
    right: &PointSet<'_, S>,
) -> Result<Matrix<S>, SciOxideError> {
    PreparedDistance::new(DistanceMetric::Euclidean)?.pairwise(left, right)
}
/// Computes Euclidean distances for pairs `i < j` in condensed order.
/// Uses [`PreparedDistance::condensed`] with its cost, storage and accuracy.
///
/// # Errors
/// As [`PreparedDistance::condensed`].
pub fn condensed<S: RealScalar>(points: &PointSet<'_, S>) -> Result<Vec<S>, SciOxideError> {
    PreparedDistance::new(DistanceMetric::Euclidean)?.condensed(points)
}
impl PreparedDistance {
    /// Computes distances for pairs `i < j` in [`condensed_index`] order.
    /// Uses default options and sequential native execution. Allocates output and staging
    /// of `n (n - 1) / 2` scalars; cost and accuracy match [`Self::condensed_into`].
    ///
    /// # Errors
    /// As [`Self::condensed_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn condensed<S: RealScalar>(
        &self,
        points: &PointSet<'_, S>,
    ) -> Result<Vec<S>, SciOxideError> {
        let options = DistanceOptions::new();
        let request = DistanceWorkspaceRequest::condensed(points.len(), options);
        let requirements = DistanceWorkspace::<S>::workspace_requirements(request)?;
        let mut workspace = DistanceWorkspace::workspace(request)?;
        let mut output = try_filled(
            requirements.scalar_len(),
            S::ZERO,
            scioxide_core::quantity!("condensed_output"),
        )?;
        self.condensed_into(
            points,
            options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}
/// Expands condensed distances into a symmetric row-major `n × n` matrix.
/// Uses default options and [`condensed_to_square_into`], allocating `2 n²` scalars in `O(n²)` time.
///
/// # Errors
/// As [`condensed_to_square_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
pub fn condensed_to_square<S: RealScalar>(
    n: usize,
    values: &[S],
) -> Result<Matrix<S>, SciOxideError> {
    let options = DistanceOptions::new();
    let request = DistanceWorkspaceRequest::square(n, options);
    let requirements = DistanceWorkspace::<S>::workspace_requirements(request)?;
    let mut workspace = DistanceWorkspace::workspace(request)?;
    let mut output = try_filled(
        requirements.scalar_len(),
        S::ZERO,
        scioxide_core::quantity!("square_output"),
    )?;
    condensed_to_square_into(
        n,
        values,
        options,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Matrix::from_vec(n, n, output)
}
/// Extracts the upper triangle of a distance matrix in condensed order.
/// Uses default options and [`square_to_condensed_into`], allocating output and staging in `O(n²)` time.
///
/// # Errors
/// Returns `DimensionMismatch` for a nonsquare matrix before allocating; otherwise
/// as [`square_to_condensed_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
pub fn square_to_condensed<'a, S: RealScalar>(
    matrix: impl Into<MatrixView<'a, S>>,
    tolerance: S,
) -> Result<Vec<S>, SciOxideError> {
    let matrix = matrix.into();
    if matrix.rows() != matrix.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [matrix.rows(), matrix.rows()],
            [matrix.rows(), matrix.cols()],
        ));
    }
    let options = DistanceOptions::new();
    let request = DistanceWorkspaceRequest::condensed(matrix.rows(), options);
    let requirements = DistanceWorkspace::<S>::workspace_requirements(request)?;
    let mut workspace = DistanceWorkspace::workspace(request)?;
    let mut output = try_filled(
        requirements.scalar_len(),
        S::ZERO,
        scioxide_core::quantity!("condensed_output"),
    )?;
    square_to_condensed_into(
        matrix,
        tolerance,
        options,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
