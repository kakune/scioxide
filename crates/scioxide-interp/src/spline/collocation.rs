//! Interpolation of tensor-grid samples by a tensor B-spline, solved axis by axis.

#![allow(
    clippy::needless_range_loop,
    reason = "axis loops index several parallel per-axis arrays"
)]

use crate::NodeSeparation;
use crate::validate::{validate_finite, validate_length, validate_nodes};
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, ExecutionContext, RealScalar, SciOxideError,
};
use scioxide_linalg::{
    BandedLuDecomposition, BandedMatrix, DirectSolveOptions, DirectSolveTermination,
    LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, RankPolicy, StructuredOperation,
    StructuredWorkspaceRequest, VectorView, VectorViewMut,
};

use super::basis::{BSplineBasis, KnotSide, local_basis};
use super::bspline::{BSpline, control_count};
use crate::values::zeroed;

/// Describes the scratch that [`TensorSplineInterpolationPlan::fit_into`] needs
/// for values of a given number of real components.
///
/// Get it from [`TensorSplineInterpolationPlan::workspace_requirements`] and
/// pass it to [`TensorSplineInterpolationWorkspace::check`] or
/// [`TensorSplineInterpolationWorkspace::try_reserve`]; see
/// [`TensorSplineInterpolationWorkspace`] for an example.
#[derive(Debug, Clone)]
#[must_use]
pub struct TensorSplineInterpolationRequirements {
    dimension: usize,
    length: usize,
    linalg: LinalgRequirements,
}

impl TensorSplineInterpolationRequirements {
    /// Returns the number of real components per value.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Holds reusable scratch for [`TensorSplineInterpolationPlan::fit_into`].
///
/// Create it with [`TensorSplineInterpolationPlan::workspace`]. It holds two
/// buffers of the fitted coefficient length and the banded LU solve scratch.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext};
/// use scioxide_interp::{BSplineBasis, TensorSplineInterpolationPlan};
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let plan = TensorSplineInterpolationPlan::new([basis], [vec![0.0, 1.0]])?;
/// let requirements = plan.workspace_requirements(2)?;
/// let mut workspace = plan.workspace(2)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// let mut coefficients = [0.0; 4];
/// plan.fit_into(
///     &[[1.0, 2.0], [3.0, 4.0]],
///     &mut coefficients,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(coefficients, [1.0, 2.0, 3.0, 4.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct TensorSplineInterpolationWorkspace<S: RealScalar> {
    data: Vec<S>,
    stage: Vec<S>,
    linalg: LinalgWorkspace<'static, S>,
}

impl<S: RealScalar> TensorSplineInterpolationWorkspace<S> {
    /// Returns whether this workspace covers `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `ResourceLimit` if the banded LU scratch size cannot
    ///   be computed.
    pub fn check(
        &self,
        requirements: &TensorSplineInterpolationRequirements,
    ) -> Result<Capacity, SciOxideError> {
        let linalg = self.linalg.check(&requirements.linalg)?;
        Ok(
            if linalg == Capacity::Sufficient
                && self.data.len() >= requirements.length
                && self.stage.len() >= requirements.length
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows the workspace in place to cover `requirements`.
    ///
    /// On failure the existing capacity is kept.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `ResourceLimit` if the size cannot be computed.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &TensorSplineInterpolationRequirements,
    ) -> Result<(), SciOxideError> {
        for buffer in [&mut self.data, &mut self.stage] {
            if buffer.len() < requirements.length {
                buffer
                    .try_reserve(requirements.length - buffer.len())
                    .map_err(|_| {
                        SciOxideError::allocation_failure(
                            scioxide_core::quantity!("tensor_interpolation_scratch"),
                            requirements.length,
                        )
                    })?;
                buffer.resize(requirements.length, S::ZERO);
            }
        }
        self.linalg.try_reserve(&requirements.linalg)
    }
}

/// Interpolates samples on a tensor grid by a tensor-product B-spline.
///
/// Given a [`BSplineBasis`] and `n_a` nodes `x_{a,i}` on each axis `a`, where
/// `n_a` is the number of basis functions, the plan finds the coefficients of
/// the [`BSpline`] `f` with `f(x_{1,i_1}, …, x_{D,i_D}) = y_{i_1…i_D}` at every
/// grid point. The coefficient tensor solves `(A_1 ⊗ ⋯ ⊗ A_D) c = y` with the
/// axis collocation matrices `A_a[i][j] = N_j(x_{a,i})`.
///
/// Preparation assembles each `A_a` from local basis rows in its band and
/// factors it once by banded LU, at a cost of `O(Σ_a n_a (p_a + 1)²)` and
/// `O(Σ_a n_a (p_a + 1))` storage. A fit solves with one axis factor at a time
/// along every grid line and never forms the Kronecker product. Axis nodes may
/// be increasing or decreasing; samples stay in the order you give. The
/// collocation matrix is nonsingular exactly when the nodes satisfy the
/// Schoenberg–Whitney condition `N_i(x_{a,i}) > 0` for every `i`, which the
/// plan checks.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     BSplineBasis, ExtrapolationPolicy, Interpolant, TensorSplineInterpolationPlan,
/// };
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let plan = TensorSplineInterpolationPlan::new(
///     [basis.clone(), basis],
///     [vec![1.0, 0.0], vec![0.0, 1.0]],
/// )?;
/// // f(x, y) = x + 2y in the caller's order x = [1, 0], y = [0, 1].
/// let spline = plan.fit(&[1.0, 3.0, 0.0, 2.0])?;
/// assert!((spline.evaluate([0.25, 0.75], ExtrapolationPolicy::Reject)? - 1.75).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
/// chapters XIII and XVII.
#[derive(Debug, Clone)]
#[must_use]
pub struct TensorSplineInterpolationPlan<S: RealScalar, const D: usize> {
    bases: [BSplineBasis<S>; D],
    nodes: [Vec<S>; D],
    descending: [bool; D],
    factors: Vec<BandedLuDecomposition<S>>,
    controls: usize,
}

impl<S: RealScalar, const D: usize> TensorSplineInterpolationPlan<S, D> {
    /// Returns an upper bound on the bytes that [`Self::new`] allocates for these
    /// bases, without allocating.
    ///
    /// The bound covers the bases, nodes, LU factors and their bands, pivot
    /// indices, assembly rows and factorization scratch, which is
    /// `O(Σ_a n_a (p_a + 1))` for `n_a` controls and degree `p_a` per axis. It
    /// excludes allocator overhead and unused capacity of the input vectors.
    /// The nodes themselves are checked by [`Self::new`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D = 0`.
    /// - `SizeOverflow` or `ResourceLimit` if the size cannot be computed.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSplineBasis, TensorSplineInterpolationPlan};
    /// let bases = [BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?];
    /// assert!(TensorSplineInterpolationPlan::preparation_bytes(&bases)? > 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn preparation_bytes(bases: &[BSplineBasis<S>; D]) -> Result<usize, SciOxideError> {
        control_count(bases)?;
        let q = scioxide_core::quantity!("spline_preparation_bytes");
        let mut bytes = checked_mul(D, core::mem::size_of::<BandedLuDecomposition<S>>(), q)?;
        for basis in bases {
            let n = basis.control_count();
            let p = basis.degree();
            let band = checked_add(checked_mul(2, p, q)?, 1, q)?;
            let factor = checked_add(checked_mul(3, p, q)?, 1, q)?;
            // Assembly and original diagnostic bands, compact LU factors,
            // transferred nodes, basis storage and the reusable local row.
            let scalars = [
                super::preparation::basis_scalars(n, p)?,
                checked_mul(n, checked_add(checked_mul(2, band, q)?, factor, q)?, q)?,
                n,
                checked_add(p, 1, q)?,
            ]
            .into_iter()
            .try_fold(0, |total, count| checked_add(total, count, q))?;
            let scratch = BandedLuDecomposition::<S>::workspace_requirements(
                &StructuredWorkspaceRequest::factor(n, p, p, &LinalgExecutionOptions::new()),
            )?
            .required_bytes::<S>()?;
            bytes = checked_add(
                bytes,
                checked_mul(scalars, core::mem::size_of::<S>(), q)?,
                q,
            )?;
            bytes = checked_add(bytes, checked_mul(n, core::mem::size_of::<usize>(), q)?, q)?;
            bytes = checked_add(bytes, scratch, q)?;
        }
        Ok(bytes)
    }

    /// Checks the axis nodes and factors the collocation matrix of each axis.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D = 0`, or the nodes of an axis are not strictly
    ///   monotone.
    /// - `DimensionMismatch` if an axis has a node count other than its number
    ///   of basis functions.
    /// - `NonFinite` if a node is NaN or infinite.
    /// - `DomainError` (`OutsideDomain`) if a node is outside the basis domain.
    /// - `Singular` if the nodes violate the Schoenberg–Whitney condition, or
    ///   the LU factorization breaks down.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(bases: [BSplineBasis<S>; D], mut nodes: [Vec<S>; D]) -> Result<Self, SciOxideError> {
        let controls = control_count(&bases)?;
        let mut descending = [false; D];
        let mut factors = try_with_capacity(D, scioxide_core::quantity!("factors"))?;
        for axis in 0..D {
            let basis = &bases[axis];
            validate_length(basis.control_count(), nodes[axis].len())?;
            validate_finite(&nodes[axis])?;
            if !nodes[axis]
                .iter()
                .all(|&node| basis.domain().contains(node))
            {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("nodes"),
                    scioxide_core::DomainReason::OutsideDomain,
                    "tensor grid nodes must lie in the basis domain",
                ));
            }
            // Descending axes are reversed, then share the increasing-node owner.
            descending[axis] = nodes[axis].windows(2).next().is_some_and(|p| p[0] > p[1]);
            if descending[axis] {
                nodes[axis].reverse();
            }
            validate_nodes(
                &nodes[axis],
                1,
                NodeSeparation::Exact,
                scioxide_core::argument!("nodes"),
            )?;
            factors.push(collocation_factor(basis, &nodes[axis])?);
        }
        Ok(Self {
            bases,
            nodes,
            descending,
            factors,
            controls,
        })
    }

    /// Returns the axis bases.
    pub const fn bases(&self) -> &[BSplineBasis<S>; D] {
        &self.bases
    }

    /// Returns the nodes of each axis in increasing order.
    pub const fn nodes(&self) -> &[Vec<S>; D] {
        &self.nodes
    }

    /// Returns, for each axis, whether its nodes were given in decreasing order.
    pub const fn descending_axes(&self) -> [bool; D] {
        self.descending
    }

    /// Returns the workspace requirements for values of `dimension` real components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `dimension` is zero.
    /// - `SizeOverflow` or `ResourceLimit` if the size cannot be computed.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<TensorSplineInterpolationRequirements, SciOxideError> {
        if dimension == 0 {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("dimension"),
                "tensor interpolation values need at least one real component",
            ));
        }
        let longest = self
            .bases
            .iter()
            .map(BSplineBasis::control_count)
            .max()
            .unwrap_or(0);
        Ok(TensorSplineInterpolationRequirements {
            dimension,
            length: checked_mul(
                self.controls,
                dimension,
                scioxide_core::quantity!("samples"),
            )?,
            linalg: BandedLuDecomposition::<S>::workspace_requirements(
                &StructuredWorkspaceRequest::solve(longest, &LinalgExecutionOptions::new()),
            )?,
        })
    }

    /// Allocates a workspace for values of `dimension` real components.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<TensorSplineInterpolationWorkspace<S>, SciOxideError> {
        let mut workspace = TensorSplineInterpolationWorkspace {
            data: Vec::new(),
            stage: Vec::new(),
            linalg: LinalgWorkspace::new(),
        };
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Returns the interpolating spline of grid samples in row-major order.
    ///
    /// The last axis varies fastest, and each axis runs in the direction its
    /// nodes were given. This runs [`Self::fit_into`] on the calling thread.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::fit_into`]; an empty `values` is a
    ///   `DimensionMismatch`.
    /// - `AllocationFailure` if the workspace or spline cannot be allocated.
    pub fn fit<V: AffineValue<Scale = S>>(
        &self,
        values: &[V],
    ) -> Result<BSpline<S, V::Owned, D>, SciOxideError> {
        let template = values
            .first()
            .ok_or_else(|| crate::error::shape(self.controls, 0))?;
        let dimension = template.dimension()?;
        let mut workspace = self.workspace(dimension)?;
        let mut coefficients = try_filled(
            checked_mul(
                self.controls,
                dimension,
                scioxide_core::quantity!("coefficients"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("coefficients"),
        )?;
        self.fit_into(
            values,
            coefficients.as_mut_slice(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        let mut bases = try_with_capacity(D, scioxide_core::quantity!("bases"))?;
        for basis in &self.bases {
            bases.push(basis.try_clone()?);
        }
        let bases = <[BSplineBasis<S>; D]>::try_from(bases)
            .map_err(|bases: Vec<_>| crate::error::shape(D, bases.len()))?;
        BSpline::from_parts(bases, coefficients, dimension, zeroed(template)?)
    }

    /// Writes the coefficients of the interpolating spline of row-major grid
    /// samples to `out`.
    ///
    /// The coefficients are in row-major multi-index order, as in
    /// [`BSpline::coefficients`], with `controls · m` real components for `m`
    /// components per value.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of samples differs from the grid
    ///   size, the samples differ in shape, `out` has the wrong shape or
    ///   `workspace` is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `Singular` if a banded solve fails, and any error of the banded solve
    ///   under `cx`.
    pub fn fit_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        values: &[V],
        out: &mut O,
        workspace: &mut TensorSplineInterpolationWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if values.len() != self.controls {
            return Err(crate::error::shape(self.controls, values.len()));
        }
        let dimension = values[0].dimension()?;
        for value in values {
            scioxide_core::check_value_shape(value, dimension)?;
            scioxide_core::validate_value(value)?;
        }
        let requirements = self.workspace_requirements(dimension)?;
        scioxide_core::check_value_shape(out, requirements.length)?;
        if workspace.check(&requirements)? != Capacity::Sufficient {
            return Err(crate::error::shape(
                requirements.length,
                workspace.data.len().min(workspace.stage.len()),
            ));
        }
        let length = requirements.length;
        let extents: [usize; D] = core::array::from_fn(|axis| self.bases[axis].control_count());
        // Gather caller samples into increasing basis order.
        for linear in 0..self.controls {
            let mut source = 0usize;
            let mut remainder = linear;
            let mut stride = self.controls;
            for axis in 0..D {
                stride /= extents[axis];
                let coordinate = remainder / stride;
                remainder %= stride;
                let coordinate = if self.descending[axis] {
                    extents[axis] - 1 - coordinate
                } else {
                    coordinate
                };
                source += coordinate * stride;
            }
            for c in 0..dimension {
                workspace.data[linear * dimension + c] = values[source].component(c)?;
            }
        }
        // The Kronecker inverse applies axis by axis: solve every lane (fixed
        // indices on the other axes, stride `inner`) with that axis's factor,
        // ping-ponging between `data` and `stage`.
        for axis in 0..D {
            let n = extents[axis];
            let inner = extents[axis + 1..].iter().product::<usize>() * dimension;
            let outer = length / (n * inner);
            workspace.stage[..length].fill(S::ZERO);
            for block in 0..outer {
                for offset in 0..inner {
                    let origin = block * n * inner + offset;
                    let report = self.factors[axis].solve_into(
                        StructuredOperation::Normal,
                        VectorView::from_slice_with_stride(&workspace.data[origin..], n, inner)?,
                        &DirectSolveOptions::new(RankPolicy::Exact),
                        VectorViewMut::from_slice_with_stride(
                            &mut workspace.stage[origin..],
                            n,
                            inner,
                        )?,
                        &mut workspace.linalg,
                        cx,
                    )?;
                    if report.termination() != DirectSolveTermination::Accepted {
                        return Err(SciOxideError::singular(None));
                    }
                }
            }
            core::mem::swap(&mut workspace.data, &mut workspace.stage);
        }
        scioxide_core::write_value(&workspace.data[..length], out)
    }
}

/// Band-LU factor of one axis collocation matrix assembled from local rows.
fn collocation_factor<S: RealScalar>(
    basis: &BSplineBasis<S>,
    nodes: &[S],
) -> Result<BandedLuDecomposition<S>, SciOxideError> {
    let n = basis.control_count();
    let degree = basis.degree();
    let mut row = try_filled(degree + 1, S::ZERO, scioxide_core::quantity!("basis_row"))?;
    // Measure the band from the nonzero pattern, then fill compact band
    // storage `band[column * width + upper + row - column]`.
    let mut lower = 0usize;
    let mut upper = 0usize;
    collocation_rows(basis, nodes, &mut row, |r, first, row| {
        // A square ordered B-spline collocation is nonsingular only when every
        // diagonal basis value is positive (Schoenberg–Whitney, with clamped
        // endpoint limits). Reject before a singular pattern can widen the band.
        if r < first || r - first > degree || row[r - first] <= S::ZERO {
            return Err(SciOxideError::singular(None));
        }
        for (offset, &value) in row.iter().enumerate() {
            let column = first + offset;
            if value != S::ZERO {
                if r >= column {
                    lower = lower.max(r - column);
                } else {
                    upper = upper.max(column - r);
                }
            }
        }
        Ok(())
    })?;
    let width = lower + upper + 1;
    let mut band = try_filled(
        checked_mul(n, width, scioxide_core::quantity!("collocation_band"))?,
        S::ZERO,
        scioxide_core::quantity!("collocation_band"),
    )?;
    collocation_rows(basis, nodes, &mut row, |r, first, row| {
        for (offset, &value) in row.iter().enumerate() {
            let column = first + offset;
            if value != S::ZERO {
                band[column * width + upper + r - column] = value;
            }
        }
        Ok(())
    })?;
    let matrix = BandedMatrix::new(n, n, lower, upper, width, band)?;
    BandedLuDecomposition::factor(&matrix)
}

/// Sweep validated increasing nodes once. Endpoint span limits come from the
/// basis kernel; interior repeated knots select the right-hand piece.
fn collocation_rows<S: RealScalar>(
    basis: &BSplineBasis<S>,
    nodes: &[S],
    row: &mut [S],
    mut visit: impl FnMut(usize, usize, &[S]) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let mut span = basis.span(basis.domain().lower(), KnotSide::Right);
    let last = basis.span(basis.domain().upper(), KnotSide::Right);
    for (r, &node) in nodes.iter().enumerate() {
        while span < last && basis.knots()[span + 1] <= node {
            span += 1;
        }
        local_basis(basis.knots(), basis.degree(), span, node, 0, row)?;
        visit(r, span - basis.degree(), row)?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{ExtrapolationPolicy, Interpolant};

    #[test]
    fn cubic_interpolation_reproduces_samples() {
        let basis = BSplineBasis::new(vec![0., 0., 0., 0., 1., 2., 3., 3., 3., 3.], 3).unwrap();
        let nodes = vec![0., 0.5, 1.2, 1.8, 2.5, 3.];
        let plan = TensorSplineInterpolationPlan::new([basis], [nodes.clone()]).unwrap();
        let values: Vec<[f64; 2]> = nodes.iter().map(|&x| [x * x, 1. - x]).collect();
        let spline = plan.fit(&values).unwrap();
        for (&x, value) in nodes.iter().zip(&values) {
            let actual = spline.evaluate([x], ExtrapolationPolicy::Reject).unwrap();
            assert!((actual[0] - value[0]).abs() < 1e-14);
            assert!((actual[1] - value[1]).abs() < 1e-14);
        }
    }

    #[test]
    fn rejects_bad_axes_and_small_workspaces() {
        let basis = BSplineBasis::new(vec![0., 0., 1., 1.], 1).unwrap();
        assert!(TensorSplineInterpolationPlan::new([basis.clone()], [vec![0., 0.]]).is_err());
        assert!(TensorSplineInterpolationPlan::new([basis.clone()], [vec![0.]]).is_err());
        let plan = TensorSplineInterpolationPlan::new([basis], [vec![0., 1.]]).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut out = [5.; 4];
        assert_eq!(
            plan.fit_into(
                &[[1., 2.], [3., 4.]],
                &mut out,
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .unwrap_err()
            .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(out, [5.; 4]);
    }
}
