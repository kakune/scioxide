//! Storage of interpolated values and the query workspace.
//!
//! Interpolants store values node by node, with the real components of each
//! value adjacent (`[node][component]`). This is the order in which
//! [`scioxide_core::write_value`] and the slice implementations of
//! [`AffineValue`] traverse values. Queries compute all components in a
//! workspace and write them to the output at the end.

use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, RealScalar, Scaled, SciOxideError, check_value_shape,
};

/// Number of scalars a workspace holds without heap storage.
const INLINE: usize = 16;

/// Holds the scratch storage of one query on most interpolants.
///
/// Create one with [`crate::Interpolant::workspace`]. Up to 16 scalars are
/// stored inline, so the owned query forms of interpolants with small values do
/// not allocate. A workspace created for one interpolant serves any other
/// interpolant of the same type whose need it covers. A workspace stays usable
/// after a failed query.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BilinearInterpolator, ExtrapolationPolicy, Interpolant};
/// // The constant two-component value [0, 1] on a 2 × 2 grid.
/// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![[0.0, 1.0]; 4])?;
/// let mut workspace = grid.workspace()?;
/// let mut out = [0.0; 2];
/// grid.evaluate_into(
///     [0.5, 0.5],
///     ExtrapolationPolicy::Reject,
///     &mut out,
///     &mut workspace,
/// )?;
/// assert_eq!(out, [0.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct QueryWorkspace<S: RealScalar> {
    inline: [S; INLINE],
    heap: Vec<S>,
    scaled_heap: Vec<Scaled<S>>,
}

impl<S: RealScalar> QueryWorkspace<S> {
    /// Scratch for `len` scalars.
    pub(crate) fn new(len: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            inline: [S::ZERO; INLINE],
            heap: if len > INLINE {
                try_filled(len, S::ZERO, scioxide_core::quantity!("query_workspace"))?
            } else {
                Vec::new()
            },
            scaled_heap: Vec::new(),
        })
    }

    /// Scratch for `len` scalars and `scaled` values with an extended exponent
    /// range, used by rational spline queries.
    pub(crate) fn with_scaled(len: usize, scaled: usize) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new(len)?;
        workspace.scaled_heap = try_filled(
            scaled,
            Scaled::ZERO,
            scioxide_core::quantity!("query_workspace"),
        )?;
        Ok(workspace)
    }

    pub(crate) fn scaled_capacity(&self, len: usize, scaled: usize) -> Capacity {
        if self.capacity(len) == Capacity::Sufficient && scaled <= self.scaled_heap.len() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }

    pub(crate) fn scratch_with_scaled(
        &mut self,
        len: usize,
        scaled: usize,
    ) -> Result<(&mut [S], &mut [Scaled<S>]), SciOxideError> {
        if self.capacity(len) != Capacity::Sufficient {
            return Err(crate::error::shape(len, self.len()));
        }
        if self.scaled_heap.len() < scaled {
            return Err(crate::error::shape(scaled, self.scaled_heap.len()));
        }
        let native = if len <= INLINE {
            &mut self.inline[..len]
        } else {
            &mut self.heap[..len]
        };
        let extended = &mut self.scaled_heap[..scaled];
        Ok((native, extended))
    }

    /// Number of scalars available.
    const fn len(&self) -> usize {
        if self.heap.len() > INLINE {
            self.heap.len()
        } else {
            INLINE
        }
    }

    /// Returns whether at least `len` scalars are available.
    pub(crate) const fn capacity(&self, len: usize) -> Capacity {
        if self.len() >= len {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }

    /// Returns the first `len` scalars, or `DimensionMismatch` if fewer exist.
    pub(crate) fn scratch(&mut self, len: usize) -> Result<&mut [S], SciOxideError> {
        if len <= INLINE {
            return Ok(&mut self.inline[..len]);
        }
        let available = self.len();
        self.heap
            .get_mut(..len)
            .ok_or_else(|| crate::error::shape(len, available))
    }
}

/// Checked values stored node by node with adjacent components.
#[derive(Debug, Clone)]
pub(crate) struct Flattened<S, V> {
    /// Real components per value.
    pub(crate) dimension: usize,
    /// `[sample][component]` components.
    pub(crate) components: Vec<S>,
    /// First sample, kept only for its shape (see [`zeroed`]).
    pub(crate) template: V,
}

/// Flatten equally shaped, finite affine values. `expected` is the required sample count.
pub(crate) fn flatten<S: RealScalar, V: AffineValue<Scale = S>>(
    values: Vec<V>,
    expected: usize,
) -> Result<Flattened<S, V>, SciOxideError> {
    if values.len() != expected {
        return Err(crate::error::shape(expected, values.len()));
    }
    let first = values
        .first()
        .ok_or_else(|| crate::error::shape(expected.max(1), 0))?;
    let dimension = first.dimension()?;
    let mut components = try_with_capacity(
        checked_mul(
            values.len(),
            dimension,
            scioxide_core::quantity!("value_components"),
        )?,
        scioxide_core::quantity!("value_components"),
    )?;
    for value in &values {
        check_value_shape(value, dimension)?;
        for component in 0..dimension {
            let component = value.component(component)?;
            if !component.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "value components must be finite",
                ));
            }
            components.push(component);
        }
    }
    let template = values
        .into_iter()
        .next()
        .ok_or_else(|| crate::error::shape(1, 0))?;
    Ok(Flattened {
        dimension,
        components,
        template,
    })
}

/// Owned copy of `value` with every real component set to zero.
pub(crate) fn zeroed<V: AffineValue + ?Sized>(value: &V) -> Result<V::Owned, SciOxideError> {
    let mut owned = value.try_to_owned()?;
    owned.for_each_component_mut(|component| {
        *component = <V::Scale as scioxide_core::AlgebraicScalar>::ZERO
    });
    Ok(owned)
}

/// Gather component `component` of node-major values into `column`.
pub(crate) fn gather<S: Copy>(
    components: &[S],
    dimension: usize,
    component: usize,
    column: &mut [S],
) {
    for (slot, value) in column
        .iter_mut()
        .zip(components.iter().skip(component).step_by(dimension))
    {
        *slot = *value;
    }
}

/// Scatter `column` into component `component` of node-major values.
pub(crate) fn scatter<S: Copy>(
    column: &[S],
    dimension: usize,
    component: usize,
    components: &mut [S],
) {
    for (value, slot) in column
        .iter()
        .zip(components.iter_mut().skip(component).step_by(dimension))
    {
        *slot = *value;
    }
}

/// Require an output of exactly `n` entries.
pub(crate) fn length<S>(out: &[S], n: usize) -> Result<(), SciOxideError> {
    if out.len() != n {
        return Err(crate::error::shape(n, out.len()));
    }
    Ok(())
}

/// Writes checked, finite components to `out`. `AffineValueMut` keeps its
/// component count and order fixed during the borrow, so no second shape or
/// finiteness check is needed.
#[inline]
pub(crate) fn commit_finite<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    components: &[S],
    out: &mut O,
) {
    debug_assert!(matches!(out.dimension(), Ok(n) if n == components.len()));
    debug_assert!(components.iter().all(|v| v.is_finite()));
    let mut components = components.iter();
    out.for_each_component_mut(|slot| {
        if let Some(&value) = components.next() {
            *slot = value;
        }
    });
}

/// Fills a new owned value component by component and returns the first error.
pub(crate) fn fill_owned<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    output: &mut O,
    mut calculate: impl FnMut(usize) -> Result<S, SciOxideError>,
) -> Result<(), SciOxideError> {
    // `for_each_component_mut` cannot short-circuit, so the first error is
    // latched and later components are skipped. Callers either own the output
    // or commit a staged copy afterwards.
    let mut index = 0;
    let mut result = Ok(());
    output.for_each_component_mut(|slot| {
        if result.is_ok() {
            match calculate(index) {
                Ok(value) => *slot = value,
                Err(error) => result = Err(error),
            }
        }
        index += 1;
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use scioxide_core::Complex;

    #[test]
    fn flatten_is_node_major_and_validates_shape() {
        let values = vec![Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)];
        let flat = flatten(values.clone(), 2).unwrap();
        assert_eq!(flat.dimension, 2);
        assert_eq!(flat.components, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(zeroed(&flat.template).unwrap(), Complex::new(0.0, 0.0));
        assert!(flatten(values, 3).is_err());
        assert!(flatten(vec![vec![1.0_f64], vec![1.0, 2.0]], 2).is_err());
        assert!(flatten(vec![f64::NAN], 1).is_err());
    }

    #[test]
    fn gather_and_scatter_round_trip() {
        let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut column = [0.0; 3];
        gather(&values, 2, 1, &mut column);
        assert_eq!(column, [2.0, 4.0, 6.0]);
        scatter(&[7.0, 8.0, 9.0], 2, 0, &mut values);
        assert_eq!(values, [7.0, 2.0, 8.0, 4.0, 9.0, 6.0]);
    }

    #[test]
    fn query_workspace_reports_capacity() {
        let mut small = QueryWorkspace::<f64>::new(2).unwrap();
        assert_eq!(small.capacity(INLINE), Capacity::Sufficient);
        assert_eq!(small.capacity(INLINE + 1), Capacity::NeedsGrowth);
        assert!(small.scratch(INLINE + 1).is_err());
        assert_eq!(small.scratch(1).unwrap().len(), 1);
        let mut large = QueryWorkspace::<f64>::new(INLINE + 4).unwrap();
        assert_eq!(large.capacity(INLINE + 4), Capacity::Sufficient);
        assert_eq!(large.scratch(INLINE + 4).unwrap().len(), INLINE + 4);
        assert!(large.scratch(INLINE + 5).is_err());
    }
    struct SnapshotValue(Cell<bool>);
    impl AffineValue for SnapshotValue {
        type Scale = f64;
        type Owned = f64;
        fn dimension(&self) -> Result<usize, SciOxideError> {
            Ok(1)
        }
        fn component(&self, index: usize) -> Result<f64, SciOxideError> {
            if index != 0 {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("component"),
                    index,
                    1,
                ));
            }
            Ok(if self.0.replace(true) { f64::NAN } else { 1.0 })
        }
        fn try_to_owned(&self) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
    }

    #[test]
    fn validates_the_captured_component_without_rereading_it() {
        let captured = flatten(vec![SnapshotValue(Cell::new(false))], 1).unwrap();
        assert_eq!(captured.components, [1.0]);
    }
}
