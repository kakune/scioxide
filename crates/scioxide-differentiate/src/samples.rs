//! Evaluation-local Hessian samples. Axis storage survives coordinate pairs;
//! mixed storage survives refinement levels of only the current pair.

use crate::workspace;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{InvalidInputReason, NumericScalar, RealScalar, SciOxideError};
pub(super) fn same_bits<S: RealScalar>(a: S, b: S) -> bool {
    a == b && (a != S::ZERO || a.is_sign_negative() == b.is_sign_negative())
}

#[derive(Debug, Clone, Copy)]
enum Key<S> {
    Base,
    Axis(usize, S),
    Mixed(usize, S, usize, S),
}
impl<S: RealScalar> Key<S> {
    fn at_axes(base: &[S], point: &[S], first: usize, second: Option<usize>) -> Self {
        let mut key = Self::Base;
        for axis in [Some(first), second].into_iter().flatten() {
            if same_bits(base[axis], point[axis]) {
                continue;
            }
            key = match key {
                Self::Base => Self::Axis(axis, point[axis]),
                Self::Axis(i, a) => Self::Mixed(i, a, axis, point[axis]),
                mixed => mixed,
            };
        }
        key
    }
    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Base, Self::Base) => true,
            (Self::Axis(i, a), Self::Axis(j, b)) => i == j && same_bits(a, b),
            (Self::Mixed(i, a, j, b), Self::Mixed(k, c, l, d)) => {
                i == k && j == l && same_bits(a, c) && same_bits(b, d)
            }
            _ => false,
        }
    }
    fn at(base: &[S], point: &[S]) -> Result<Self, SciOxideError> {
        let mut key = Self::Base;
        for (i, (&a, &b)) in base.iter().zip(point).enumerate() {
            if same_bits(a, b) {
                continue;
            }
            key = match key {
                Self::Base => Self::Axis(i, b),
                Self::Axis(j, c) => Self::Mixed(j, c, i, b),
                Self::Mixed(..) => {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("workspace"),
                        InvalidInputReason::Inconsistent,
                        "a Hessian sample must change at most two coordinates",
                    ));
                }
            };
        }
        Ok(key)
    }
}

#[derive(Debug, Clone, Copy)]
struct Sample<T: NumericScalar> {
    key: Key<T::Real>,
    value: T,
}

// Hessians have exactly one callback output. One vector owns two disjoint
// bounded regions, so storage does not enlarge downstream workspace enums or
// add an indirection/allocation solely for inline metadata.
#[derive(Debug, Clone)]
pub(super) struct HessianSamples<T: NumericScalar> {
    samples: Vec<Sample<T>>,
    axis_lengths: Vec<usize>,
    axis_stride: usize,
    axis_capacity: usize,
    mixed_len: usize,
}
impl<T: NumericScalar> Default for HessianSamples<T> {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            axis_lengths: Vec::new(),
            axis_stride: 0,
            axis_capacity: 0,
            mixed_len: 0,
        }
    }
}
// Every fallible method reports failures
// that owns the evaluation or workspace.
impl<T: NumericScalar> HessianSamples<T> {
    fn axis_capacity(inputs: usize, nodes: usize) -> Result<usize, SciOxideError> {
        if nodes == 0 {
            Ok(0)
        } else {
            checked_add(
                1,
                checked_mul(inputs, nodes, scioxide_core::quantity!("hessian_samples"))?,
                scioxide_core::quantity!("hessian_samples"),
            )
        }
    }
    pub(super) fn has_capacity(
        &self,
        inputs: usize,
        axis_nodes: usize,
        mixed_nodes: usize,
    ) -> bool {
        let axis = if axis_nodes == 0 {
            Some(0)
        } else {
            inputs
                .checked_mul(axis_nodes)
                .and_then(|n| n.checked_add(1))
        };
        axis.and_then(|n| n.checked_add(mixed_nodes))
            .is_some_and(|n| self.samples.capacity() >= n)
            && (axis_nodes == 0
                || inputs
                    .checked_add(1)
                    .is_some_and(|n| self.axis_lengths.capacity() >= n))
    }
    pub(super) fn projected_bytes(
        &self,
        inputs: usize,
        axis_nodes: usize,
        mixed_nodes: usize,
    ) -> Result<usize, SciOxideError> {
        let samples = checked_add(
            Self::axis_capacity(inputs, axis_nodes)?,
            mixed_nodes,
            scioxide_core::quantity!("hessian_samples"),
        )?;
        let lengths = if axis_nodes == 0 {
            0
        } else {
            checked_add(inputs, 1, scioxide_core::quantity!("hessian_samples"))?
        };
        self.payload_bytes(samples, lengths)
    }
    #[inline]
    fn payload_bytes(&self, samples: usize, lengths: usize) -> Result<usize, SciOxideError> {
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<Sample<T>>(
                    [(self.samples.capacity(), samples)],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<usize>(
                    [(self.axis_lengths.capacity(), lengths)],
                    workspace::BYTES,
                )?,
            ],
            workspace::BYTES,
        )
    }
    #[inline]
    pub(super) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.payload_bytes(0, 0)
    }
    pub(super) fn reserve(
        &mut self,
        inputs: usize,
        axis_nodes: usize,
        mixed_nodes: usize,
    ) -> Result<(), SciOxideError> {
        let samples = checked_add(
            Self::axis_capacity(inputs, axis_nodes)?,
            mixed_nodes,
            scioxide_core::quantity!("hessian_samples"),
        )?;
        let lengths = if axis_nodes == 0 {
            0
        } else {
            checked_add(inputs, 1, scioxide_core::quantity!("hessian_samples"))?
        };
        let sample_extra = samples.saturating_sub(self.samples.len());
        let length_extra = lengths.saturating_sub(self.axis_lengths.len());
        scioxide_core::backend::try_reserve(
            &mut self.samples,
            sample_extra,
            scioxide_core::quantity!("hessian_samples"),
        )?;
        scioxide_core::backend::try_reserve(
            &mut self.axis_lengths,
            length_extra,
            scioxide_core::quantity!("hessian_lengths"),
        )?;
        self.configure(inputs, axis_nodes, mixed_nodes)
    }
    pub(super) fn configure(
        &mut self,
        inputs: usize,
        axis_nodes: usize,
        mixed_nodes: usize,
    ) -> Result<(), SciOxideError> {
        self.axis_capacity = Self::axis_capacity(inputs, axis_nodes)?;
        self.axis_stride = axis_nodes;
        self.samples.resize(
            checked_add(
                self.axis_capacity,
                mixed_nodes,
                scioxide_core::quantity!("hessian_samples"),
            )?,
            Sample {
                key: Key::Base,
                value: T::ZERO,
            },
        );
        self.axis_lengths.resize(
            if axis_nodes == 0 {
                0
            } else {
                checked_add(inputs, 1, scioxide_core::quantity!("hessian_samples"))?
            },
            0,
        );
        self.clear();
        Ok(())
    }
    pub(super) fn clear(&mut self) {
        self.axis_lengths.fill(0);
        self.clear_pair();
    }
    pub(super) fn clear_pair(&mut self) {
        self.mixed_len = 0;
    }
    fn region(&self, key: Key<T::Real>) -> (usize, usize) {
        match key {
            Key::Mixed(..) => (self.axis_capacity, self.mixed_len),
            Key::Axis(axis, _) => (1 + axis * self.axis_stride, self.axis_lengths[axis]),
            Key::Base => (0, self.axis_lengths.last().copied().unwrap_or(0)),
        }
    }
    pub(super) fn find(
        &self,
        base: &[T::Real],
        point: &[T::Real],
    ) -> Result<Option<T>, SciOxideError> {
        let key = Key::at(base, point)?;
        Ok(self.lookup(key))
    }
    fn lookup(&self, key: Key<T::Real>) -> Option<T> {
        let (start, len) = self.region(key);
        self.samples[start..start + len]
            .iter()
            .find(|sample| sample.key.same(key))
            .map(|sample| sample.value)
    }
    pub(super) fn find_axis(&self, axis: usize, node: T::Real) -> Option<T> {
        self.lookup(Key::Axis(axis, node))
    }
    pub(super) fn insert_axes(
        &mut self,
        base: &[T::Real],
        point: &[T::Real],
        first: usize,
        second: Option<usize>,
        value: T,
    ) -> Result<(), SciOxideError> {
        self.store(Key::at_axes(base, point, first, second), value)
    }
    pub(super) fn insert(
        &mut self,
        base: &[T::Real],
        point: &[T::Real],
        value: T,
    ) -> Result<(), SciOxideError> {
        let key = Key::at(base, point)?;
        self.store(key, value)
    }
    fn store(&mut self, key: Key<T::Real>, value: T) -> Result<(), SciOxideError> {
        let (start, capacity, len) = match key {
            Key::Mixed(..) if self.samples.len() == self.axis_capacity => return Ok(()),
            Key::Mixed(..) => (
                self.axis_capacity,
                self.samples.len() - self.axis_capacity,
                &mut self.mixed_len,
            ),
            Key::Axis(axis, _) => (
                1 + axis * self.axis_stride,
                self.axis_stride,
                &mut self.axis_lengths[axis],
            ),
            Key::Base => {
                let i = self.axis_lengths.len().checked_sub(1).ok_or_else(|| {
                    SciOxideError::invalid_input(
                        scioxide_core::argument!("workspace"),
                        InvalidInputReason::Inconsistent,
                        "no axis storage",
                    )
                })?;
                (0, 1, &mut self.axis_lengths[i])
            }
        };
        if *len == capacity {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("workspace"),
                InvalidInputReason::Inconsistent,
                "insufficient reserved sample storage",
            ));
        }
        self.samples[start + *len] = Sample { key, value };
        *len += 1;
        Ok(())
    }
    pub(super) fn bytes_for(
        inputs: usize,
        axis_nodes: usize,
        mixed_nodes: usize,
    ) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("hessian_samples");
        let samples = checked_add(
            Self::axis_capacity(inputs, axis_nodes)?,
            mixed_nodes,
            quantity,
        )?;
        let lengths = if axis_nodes == 0 {
            0
        } else {
            checked_add(inputs, 1, quantity)?
        };
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<Sample<T>>([(0, samples)], quantity)?,
                workspace::buffer_bytes::<usize>([(0, lengths)], quantity)?,
            ],
            quantity,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::stencil::{MAX_STENCIL_POINTS, mixed_stencil_derivative, stencil_derivative};
    use crate::{
        DerivativeRequest, DerivativeShape, DifferenceScheme, FiniteDifference,
        FiniteDifferenceWorkspace,
    };

    use super::*;
    use alloc::vec;

    fn callback(x: &[f64]) -> f64 {
        x.iter().map(|v| v.sin() + v * v).sum::<f64>()
            + x.windows(2).map(|v| v[0] * v[1]).sum::<f64>()
    }

    #[test]
    fn fixed_counts_and_previous_contraction_bits() {
        for n in [2, 6] {
            for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
                let x = vec![1.0; n];
                let method = FiniteDifference::default().with_scheme(scheme);
                let mut workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
                    &method,
                    DerivativeShape::hessian(n),
                ))
                .unwrap();
                let mut actual = vec![0.0; n * n];
                let mut calls = 0;
                let report = method
                    .hessian_into(
                        |p: &[f64], y: &mut [f64]| {
                            calls += 1;
                            y[0] = callback(p);
                            Ok(())
                        },
                        &x,
                        None,
                        None,
                        &mut actual,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                let expected_calls = match scheme {
                    DifferenceScheme::Forward => 1 + 2 * n + n * (n - 1) / 2,
                    DifferenceScheme::Central => 1 + 2 * n * n,
                };
                assert_eq!(calls, expected_calls);
                assert_eq!(report.evaluations(), calls);
                // Reproduce the previous uncached sample/contraction order.
                // The arithmetic kernels are intentionally shared: this checks
                // that reuse changes callbacks, not numerical transformations.
                let base = callback(&x);
                let mut previous = vec![0.0; n * n];
                for i in 0..n {
                    let axis = workspace.data.second_stencils[i];
                    let mut values = [0.0; MAX_STENCIL_POINTS];
                    for (k, v) in values[..axis.len].iter_mut().enumerate() {
                        let mut point = x.clone();
                        point[i] += axis.offsets[k];
                        *v = if axis.offsets[k] == 0.0 {
                            base
                        } else {
                            callback(&point)
                        };
                    }
                    previous[i * n + i] = stencil_derivative(
                        axis,
                        &values[..axis.len],
                        base,
                        &mut scioxide_core::RangeAccumulator::new(),
                    )
                    .unwrap();
                    for j in i + 1..n {
                        let a = workspace.data.first_stencils[i];
                        let b = workspace.data.first_stencils[j];
                        let mut values = [0.0; MAX_STENCIL_POINTS * MAX_STENCIL_POINTS];
                        for p in 0..a.len {
                            for q in 0..b.len {
                                let mut point = x.clone();
                                point[i] += a.offsets[p];
                                point[j] += b.offsets[q];
                                values[p * b.len + q] =
                                    if a.offsets[p] == 0.0 && b.offsets[q] == 0.0 {
                                        base
                                    } else {
                                        callback(&point)
                                    };
                            }
                        }
                        let value = mixed_stencil_derivative(
                            a,
                            b,
                            &values[..a.len * b.len],
                            base,
                            &mut scioxide_core::RangeAccumulator::new(),
                        )
                        .unwrap();
                        previous[i * n + j] = value;
                        previous[j * n + i] = value;
                    }
                }
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    previous.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn cache_distinguishes_signed_zero_and_resets_validity() {
        let mut samples = HessianSamples::<f64>::default();
        samples.reserve(2, 4, 4).unwrap();
        samples.insert(&[-0.0, 1.0], &[-0.0, 1.0], 2.0).unwrap();
        assert_eq!(samples.find(&[-0.0, 1.0], &[-0.0, 1.0]).unwrap(), Some(2.0));
        assert!(samples.find(&[-0.0, 1.0], &[0.0, 1.0]).unwrap().is_none());
        samples.insert(&[-0.0, 1.0], &[0.0, 1.0], 3.0).unwrap();
        samples.clear();
        assert!(samples.find(&[-0.0, 1.0], &[0.0, 1.0]).unwrap().is_none());
    }
    #[test]
    fn hessian_cache_requires_each_buffer_and_preserves_overflow_quantities() {
        let mut samples = HessianSamples::<scioxide_core::Complex64>::default();
        samples.reserve(2, 6, 8).unwrap();
        assert!(samples.has_capacity(2, 6, 8));
        let retained = samples.retained_bytes().unwrap();
        assert_eq!(
            retained,
            HessianSamples::<scioxide_core::Complex64>::bytes_for(2, 6, 8).unwrap()
        );
        samples.samples = Vec::with_capacity(20); // 1 + 2*6 + 8 needs 21 slots.
        samples.axis_lengths.reserve(1024);
        assert!(!samples.has_capacity(2, 6, 8));
        assert!(samples.retained_bytes().unwrap() > retained);
        samples.reserve(2, 6, 8).unwrap();
        samples.axis_lengths = Vec::with_capacity(2); // needs inputs + 1.
        samples.samples.reserve(1024);
        assert!(!samples.has_capacity(2, 6, 8));
        let error =
            HessianSamples::<scioxide_core::Complex64>::bytes_for(usize::MAX, 2, 0).unwrap_err();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow {  context }
            if context == scioxide_core::quantity!("hessian_samples"))
        );
        let error = samples.projected_bytes(0, 0, usize::MAX).unwrap_err();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow {  context }
            if context == workspace::BYTES)
        );
    }
}
