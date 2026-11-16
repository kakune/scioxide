//! Packed microkernel size checks, output tiles, and portable arithmetic.

use scioxide_core::backend::{KernelDispatch, ScalarSlice, ScalarSliceMut, ScalarValue};
use scioxide_core::{AlgebraicScalar, NumericScalar};

use crate::dense::MatrixViewMut;

use scioxide_core::backend::{component_count, components_mut};

type MicrokernelF32 = fn(&[f32], &[f32], &mut [f32], usize, usize, f32, KernelDispatch) -> bool;
type MicrokernelF64 = fn(&[f64], &[f64], &mut [f64], usize, usize, f64, KernelDispatch) -> bool;

#[derive(Clone, Copy)]
// Scalar-only targets cannot construct an architecture leaf.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
    expect(
        dead_code,
        reason = "scalar-only targets cannot construct an architecture leaf variant"
    )
)]
enum MicrokernelLeaf {
    F32(MicrokernelF32),
    F64(MicrokernelF64),
}

/// A detected architecture leaf together with the slice contract it requires.
///
/// The verified dispatch travels with the callback. Both [`Self::apply`] and
/// each safe ISA callback check the panel spans; callbacks also check their ISA token.
/// Leaves are real: complex products pass component panels and the leaf's
/// `cols` real columns hold `cols / 2` interleaved complex entries.
#[derive(Clone, Copy)]
pub(crate) struct Microkernel {
    leaf: MicrokernelLeaf,
    dispatch: KernelDispatch,
    rows: usize,
    cols: usize,
}

impl Microkernel {
    /// Native leaves always consume full padded component panels. Adapt their
    /// stores to partial, strided, or masked destinations without reading
    /// excluded values. `tile` is in scalars of `S`; a non-real complex `beta`
    /// scales gathered entries before a unit-beta leaf call.
    // Specialize admission and the full-tile path in the packed block loop,
    // independent of which codegen unit owns the other scalar instantiations.
    #[inline]
    pub(super) fn apply_tile<S: NumericScalar>(
        self,
        packed_a: &[S::Real],
        packed_b: &[S::Real],
        output: &mut MatrixViewMut<'_, S>,
        tile: &super::OutputTile,
        beta: S,
    ) -> bool {
        let count = component_count::<S>();
        let cols = self.cols / count;
        let real_beta = beta.imaginary_component() == S::Real::ZERO;
        let leaf_beta = if real_beta {
            beta.real_component()
        } else {
            S::Real::ONE
        };
        let depth = count * tile.shared;
        if tile.rows == self.rows
            && tile.cols == cols
            && output.col_stride == 1
            && tile.triangle.is_none()
            && real_beta
        {
            let offset = tile.row_start * output.row_stride + tile.col_start;
            return self.apply(
                packed_a,
                packed_b,
                components_mut(&mut output.data[offset..]),
                count * output.row_stride,
                depth,
                leaf_beta,
            );
        }
        // The largest supported register tile is AVX-512 f32, 4 by 32 real
        // components. This bound is also exercised for every backend by
        // conformance tests.
        let mut scratch = [S::ZERO; 128];
        let Some(scratch) = scratch.get_mut(..self.rows * cols) else {
            return false;
        };
        if beta != S::ZERO {
            for row in 0..tile.rows {
                for col in 0..tile.cols {
                    if tile.contains(row, col) {
                        let value = output.get_element(tile.row_start + row, tile.col_start + col);
                        scratch[row * cols + col] = if real_beta { value } else { beta * value };
                    }
                }
            }
        }
        if !self.apply(
            packed_a,
            packed_b,
            components_mut(scratch),
            self.cols,
            depth,
            leaf_beta,
        ) {
            return false;
        }
        for row in 0..tile.rows {
            for col in 0..tile.cols {
                if tile.contains(row, col) {
                    output.set_element(
                        tile.row_start + row,
                        tile.col_start + col,
                        scratch[row * cols + col],
                    );
                }
            }
        }
        true
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    pub(super) const fn new_f32(
        leaf: MicrokernelF32,
        rows: usize,
        cols: usize,
        dispatch: KernelDispatch,
    ) -> Self {
        Self {
            leaf: MicrokernelLeaf::F32(leaf),
            rows,
            cols,
            dispatch,
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    pub(super) const fn new_f64(
        leaf: MicrokernelF64,
        rows: usize,
        cols: usize,
        dispatch: KernelDispatch,
    ) -> Self {
        Self {
            leaf: MicrokernelLeaf::F64(leaf),
            rows,
            cols,
            dispatch,
        }
    }

    /// Real register tile `(rows, cols)` of the leaf.
    #[cfg(test)]
    pub(super) const fn shape(self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub(super) fn apply<S: NumericScalar>(
        self,
        packed_a: &[S],
        packed_b: &[S],
        output: &mut [S],
        row_stride: usize,
        depth: usize,
        beta: S,
    ) -> bool {
        if !panel_fits(
            self.rows,
            self.cols,
            packed_a.len(),
            packed_b.len(),
            output.len(),
            row_stride,
            depth,
        ) {
            return false;
        }
        match self.leaf {
            MicrokernelLeaf::F32(leaf) => {
                let (
                    ScalarSlice::F32(packed_a),
                    ScalarSlice::F32(packed_b),
                    ScalarSliceMut::F32(output),
                    ScalarValue::F32(beta),
                ) = (
                    S::slice(packed_a),
                    S::slice(packed_b),
                    S::slice_mut(output),
                    beta.value(),
                )
                else {
                    return false;
                };
                leaf(
                    packed_a,
                    packed_b,
                    output,
                    row_stride,
                    depth,
                    beta,
                    self.dispatch,
                )
            }
            MicrokernelLeaf::F64(leaf) => {
                let (
                    ScalarSlice::F64(packed_a),
                    ScalarSlice::F64(packed_b),
                    ScalarSliceMut::F64(output),
                    ScalarValue::F64(beta),
                ) = (
                    S::slice(packed_a),
                    S::slice(packed_b),
                    S::slice_mut(output),
                    beta.value(),
                )
                else {
                    return false;
                };
                leaf(
                    packed_a,
                    packed_b,
                    output,
                    row_stride,
                    depth,
                    beta,
                    self.dispatch,
                )
            }
        }
    }
}

/// Overflow-safe packed-panel and output-span size check, also made by each ISA leaf.
pub(super) fn panel_fits(
    rows: usize,
    cols: usize,
    a: usize,
    b: usize,
    c: usize,
    stride: usize,
    depth: usize,
) -> bool {
    let Some(a_len) = depth.checked_mul(rows) else {
        return false;
    };
    let Some(b_len) = depth.checked_mul(cols) else {
        return false;
    };
    let Some(c_len) = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|offset| offset.checked_add(cols))
    else {
        return false;
    };
    stride >= cols && a >= a_len && b >= b_len && c >= c_len
}

/// Portable evaluation of the same component panels as the native leaves.
pub(super) fn scalar_microkernel<S: NumericScalar>(
    packed_a: &[S::Real],
    packed_b: &[S::Real],
    output: &mut MatrixViewMut<'_, S>,
    tile: &crate::kernel::OutputTile,
    beta: S,
) {
    let &crate::kernel::OutputTile {
        row_start,
        col_start,
        rows,
        cols,
        shared,
        mr,
        nr,
        ..
    } = tile;

    // Each lane keeps exactly the same inner-product order as the scalar
    // reference. Independent adjacent columns let the compiler vectorize on
    // baseline ISAs without reassociation or a hidden processor capability.
    // The largest architecture panel has 32 real columns.
    let count = component_count::<S>();
    let (lanes, width) = (count * nr, count * cols);
    for row in 0..rows {
        let mut sums = [S::Real::ZERO; 32];
        for inner in 0..count * shared {
            let a = packed_a[inner * mr + row];
            sums[..width]
                .iter_mut()
                .zip(&packed_b[inner * lanes..inner * lanes + width])
                .for_each(|(sum, &b)| *sum += a * b);
        }
        for col in 0..cols {
            if !tile.contains(row, col) {
                continue;
            }
            let sum = if S::IS_REAL {
                S::from_real(sums[col])
            } else {
                S::from_components(sums[2 * col], sums[2 * col + 1])
            };
            let prior = if beta == S::ZERO {
                S::ZERO
            } else {
                beta * output.get_element(row_start + row, col_start + col)
            };
            output.set_element(row_start + row, col_start + col, sum + prior);
        }
    }
}

#[cfg(all(
    test,
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    use super::*;

    fn fill_leaf(
        _packed_a: &[f64],
        _packed_b: &[f64],
        output: &mut [f64],
        row_stride: usize,
        _depth: usize,
        _beta: f64,
        _dispatch: KernelDispatch,
    ) -> bool {
        output[0] = 1.0;
        output[row_stride] = 2.0;
        true
    }

    #[test]
    fn dispatch_boundary_rejects_incomplete_panels_and_output_rows() {
        let kernel = Microkernel::new_f64(
            fill_leaf,
            2,
            1,
            KernelDispatch::detect(&scioxide_core::PortableProcessor),
        );
        let mut output = [0.0; 2];

        assert!(!kernel.apply(&[1.0], &[1.0], &mut output, 1, 1, 0.0));
        assert!(!kernel.apply(&[1.0, 1.0], &[1.0], &mut output[..1], 1, 1, 0.0,));
        assert_eq!(output, [0.0, 0.0]);
    }

    #[test]
    fn dispatch_boundary_invokes_leaf_after_validation() {
        let kernel = Microkernel::new_f64(
            fill_leaf,
            2,
            1,
            KernelDispatch::detect(&scioxide_core::PortableProcessor),
        );
        let mut output = [0.0; 2];

        assert!(kernel.apply(&[1.0, 1.0], &[1.0], &mut output, 1, 1, 0.0));
        assert_eq!(output, [1.0, 2.0]);
    }
}
