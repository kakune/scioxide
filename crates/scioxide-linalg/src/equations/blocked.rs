//! Block-scaled Standard products and quasi-triangular matrix equations.
use crate::kernel::frames::{Framed, Grid, Input, exponent_sum, lost, shift};
use crate::kernel::sylvester;
use crate::kernel::{GemmConfig, WorkspaceGemmProblem, standard};
use crate::{
    DenseWorkspaceRequest, FiniteMatrixView, KernelContext, LinalgExecutionOptions,
    LinalgRequirements, LinalgWorkspace, MatrixView, MatrixViewMut,
};
use scioxide_core::{ExecutionPolicy, NumericScalar, Scaled, ScaledValue, SciOxideError};

pub(crate) const TILE: usize = 32;
pub(crate) const ELEMENTS: usize = (TILE + 1) * (TILE + 1);
pub(crate) const PACKING: usize = 4 * ELEMENTS;
pub(super) const SCRATCH: usize = 4 * ELEMENTS;

pub(crate) fn packing_requirements<S: NumericScalar>(
    execution: &LinalgExecutionOptions,
    kernels: Option<KernelContext<'_>>,
) -> Result<LinalgRequirements, SciOxideError> {
    let options = execution.with_execution_policy(ExecutionPolicy::Sequential);
    let mut request = DenseWorkspaceRequest::product(TILE + 1, TILE + 1, TILE + 1, &options)
        .with_provider_packing();
    if let Some(kernels) = kernels {
        request = request.for_kernels(kernels);
    }
    crate::kernel::product_requirements::<S>(&request)
}

/// Fixed panel cuts, adjusted once so forward and adjoint jobs use the same grid.
pub(super) fn partition<S: NumericScalar>(a: MatrixView<'_, S>, cuts: &mut [usize]) {
    let n = a.rows();
    for (i, cut) in cuts.iter_mut().enumerate() {
        *cut = (i * TILE).min(n);
        if S::IS_REAL && *cut > 0 && *cut < n && a.get_element(*cut, *cut - 1) != S::ZERO {
            if *cut + 1 == n {
                *cut -= 1;
            } else {
                *cut += 1;
            }
        }
    }
}

pub(crate) struct Products<'a, 'w, 'k, S: NumericScalar> {
    pub(crate) packing: &'a mut [S],
    pub(crate) workspace: &'a mut LinalgWorkspace<'w, S>,
    pub(crate) kernels: KernelContext<'k>,
    pub(crate) policy: ExecutionPolicy,
}
impl<S: NumericScalar> Products<'_, '_, '_, S> {
    pub(super) fn product(
        &mut self,
        a: Input<'_, S>,
        b: Input<'_, S>,
        output: &mut Framed<'_, S>,
        admit: &mut impl FnMut() -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let (_, k) = a.shape();
        let (left, rest) = crate::workspace::split_mut(self.packing, ELEMENTS)?;
        let (right, rest) = crate::workspace::split_mut(rest, ELEMENTS)?;
        let (result, accumulated) = crate::workspace::split_mut(rest, ELEMENTS)?;
        for r in 0..output.grid.rows.len() - 1 {
            for c in 0..output.grid.cols.len() - 1 {
                let row = output.grid.rows[r];
                let col = output.grid.cols[c];
                let mut tile = output.tile(r, c);
                tile.clear();
                for h in (0..k).step_by(TILE) {
                    admit()?;
                    let shared = (k - h).min(TILE);
                    let aa = a.block(row, h, tile.rows, shared)?;
                    let bb = b.block(h, col, shared, tile.cols)?;
                    let (ae, ar) = aa.pack(left)?;
                    let (be, br) = bb.pack(right)?;
                    if ar || br {
                        tile.promote()?;
                        for i in 0..tile.rows {
                            for j in 0..tile.cols {
                                let mut sum = ScaledValue::ZERO;
                                for l in 0..shared {
                                    sum = sum.add(aa.get(i, l)?.mul(bb.get(l, j)?)?)?;
                                }
                                let dest = &mut tile.wide[i * tile.stride + j];
                                *dest = dest.add(sum)?;
                            }
                        }
                        continue;
                    }
                    crate::kernel::gemm_workspace_with_context(
                        WorkspaceGemmProblem {
                            left: FiniteMatrixView::from_workspace(MatrixView::from_slice(
                                tile.rows,
                                shared,
                                &left[..tile.rows * shared],
                            )?),
                            right: FiniteMatrixView::from_workspace(MatrixView::from_slice(
                                shared,
                                tile.cols,
                                &right[..shared * tile.cols],
                            )?),
                            output: MatrixViewMut::from_slice(
                                tile.rows,
                                tile.cols,
                                &mut result[..tile.rows * tile.cols],
                            )?,
                            config: GemmConfig {
                                packed: S::IS_REAL && tile.rows >= 8 && tile.cols >= 8,
                                ..GemmConfig::ordinary()
                            },
                        },
                        self.policy,
                        self.workspace,
                        self.kernels,
                    )?;
                    tile.add_plain(result, exponent_sum(ae, be)?, S::ONE, accumulated)?;
                }
            }
        }
        Ok(())
    }
}

/// Panel updates and a shared scalar-domain Sylvester/Stein leaf recurrence.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn solve<S: NumericScalar>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    sign: S,
    x: &mut Framed<'_, S>,
    cached: &mut Framed<'_, S>,
    scalars: &mut [S],
    wide: &mut [ScaledValue<S>],
    left_adjoint: bool,
    right_adjoint: bool,
    discrete: bool,
    products: &mut Products<'_, '_, '_, S>,
    admit: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<bool, SciOxideError> {
    let (m, n) = (a.rows(), b.rows());
    let av = if left_adjoint { a.adjoint() } else { a };
    let bv = if right_adjoint { b.adjoint() } else { b };
    let (leaf_values, rest) = crate::workspace::split_mut(scalars, ELEMENTS)?;
    let (update_values, rest) = crate::workspace::split_mut(rest, ELEMENTS)?;
    let (intermediate_values, leaf_cache) = crate::workspace::split_mut(rest, ELEMENTS)?;
    let (leaf_wide, rest) = crate::workspace::split_mut(wide, ELEMENTS)?;
    let (update_wide, rest) = crate::workspace::split_mut(rest, ELEMENTS)?;
    let (intermediate_wide, wide_cache) = crate::workspace::split_mut(rest, ELEMENTS)?;
    let grid = x.grid;
    let row_blocks = grid.rows.len() - 1;
    let col_blocks = grid.cols.len() - 1;
    for cc in 0..col_blocks {
        let ci = if right_adjoint {
            col_blocks - 1 - cc
        } else {
            cc
        };
        let (c, q) = (grid.cols[ci], grid.cols[ci + 1] - grid.cols[ci]);
        for rr in 0..row_blocks {
            admit()?;
            let ri = if left_adjoint {
                rr
            } else {
                row_blocks - 1 - rr
            };
            let (r, p) = (grid.rows[ri], grid.rows[ri + 1] - grid.rows[ri]);
            let cuts_r = [0, p];
            let cuts_c = [0, q];
            let local = Grid {
                rows: &cuts_r,
                cols: &cuts_c,
            };
            let mut scales = [0; 3];
            let (s1, ss) = crate::workspace::split_mut(&mut scales, 1)?;
            let (s2, s3) = crate::workspace::split_mut(ss, 1)?;
            let mut leaf = Framed::new(
                &mut leaf_values[..p * q],
                &mut leaf_wide[..p * q],
                s1,
                local,
            );
            let mut update = Framed::new(
                &mut update_values[..p * q],
                &mut update_wide[..p * q],
                s2,
                local,
            );
            let mut intermediate = Framed::new(
                &mut intermediate_values[..p * q],
                &mut intermediate_wide[..p * q],
                s3,
                local,
            );
            leaf.tile(0, 0)
                .copy(x.input()?.block(r, c, p, q)?, products.packing)?;
            let rows = if left_adjoint { 0..r } else { r + p..m };
            let cols = if right_adjoint { c + q..n } else { 0..c };
            if !cols.is_empty() {
                products.product(
                    if discrete {
                        cached.input()?
                    } else {
                        x.input()?
                    }
                    .block(r, cols.start, p, cols.len())?,
                    Input::Plain(bv.block(cols.start, c, cols.len(), q)?),
                    &mut update,
                    admit,
                )?;
                let (packing, scratch) = crate::workspace::split_mut(products.packing, ELEMENTS)?;
                leaf.tile(0, 0).add(
                    update.input()?,
                    if !discrete { -sign } else { -S::ONE },
                    packing,
                    scratch,
                )?;
            }
            if !rows.is_empty() {
                products.product(
                    Input::Plain(av.block(r, rows.start, p, rows.len())?),
                    x.input()?.block(rows.start, c, rows.len(), q)?,
                    &mut intermediate,
                    admit,
                )?;
                if discrete {
                    products.product(
                        intermediate.input()?,
                        Input::Plain(bv.block(c, c, q, q)?),
                        &mut update,
                        admit,
                    )?;
                }
                let (packing, scratch) = crate::workspace::split_mut(products.packing, ELEMENTS)?;
                leaf.tile(0, 0).add(
                    if discrete {
                        update.input()?
                    } else {
                        intermediate.input()?
                    },
                    -S::ONE,
                    packing,
                    scratch,
                )?;
            }
            if !sylvester::framed(
                sylvester::Recurrence {
                    a: a.block(r, r, p, p)?,
                    b: b.block(c, c, q, q)?,
                    sign,
                    left_adjoint,
                    right_adjoint,
                    discrete,
                },
                &mut leaf.tile(0, 0),
                &mut products.packing[..ELEMENTS],
                leaf_cache,
                wide_cache,
                admit,
            )? {
                return Ok(false);
            }
            x.tile(ri, ci).copy(leaf.input()?, products.packing)?;
        }
        if discrete {
            for ri in 0..row_blocks {
                let (r, p) = (grid.rows[ri], grid.rows[ri + 1] - grid.rows[ri]);
                let cr = [0, p];
                let cc = [0, q];
                let mut scale = [0];
                let mut update = Framed::new(
                    &mut update_values[..p * q],
                    &mut update_wide[..p * q],
                    &mut scale,
                    Grid {
                        rows: &cr,
                        cols: &cc,
                    },
                );
                products.product(
                    Input::Plain(av.block(r, 0, p, m)?),
                    x.input()?.block(0, c, m, q)?,
                    &mut update,
                    admit,
                )?;
                cached
                    .tile(ri, ci)
                    .copy(update.input()?, products.packing)?;
            }
        }
    }
    Ok(true)
}

/// Squared Frobenius residual and squared term norms for one output tile.
/// Ordinary reductions operate in binary frames; recovery preserves cancellation.
#[expect(
    clippy::type_complexity,
    reason = "the tuple groups the distinct numerical outputs of this operation"
)]
pub(super) fn residual_tile<S: NumericScalar, const N: usize>(
    terms: [Input<'_, S>; N],
    signs: [S; N],
    packing: &mut [S],
) -> Result<(Scaled<S::Real>, [Scaled<S::Real>; N]), SciOxideError> {
    let (rows, cols) = terms[0].shape();
    let count = rows * cols;
    let mut scales = [0; N];
    let mut recover = false;
    for (i, input) in terms.iter().enumerate() {
        let (e, r) = input.pack(&mut packing[i * ELEMENTS..(i + 1) * ELEMENTS])?;
        scales[i] = e;
        recover |= r;
    }
    let common = scales.iter().copied().max().unwrap_or(0);
    let mut residual = [S::ZERO; ELEMENTS];
    if !recover {
        for k in 0..count {
            for i in 0..N {
                let value = packing[i * ELEMENTS + k];
                let aligned = shift(value, i128::from(scales[i]) - i128::from(common));
                recover |= lost(value, aligned);
                residual[k] = residual[k] + signs[i] * aligned;
            }
            recover |= !residual[k].has_finite_components();
        }
    }
    let mut norms = [Scaled::ZERO; N];
    if !recover {
        for i in 0..N {
            let v = Scaled::from_parts(
                standard::norm_workspace(
                    packing[i * ELEMENTS..i * ELEMENTS + count].iter().copied(),
                )?,
                scales[i],
            )?;
            norms[i] = v.mul(v)?;
        }
        let v = Scaled::from_parts(
            standard::norm_workspace(residual[..count].iter().copied())?,
            common,
        )?;
        return Ok((v.mul(v)?, norms));
    }
    let mut error = Scaled::ZERO;
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = ScaledValue::ZERO;
            for i in 0..N {
                let value = terms[i].get(r, c)?;
                let magnitude = value.magnitude()?;
                norms[i] = norms[i].add(magnitude.mul(magnitude)?)?;
                sum = if signs[i] == S::ONE {
                    sum.add(value)?
                } else {
                    sum.sub(value)?
                };
            }
            let magnitude = sum.magnitude()?;
            error = error.add(magnitude.mul(magnitude)?)?;
        }
    }
    Ok((error, norms))
}
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn ordinary_products_leave_recovery_storage_untouched() -> Result<(), SciOxideError> {
        let mut workspace = LinalgWorkspace::<f64>::new();
        workspace.try_reserve(&packing_requirements::<f64>(
            &LinalgExecutionOptions::new(),
            None,
        )?)?;
        let mut packing = vec![0.; PACKING];
        let mut products = Products {
            packing: &mut packing,
            workspace: &mut workspace,
            kernels: KernelContext::sequential(),
            policy: ExecutionPolicy::Sequential,
        };
        let cuts = [0, 2];
        let mut values = [0.; 4];
        let marker = ScaledValue::from_value(7.)?;
        let mut recovery = [marker; 4];
        let mut scales = [0];
        let mut result = Framed::new(
            &mut values,
            &mut recovery,
            &mut scales,
            Grid {
                rows: &cuts,
                cols: &cuts,
            },
        );
        let a = MatrixView::from_slice(2, 2, &[1., 2., 3., 4.])?;
        products.product(
            Input::Plain(a),
            Input::Plain(a),
            &mut result,
            &mut || Ok(()),
        )?;
        assert_eq!(result.values, &[7., 10., 15., 22.]);
        assert_eq!(result.scales, &[0]);
        assert!(result.wide.iter().all(|&v| v == marker));
        Ok(())
    }

    #[test]
    fn residual_retains_tiny_tail_after_large_cancellation() -> Result<(), SciOxideError> {
        let a = [1e308_f64];
        let b = [1e308_f64];
        let c = [1e-308_f64];
        let mut packing = vec![0.; PACKING];
        let (error, _) = residual_tile(
            [
                Input::Plain(MatrixView::from_slice(1, 1, &a)?),
                Input::Plain(MatrixView::from_slice(1, 1, &b)?),
                Input::Plain(MatrixView::from_slice(1, 1, &c)?),
            ],
            [1., -1., 1.],
            &mut packing,
        )?;
        assert!((error.sqrt()?.round()? / 1e-308 - 1.).abs() < 16. * f64::EPSILON);
        Ok(())
    }

    #[test]
    fn a_tiny_product_uses_one_frame_and_can_be_reused() -> Result<(), SciOxideError> {
        let mut workspace = LinalgWorkspace::<f64>::new();
        workspace.try_reserve(&packing_requirements::<f64>(
            &LinalgExecutionOptions::new(),
            None,
        )?)?;
        let mut packing = vec![0.; PACKING];
        let mut products = Products {
            packing: &mut packing,
            workspace: &mut workspace,
            kernels: KernelContext::sequential(),
            policy: ExecutionPolicy::Sequential,
        };
        let cuts = [0, 1];
        let mut values = [0.];
        let marker = ScaledValue::from_value(7.)?;
        let mut recovery = [marker];
        let mut scale = [0];
        let mut output = Framed::new(
            &mut values,
            &mut recovery,
            &mut scale,
            Grid {
                rows: &cuts,
                cols: &cuts,
            },
        );
        for value in [1e-300, 2.] {
            let input = [value];
            let view = MatrixView::from_slice(1, 1, &input)?;
            products.product(
                Input::Plain(view),
                Input::Plain(view),
                &mut output,
                &mut || Ok(()),
            )?;
            assert!(
                output
                    .scales
                    .iter()
                    .all(|&s| s != crate::kernel::frames::WIDE)
            );
            assert_eq!(output.wide, &[marker]);
            let restored = output
                .input()?
                .get(0, 0)?
                .div(ScaledValue::from_value(value)?)?
                .round()?;
            assert!((restored / value - 1.).abs() < 16. * f64::EPSILON);
        }
        Ok(())
    }

    #[test]
    fn products_preserve_overflowing_partial_sums_and_graded_operands() -> Result<(), SciOxideError>
    {
        let mut workspace = LinalgWorkspace::<f64>::new();
        workspace.try_reserve(&packing_requirements::<f64>(
            &LinalgExecutionOptions::new(),
            None,
        )?)?;
        let mut packing = vec![0.; PACKING];
        let mut products = Products {
            packing: &mut packing,
            workspace: &mut workspace,
            kernels: KernelContext::sequential(),
            policy: ExecutionPolicy::Sequential,
        };
        let left = vec![1e308; 65];
        let right: alloc::vec::Vec<_> = (0..65)
            .map(|k| if (32..64).contains(&k) { -1. } else { 1. })
            .collect();
        let cuts = [0, 1];
        let mut values = [0.0];
        let mut wide = [ScaledValue::ZERO];
        let mut scales = [0];
        let mut output = Framed::new(
            &mut values,
            &mut wide,
            &mut scales,
            Grid {
                rows: &cuts,
                cols: &cuts,
            },
        );
        products.product(
            Input::Plain(MatrixView::from_slice(1, 65, &left)?),
            Input::Plain(MatrixView::from_slice(65, 1, &right)?),
            &mut output,
            &mut || Ok(()),
        )?;
        assert_eq!(output.input()?.get(0, 0)?.round()?, 1e308);
        products.product(
            Input::Plain(MatrixView::from_slice(1, 2, &[1e308, 1e-308])?),
            Input::Plain(MatrixView::from_slice(2, 1, &[1e-308, 1e308])?),
            &mut output,
            &mut || Ok(()),
        )?;
        assert!((output.input()?.get(0, 0)?.round()? - 2.).abs() < 16. * f64::EPSILON);
        products.product(
            Input::Plain(MatrixView::from_slice(1, 2, &[1., 1e-300])?),
            Input::Plain(MatrixView::from_slice(2, 1, &[0., 1e-300])?),
            &mut output,
            &mut || Ok(()),
        )?;
        // The product is below IEEE range, but division in the surrounding
        // equation can make it representable again.
        let restored = output
            .input()?
            .get(0, 0)?
            .div(ScaledValue::<f64>::from_value(1e-300)?)?
            .round()?;
        assert!((restored / 1e-300 - 1.).abs() < 16. * f64::EPSILON);
        let left: alloc::vec::Vec<_> = (0..65)
            .map(|k| {
                if k < 32 {
                    1e308
                } else if k < 64 {
                    -1e308
                } else {
                    1e-308
                }
            })
            .collect();
        let right: alloc::vec::Vec<_> = (0..65)
            .map(|k| if k < 64 { 1e308 } else { 1e-308 })
            .collect();
        products.product(
            Input::Plain(MatrixView::from_slice(1, 65, &left)?),
            Input::Plain(MatrixView::from_slice(65, 1, &right)?),
            &mut output,
            &mut || Ok(()),
        )?;
        // Two overflowing tiles cancel, leaving a tiny tail whose alignment
        // to their old frame would otherwise erase it.
        let restored = output
            .input()?
            .get(0, 0)?
            .div(ScaledValue::<f64>::from_value(1e-308)?)?
            .round()?;
        assert!((restored / 1e-308 - 1.).abs() < 16. * f64::EPSILON);
        Ok(())
    }
}
