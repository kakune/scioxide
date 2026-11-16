//! Workspace and transform sizes of prepared operations, computed from their
//! shapes alone.
use super::crossover;
use crate::util::invalid;
use scioxide_core::backend::{checked_add, checked_next_power_of_two};
use scioxide_core::{InvalidInputReason, NumericScalar, SciOxideError};

/// Describes one operation and its operand lengths for a
/// [`PolynomialOperationPlan`](crate::PolynomialOperationPlan).
///
/// Lengths count ascending monomial coefficients, including trailing zeros
/// where the operation accepts them. Every length must be positive. A plan
/// accepts only the exact lengths it was prepared for, not smaller ones, so
/// declare every shape you need. A `Multiply` shape also covers the swapped
/// operand order. Several shapes in one plan share the FFT plans and take the
/// largest of their scratch requirements.
///
/// ```
/// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
///
/// let requests = [PolynomialOperationRequest::Divide {
///     dividend_len: 8,
///     divisor_len: 3,
/// }];
/// let plan = PolynomialOperationPlan::<f64>::new(&requests)?;
/// let requirements = plan.workspace_requirements(&requests)?;
/// assert_eq!(requirements.fft_len(), 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolynomialOperationRequest {
    /// Product `a·b`, computed by
    /// [`PolynomialOperationPlan::multiply_into`](crate::PolynomialOperationPlan::multiply_into).
    Multiply {
        /// Number of coefficients of the left operand.
        left_len: usize,
        /// Number of coefficients of the right operand.
        right_len: usize,
    },
    /// Composition `p(q(x))`, computed by
    /// [`PolynomialOperationPlan::compose_into`](crate::PolynomialOperationPlan::compose_into).
    Compose {
        /// Number of coefficients of the outer polynomial `p`.
        outer_len: usize,
        /// Number of coefficients of the inner polynomial `q`.
        inner_len: usize,
    },
    /// Taylor shift `p(x + s)`, computed by
    /// [`PolynomialOperationPlan::taylor_shift_into`](crate::PolynomialOperationPlan::taylor_shift_into).
    TaylorShift {
        /// Number of coefficients of `p`.
        len: usize,
    },
    /// Division with remainder, computed by
    /// [`PolynomialOperationPlan::div_rem_into`](crate::PolynomialOperationPlan::div_rem_into).
    Divide {
        /// Number of coefficients of the dividend.
        dividend_len: usize,
        /// Number of coefficients of the divisor.
        divisor_len: usize,
    },
}

impl PolynomialOperationRequest {
    /// Returns the key used for preparation and lookup: multiplication
    /// operands are ordered shorter first, and other requests are unchanged.
    #[inline(always)]
    pub(super) const fn canonical(self) -> Self {
        match self {
            Self::Multiply {
                left_len,
                right_len,
            } if right_len < left_len => Self::Multiply {
                left_len: right_len,
                right_len: left_len,
            },
            request => request,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Layout {
    pub coefficients: usize,
    pub scaled: usize,
    // Bit k selects a length-2^k transform. No heap or oversized FFT plan is
    // needed to inspect requirements at large or overflowing shapes.
    pub transforms: usize,
}

#[derive(Clone, Copy, Default)]
struct KernelLayout {
    scratch: usize,
    transforms: usize,
}

const SCRATCH: scioxide_core::Quantity = scioxide_core::quantity!("coefficient_scratch");

fn product<C: NumericScalar>(a: usize, b: usize) -> Result<usize, SciOxideError> {
    let output = checked_add(a, b - 1, scioxide_core::quantity!("product_coefficients"))?;
    if a.min(b) < crossover::<C>() {
        return Ok(0);
    }
    let n = checked_next_power_of_two(output, scioxide_core::quantity!("transform_length"))?;
    Ok(1 << n.ilog2())
}

fn length(p: usize, q: usize) -> Result<usize, SciOxideError> {
    crate::composition::length(
        p,
        q,
        scioxide_core::argument!("outer_len"),
        scioxide_core::argument!("inner_len"),
    )
}

/// Layouts of binary blocks, indexed by `log2` of the outer length.
type BlockLayouts = [Option<KernelLayout>; usize::BITS as usize];

/// Computes the stack and transforms of the composition recursion by walking
/// the nodes of [`crate::composition::Split`] as execution does. Every low child
/// is a power-of-two block, and a block's children are two half blocks, so memoizing
/// blocks by exponent leaves one uncached path along the remainders: `O(log p)`
/// nodes in all.
fn composition<C: NumericScalar>(
    p: usize,
    q: usize,
    blocks: &mut BlockLayouts,
) -> Result<KernelLayout, SciOxideError> {
    let block = p.is_power_of_two().then(|| p.ilog2() as usize);
    if let Some(layout) = block.and_then(|exponent| blocks[exponent]) {
        return Ok(layout);
    }
    let Some(node) = crate::composition::Split::of(
        p,
        q,
        scioxide_core::argument!("outer_len"),
        scioxide_core::argument!("inner_len"),
    )?
    else {
        return Ok(KernelLayout::default());
    };
    let left = composition::<C>(node.block, q, blocks)?;
    let right = composition::<C>(p - node.block, q, blocks)?;
    let local = checked_add(node.low, node.high, SCRATCH)?;
    let layout = KernelLayout {
        scratch: checked_add(local, left.scratch.max(right.scratch), SCRATCH)?,
        transforms: left.transforms | right.transforms | product::<C>(node.high, node.power)?,
    };
    if let Some(exponent) = block {
        blocks[exponent] = Some(layout);
    }
    Ok(layout)
}

pub(super) fn analyze<C: NumericScalar>(
    request: PolynomialOperationRequest,
) -> Result<Layout, SciOxideError> {
    use PolynomialOperationRequest::*;
    let lengths = match request {
        Multiply {
            left_len,
            right_len,
        } => [
            (left_len, scioxide_core::argument!("left_len")),
            (right_len, scioxide_core::argument!("right_len")),
        ],
        Compose {
            outer_len,
            inner_len,
        } => [
            (outer_len, scioxide_core::argument!("outer_len")),
            (inner_len, scioxide_core::argument!("inner_len")),
        ],
        TaylorShift { len } => [(len, scioxide_core::argument!("len")); 2],
        Divide {
            dividend_len,
            divisor_len,
        } => [
            (dividend_len, scioxide_core::argument!("dividend_len")),
            (divisor_len, scioxide_core::argument!("divisor_len")),
        ],
    };
    for (len, argument) in lengths {
        if len == 0 {
            return Err(invalid(
                argument,
                InvalidInputReason::Empty,
                "coefficient count must be positive",
            ));
        }
    }
    match request {
        Multiply {
            left_len,
            right_len,
        } => Ok(Layout {
            coefficients: checked_add(left_len, right_len - 1, SCRATCH)?,
            transforms: product::<C>(left_len, right_len)?,
            ..Layout::default()
        }),
        Divide {
            dividend_len: a,
            divisor_len: b,
        } => Ok(Layout {
            coefficients: checked_add(a, a.saturating_sub(b - 1).max(1), SCRATCH)?,
            scaled: checked_add(a, b, scioxide_core::quantity!("scaled_division_scratch"))?,
            ..Layout::default()
        }),
        Compose {
            outer_len,
            inner_len,
        } => composed_layout::<C>(outer_len, inner_len),
        TaylorShift { len } => composed_layout::<C>(len, 2),
    }
}

fn composed_layout<C: NumericScalar>(p: usize, q: usize) -> Result<Layout, SciOxideError> {
    let output = length(p, q)?;
    let kernel = composition::<C>(p, q, &mut [None; usize::BITS as usize])?;
    let powers = crate::composition::powers_len(p, q)?;
    let mut transforms = kernel.transforms;
    if p > 2 && q > 1 {
        let mut exponent = 2;
        let largest = crate::composition::split(p);
        loop {
            let half = length(exponent / 2 + 1, q)?;
            transforms |= product::<C>(half, half)?;
            if exponent == largest {
                break;
            }
            exponent *= 2;
        }
    }
    Ok(Layout {
        coefficients: checked_add(
            checked_add(output, powers, SCRATCH)?,
            kernel.scratch,
            SCRATCH,
        )?,
        transforms,
        scaled: checked_add(output, output, SCRATCH)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{Complex, RealScalar, SciOxideErrorKind};

    fn storage_bounds<C: NumericScalar>() {
        use PolynomialOperationRequest::*;
        for power in 0..=20 {
            let n = 1_usize << power;
            for b in [1, n, n.saturating_sub(1).max(1)] {
                let division = analyze::<C>(Divide {
                    dividend_len: n,
                    divisor_len: b,
                })
                .unwrap();
                assert_eq!(division.coefficients, n + n.saturating_sub(b - 1).max(1));
                assert_eq!(division.scaled, n + b);
                assert_eq!(division.transforms, 0);
            }
            let shift = analyze::<C>(TaylorShift { len: n }).unwrap();
            assert!((n..=8 * n).contains(&shift.coefficients));
            assert_eq!(shift.scaled, 2 * n);
            let linear_composition = analyze::<C>(Compose {
                outer_len: n,
                inner_len: 2,
            })
            .unwrap();
            assert_eq!(shift, linear_composition);
            let constant = analyze::<C>(Compose {
                outer_len: n,
                inner_len: 1,
            })
            .unwrap();
            assert_eq!(constant.coefficients, 1);
            assert_eq!(constant.transforms, 0);
            let product = analyze::<C>(Multiply {
                left_len: n,
                right_len: n,
            })
            .unwrap();
            assert_eq!(product.coefficients, 2 * n - 1);
            assert_eq!(product.scaled, 0);
            assert_eq!(
                product.transforms,
                if n < crossover::<C>() { 0 } else { 2 * n }
            );
            let unbalanced = analyze::<C>(Multiply {
                left_len: n,
                right_len: 1,
            })
            .unwrap();
            assert_eq!(unbalanced.coefficients, n);
            assert_eq!(unbalanced.transforms, 0);
            if n > 2 {
                // Odd outer lengths walk a remainder path beside memoized blocks.
                let composition = analyze::<C>(Compose {
                    outer_len: n / 2 + 1,
                    inner_len: 3,
                })
                .unwrap();
                assert!(composition.coefficients > n + 1);
                assert!(composition.coefficients <= 8 * (n + 1));
                assert_eq!(composition.scaled, 2 * (n + 1));
            }
        }
    }

    #[test]
    fn operation_storage_bounds_through_two_to_twenty() {
        storage_bounds::<f32>();
        storage_bounds::<f64>();
        storage_bounds::<Complex<f32>>();
        storage_bounds::<Complex<f64>>();
    }

    /// Admitted normal inputs exercise the full recursive coefficient storage;
    /// range recovery and its separate buffers are covered by integration tests.
    fn composition_requirements_equal_execution_usage<C: NumericScalar>() {
        use PolynomialOperationRequest::*;
        let real = |value| C::Real::try_from_f64_exact(value).unwrap();
        let context = scioxide_core::ExecutionContext::sequential();
        let sentinel = C::from_real(C::Real::NAN);
        for outer_len in 1..=70 {
            for inner_len in 1..=5 {
                let request = Compose {
                    outer_len,
                    inner_len,
                };
                let plan = super::super::PolynomialOperationPlan::<C>::new(&[request]).unwrap();
                let requirements = plan.workspace_requirements(&[request]).unwrap();
                let mut workspace = plan.workspace(&[request]).unwrap();
                assert_eq!(workspace.coefficients.len(), requirements.coefficient_len());
                workspace.coefficients.fill(sentinel);
                let p = vec![C::from_real(real(0.5)); outer_len];
                let mut q = vec![C::from_real(real(0.25)); inner_len];
                q[inner_len - 1] = C::ONE;
                let mut output = vec![sentinel; (outer_len - 1) * (inner_len - 1) + 1];
                plan.compose_into(&p, &q, &mut output, &mut workspace, context)
                    .unwrap();
                assert!(
                    workspace
                        .coefficients
                        .iter()
                        .all(|v| v.has_finite_components())
                        && output.iter().all(|v| v.has_finite_components()),
                    "unused composition slot at ({outer_len}, {inner_len})"
                );
            }
            let request = TaylorShift { len: outer_len };
            let plan = super::super::PolynomialOperationPlan::<C>::new(&[request]).unwrap();
            let mut workspace = plan.workspace(&[request]).unwrap();
            workspace.coefficients.fill(sentinel);
            let mut output = vec![sentinel; outer_len];
            plan.taylor_shift_into(
                &vec![C::ONE; outer_len],
                C::from_real(real(0.5)),
                &mut output,
                &mut workspace,
                context,
            )
            .unwrap();
            assert!(
                workspace
                    .coefficients
                    .iter()
                    .all(|v| v.has_finite_components())
                    && output.iter().all(|v| v.has_finite_components()),
                "unused Taylor-shift slot at {outer_len}"
            );
        }
    }

    #[test]
    fn composition_layout_matches_the_execution_recursion() {
        composition_requirements_equal_execution_usage::<f64>();
        composition_requirements_equal_execution_usage::<Complex<f64>>();
    }

    #[test]
    fn invalid_lengths_and_overflow_are_structured_errors() {
        use PolynomialOperationRequest::*;
        for request in [
            Multiply {
                left_len: 0,
                right_len: 1,
            },
            Multiply {
                left_len: 1,
                right_len: 0,
            },
            Compose {
                outer_len: 0,
                inner_len: 2,
            },
            Compose {
                outer_len: 2,
                inner_len: 0,
            },
            TaylorShift { len: 0 },
            Divide {
                dividend_len: 0,
                divisor_len: 1,
            },
            Divide {
                dividend_len: 1,
                divisor_len: 0,
            },
        ] {
            assert_eq!(
                analyze::<f64>(request).unwrap_err().kind(),
                SciOxideErrorKind::InvalidInput
            );
        }
        for request in [
            Multiply {
                left_len: usize::MAX,
                right_len: 2,
            },
            Compose {
                outer_len: usize::MAX,
                inner_len: 3,
            },
            TaylorShift { len: usize::MAX },
            Divide {
                dividend_len: usize::MAX,
                divisor_len: 1,
            },
        ] {
            assert_eq!(
                analyze::<f64>(request).unwrap_err().kind(),
                SciOxideErrorKind::SizeOverflow
            );
        }
    }
}
