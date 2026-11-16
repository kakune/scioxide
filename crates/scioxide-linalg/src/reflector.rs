//! Householder reflectors: generation, compact WY block factors and blocked
//! application to validated workspace values.
//!
//! `H = I - tau v v^H`, `v[0] = 1`, and `H^H x = beta e_0` with real beta.
//! Blocks of reflectors use the compact WY representation `I - V T V^H`
//! (Schreiber and Van Loan, 1989).
//! Complex tau is essential: H need not be Hermitian. Only powers of two
//! rescale inputs, including the safe-minimum loop for subnormal vectors.

use crate::workspace::storage::MatrixStorage;
use crate::{
    Accuracy, FiniteMatrixView, FiniteVectorView, KernelContext, MatrixView, MatrixViewMut,
    VectorViewMut,
    kernel::{self, GemmConfig, GemmOp, LinalgWorkspace, standard},
};
use scioxide_core::{AlgebraicScalar, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Reflector<S: NumericScalar> {
    pub(crate) tau: S,
    pub(crate) beta: S::Real,
}

/// Input is a finite factorization candidate. On success it holds v with an
/// explicit leading one; callers store beta in their packed diagonal slot.
/// Narrow real QR can keep the unit frame when squares, the denominator and
/// its reciprocal have headroom. The maximum is gathered with the source column.
pub(crate) fn generate_narrow<S: NumericScalar>(
    vector: &mut [S],
    maximum: S::Real,
) -> Result<Reflector<S>, SciOxideError> {
    if !S::IS_REAL
        || vector.is_empty()
        || standard::product_scales(maximum, maximum, vector.len())? != (0, 0)
    {
        return generate(VectorViewMut::from_slice(vector));
    }
    generate_bounded_real(vector)
}

/// Real inputs whose sum of squares, signed denominator and its reciprocal
/// have normal-range headroom. In particular a nonzero two/three-entry vector
/// normalized by its 1-norm needs no further range setup. Other callers must
/// establish the bound, as [`generate_narrow`] does.
#[inline(always)]
pub(crate) fn generate_bounded_real<S: NumericScalar>(
    vector: &mut [S],
) -> Result<Reflector<S>, SciOxideError> {
    debug_assert!(S::IS_REAL && !vector.is_empty());
    let alpha = vector[0];
    if vector.iter().skip(1).all(|&x| x == S::ZERO) {
        vector[0] = S::ONE;
        return Ok(Reflector {
            tau: S::ZERO,
            beta: alpha.real_component(),
        });
    }
    let square = match vector {
        [a, b] => *a * *a + *b * *b,
        [a, b, c] => (*a * *a + *b * *b) + *c * *c,
        _ => standard::dot(vector.iter().copied().map(|x| (x, x))),
    };
    let norm = square.real_component().sqrt();
    let beta = -norm.copysign(alpha.real_component());
    let denominator = alpha - S::from_real(beta);
    let tau = -crate::rotation::divide_real(denominator, beta);
    let reciprocal = crate::checked::finite_candidate(S::ONE / denominator)?;
    vector.iter_mut().skip(1).for_each(|x| *x = *x * reciprocal);
    vector[0] = S::ONE;
    Ok(Reflector { tau, beta })
}

pub(crate) fn generate<S: NumericScalar>(
    mut vector: VectorViewMut<'_, S>,
) -> Result<Reflector<S>, SciOxideError> {
    if vector.is_empty() {
        return Ok(Reflector {
            tau: S::ZERO,
            beta: S::Real::ZERO,
        });
    }
    let alpha = vector.as_view().get_element(0);
    if vector.iter().skip(1).all(|&x| x == S::ZERO) && alpha.imaginary_component() == S::Real::ZERO
    {
        vector.set_element(0, S::ONE);
        return Ok(Reflector {
            tau: S::ZERO,
            beta: alpha.real_component(),
        });
    }
    let mut maximum = vector
        .iter()
        .fold(S::Real::ZERO, |a, &b| a.max(standard::magnitude(b)));
    let safe = S::Real::MIN_POSITIVE / S::Real::EPSILON;
    let lift = -standard::exponent(safe)?;
    let mut lifted = 0;
    while maximum < safe && maximum != S::Real::ZERO {
        vector
            .iter_mut()
            .for_each(|x| *x = standard::scale(*x, lift));
        maximum = maximum.scalbn(lift);
        lifted += lift;
    }
    let exponent = standard::exponent(maximum)?;
    let multiplier = S::Real::ONE.scalbn(-exponent);
    if multiplier.is_finite() && multiplier != S::Real::ZERO {
        // Multiplication by a representable power of two gives the same
        // rounded components as scalbn, without decoding every entry again.
        vector.iter_mut().for_each(|x| {
            *x = S::from_components(
                x.real_component() * multiplier,
                x.imaginary_component() * multiplier,
            );
        });
    } else {
        vector
            .iter_mut()
            .for_each(|x| *x = standard::scale(*x, -exponent));
    }
    let norm = standard::norm_workspace(vector.iter().copied())?;
    let alpha = vector.as_view().get_element(0);
    let beta = -norm.copysign(alpha.real_component());
    let denominator = alpha - S::from_real(beta);
    // Reuse the bounded denominator in xLARFG's (beta - alpha) / beta.
    let tau = -crate::rotation::divide_real(denominator, beta);
    // The denominator is bounded after binary normalization, so complex
    // division cannot square an extreme exponent.
    let reciprocal = crate::checked::finite_candidate(S::ONE / denominator)?;
    // The normalized denominator has magnitude at least one half. Its
    // reciprocal and the resulting reflector entries are bounded by two;
    // one division followed by contiguous products avoids per-entry division.
    vector.iter_mut().skip(1).for_each(|x| *x = *x * reciprocal);
    vector.set_element(0, S::ONE);
    Ok(Reflector {
        tau,
        beta: standard::restore(beta, exponent - lifted)?,
    })
}

/// Apply H on the left. Transpose/conjugate views and tau select other sides
/// and adjoints without a second arithmetic implementation. v is explicit.
pub(crate) fn apply<S: NumericScalar>(
    vector: FiniteVectorView<'_, S>,
    tau: S,
    mut output: MatrixViewMut<'_, S>,
) -> Result<(), SciOxideError> {
    if tau == S::ZERO {
        return Ok(());
    }
    debug_assert_eq!(vector.len(), output.rows());
    for col in 0..output.cols() {
        let input = FiniteVectorView::from_workspace(output.as_view().col(col)?);
        // Generated Householder vectors and tau have magnitude at most two.
        // The product bound includes accumulation and update headroom. In the
        // unit frame there is no reason to scale or check individual entries.
        if standard::product_scales(S::Real::TWO, input.maximum, input.len())? == (0, 0) {
            let dot = vector
                .iter()
                .zip(input.iter())
                .enumerate()
                .fold(S::ZERO, |sum, (row, (v, x))| {
                    sum + if row == 0 { x } else { v.conj() * x }
                })
                * tau;
            for row in 0..output.rows() {
                let v = if row == 0 {
                    S::ONE
                } else {
                    vector.as_view().get_element(row)
                };
                output.set_element(row, col, output.get_element(row, col) - v * dot);
            }
            continue;
        }
        let projection = project(vector, input, tau)?;
        for row in 0..output.rows() {
            let v = if row == 0 {
                S::ONE
            } else {
                vector.as_view().get_element(row)
            };
            output.set_element(
                row,
                col,
                update(output.get_element(row, col), v, projection)?,
            );
        }
    }
    Ok(())
}

/// Apply a reflector to a narrow row-contiguous panel.
/// Scratch is reserved by the factorization driver. Extreme columns keep
/// their individual binary frames through the elementary application above.
/// `maximum` bounds the input components. The returned bound includes the
/// projection's componentwise growth and can be reused for a remaining subpanel.
pub(crate) fn apply_panel<S: NumericScalar>(
    vector: FiniteVectorView<'_, S>,
    tau: S,
    mut output: MatrixViewMut<'_, S>,
    maximum: S::Real,
    work: &mut impl MatrixStorage<S>,
) -> Result<S::Real, SciOxideError> {
    if tau == S::ZERO || output.cols() == 0 {
        return Ok(maximum);
    }
    debug_assert_eq!(output.col_stride(), 1);
    if standard::product_scales(S::Real::TWO, maximum, output.rows())? != (0, 0) {
        apply(vector, tau, output.reborrow())?;
        return Ok(FiniteMatrixView::from_workspace(output.as_view()).maximum);
    }
    if S::IS_REAL {
        // Fixed small widths keep projections in registers and unroll the
        // contiguous column updates used by tall polynomial designs.
        macro_rules! narrow {
            ($($n:literal),*) => {
                match output.cols() {
                    $($n => narrow_panel::<S, $n>(vector, tau, output.reborrow(), maximum),)*
                    _ => None,
                }
            };
        }
        if let Some(maximum) = narrow!(1, 2, 3, 4, 5, 6, 7, 8) {
            return Ok(maximum);
        }
    }
    work.try_resize_flat(1, output.cols())?;
    work.as_mut_slice().fill(S::ZERO);
    for row in 0..output.rows() {
        let coefficient = if row == 0 {
            S::ONE
        } else {
            vector.as_view().get_element(row).conj()
        };
        let offset = row * output.row_stride();
        let values = &output.data[offset..offset + output.cols()];
        work.as_mut_slice()
            .iter_mut()
            .zip(values)
            .for_each(|(dot, &value)| *dot = *dot + coefficient * value);
    }
    work.as_mut_slice()
        .iter_mut()
        .for_each(|dot| *dot = tau * *dot);
    let projection_maximum = work.as_slice().iter().fold(S::Real::ZERO, |bound, &value| {
        bound.max(standard::magnitude(value))
    });
    // Each component of v is bounded by two, so even a complex product has
    // components bounded by 4*max(dot). A further factor of two covers rounding
    // in both this bound and the update. Reuse it instead of scanning every
    // shrinking panel. If the conservative bound overflows, the elementary
    // range-safe path still accepts representable results.
    let four = S::Real::TWO + S::Real::TWO;
    let maximum = (maximum + four * projection_maximum) * S::Real::TWO;
    if !maximum.is_finite() {
        apply(vector, tau, output.reborrow())?;
        return Ok(FiniteMatrixView::from_workspace(output.as_view()).maximum);
    }
    for row in 0..output.rows() {
        let coefficient = if row == 0 {
            S::ONE
        } else {
            vector.as_view().get_element(row)
        };
        let offset = row * output.row_stride();
        let cols = output.cols();
        output.data[offset..offset + cols]
            .iter_mut()
            .zip(work.as_slice())
            .for_each(|(value, &dot)| *value = *value - coefficient * dot);
    }
    Ok(maximum)
}

/// Same row-order arithmetic as `apply_panel`, with fixed stack projections.
/// A failed growth bound leaves the output untouched for the general replay.
#[inline]
fn narrow_panel<S: NumericScalar, const COLS: usize>(
    vector: FiniteVectorView<'_, S>,
    tau: S,
    output: MatrixViewMut<'_, S>,
    maximum: S::Real,
) -> Option<S::Real> {
    let mut dots = [S::ZERO; COLS];
    for row in 0..output.rows() {
        let coefficient = if row == 0 {
            S::ONE
        } else {
            vector.as_view().get_element(row).conj()
        };
        let offset = row * output.row_stride();
        for (dot, &value) in dots.iter_mut().zip(&output.data[offset..offset + COLS]) {
            *dot = *dot + coefficient * value;
        }
    }
    let dots = dots.map(|dot| tau * dot);
    let projection_maximum = dots.iter().fold(S::Real::ZERO, |bound, &dot| {
        bound.max(standard::magnitude(dot))
    });
    let four = S::Real::TWO + S::Real::TWO;
    let maximum = (maximum + four * projection_maximum) * S::Real::TWO;
    if !maximum.is_finite() {
        return None;
    }
    for row in 0..output.rows() {
        let coefficient = if row == 0 {
            S::ONE
        } else {
            vector.as_view().get_element(row)
        };
        let offset = row * output.row_stride();
        for (value, &dot) in output.data[offset..offset + COLS].iter_mut().zip(&dots) {
            *value = *value - coefficient * dot;
        }
    }
    Some(maximum)
}

/// Pure projection shared by disjoint views and packed in-place factors.
pub(crate) fn project<S: NumericScalar>(
    vector: FiniteVectorView<'_, S>,
    input: FiniteVectorView<'_, S>,
    tau: S,
) -> Result<(S, i32), SciOxideError> {
    let (_, e) = standard::product_scales(S::Real::TWO, input.maximum, input.len())?;
    let dot = vector
        .iter()
        .zip(input.iter())
        .enumerate()
        .fold(S::ZERO, |sum, (row, (v, x))| {
            (if row == 0 { S::ONE } else { v.conj() }) * standard::scale(x, -e) + sum
        });
    Ok((crate::checked::finite_candidate(tau * dot)?, e))
}

/// Restore one candidate entry after subtracting its reflector projection.
pub(crate) fn update<S: NumericScalar>(
    value: S,
    vector: S,
    projection: (S, i32),
) -> Result<S, SciOxideError> {
    let (dot, e) = projection;
    standard::restore(-vector * dot + standard::scale(value, -e), e)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Forward,
    Backward,
}

/// Explicit V permits both forward and backward products without encoding
/// packed-factor layout in the compact-WY recurrence.
/// Build T transposed so each projection accumulates into a contiguous row;
/// transpose the small completed factor once before returning. All
/// projections accumulate in one pass over the rows of V, each in row order.
pub(crate) fn form_t<S: NumericScalar>(
    vectors: MatrixView<'_, S>,
    tau: impl Clone + ExactSizeIterator<Item = S>,
    direction: Direction,
    triangular: &mut impl MatrixStorage<S>,
) -> Result<(), SciOxideError> {
    let n = tau.len();
    triangular.try_resize_flat(n, n)?;
    let t = triangular.as_mut_slice();
    t.fill(S::ZERO);
    // The diagonal holds each coefficient; zero coefficients need no projection.
    tau.enumerate()
        .for_each(|(j, coefficient)| t[j * n + j] = coefficient);
    let range = |j: usize| match direction {
        Direction::Forward => 0..j,
        Direction::Backward => j + 1..n,
    };
    let entry = |row: usize, col: usize| vectors.get_element(row, col);
    for row in 0..vectors.rows() {
        // Contiguous rows are read as slices so the updates vectorize.
        let contiguous = (vectors.col_stride() == 1 && !vectors.is_conjugated())
            .then(|| &vectors.data[row * vectors.row_stride()..][..n]);
        for j in 0..n {
            if t[j * n + j] == S::ZERO {
                continue;
            }
            let range = range(j);
            let projection = &mut t[j * n + range.start..j * n + range.end];
            match contiguous {
                Some(values) => {
                    let value = values[j];
                    for (dot, &left) in projection.iter_mut().zip(&values[range]) {
                        *dot = left.conj() * value + *dot;
                    }
                }
                None => {
                    let value = entry(row, j);
                    for (dot, i) in projection.iter_mut().zip(range) {
                        *dot = entry(row, i).conj() * value + *dot;
                    }
                }
            }
        }
    }
    for step in 0..n {
        let j = match direction {
            Direction::Forward => step,
            Direction::Backward => n - 1 - step,
        };
        let coefficient = t[j * n + j];
        if coefficient == S::ZERO {
            continue;
        }
        let range = range(j);
        t[j * n + range.start..j * n + range.end]
            .iter_mut()
            .for_each(|dot| *dot = -coefficient * *dot);
        for step in 0..range.len() {
            let i = match direction {
                Direction::Forward => step,
                Direction::Backward => n - 1 - step,
            };
            let inner = match direction {
                Direction::Forward => i..j,
                Direction::Backward => j + 1..i + 1,
            };
            let value = inner.fold(S::ZERO, |sum, k| t[k * n + i] * t[j * n + k] + sum);
            t[j * n + i] = crate::checked::finite_candidate(value)?;
        }
    }
    for row in 0..n {
        for col in 0..row {
            t.swap(row * n + col, col * n + row);
        }
    }
    Ok(())
}

/// Pack reflectors `start..end` into an explicit unit-lower V and form their
/// T for `direction`; backward factors use conjugated coefficients. Column
/// `j` of `packed` holds the tail of reflector `j` below row `j + offset`.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(crate) fn pack<S: NumericScalar>(
    packed: MatrixView<'_, S>,
    tau: &[S],
    start: usize,
    end: usize,
    offset: usize,
    direction: Direction,
    vectors: &mut impl MatrixStorage<S>,
    triangular: &mut impl MatrixStorage<S>,
) -> Result<(), SciOxideError> {
    let rows = packed.rows() - start - offset;
    let n = end - start;
    vectors.try_resize_flat(rows, n)?;
    // Pack contiguous row prefixes rather than walking each reflector column.
    // This exposes copies directly and avoids zeroing entries overwritten below.
    for (i, row) in vectors
        .as_mut_slice()
        .chunks_exact_mut(n.max(1))
        .enumerate()
    {
        let tail = i.min(n);
        if packed.col_stride() == 1 && !packed.is_conjugated() {
            let source = (start + offset + i) * packed.row_stride() + start;
            row[..tail].copy_from_slice(&packed.data[source..source + tail]);
        } else {
            row[..tail].iter_mut().enumerate().for_each(|(j, value)| {
                *value = packed.get_element(start + offset + i, start + j);
            });
        }
        row[tail..].fill(S::ZERO);
        if i < n {
            row[i] = S::ONE;
        }
    }
    form_t(
        vectors.try_view()?,
        tau[start..end].iter().copied().map(|t| match direction {
            Direction::Forward => t,
            Direction::Backward => t.conj(),
        }),
        direction,
        triangular,
    )
}

/// Blocked reflector product from the left: `C := C - V op(T) (V^H C)` for
/// explicit V and the block factor T of `direction`. `bound`, when known, is an
/// upper bound on the components of C that replaces measuring it.
/// `projection_rows`, when supplied, asserts that all later rows of C are zero
/// before this update.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(crate) fn apply_block<S: NumericScalar>(
    vectors: &impl MatrixStorage<S>,
    triangular: &impl MatrixStorage<S>,
    mut output: MatrixViewMut<'_, S>,
    bound: Option<S::Real>,
    projection_rows: Option<usize>,
    operation_kind: GemmOp,
    direction: Direction,
    work: &mut impl MatrixStorage<S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    #[cfg(not(feature = "blas"))]
    let _ = direction;
    let context = context.with_accuracy(Accuracy::Standard);
    work.try_resize_flat(vectors.cols(), output.cols())?;
    let vectors = FiniteMatrixView::from_workspace(vectors.try_view()?);
    let triangular = FiniteMatrixView::from_workspace(triangular.try_view()?);
    // C is unchanged until the final update, which reuses this bound.
    // Rows excluded from the projection are known zero, so this also gives
    // the range of the complete output before its first update.
    let active = projection_rows.unwrap_or(output.rows());
    let projected = output.as_view().block(0, 0, active, output.cols())?;
    let target = match bound {
        Some(bound) => FiniteMatrixView::from_validated(projected, bound),
        None => FiniteMatrixView::from_workspace(projected),
    };
    let prior = target.maximum;
    kernel::gemm_workspace_with_context(
        kernel::WorkspaceGemmProblem {
            left: vectors.block(0, 0, active, vectors.cols())?,
            right: target,
            output: work.try_view_mut()?,
            config: GemmConfig {
                left_op: GemmOp::Adjoint,
                ..GemmConfig::ordinary()
            },
        },
        policy,
        workspace,
        context,
    )?;
    // T has an explicitly zero-filled unused triangle, established by
    // form_t. TRMM therefore shares the bounded packed GEMM child without
    // another input scan or public transactional copy pipeline.
    workspace.require_product(work.rows(), triangular.cols(), work.cols(), policy, context)?;
    kernel::structured_product::with_candidate(
        workspace,
        work.rows() * work.cols(),
        |scratch, workspace| {
            #[cfg(feature = "blas")]
            if let Some(dispatch) = context.blas() {
                let capabilities = dispatch.capabilities()?;
                let layout = crate::blas::dispatch::preferred_layout(&capabilities);
                let values = work.as_slice();
                for row in 0..work.rows() {
                    for col in 0..work.cols() {
                        scratch[crate::blas::dispatch::matrix_index(
                            layout,
                            work.rows(),
                            work.cols(),
                            row,
                            col,
                        )] = values[row * work.cols() + col];
                    }
                }
                let triangle = match direction {
                    Direction::Forward => scioxide_core::Triangle::Upper,
                    Direction::Backward => scioxide_core::Triangle::Lower,
                };
                let route =
                    crate::blas::dispatch::triangular(crate::blas::dispatch::BlasTriangular {
                        accuracy: context.accuracy(),
                        triangular: triangular.as_view(),
                        op: operation_kind,
                        packed: workspace.scalar_mut(kernel::PACKED_A),
                        candidate: scratch,
                        rows: work.rows(),
                        cols: work.cols(),
                        side: crate::MatrixSide::Left,
                        triangle,
                        diagonal: scioxide_core::DiagonalKind::Explicit,
                        alpha: S::ONE,
                        solve: false,
                        dispatch,
                        capabilities: &capabilities,
                        workers: kernel::provider_worker_bound::<S>(
                            work.rows(),
                            triangular.rows(),
                            work.cols(),
                            policy,
                            context,
                        )?,
                    })?;
                if route == crate::blas::dispatch::BlasRoute::Provider {
                    for row in 0..work.rows() {
                        for col in 0..work.cols() {
                            let value = scratch[crate::blas::dispatch::matrix_index(
                                layout,
                                work.rows(),
                                work.cols(),
                                row,
                                col,
                            )];
                            let index = row * work.cols() + col;
                            work.as_mut_slice()[index] = value;
                        }
                    }
                    return Ok(());
                }
            }
            kernel::gemm_workspace_with_context(
                kernel::WorkspaceGemmProblem {
                    left: triangular,
                    right: FiniteMatrixView::from_workspace(work.try_view()?),
                    output: MatrixViewMut::from_slice(work.rows(), work.cols(), scratch)?,
                    config: GemmConfig {
                        left_op: operation_kind,
                        ..GemmConfig::ordinary()
                    },
                },
                policy,
                workspace,
                context,
            )?;
            work.as_mut_slice().copy_from_slice(scratch);
            Ok(())
        },
    )?;
    kernel::gemm_workspace_with_context(
        kernel::WorkspaceGemmProblem {
            left: vectors,
            right: FiniteMatrixView::from_workspace(work.try_view()?),
            output: output.reborrow(),
            config: GemmConfig {
                alpha: -S::ONE,
                beta: S::ONE,
                prior: Some(prior),
                ..GemmConfig::ordinary()
            },
        },
        policy,
        workspace,
        context,
    )
}

/// Reflector group width of every blocked compact-WY application, independent
/// of scalar and ISA dispatch.
pub(crate) const BLOCK_WIDTH: usize = 32;

/// Product of packed column reflectors `Q = H(0) H(1) ... H(k-1)` that
/// left-multiplies a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Product {
    /// `Q C`: groups run last to first with forward block factors.
    Q,
    /// `Q^H C`: groups run first to last with backward block factors built
    /// from conjugated coefficients.
    Adjoint,
}

/// Target of a blocked application and the product it receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// A general matrix receives the product over every column.
    Matrix(Product),
    /// `Q C` with only this many leading rows initially nonzero. After the
    /// last reflector group acts, the target is treated as fully populated.
    Padded(usize),
    /// The identity becomes `Q`. Below the unit row of reflector `j` every
    /// column left of that row is still zero, so the reflector only reaches
    /// columns from its unit row on. This forms `Q` explicitly.
    Identity,
}

impl Target {
    /// First column that the reflector with unit row `unit_row` can change.
    pub(crate) const fn first_column(self, unit_row: usize) -> usize {
        match self {
            Self::Matrix(_) | Self::Padded(_) => 0,
            Self::Identity => unit_row,
        }
    }

    const fn product(self) -> Product {
        match self {
            Self::Matrix(product) => product,
            Self::Identity | Self::Padded(_) => Product::Q,
        }
    }
}

/// Packed column reflectors: `H(j)` has its unit at row `j + offset` of
/// column `j`, its tail below it and coefficient `tau[j]`. `factors` may be
/// an adjoint view, which conjugates the stored tails.
#[derive(Clone, Copy)]
pub(crate) struct Packed<'a, S: NumericScalar> {
    pub(crate) factors: MatrixView<'a, S>,
    pub(crate) tau: &'a [S],
    pub(crate) offset: usize,
}

/// Compact-WY block storage with capacity for a
/// `rows x BLOCK_WIDTH` V, a `BLOCK_WIDTH x BLOCK_WIDTH` T and a
/// `BLOCK_WIDTH x cols` product.
pub(crate) struct CompactWy<'s, M> {
    pub(crate) vectors: &'s mut M,
    pub(crate) triangular: &'s mut M,
    pub(crate) work: &'s mut M,
}

/// Left-multiply `output` by `Q` or `Q^H` of packed column reflectors in
/// aligned groups of [`BLOCK_WIDTH`].
///
/// Group `g` aggregates reflectors `gW..min((g + 1)W, k)` in both products,
/// so `Q` and `Q^H` share identical block factors. Each group updates only
/// the rows from its first unit row on and, for [`Target::Identity`], only
/// the columns from that row on. Execution stops are checked before every
/// group; storage is resized within the capacity lent by the caller.
///
/// # Errors
/// Returns `DimensionMismatch` unless `output` has the rows of `factors` and
/// every unit row lies inside them (an empty `tau` has none), plus storage,
/// execution and arithmetic errors of the block products. A failure can leave
/// `output` partially updated; callers apply it to staging storage.
/// Row-contiguous targets with both extents at most 64, or at most eight
/// reflectors and eight columns, use elementary panel applications in the same
/// reflector order, avoiding compact-WY preparation for narrow tall problems.
pub(crate) fn apply_blocked<S: NumericScalar, M: MatrixStorage<S>>(
    packed: Packed<'_, S>,
    target: Target,
    mut output: MatrixViewMut<'_, S>,
    blocks: CompactWy<'_, M>,
    workspace: &mut LinalgWorkspace<'_, S>,
    execution: kernel::KernelExecution<KernelContext<'_>>,
) -> Result<(), SciOxideError> {
    let Packed {
        factors,
        tau,
        offset,
    } = packed;
    let CompactWy {
        vectors,
        triangular,
        work,
    } = blocks;
    let kernel::KernelExecution { policy, context } = execution;
    let rows = factors.rows();
    let units_fit = tau.is_empty() || tau.len() + offset <= rows;
    if output.rows() != rows || !units_fit {
        return Err(SciOxideError::dimension_mismatch(
            [rows, tau.len() + offset],
            [output.rows(), output.cols()],
        ));
    }
    let cols = output.cols();
    let product = target.product();
    let elementary =
        ((rows <= 64 && cols <= 64) || (tau.len() <= 8 && cols <= 8)) && output.col_stride() == 1;
    #[cfg(feature = "blas")]
    let elementary = elementary && context.blas().is_none();
    if elementary {
        // Short factors do not amortize forming T and three checked matrix
        // products. Replay the same reflectors with contiguous panel updates.
        let mut maximum = FiniteMatrixView::from_workspace(output.as_view()).maximum;
        let invariant = maximum
            * S::Real::ONE.scalbn((usize::BITS - rows.leading_zeros() + 1).div_ceil(2) as i32 + 1);
        let invariant = invariant.is_finite().then_some(invariant);
        for step in 0..tau.len() {
            context.execution().check()?;
            let index = match product {
                Product::Q => tau.len() - 1 - step,
                Product::Adjoint => step,
            };
            let unit = index + offset;
            vectors.try_resize_flat(rows - unit, 1)?;
            vectors.as_mut_slice()[0] = S::ONE;
            for (value, row) in vectors
                .as_mut_slice()
                .iter_mut()
                .skip(1)
                .zip(unit + 1..rows)
            {
                *value = factors.get_element(row, index);
            }
            let coefficient = match product {
                Product::Q => tau[index],
                Product::Adjoint => tau[index].conj(),
            };
            let first = target.first_column(unit);
            maximum = apply_panel(
                FiniteVectorView::from_validated(
                    crate::VectorView::from_slice(vectors.as_slice()),
                    S::Real::TWO,
                ),
                coefficient,
                output.block_mut(unit, first, rows - unit, cols - first)?,
                invariant.unwrap_or(maximum),
                work,
            )?;
        }
        return Ok(());
    }
    let direction = match product {
        Product::Q => Direction::Forward,
        Product::Adjoint => Direction::Backward,
    };
    let mut group = |index: usize| -> Result<(), SciOxideError> {
        context.execution().check()?;
        let start = index * BLOCK_WIDTH;
        let end = (start + BLOCK_WIDTH).min(tau.len());
        pack(
            factors,
            tau,
            start,
            end,
            offset,
            direction,
            &mut *vectors,
            &mut *triangular,
        )?;
        let unit_row = start + offset;
        let first = target.first_column(unit_row);
        apply_block(
            &*vectors,
            &*triangular,
            output.block_mut(unit_row, first, rows - unit_row, cols - first)?,
            None,
            match target {
                Target::Padded(leading) if end == tau.len() => {
                    Some(leading.saturating_sub(unit_row))
                }
                _ => None,
            },
            GemmOp::None,
            direction,
            &mut *work,
            policy,
            workspace,
            context,
        )
    };
    let groups = 0..tau.len().div_ceil(BLOCK_WIDTH);
    match product {
        Product::Q => groups.rev().try_for_each(&mut group),
        Product::Adjoint => { groups }.try_for_each(&mut group),
    }
}

/// Plain `X := H X` with `H = I - tau v v^H` on rows `first..first + v.len()`
/// of row-major `data` (row stride `stride`) over `columns`.
///
/// For the short explicit reflectors of bulge chases on normalized workspace
/// values, where no intermediate can overflow; callers check finiteness once.
/// The generated vector has two or three entries with `v[0] = 1`. Pass
/// `conj(tau)` for `H^H`; zero tau also permits a zero skipped-step record.
/// Preweight the update coefficients once per panel, keeping the elementary
/// two/three-lane arithmetic explicit for vectorization across columns.
#[inline(always)]
pub(crate) fn reflect_rows<S: NumericScalar>(
    v: &[S],
    tau: S,
    data: &mut [S],
    stride: usize,
    first: usize,
    columns: core::ops::Range<usize>,
) -> Result<(), SciOxideError> {
    if tau == S::ZERO {
        return Ok(());
    }
    debug_assert!((v.len() == 2 || v.len() == 3) && v[0] == S::ONE);
    let third = v.len() == 3;
    let v1 = v[1];
    let v2 = if third { v[2] } else { S::ZERO };
    let w1 = tau * v1;
    let w2 = tau * v2;
    // Disjoint contiguous rows expose independent columns to vectorization;
    // no indexed alias or bounds checks remain in the elementary update.
    let (before, c) = crate::workspace::split_mut(data, (first + 2) * stride)?;
    let (before, b) = crate::workspace::split_mut(before, (first + 1) * stride)?;
    let a = &mut before[first * stride + columns.start..first * stride + columns.end];
    let b = &mut b[columns.clone()];
    if third {
        let c = &mut c[columns];
        for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
            let dot = (*a + v1.conj() * *b) + v2.conj() * *c;
            *c = *c - w2 * dot;
            *b = *b - w1 * dot;
            *a = *a - tau * dot;
        }
    } else {
        for (a, b) in a.iter_mut().zip(b) {
            let dot = *a + v1.conj() * *b;
            *b = *b - w1 * dot;
            *a = *a - tau * dot;
        }
    }

    Ok(())
}

/// Plain `X := X H` with `H = I - tau v v^H` on columns
/// `first..first + v.len()` of row-major `data` over `rows`; the arithmetic
/// contract is that of [`reflect_rows`].
#[inline(always)]
pub(crate) fn reflect_columns<S: NumericScalar>(
    v: &[S],
    tau: S,
    data: &mut [S],
    stride: usize,
    first: usize,
    rows: core::ops::Range<usize>,
) {
    if tau == S::ZERO {
        return;
    }
    debug_assert!((v.len() == 2 || v.len() == 3) && v[0] == S::ONE);
    let third = v.len() == 3;
    let v1 = v[1];
    let v2 = if third { v[2] } else { S::ZERO };
    let w1 = tau * v1;
    let w2 = tau * v2;
    for row in rows {
        let start = row * stride + first;
        let x = &mut data[start..start + v.len()];
        if let [a, b, c] = x {
            let dot = (tau * *a + w1 * *b) + w2 * *c;
            *a = *a - dot;
            *b = *b - dot * v1.conj();
            *c = *c - dot * v2.conj();
        } else if let [a, b] = x {
            let dot = tau * *a + w1 * *b;
            *a = *a - dot;
            *b = *b - dot * v1.conj();
        }
    }
}

/// Replay a sweep's deferred right transformations over independent row
/// tiles, preserving step order within every row. For triangular replay, row r
/// takes only steps k >= r; unrestricted vector rows take the entire sweep.
/// The callback applies one recorded step to the supplied active rows. Storage
/// and transformation kinds remain with the sweep; replay needs no scratch.
pub(crate) fn replay<S>(
    data: &mut [S],
    stride: usize,
    rows: core::ops::Range<usize>,
    steps: core::ops::Range<usize>,
    triangular: bool,
    mut apply: impl FnMut(&mut [S], usize, core::ops::Range<usize>),
) {
    // Sixteen independent chains hide short-reflector update latency while
    // retaining a narrow window of each row in cache.
    const CHUNK: usize = 16;
    for start in rows.clone().step_by(CHUNK) {
        let end = (start + CHUNK).min(rows.end);
        let first = if triangular {
            start.max(steps.start)
        } else {
            steps.start
        };
        let block = &mut data[start * stride..end * stride];
        for step in first..steps.end {
            let active = if triangular {
                (step + 1).min(end) - start
            } else {
                end - start
            };
            apply(block, step, 0..active);
        }
    }
}

pub(crate) fn swap_columns<S: NumericScalar>(mut matrix: MatrixViewMut<'_, S>, a: usize, b: usize) {
    for row in 0..matrix.rows() {
        let x = matrix.get_element(row, a);
        let y = matrix.get_element(row, b);
        matrix.set_element(row, a, y);
        matrix.set_element(row, b, x);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::Matrix;

    use super::*;
    use scioxide_core::Complex;

    fn short_generation<S: NumericScalar>() {
        let real = |x| S::Real::try_from_f64_exact(x).unwrap();
        let initial = [
            S::from_real(real(0.25)),
            S::from_components(real(-0.5), if S::IS_REAL { real(0.) } else { real(0.25) }),
            S::from_components(real(1.), if S::IS_REAL { real(0.) } else { real(-0.5) }),
        ];
        for d in [2, 3] {
            for scale in [
                S::Real::MIN_POSITIVE,
                real(0.25),
                S::Real::ONE,
                S::Real::MAX / real(8.),
            ] {
                let mut v = initial.map(|x| crate::rotation::scale_real(x, scale));
                let h = generate(VectorViewMut::from_slice(&mut v[..d])).unwrap();
                let mut x = Matrix::from_vec(d, 1, initial[..d].to_vec()).unwrap();
                apply(
                    FiniteVectorView::from_workspace(crate::VectorView::from_slice(&v[..d])),
                    h.tau.conj(),
                    x.view_mut(),
                )
                .unwrap();
                let bound = real(64.) * S::Real::EPSILON;
                let norm = initial[..d]
                    .iter()
                    .fold(S::Real::ZERO, |n, x| n.hypot(x.abs()));
                assert!((h.beta.abs() / scale - norm).abs() <= bound);
                assert!((x.as_slice()[0] - S::from_real(h.beta / scale)).abs() <= bound);
                assert!(x.as_slice()[1..].iter().all(|x| x.abs() <= bound));
            }
        }
    }

    #[test]
    fn short_generation_preserves_annihilation_across_binary_frames() {
        short_generation::<f32>();
        short_generation::<f64>();
        short_generation::<Complex<f32>>();
        short_generation::<Complex<f64>>();
    }

    fn bounded_short_generation<R: RealScalar>() {
        let real = |x| R::try_from_f64_exact(x).unwrap();
        let tiny = R::MIN_POSITIVE.scalbn(1 - R::MANTISSA_DIGITS as i32);
        for initial in [
            [real(0.5), real(-0.25), real(0.25)],
            [real(-0.5), real(0.25), real(-0.25)],
            [-R::ZERO, R::ONE, tiny],
            [R::ONE, tiny, tiny],
            [R::ONE, R::ZERO, -R::ZERO],
            [-R::ONE, -R::ZERO, R::ZERO],
        ] {
            for d in [2, 3] {
                let scale = initial[..d].iter().fold(R::ZERO, |sum, x| sum + x.abs());
                let mut v = initial.map(|x| x / scale);
                let input = v;
                let h = generate_bounded_real(&mut v[..d]).unwrap();
                let norm = input[..d].iter().fold(R::ZERO, |sum, x| sum.hypot(*x));
                let dot = v[..d]
                    .iter()
                    .zip(&input[..d])
                    .fold(R::ZERO, |sum, (v, x)| sum + *v * *x);
                let bound = real(32.) * R::EPSILON;
                for j in 0..d {
                    let transformed = input[j] - h.tau * v[j] * dot;
                    let expected = if j == 0 { h.beta } else { R::ZERO };
                    assert!((transformed - expected).abs() <= bound);
                }
                assert!((h.beta.abs() - norm).abs() <= bound);
                if h.tau != R::ZERO {
                    let square = v[..d].iter().fold(R::ZERO, |sum, x| sum + *x * *x);
                    assert!((h.tau * square - R::TWO).abs() <= bound);
                }
            }
        }
    }

    #[test]
    fn bounded_bulge_reflectors_annihilate_and_preserve_orthogonality() {
        bounded_short_generation::<f32>();
        bounded_short_generation::<f64>();
    }

    #[test]
    fn deferred_replay_matches_elementary_reflectors_across_row_tiles() {
        for n in [3, 7, 8, 9, 16, 17, 31, 32, 33] {
            let mut records = alloc::vec::Vec::new();
            for k in 1..n - 1 {
                let d = 3.min(n - k);
                let mut v = [
                    Complex::new(1., 0.5),
                    Complex::new(-0.25, 0.75),
                    Complex::new(0.5, -0.25),
                ];
                let h = generate(VectorViewMut::from_slice(&mut v[..d])).unwrap();
                records.push((v, if k % 3 == 0 { Complex::ZERO } else { h.tau }));
            }
            for triangular in [false, true] {
                let mut actual = Matrix::from_vec(
                    n,
                    n,
                    (0..n * n)
                        .map(|i| Complex::new((i % 7) as f64 / 8., (i % 5) as f64 / 16.))
                        .collect(),
                )
                .unwrap();
                let mut expected = actual.clone();
                replay(
                    actual.as_mut_slice(),
                    n,
                    1..n,
                    1..n - 1,
                    triangular,
                    |data, k, rows| {
                        let (v, tau) = &records[k - 1];
                        reflect_columns(&v[..3.min(n - k)], *tau, data, n, k, rows);
                    },
                );
                for (k, (v, tau)) in (1..n - 1).zip(&records) {
                    let end = if triangular { k + 1 } else { n };
                    let d = 3.min(n - k);
                    apply(
                        FiniteVectorView::from_workspace(crate::VectorView::from_slice(&v[..d]))
                            .conjugate(),
                        *tau,
                        expected
                            .view_mut()
                            .block_mut(1, k, end - 1, d)
                            .unwrap()
                            .transpose(),
                    )
                    .unwrap();
                }
                for (&a, &b) in actual.as_slice().iter().zip(expected.as_slice()) {
                    assert!((a - b).norm() <= 32. * n as f64 * f64::EPSILON);
                }
            }
        }
    }

    #[test]
    fn real_tail_and_safe_minimum() {
        for scale in [f64::from_bits(1), 1e-300, 1.0, 1e300] {
            for initial in [[3.0, 4.0, 0.0], [-3., 4., 0.], [0., 3., 4.], [3., 0., 0.]] {
                let mut vector = initial.map(|x| x * scale);
                let h = generate(VectorViewMut::from_slice(&mut vector)).unwrap();
                let mut product = Matrix::from_vec(3, 1, initial.to_vec()).unwrap();
                apply(
                    FiniteVectorView::from_workspace(crate::VectorView::from_slice(&vector)),
                    h.tau,
                    product.view_mut(),
                )
                .unwrap();
                let x = product.as_slice();
                assert!(x[1].abs().max(x[2].abs()) < 1e-13);
                assert!(
                    (x[0] - h.beta / scale).abs()
                        < if scale < f64::MIN_POSITIVE {
                            1.0
                        } else {
                            1e-13
                        }
                );
            }
        }
    }

    #[test]
    fn complex_phase_is_not_a_hermitian_reflector() {
        for scale in [1e-300, 1.0, 1e300] {
            for initial in [
                [Complex::new(0., 2.), Complex::new(0., 0.)],
                [Complex::new(2., -3.), Complex::new(-4., 1.)],
            ] {
                let mut v = initial.map(|x| x * scale);
                let h = generate(VectorViewMut::from_slice(&mut v)).unwrap();
                let mut x = Matrix::from_vec(2, 1, initial.to_vec()).unwrap();
                apply(
                    FiniteVectorView::from_workspace(crate::VectorView::from_slice(&v)),
                    h.tau.conj(),
                    x.view_mut(),
                )
                .unwrap();
                assert!(x.as_slice()[1].norm() < 1e-13);
                assert!((x.as_slice()[0] - Complex::new(h.beta / scale, 0.)).norm() < 1e-13);
                assert!((h.tau - Complex::new(1., 0.)).norm() <= 1. + 1e-14);
            }
        }
    }

    #[test]
    fn binary32_subnormal_and_overflow() {
        for scale in [f32::from_bits(1), 1e-30, 1., 1e30] {
            let mut v = [3. * scale, 4. * scale];
            let h = generate(VectorViewMut::from_slice(&mut v)).unwrap();
            assert!((h.beta / scale + 5.).abs() <= 1e-5);
            assert!((h.tau - 1.6).abs() < 1e-5);
        }
        let mut v = [f64::MAX, f64::MAX];
        assert!(generate(VectorViewMut::from_slice(&mut v)).is_err());
        let mut zero = [0.0_f64; 3];
        let h = generate(VectorViewMut::from_slice(&mut zero)).unwrap();
        assert_eq!(h.tau, 0.);
        assert_eq!(h.beta, 0.);
    }
    #[test]
    fn packed_reflector_rows_preserve_offsets_and_logical_transforms() -> Result<(), SciOxideError>
    {
        let packed = Matrix::from_vec(
            9,
            9,
            (0..81)
                .map(|i| Complex::new(i as f64 / 81.0, (i % 5) as f64 / 10.0))
                .collect(),
        )?;
        let tau = [Complex::ZERO; 9];
        for view in [
            packed.view(),
            packed.view().transpose(),
            packed.view().adjoint(),
        ] {
            for (start, offset, width) in [(0, 0, 0), (1, 0, 3), (2, 1, 4), (5, 1, 3)] {
                let mut vectors = Matrix::from_vec(9, 9, alloc::vec![Complex::ONE; 81])?;
                let mut triangular = Matrix::empty();
                pack(
                    view,
                    &tau,
                    start,
                    start + width,
                    offset,
                    Direction::Forward,
                    &mut vectors,
                    &mut triangular,
                )?;
                assert_eq!(
                    (vectors.rows(), vectors.cols()),
                    (9 - start - offset, width)
                );
                for i in 0..vectors.rows() {
                    for j in 0..width {
                        let expected = if i == j {
                            Complex::ONE
                        } else if i < j {
                            Complex::ZERO
                        } else {
                            view.get_element(start + offset + i, start + j)
                        };
                        assert_eq!(vectors.get_element(i, j), expected);
                    }
                }
                assert!(
                    triangular
                        .as_slice()
                        .iter()
                        .all(|&value| value == Complex::ZERO)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn compact_wy_matches_elementary_products_in_both_orders() {
        let mut v = Matrix::zeros(4, 3).unwrap();
        let mut tau = [Complex::new(0.0_f64, 0.0); 3];
        for (j, t) in tau.iter_mut().enumerate() {
            let mut column = [
                Complex::new(j as f64 + 1., 0.5),
                Complex::new(2., -1.),
                Complex::new(-0.5, 1.),
                Complex::new(0.25, j as f64),
            ];
            *t = generate(VectorViewMut::from_slice(&mut column))
                .unwrap()
                .tau;
            for (i, &x) in column.iter().enumerate() {
                v.as_mut_slice()[i * 3 + j] = x;
            }
        }
        let initial = Matrix::from_vec(
            4,
            3,
            (0..12)
                .map(|i| Complex::new(i as f64 / 7., (i % 3) as f64))
                .collect(),
        )
        .unwrap();
        for direction in [Direction::Forward, Direction::Backward] {
            for zero_middle in [false, true] {
                let mut tau = tau;
                if zero_middle {
                    tau[1] = Complex::ZERO;
                }
                let mut t = Matrix::empty();
                form_t(v.view(), tau.iter().copied(), direction, &mut t).unwrap();
                let mut expected = initial.clone();
                for step in 0..3 {
                    let j = if direction == Direction::Forward {
                        2 - step
                    } else {
                        step
                    };
                    apply(
                        FiniteVectorView::from_workspace(v.view().col(j).unwrap()),
                        tau[j],
                        expected.view_mut(),
                    )
                    .unwrap();
                }
                let mut actual = initial.view().transpose().try_to_matrix().unwrap();
                apply_block(
                    &v,
                    &t,
                    actual.view_mut().transpose(),
                    None,
                    None,
                    GemmOp::None,
                    direction,
                    &mut Matrix::empty(),
                    ExecutionPolicy::Sequential,
                    &mut LinalgWorkspace::workspace(&crate::DenseWorkspaceRequest::structured(
                        4,
                        4,
                        4,
                        &crate::LinalgExecutionOptions::new(),
                    ))
                    .unwrap(),
                    KernelContext::sequential(),
                )
                .unwrap();
                for i in 0..4 {
                    for j in 0..3 {
                        assert!(
                            (actual.get_element(j, i) - expected.get_element(i, j)).norm() < 1e-12
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn plain_short_reflections_match_the_checked_kernel() -> Result<(), SciOxideError> {
        let n = 6;
        let data: alloc::vec::Vec<Complex<f64>> = (0..n * n)
            .map(|i| Complex::new((i % 5) as f64 * 0.3 - 0.6, (i % 4) as f64 * 0.2 - 0.3))
            .collect();
        let mut v = [
            Complex::new(0.5, 0.25),
            Complex::new(-1.0, 0.5),
            Complex::new(0.25, -0.75),
        ];
        let h = generate(VectorViewMut::from_slice(&mut v)).unwrap();
        let vector = FiniteVectorView::from_workspace(crate::VectorView::from_slice(&v));
        // Left: X := H^H X on rows 2..5; right: X := X H on columns 1..4.
        let mut plain = data.clone();
        let mut checked = data;
        reflect_rows(&v, h.tau.conj(), &mut plain, n, 2, 0..n)?;
        apply(
            vector,
            h.tau.conj(),
            MatrixViewMut::from_slice(n, n, &mut checked)
                .unwrap()
                .block_mut(2, 0, 3, n)
                .unwrap(),
        )
        .unwrap();
        reflect_columns(&v, h.tau, &mut plain, n, 1, 0..n);
        apply(
            vector.conjugate(),
            h.tau,
            MatrixViewMut::from_slice(n, n, &mut checked)
                .unwrap()
                .block_mut(0, 1, n, 3)
                .unwrap()
                .transpose(),
        )
        .unwrap();
        plain.iter().zip(&checked).for_each(|(a, b)| {
            assert!((*a - *b).norm() < 1e-15, "{a:?} {b:?}");
        });
        Ok(())
    }

    #[test]
    fn swaps_logical_columns_without_touching_padding() {
        let mut entries = [1.0_f64, 2., 99., 3., 4., 99.];
        swap_columns(
            MatrixViewMut::from_slice_with_strides(2, 2, &mut entries, 3, 1).unwrap(),
            0,
            1,
        );
        assert_eq!(entries, [2., 1., 99., 4., 3., 99.]);
        swap_columns(
            MatrixViewMut::<f64>::from_slice(0, 2, &mut []).unwrap(),
            0,
            1,
        );
    }
}
