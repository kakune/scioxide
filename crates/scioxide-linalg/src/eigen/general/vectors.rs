//! Schur eigenvector recovery, back-transformation, residuals and conditioning.

use super::*;

/// Eigenvalue block containing Schur index `index`: its start, size, and
/// whether `index` is the conjugate (second) member of a real pair.
pub(super) fn member<W: NumericScalar>(
    values: &[Complex<W::Real>],
    index: usize,
) -> (usize, usize, bool) {
    if !W::IS_REAL {
        (index, 1, false)
    } else if values[index].im > W::Real::ZERO {
        (index, 2, false)
    } else if values[index].im < W::Real::ZERO {
        (index - 1, 2, true)
    } else {
        (index, 1, false)
    }
}

/// Eigenvalue blocks `(start, size)` in Schur order.
fn blocks<W: NumericScalar>(
    values: &[Complex<W::Real>],
) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut start = 0;
    core::iter::from_fn(move || {
        (start < values.len()).then(|| {
            let (_, size, _) = member::<W>(values, start);
            let block = (start, size);
            start += size;
            block
        })
    })
}

/// Complex entry `row` of the vector of block `(start, size)`.
pub(super) fn entry<W: NumericScalar>(
    vectors: &[W],
    n: usize,
    (start, size): (usize, usize),
    row: usize,
) -> Complex<W::Real> {
    let value = crate::eigen::into_complex(vectors[start * n + row]);
    if size == 2 {
        Complex::new(value.re, vectors[(start + 1) * n + row].real_component())
    } else {
        value
    }
}

/// Euclidean norm of the complex vector of a block. A real pair stores its
/// real and imaginary parts as adjacent columns, so the block is contiguous.
fn vector_norm<W: NumericScalar>(
    vectors: &[W],
    n: usize,
    (start, size): (usize, usize),
) -> Result<W::Real, SciOxideError> {
    crate::kernel::finite::norm(&vectors[start * n..(start + size) * n])
}

/// Distance below which a computed eigenvector is numerically dependent on
/// the earlier ones: `max(n,1) * sqrt(threshold(1))` at machine precision.
fn independence_cutoff<R: RealScalar>(n: usize) -> Result<R, SciOxideError> {
    Ok(Tolerance::<R>::from_machine_epsilon()
        .threshold(R::ONE)?
        .sqrt()
        * crate::checked::rounded_count::<R>(n.max(1))?)
}

/// Recover normalized right vectors from an existing, unbalanced Schur form.
/// Scratch is four order-squared arrays plus two vectors. Reuses the dense
/// eigensolver's substitution, phase and independence policies without a second QR.
pub(crate) fn right_vectors_from_schur<S: NumericScalar>(
    factor: &crate::SchurDecomposition<S>,
    output: &mut crate::Matrix<Complex<S::Real>>,
    scratch: &mut [S],
    phases: &mut [Complex<S::Real>],
    context: KernelContext<'_>,
) -> Result<(), SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let n = factor.t().rows();
    output.try_resize_flat(n, n)?;
    let (t, rest) = crate::workspace::split_mut(scratch, n * n)?;
    let (x, rest) = crate::workspace::split_mut(rest, n * n)?;
    let (mapped, rest) = crate::workspace::split_mut(rest, n * n)?;
    let (basis, work) = crate::workspace::split_mut(rest, n * n)?;
    t.copy_from_slice(factor.t().as_slice());
    let scale = crate::kernel::standard::slice_maximum(t);
    super::normalize_slice(t, scale);
    let minimum = Tolerance::<S::Real>::from_machine_epsilon()
        .threshold(crate::kernel::standard::slice_maximum(t))?
        .max(S::Real::MIN_POSITIVE);
    schur_vectors::eigenvectors(t, n, Side::Right, minimum, x, work, context.dispatch())?;
    for (source, target) in x
        .chunks_exact(n.max(1))
        .zip(mapped.chunks_exact_mut(n.max(1)))
    {
        crate::kernel::level2::gemv_bounded(
            FiniteMatrixView::from_workspace(factor.q().view()),
            crate::FiniteVectorView::from_workspace(crate::VectorView::from_slice(source)),
            target,
            ExecutionPolicy::Sequential,
            context,
        )?;
    }
    validate_vectors(mapped)?;
    independence(
        mapped,
        basis,
        factor.eigenvalues(),
        n,
        independence_cutoff(n)?,
        context.dispatch(),
    )?;
    normalizing_phases(mapped, factor.eigenvalues(), n, &mut phases[..n])?;
    for (start, size) in blocks::<S>(factor.eigenvalues()) {
        for row in 0..n {
            let value = entry(mapped, n, (start, size), row) * phases[start];
            output.as_mut_slice()[row * n + start] = value;
            if size == 2 {
                output.as_mut_slice()[row * n + start + 1] = value.conj();
            }
        }
    }
    Ok(())
}

/// Everything after the Schur form: vectors of every requested side,
/// back-transformation, balancing, residuals, independence, phases,
/// conditions and the output order.
pub(super) fn vectors_candidate<S, W, R>(
    matrix: FiniteMatrixView<'_, S>,
    options: &GeneralEigenOptions<R>,
    stage: &mut Stage<'_, W>,
    small: &mut Small<'_, R>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: NumericScalar<Real = R>,
    W: NumericScalar<Real = R>,
    R: RealScalar,
    Complex<R>: NumericScalar<Real = R>,
{
    let n = matrix.rows();
    let selection = options.eigenvectors();
    let policy = options.execution_options().execution_policy();
    let mut index = 0;
    for_each_schur_block(stage.t, n, |_, block| {
        small.values[index..index + block.len()].copy_from_slice(block);
        index += block.len();
    })?;
    let context = KernelContext::new(solve.execution()).with_accuracy(Accuracy::Standard);
    let minimum = Tolerance::<R>::from_machine_epsilon()
        .threshold(crate::kernel::standard::slice_maximum(stage.t))?
        .max(R::MIN_POSITIVE);
    let cutoff = independence_cutoff::<R>(n)?;
    let requested = || {
        [Side::Right, Side::Left]
            .into_iter()
            .filter(move |side| match side {
                Side::Right => selection.includes_right(),
                Side::Left => selection.includes_left(),
            })
    };
    for side in requested() {
        solve.execution().check()?;
        let target = match side {
            Side::Right => &mut *stage.right,
            Side::Left => &mut *stage.left,
        };
        schur_vectors::eigenvectors(
            stage.t,
            n,
            side,
            minimum,
            stage.x,
            stage.work,
            context.dispatch(),
        )?;
        back_transform(
            stage.x,
            stage.z,
            target,
            n,
            side,
            policy,
            &mut stage.gemm,
            context,
        )?;
        validate_vectors(target)?;
        // Diagonal similarity may make a perfectly independent basis nearly
        // parallel in input coordinates. Test the balanced basis before xGEBAK;
        // the original-coordinate residual remains authoritative afterwards.
        independence(target, stage.x, small.values, n, cutoff, context.dispatch())?;
        undo_balance(
            target,
            small.values,
            small.scale,
            small.permutation,
            side,
            stage.work,
        )?;
        validate_vectors(target)?;
    }
    // The Schur factor is no longer needed; its storage holds the residual
    // product. Keep values in Schur units so real-pair block markers survive
    // even if a very small imaginary part underflows on scale restoration.
    for side in requested() {
        solve.execution().check()?;
        let (vectors, residuals, phases) = match side {
            Side::Right => (
                &*stage.right,
                &mut *small.right_residuals,
                &mut *small.right_phases,
            ),
            Side::Left => (
                &*stage.left,
                &mut *small.left_residuals,
                &mut *small.left_phases,
            ),
        };
        relative_residuals(
            stage.a,
            vectors,
            (small.values, stage.balance_scale),
            n,
            side,
            stage.t,
            residuals,
            (policy, &mut stage.gemm, context),
        )?;
        validate_relative_residuals(residuals, options)?;
        normalizing_phases(vectors, small.values, n, phases)?;
    }
    if selection == GeneralEigenvectorSelection::Both {
        eigenvalue_conditions(
            stage.right,
            stage.left,
            small.values,
            n,
            small.eigenvalue_conditions,
        )?;
    }
    small.eigenvalues.copy_from_slice(small.values);
    rescale_complex_values(small.eigenvalues, [stage.balance_scale, matrix.maximum])?;
    validate_complex_output(small.eigenvalues)?;
    let norm = stage.a.chunks_exact(n.max(1)).fold(R::ZERO, |norm, row| {
        norm.max(row.iter().fold(R::ZERO, |sum, value| sum + value.abs()))
    });
    compute_eigenvector_conditions(
        small.values,
        stage.balance_scale,
        norm,
        small.eigenvector_conditions,
    )?;
    small
        .order
        .iter_mut()
        .enumerate()
        .for_each(|(position, index)| *index = position);
    let eigenvalues = &*small.eigenvalues;
    small.order.sort_unstable_by(|&left, &right| {
        compare_complex(eigenvalues[left], eigenvalues[right], options.ordering())
            .then(left.cmp(&right))
    });
    Ok(())
}

/// Rows of `target` become `Z x` for the Schur-coordinate rows `x` of
/// `schur` (`target = schur Z^T`), in blocks of [`BLOCK`] vectors restricted
/// to the columns their rows can reach: right vectors end at the block's
/// last pair partner, left vectors start at its first.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn back_transform<W: NumericScalar>(
    schur: &[W],
    z: &[W],
    target: &mut [W],
    n: usize,
    side: Side,
    policy: ExecutionPolicy,
    gemm: &mut LinalgWorkspace<'_, W>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    for start in (0..n).step_by(BLOCK) {
        let rows = BLOCK.min(n - start);
        let (first, last) = match side {
            Side::Right => (0, (start + rows + 1).min(n)),
            Side::Left => (start.saturating_sub(1), n),
        };
        gemm_workspace_with_context(
            WorkspaceGemmProblem {
                left: FiniteMatrixView::from_workspace(
                    MatrixView::from_slice(n, n, schur)?.block(start, first, rows, last - first)?,
                ),
                right: FiniteMatrixView::from_workspace(MatrixView::from_slice(n, n, z)?.block(
                    0,
                    first,
                    n,
                    last - first,
                )?),
                output: MatrixViewMut::from_slice(n, n, target)?.block_mut(start, 0, rows, n)?,
                config: GemmConfig {
                    right_op: GemmOp::Transpose,
                    ..GemmConfig::ordinary()
                },
            },
            policy,
            gemm,
            context,
        )?;
    }
    Ok(())
}

/// Map balanced-coordinate vectors back to the input: balancing
/// produced `B = D^{-1} P^T A P D`, so right vectors are `P D x` and left
/// vectors `P D^{-1} y`. A common binary scale per vector (shared by a real
/// conjugate pair) bounds the result before rounding, so neither an intermediate
/// product nor the later normalization can overflow. `row` holds `n` elements.
fn undo_balance<W: NumericScalar>(
    vectors: &mut [W],
    values: &[Complex<W::Real>],
    scale: &[W::Real],
    permutation: &[usize],
    side: Side,
    row: &mut [W],
) -> Result<(), SciOxideError> {
    use scioxide_core::Scaled;
    let n = scale.len();
    let component = |value, factor| {
        let value = Scaled::from_value(value)?;
        let factor = Scaled::from_value(factor)?;
        match side {
            Side::Right => value.mul(factor),
            Side::Left => value.div(factor),
        }
    };
    for (start, size) in blocks::<W>(values) {
        let block = &mut vectors[start * n..(start + size) * n];
        let common = block
            .iter()
            .enumerate()
            .try_fold(None, |common, (i, value)| {
                let magnitude = crate::kernel::standard::magnitude(*value);
                let value = component(magnitude, scale[i % n])?;
                Ok::<_, SciOxideError>(if value.is_zero() {
                    common
                } else {
                    Some(common.map_or(value.exponent(), |e: i64| e.max(value.exponent())))
                })
            })?
            .unwrap_or(0);
        for vector in block.chunks_exact_mut(n) {
            for ((&value, &factor), &original) in vector.iter().zip(scale).zip(permutation) {
                let project = |value| {
                    let value = component(value, factor)?;
                    Scaled::from_parts(value.significand(), value.exponent() - common)?.round()
                };
                row[original] = W::from_components(
                    project(value.real_component())?,
                    project(value.imaginary_component())?,
                );
            }
            vector.copy_from_slice(&row[..n]);
        }
    }
    Ok(())
}

fn validate_vectors<W: NumericScalar>(vectors: &[W]) -> Result<(), SciOxideError> {
    vectors
        .iter()
        .all(|value| value.has_finite_components())
        .then_some(())
        .ok_or_else(|| {
            SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "computed eigenvector is not finite",
            )
        })
}

/// Normwise relative residuals `‖A x - μ x‖∞ / ((‖A‖∞ + |μ|) ‖x‖∞)` of the
/// vector rows, with `μ = λ` for right vectors and `A^H`, `μ = conj(λ)` for
/// left vectors, all in normalized units. One product forms
/// `P = A X - X M`: P is pre-filled with `-X M` (a real pair's columns in
/// real form) and the Standard product accumulates `A X` into it.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn relative_residuals<W, R>(
    a: &[W],
    vectors: &[W],
    (values, value_scale): (&[Complex<R>], R),
    n: usize,
    side: Side,
    product: &mut [W],
    residuals: &mut [R],
    (policy, gemm, context): (
        ExecutionPolicy,
        &mut LinalgWorkspace<'_, W>,
        KernelContext<'_>,
    ),
) -> Result<(), SciOxideError>
where
    W: NumericScalar<Real = R>,
    R: RealScalar,
{
    let shift = |index: usize| {
        let value = values[index];
        let value = Complex::new(
            scioxide_core::checked_float_product(value.re, value_scale)?,
            scioxide_core::checked_float_product(value.im, value_scale)?,
        );
        Ok::<_, SciOxideError>(match side {
            Side::Right => value,
            Side::Left => value.conj(),
        })
    };
    for (start, size) in blocks::<W>(values) {
        let mu = shift(start)?;
        for row in 0..n {
            if size == 2 {
                let (re, im) = (
                    vectors[start * n + row].real_component(),
                    vectors[(start + 1) * n + row].real_component(),
                );
                product[row * n + start] = W::from_real(mu.im * im - mu.re * re);
                product[row * n + start + 1] = W::from_real(-(mu.re * im + mu.im * re));
            } else {
                product[row * n + start] =
                    -(W::from_components(mu.re, mu.im) * vectors[start * n + row]);
            }
        }
    }
    gemm_workspace_with_context(
        WorkspaceGemmProblem {
            left: FiniteMatrixView::from_workspace(MatrixView::from_slice(n, n, a)?),
            right: FiniteMatrixView::from_workspace(MatrixView::from_slice(n, n, vectors)?),
            output: MatrixViewMut::from_slice(n, n, product)?,
            config: GemmConfig {
                left_op: match side {
                    Side::Right => GemmOp::None,
                    Side::Left => GemmOp::Adjoint,
                },
                right_op: GemmOp::Transpose,
                beta: W::ONE,
                ..GemmConfig::ordinary()
            },
        },
        policy,
        gemm,
        context,
    )?;
    // ‖A‖∞ is the largest row sum; ‖A^H‖∞ the largest column sum.
    let norm = (0..n).fold(R::ZERO, |norm, outer| {
        norm.max((0..n).fold(R::ZERO, |sum, inner| {
            sum + match side {
                Side::Right => a[outer * n + inner],
                Side::Left => a[inner * n + outer],
            }
            .abs()
        }))
    });
    for (start, size) in blocks::<W>(values) {
        let magnitude = |matrix: &[W], row: usize, column: usize| {
            let re = crate::eigen::into_complex(matrix[row]);
            if size == 2 {
                re.re.hypot(matrix[column].real_component())
            } else {
                re.norm()
            }
        };
        let (residual, vector) = (0..n).fold((R::ZERO, R::ZERO), |(residual, vector), row| {
            (
                residual.max(magnitude(product, row * n + start, row * n + start + 1)),
                vector.max(magnitude(vectors, start * n + row, (start + 1) * n + row)),
            )
        });
        let denominator = (norm + shift(start)?.norm()) * vector;
        let relative = if vector == R::ZERO || denominator == R::ZERO {
            residual
        } else {
            checked_float_quotient(residual, denominator)?
        };
        residuals[start..start + size].fill(relative);
    }
    Ok(())
}

/// Reject a numerically dependent basis by modified Gram–Schmidt on unit
/// copies of the vectors in the supplied coordinates. General eigen jobs use
/// balanced coordinates; unbalanced Schur jobs use input coordinates.
/// A vector whose distance from the span of
/// its predecessors is at most `cutoff` is reported as defective.
///
/// A real pair `v = r + i s` and `conj(v)` span the complexified span of r
/// and s, so the basis stays real: after projecting the earlier (real
/// orthonormal) basis out of r and s, `dist(v) = sqrt(|r|^2 + |s|^2)` and
/// `dist(conj(v)) = 2 |r| |s - (r̂·s) r̂| / dist(v)`, exactly the complex
/// Gram–Schmidt distances of `[v, conj(v)]`.
fn independence<W: NumericScalar>(
    vectors: &[W],
    basis: &mut [W],
    values: &[Complex<W::Real>],
    n: usize,
    cutoff: W::Real,
    dispatch: scioxide_core::backend::KernelDispatch,
) -> Result<(), SciOxideError> {
    let dependent = |index: usize| {
        SciOxideError::numerical_instability(
            NumericalIssue::DefectiveEigenbasis,
            alloc::format!("eigenvector {index} is linearly dependent at cutoff {cutoff:?}"),
        )
    };
    basis[..n * n].copy_from_slice(&vectors[..n * n]);
    // `left^H right`; real rows use the dispatched dot.
    let inner = |left: &[W], right: &[W]| -> Result<W, SciOxideError> {
        Ok(match W::IS_REAL {
            true => crate::kernel::arch::dot(left, right, dispatch)?,
            false => None,
        }
        .unwrap_or_else(|| {
            left.iter()
                .zip(right)
                .fold(W::ZERO, |sum, (&left, &right)| sum + left.conj() * right)
        }))
    };
    let divide = |vector: &mut [W], divisor: W::Real| {
        vector
            .iter_mut()
            .for_each(|value| *value = crate::rotation::divide_real(*value, divisor));
    };
    for (start, size) in blocks::<W>(values) {
        let length = vector_norm(basis, n, (start, size))?;
        if length == W::Real::ZERO {
            return Err(dependent(start));
        }
        let (done, rest) = crate::workspace::split_mut(basis, start * n)?;
        for row in rest[..size * n].chunks_exact_mut(n) {
            divide(row, length);
            for previous in done.chunks_exact(n) {
                let projection = inner(previous, row)?;
                row.iter_mut()
                    .zip(previous)
                    .for_each(|(value, &q)| *value = *value - projection * q);
            }
        }
        let (first, second) = crate::workspace::split_mut(&mut rest[..size * n], n)?;
        let first_norm = crate::kernel::finite::norm(first)?;
        let second_norm = crate::kernel::finite::norm(second)?;
        let distance = first_norm.hypot(second_norm);
        if distance <= cutoff {
            return Err(dependent(start));
        }
        if size == 1 {
            divide(first, distance);
            continue;
        }
        // `dist(conj(v)) <= 2 |r|`, so a negligible real part is dependent.
        if W::Real::TWO * first_norm <= cutoff {
            return Err(dependent(start + 1));
        }
        divide(first, first_norm);
        let projection = inner(first, second)?;
        second
            .iter_mut()
            .zip(&*first)
            .for_each(|(value, &q)| *value = *value - projection * q);
        let orthogonal = crate::kernel::finite::norm(second)?;
        let conjugate_distance = (W::Real::TWO * first_norm / distance) * orthogonal;
        if conjugate_distance <= cutoff {
            return Err(dependent(start + 1));
        }
        divide(second, orthogonal);
    }
    Ok(())
}

/// Factor taking each vector to unit 2-norm with its largest-magnitude entry
/// (the last on ties) real and positive; a pair's conjugate uses the
/// conjugate factor.
fn normalizing_phases<W: NumericScalar>(
    vectors: &[W],
    values: &[Complex<W::Real>],
    n: usize,
    factors: &mut [Complex<W::Real>],
) -> Result<(), SciOxideError> {
    for block in blocks::<W>(values) {
        let norm = vector_norm(vectors, n, block)?;
        let pivot = (0..n)
            .max_by(|&left, &right| {
                entry(vectors, n, block, left)
                    .norm_sqr()
                    .partial_cmp(&entry(vectors, n, block, right).norm_sqr())
                    .unwrap_or(Ordering::Equal)
            })
            .map_or(Complex::new(W::Real::ONE, W::Real::ZERO), |row| {
                entry(vectors, n, block, row)
            });
        let magnitude = pivot.norm();
        let phase = if magnitude > W::Real::ZERO {
            pivot.conj() / Complex::new(magnitude, W::Real::ZERO)
        } else {
            Complex::new(W::Real::ONE, W::Real::ZERO)
        };
        factors[block.0] = if norm > W::Real::ZERO {
            phase / Complex::new(norm, W::Real::ZERO)
        } else {
            phase
        };
    }
    Ok(())
}

/// Wilkinson's reciprocal eigenvalue conditions `|y^H x| / (|x| |y|)` from
/// the matched right and left vectors of each eigenvalue.
fn eigenvalue_conditions<W: NumericScalar>(
    right: &[W],
    left: &[W],
    values: &[Complex<W::Real>],
    n: usize,
    conditions: &mut [W::Real],
) -> Result<(), SciOxideError> {
    for block in blocks::<W>(values) {
        let overlap = (0..n).fold(Complex::new(W::Real::ZERO, W::Real::ZERO), |sum, row| {
            sum + entry(left, n, block, row).conj() * entry(right, n, block, row)
        });
        let denominator = vector_norm(right, n, block)? * vector_norm(left, n, block)?;
        let reciprocal = if denominator > W::Real::ZERO {
            (overlap.norm() / denominator).min(W::Real::ONE)
        } else {
            W::Real::ZERO
        };
        let _ = ConditionEstimate::new(reciprocal, EstimateReliability::ModelBased)?;
        conditions[block.0..block.0 + block.1].fill(reciprocal);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn schur_recovery_shares_the_dense_independence_cutoff() {
        // The eigenvectors of [[1, 1], [0, 1 + delta]] are nearly parallel; with
        // the shared `max(n,1)` factor, delta = 3.5e-8 is dependent for n = 2.
        for (delta, independent) in [(3.5e-8, false), (1e-7, true)] {
            let a = crate::Matrix::from_vec(2, 2, alloc::vec![1.0, 1.0, 0.0, 1.0 + delta]).unwrap();
            let schur = crate::SchurDecomposition::decompose(&a).unwrap();
            let mut output = crate::Matrix::empty();
            let mut scratch = alloc::vec![0.0; 64];
            let mut phases = alloc::vec![Complex::new(0.0, 0.0); 2];
            let result = right_vectors_from_schur(
                &schur,
                &mut output,
                &mut scratch,
                &mut phases,
                KernelContext::new(scioxide_core::ExecutionContext::sequential()),
            );
            assert_eq!(result.is_ok(), independent, "{delta}: {result:?}");
        }
    }

    #[test]
    fn undo_balance_bounds_products_and_keeps_real_pairs_on_one_scale() {
        let values = [Complex::new(0.0_f64, 1.0), Complex::new(0.0, -1.0)];
        let scales = [2.0_f64.powi(900), 2.0_f64.powi(-900)];
        let permutation = [1, 0];
        for side in [Side::Right, Side::Left] {
            let mut vectors = [1e200, 1e200, 1e199, -1e199];
            undo_balance(
                &mut vectors,
                &values,
                &scales,
                &permutation,
                side,
                &mut [0.0; 2],
            )
            .unwrap();
            assert!(vectors.iter().all(|v| v.is_finite()));
            let dominant = match side {
                Side::Right => 1,
                Side::Left => 0,
            };
            assert!(vectors[dominant] >= 1.0 && vectors[dominant] < 2.0);
            let expected = match side {
                Side::Right => 0.1,
                Side::Left => -0.1,
            };
            assert!(
                (vectors[dominant + 2] / vectors[dominant] - expected).abs() <= 4.0 * f64::EPSILON
            );
        }
    }

    /// Standardized quasi-triangular T of order `n` with conjugate pairs at
    /// the given starts, and a dense Z.
    fn quasi_triangular(n: usize, pairs: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            t[i * n + i] = 0.1 * i as f64 - 2.0;
            for j in i + 1..n {
                t[i * n + j] = ((i * 13 + j * 7) % 17) as f64 / 17.0 - 0.5;
            }
        }
        for &k in pairs {
            t[(k + 1) * n + k + 1] = t[k * n + k];
            t[k * n + k + 1] = 0.75;
            t[(k + 1) * n + k] = -1.25;
        }
        let z = (0..n * n)
            .map(|index| ((index * 31) % 23) as f64 / 23.0 - 0.5)
            .collect();
        (t, z)
    }

    #[test]
    fn back_transform_reaches_pairs_across_block_boundaries() {
        let n = 70;
        let (t, z) = quasi_triangular(n, &[BLOCK - 1, 2 * BLOCK - 1, n - 2]);
        let policy = ExecutionPolicy::Sequential;
        let products = products::<f64>(n, policy).unwrap();
        let mut arena = vec![0.0; products.scalar_len().unwrap()];
        let mut gemm = LinalgWorkspace::from_slices(
            LinalgSlices {
                scalars: &mut arena,
                ..LinalgSlices::default()
            },
            &products,
        )
        .unwrap();
        let context = KernelContext::new(crate::kernel::tests::test_context())
            .with_accuracy(Accuracy::Standard);
        for side in [Side::Right, Side::Left] {
            let mut x = vec![0.0; n * n];
            let mut work = vec![0.0; 2 * n];
            schur_vectors::eigenvectors(
                &t,
                n,
                side,
                f64::EPSILON,
                &mut x,
                &mut work,
                context.dispatch(),
            )
            .unwrap();
            let mut target = vec![0.0; n * n];
            back_transform(&x, &z, &mut target, n, side, policy, &mut gemm, context).unwrap();
            let scale = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for row in 0..n {
                for col in 0..n {
                    let expected = (0..n).fold(0.0, |sum, k| sum + x[row * n + k] * z[col * n + k]);
                    assert!(
                        (target[row * n + col] - expected).abs() <= 1e-12 * scale,
                        "{side:?} ({row}, {col})"
                    );
                }
            }
        }
    }

    #[test]
    fn real_pair_distances_match_complex_gram_schmidt() {
        // Rows: a real vector, a pair `re + i im` with nearly parallel parts
        // and another real vector; values mark the pair at index 1.
        let n = 4;
        let rows = [
            [1.0, 0.5, -0.25, 0.0],
            [0.5, 1.0, 0.25, -0.5],
            [0.5, 1.0 + 1e-5, 0.25, -0.5 + 2e-5],
            [0.0, 0.25, 1.0, 0.5],
        ];
        let vectors: Vec<f64> = rows.iter().flatten().copied().collect();
        let values = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(3.0, 0.0),
        ];
        // Reference: modified Gram-Schmidt on the unit complex columns
        // `[x0, v, conj(v), x3]`.
        let complex = |re: &[f64; 4], im: &[f64; 4], sign: f64| -> Vec<Complex<f64>> {
            re.iter()
                .zip(im)
                .map(|(&r, &i)| Complex::new(r, sign * i))
                .collect()
        };
        let zero = [0.0; 4];
        let mut basis: Vec<Vec<Complex<f64>>> = Vec::new();
        let mut distances = Vec::new();
        for mut column in [
            complex(&rows[0], &zero, 1.0),
            complex(&rows[1], &rows[2], 1.0),
            complex(&rows[1], &rows[2], -1.0),
            complex(&rows[3], &zero, 1.0),
        ] {
            let norm = column.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            column
                .iter_mut()
                .for_each(|v| *v /= Complex::new(norm, 0.0));
            for q in &basis {
                let projection = q
                    .iter()
                    .zip(&column)
                    .fold(Complex::new(0.0, 0.0), |sum, (q, v)| sum + q.conj() * *v);
                column
                    .iter_mut()
                    .zip(q)
                    .for_each(|(v, q)| *v -= projection * *q);
            }
            let distance = column.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            column
                .iter_mut()
                .for_each(|v| *v /= Complex::new(distance, 0.0));
            distances.push(distance);
            basis.push(column);
        }
        let smallest = distances.iter().copied().fold(f64::MAX, f64::min);
        assert!(smallest < 1e-4 && smallest == distances[2]);
        let dispatch = KernelContext::new(crate::kernel::tests::test_context()).dispatch();
        let mut basis = vec![0.0; n * n];
        let check = |cutoff: f64, basis: &mut [f64]| {
            independence(&vectors, basis, &values, n, cutoff, dispatch)
        };
        assert!(check(smallest * (1.0 - 1e-6), &mut basis).is_ok());
        let error = check(smallest * (1.0 + 1e-6), &mut basis).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::DefectiveEigenbasis,
                ..
            }
        ));
    }
}
