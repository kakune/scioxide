use super::*;
use crate::{ExplicitZeroPolicy, SparseProductOptions, SparseProductPlan};
use scioxide_core::AlgebraicScalar;
struct Resources {
    limits: AmgLimits,
    charged: usize,
    accuracy: scioxide_core::Accuracy,
}
const MATRIX: Argument = scioxide_core::argument!("matrix");
const MODES: Argument = scioxide_core::argument!("modes");
// Setup payload accounting.
impl Resources {
    fn charge(&mut self, bytes: usize) -> Result<(), SciOxideError> {
        self.charged = add(self.charged, bytes, scioxide_core::quantity!("setup_bytes"))?;
        if self.charged > self.limits.max_setup_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("setup_bytes"),
                self.charged,
                self.limits.max_setup_bytes,
            ));
        }
        Ok(())
    }
    // Conservative envelope for conversion/product plans, geometric Vec capacity growth,
    // opposite-index maps, validation copies, numerical output and temporary row paths.
    fn sparse<S: NumericScalar>(
        &mut self,
        (rows, cols): (usize, usize),
        entries: usize,
    ) -> Result<(), SciOxideError> {
        if entries > self.limits.max_entries {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("matrix_entries"),
                entries,
                self.limits.max_entries,
            ));
        }
        self.sparse_payload::<S>((rows, cols), entries)
    }
    // Incrementally admit the same conservative envelope during one symbolic pass.
    fn sparse_payload<S: NumericScalar>(
        &mut self,
        (rows, cols): (usize, usize),
        entries: usize,
    ) -> Result<(), SciOxideError> {
        let bytes = scioxide_core::quantity!("setup_bytes");
        self.charge(mul(
            mul(
                16,
                add(add(add(rows, cols, bytes)?, entries, bytes)?, 1, bytes)?,
                bytes,
            )?,
            add(size_of::<S>(), mul(2, size_of::<usize>(), bytes)?, bytes)?,
            bytes,
        )?)
    }
    fn dense<S: NumericScalar>(&mut self, rows: usize, modes: usize) -> Result<(), SciOxideError> {
        let bytes = scioxide_core::quantity!("setup_bytes");
        self.charge(mul(
            mul(
                16,
                mul(add(rows, 1, bytes)?, add(modes, 1, bytes)?, bytes)?,
                bytes,
            )?,
            add(size_of::<S>(), size_of::<usize>(), bytes)?,
            bytes,
        )?)
    }
}
// Inverse diagonal and Jacobi damping of `a`.
fn diagonal<S: NumericScalar>(
    a: &CsrMatrix<S>,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<(Vec<S>, S::Real), SciOxideError> {
    let mut inverse = try_with_capacity(a.rows(), scioxide_core::quantity!("inverse_diagonal"))?;
    let mut bound = S::Real::ONE;
    for i in 0..a.rows() {
        tick()?;
        let d = a.get(i, i)?;
        if d.imaginary_component() != S::Real::ZERO || d.real_component() <= S::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("matrix"),
                "AMG requires a positive real diagonal",
            ));
        }
        inverse.push(crate::helpers::divide(
            S::ONE,
            d,
            crate::helpers::Projection::Strict,
        )?);
        let mut row_bound = S::Real::ZERO;
        for pos in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_indices()[pos];
            let value = a.values()[pos];
            if value != a.get(j, i)?.conj() {
                return Err(invalid(
                    MATRIX,
                    "automatic AMG requires exact Hermitian storage",
                ));
            }
            row_bound += value.abs() / d.real_component();
        }
        crate::helpers::finite_with_origin(
            &[row_bound],
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;

        if row_bound > bound {
            bound = row_bound;
        }
    }
    crate::helpers::finite_with_origin(&inverse, scioxide_core::NonFiniteOrigin::Intermediate)?;
    // 4/(3*rho_bound) lies strictly inside (0,2/rho) for positive definite A.
    let weight = S::Real::try_from_f64_lossy(4. / 3.)? / bound;
    if weight <= S::Real::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "AMG damping underflow",
        ));
    }
    Ok((inverse, weight))
}
// Three-phase strength aggregation in canonical row/column order. Phase two
// recruits through phase-one members only, so attachment cannot cascade.
fn aggregate<S: NumericScalar>(
    a: &CsrMatrix<S>,
    threshold: S::Real,
    resources: &mut Resources,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<(Vec<usize>, usize), SciOxideError> {
    let n = a.rows();
    let bytes = scioxide_core::quantity!("setup_bytes");
    resources.charge(mul(
        n,
        add(
            size_of::<S::Real>(),
            mul(2, size_of::<usize>(), bytes)?,
            bytes,
        )?,
        bytes,
    )?)?;
    let mut diagonals = try_with_capacity(n, scioxide_core::quantity!("aggregate_diagonals"))?;
    for i in 0..n {
        tick()?;
        diagonals.push(a.get(i, i)?.real_component().sqrt());
    }
    // Positive diagonals have already been validated. Avoid products of diagonal
    // magnitudes, which can overflow even when the normalized strength is finite.
    let neighbors = |i: usize| {
        let diagonals = &diagonals;
        (a.row_ptr()[i]..a.row_ptr()[i + 1]).filter_map(move |pos| {
            let j = a.col_indices()[pos];
            (j != i
                && a.values()[pos] != S::ZERO
                && a.values()[pos].abs() / diagonals[i] / diagonals[j] >= threshold)
                .then_some(j)
        })
    };
    let mut ids = try_filled(n, usize::MAX, scioxide_core::quantity!("aggregate_ids"))?;
    let mut count = 0;
    // Phase one: disjoint closed strong neighbourhoods. Isolated nodes wait
    // until phase three and remain represented by singleton local bases.
    for i in 0..n {
        tick()?;
        if ids[i] != usize::MAX
            || neighbors(i).next().is_none()
            || neighbors(i).any(|j| ids[j] != usize::MAX)
        {
            continue;
        }
        ids[i] = count;
        for j in neighbors(i) {
            ids[j] = count;
        }
        count += 1;
    }
    // Phase two: freeze eligibility, and break ties by the first CSR neighbour.
    let seeds = try_copy_slice(&ids, scioxide_core::quantity!("aggregate_seeds"))?;
    for (i, id) in ids.iter_mut().enumerate() {
        tick()?;
        if *id == usize::MAX
            && let Some(group) = neighbors(i).map(|j| seeds[j]).find(|&id| id != usize::MAX)
        {
            *id = group;
        }
    }
    // Phase three: collect residual neighbourhoods without stealing members.
    for i in 0..n {
        tick()?;
        if ids[i] != usize::MAX {
            continue;
        }
        ids[i] = count;
        for j in neighbors(i) {
            if ids[j] == usize::MAX {
                ids[j] = count;
            }
        }
        count += 1;
    }
    Ok((ids, count))
}
// Twice-reorthogonalized local MGS. Locally dependent modes are represented by
// existing local columns; genuinely zero local spaces are rejected. Global input
// scaling is retained in Bc, and Q uses a fixed positive norm convention.
fn tentative<S: NumericScalar>(
    n: usize,
    b: &[S],
    k: usize,
    (ids, groups): (&[usize], usize),
    resources: &mut Resources,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<(CsrMatrix<S>, Vec<S>), SciOxideError> {
    let basis = scioxide_core::quantity!("tentative_basis");
    let modes = scioxide_core::quantity!("coarse_modes");
    resources.dense::<S>(n, k)?;
    resources.sparse::<S>((n, n), mul(n, k, basis)?)?;
    let mut q = try_filled(mul(n, k, basis)?, S::ZERO, basis)?;
    let mut bc = try_filled(mul(n, k, modes)?, S::ZERO, modes)?;
    let mut vector = try_filled(n, S::ZERO, scioxide_core::quantity!("mode_vector"))?;
    let mut first = try_filled(
        groups,
        0usize,
        scioxide_core::quantity!("aggregate_offsets"),
    )?;
    let mut ranks = try_filled(groups, 0usize, scioxide_core::quantity!("aggregate_ranks"))?;
    let mut members = try_with_capacity(n, scioxide_core::quantity!("aggregate_members"))?;
    members.extend(0..n);
    members.sort_unstable_by_key(|&i| (ids[i], i));
    let mut offset = 0;
    let mut coarse = 0;
    for group in 0..groups {
        tick()?;
        first[group] = coarse;
        let end = offset + members[offset..].partition_point(|&i| ids[i] == group);
        let local_rows = &members[offset..end];
        offset = end;
        let mut rank = 0;
        for mode in 0..k {
            let scale = local_rows
                .iter()
                .map(|&i| b[mode * n + i].abs())
                .fold(S::Real::ZERO, |a, v| if a > v { a } else { v });
            if scale == S::Real::ZERO {
                continue;
            }
            for &i in local_rows {
                vector[i] = b[mode * n + i] / S::from_real(scale);
            }
            for _ in 0..2 {
                for local in 0..rank {
                    let mut dot = ProductSum::new();
                    for &i in local_rows {
                        {
                            dot.add_product(q[local * n + i].conj(), vector[i])?;
                        }
                    }
                    let coefficient = dot.finish()?;
                    let slot = mode * n + coarse + local;
                    let mut sum = ProductSum::new();
                    sum.add_product(bc[slot], S::ONE)?;
                    sum.add_product(coefficient, S::from_real(scale))?;
                    bc[slot] = sum.finish()?;
                    for &i in local_rows {
                        {
                            let mut sum = ProductSum::new();
                            sum.add_product(vector[i], S::ONE)?;
                            sum.add_product(-coefficient, q[local * n + i])?;
                            vector[i] = sum.finish()?;
                        }
                    }
                }
            }
            let mut norm = ProductSum::new();
            for &i in local_rows {
                {
                    norm.add_product(vector[i].conj(), vector[i])?;
                }
            }
            let norm = norm.finish()?.real_component().sqrt();
            if scioxide_core::RankThreshold::new(
                S::Real::EPSILON * S::Real::try_from_f64_lossy(64.)?,
                S::Real::ZERO,
            )?
            .is_satisfied(norm, S::Real::ONE)?
            {
                continue;
            }
            if coarse + rank >= n {
                return Err(invalid(MODES, "AMG local basis exceeds dimension"));
            }
            for &i in local_rows {
                {
                    q[rank * n + i] = vector[i] / S::from_real(norm);
                }
            }
            bc[mode * n + coarse + rank] = S::from_real(norm * scale);
            rank += 1;
        }
        if rank == 0 {
            return Err(invalid(MODES, "near-nullspace vanishes on an aggregate"));
        }
        ranks[group] = rank;
        coarse += rank;
    }
    crate::helpers::finite_with_origin(&q, scioxide_core::NonFiniteOrigin::Intermediate)?;
    crate::helpers::finite_with_origin(&bc, scioxide_core::NonFiniteOrigin::Intermediate)?;
    let pointers = scioxide_core::quantity!("tentative_row_pointers");
    let columns = scioxide_core::quantity!("tentative_column_indices");
    let entries = scioxide_core::quantity!("tentative_values");
    let mut ptr = try_with_capacity(add(n, 1, pointers)?, pointers)?;
    let mut col = try_with_capacity(mul(n, k, columns)?, columns)?;
    let mut values = try_with_capacity(mul(n, k, entries)?, entries)?;
    ptr.push(0);
    for i in 0..n {
        for local in 0..ranks[ids[i]] {
            col.push(first[ids[i]] + local);
            values.push(q[local * n + i]);
        }
        ptr.push(col.len());
    }
    let mut compact = try_with_capacity(mul(coarse, k, modes)?, modes)?;
    for mode in 0..k {
        compact.extend_from_slice(&bc[mode * n..mode * n + coarse]);
    }
    Ok((CsrMatrix::from_raw(n, coarse, ptr, col, values)?, compact))
}
// Budgeted sparse product `a * b`.
fn product<S: NumericScalar>(
    a: &CsrMatrix<S>,
    b: &CsrMatrix<S>,
    resources: &mut Resources,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<CsrMatrix<S>, SciOxideError> {
    let q = scioxide_core::quantity!("product_paths");
    let base = add(a.nnz(), b.nnz(), q)?;
    resources.sparse_payload::<S>((a.rows(), b.cols()), base)?;
    let mut paths = 0;
    let plan = SparseProductPlan::with_admission(a, b, &mut |additional| {
        let Some(count) = additional else {
            return tick();
        };
        paths = add(paths, count, q)?;
        if paths > resources.limits.max_entries {
            return Err(SciOxideError::resource_limit(
                q,
                paths,
                resources.limits.max_entries,
            ));
        }
        resources.charge(mul(
            mul(16, count, q)?,
            add(size_of::<S>(), mul(2, size_of::<usize>(), q)?, q)?,
            q,
        )?)
    })?;
    let entries = add(base, paths, q)?;
    if entries > resources.limits.max_entries {
        return Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("matrix_entries"),
            entries,
            resources.limits.max_entries,
        ));
    }
    let options = SparseProductOptions::new().with_accuracy(resources.accuracy);
    resources.charge(plan.workspace_requirements(options).required_bytes::<S>()?)?;
    let mut output = plan.create_output()?;
    let mut work = plan.workspace(options)?;
    plan.evaluate_into(
        a,
        b,
        options,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )?;
    tick()?;
    Ok(output)
}
// Jacobi-smoothed interpolation.
fn interpolate<S: NumericScalar>(
    (a, t): (&CsrMatrix<S>, &CsrMatrix<S>),
    inverse: &[S],
    weight: S::Real,
    resources: &mut Resources,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<CsrMatrix<S>, SciOxideError> {
    let mut p = product(a, t, resources, tick)?;
    let mut values = p.workspace(())?;
    let pattern = p.structural_pattern().to_owned()?;
    p.update_values(
        |old, next| {
            for (i, &diagonal_inverse) in inverse.iter().enumerate() {
                tick()?;
                for pos in pattern.view().row_ptr()[i]..pattern.view().row_ptr()[i + 1] {
                    let col = pattern.view().col_indices()[pos];
                    let scaled = old[pos] * diagonal_inverse;
                    crate::helpers::finite_with_origin(
                        &[scaled],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                    let mut sum = ProductSum::new();
                    sum.add_product(t.get(i, col)?, S::ONE)?;
                    sum.add_product(-scaled, S::from_real(weight))?;
                    next[pos] = sum.finish()?;
                }
            }
            Ok::<_, SciOxideError>(())
        },
        &mut values,
    )?;
    Ok(p)
}
// Galerkin products differ in their two accumulation orders. Average conjugate
// pairs (and discard diagonal imaginary roundoff), keeping an exact Hermitian
// representation for the next setup stage and existing Cholesky contracts.
fn hermitianize<S: NumericScalar>(a: &mut CsrMatrix<S>) -> Result<(), SciOxideError> {
    let half = scalar::<S>(0.5)?;
    let mut values = a.workspace(())?;
    let pattern = a.structural_pattern().to_owned()?;
    a.update_values(
        |old, next| {
            next.copy_from_slice(old);
            for i in 0..pattern.view().rows() {
                for pos in pattern.view().row_ptr()[i]..pattern.view().row_ptr()[i + 1] {
                    let j = pattern.view().col_indices()[pos];
                    if j < i {
                        continue;
                    }
                    if i == j {
                        next[pos] = S::from_real(old[pos].real_component());
                        continue;
                    }
                    let start = pattern.view().row_ptr()[j];
                    let mirror = start
                        + pattern.view().col_indices()[start..pattern.view().row_ptr()[j + 1]]
                            .binary_search(&i)
                            .map_err(|_| {
                                invalid(MATRIX, "Galerkin structure lacks conjugate entry")
                            })?;
                    let mut sum = ProductSum::new();
                    sum.add_product(old[pos], half)?;
                    sum.add_product(old[mirror].conj(), half)?;
                    let v = sum.finish()?;
                    next[pos] = v;
                    next[mirror] = v.conj();
                }
            }
            Ok::<_, SciOxideError>(())
        },
        &mut values,
    )
}
// Smoothed-aggregation setup shared by the public constructors.
pub(super) fn build<S: NumericScalar>(
    matrix: SparseView<'_, S>,
    near_nullspace: BlockVectorView<'_, S>,
    options: AmgSetupOptions<S::Real>,
    limits: AmgLimits,
    accuracy: scioxide_core::Accuracy,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<AmgHierarchy<S>, SciOxideError> {
    tick()?;
    let n = matrix.rows();
    let (modes, k) = (near_nullspace.as_slice(), near_nullspace.right_hand_sides());
    crate::helpers::exact(matrix.cols(), n)?;
    crate::helpers::exact(near_nullspace.rows(), n)?;
    let argument = if k == 0 {
        Some(MODES)
    } else if options.coarse_size == 0 {
        Some(scioxide_core::argument!("setup.coarse_size"))
    } else if limits.max_levels == 0 || limits.max_levels > 64 {
        Some(scioxide_core::argument!("limits.max_levels"))
    } else if !options.strength.is_finite()
        || options.strength < S::Real::ZERO
        || options.strength > S::Real::ONE
    {
        Some(scioxide_core::argument!("setup.strength"))
    } else {
        None
    };
    if let Some(argument) = argument {
        return Err(invalid(
            argument,
            "invalid AMG shape, modes, strength or level limit",
        ));
    }
    finite(modes)?;

    let mut resources = Resources {
        limits,
        charged: 0,
        accuracy,
    };
    resources.charge(mul(
        limits.max_levels,
        size_of::<Level<S>>(),
        scioxide_core::quantity!("setup_bytes"),
    )?)?;
    resources.sparse::<S>((n, n), matrix.values().len())?;
    resources.dense::<S>(n, k)?;
    let mut a = matrix.to_csr(ExplicitZeroPolicy::Drop)?;
    let mut b = try_copy_slice(modes, scioxide_core::quantity!("level_modes"))?;
    let mut levels = try_with_capacity(limits.max_levels, scioxide_core::quantity!("levels"))?;
    let termination = loop {
        resources.dense::<S>(a.rows(), k)?;
        let (inverse, weight) = diagonal(&a, tick)?;
        let mut level = Level {
            a,
            inverse,
            weight,
            modes: b,
            aggregates: Vec::new(),
            tentative: None,
            p: None,
            r: None,
        };
        if level.a.rows() <= options.coarse_size {
            levels.push(level);
            break AmgSetupTermination::CoarseSize;
        }
        if levels.len() + 1 == limits.max_levels {
            levels.push(level);
            break AmgSetupTermination::LevelLimit;
        }
        let (ids, groups) = aggregate(&level.a, options.strength, &mut resources, tick)?;
        let (t, bc) = tentative(
            level.a.rows(),
            &level.modes,
            k,
            (&ids, groups),
            &mut resources,
            tick,
        )?;
        if t.cols() >= t.rows() {
            levels.push(level);
            break AmgSetupTermination::Stagnation;
        }
        let p = interpolate((&level.a, &t), &level.inverse, weight, &mut resources, tick)?;
        resources.sparse::<S>((p.cols(), p.rows()), p.nnz())?;
        let r = p.adjoint()?.to_csr()?;
        let ap = product(&level.a, &p, &mut resources, tick)?;
        a = product(&r, &ap, &mut resources, tick)?;
        hermitianize(&mut a)?;
        b = bc;
        level.aggregates = ids;
        level.tentative = Some(t);
        level.p = Some(p);
        level.r = Some(r);
        levels.push(level);
    };
    let mut diagnostics = AmgDiagnostics {
        levels: levels.len(),
        total_rows: 0,
        operator_entries: 0,
        coarsening_ratio: levels
            .get(1)
            .map(|coarse| n as f64 / coarse.a.rows() as f64),
        operator_complexity: None,
        transfer_entries: 0,
        setup_bytes_bound: resources.charged,
        termination,
    };
    let rows = scioxide_core::quantity!("diagnostics.total_rows");
    let operators = scioxide_core::quantity!("diagnostics.operator_entries");
    let transfers = scioxide_core::quantity!("diagnostics.transfer_entries");
    for level in &levels {
        diagnostics.total_rows = add(diagnostics.total_rows, level.a.rows(), rows)?;
        diagnostics.operator_entries = add(diagnostics.operator_entries, level.a.nnz(), operators)?;
        if let Some(p) = &level.p {
            diagnostics.transfer_entries = add(
                diagnostics.transfer_entries,
                mul(2, p.nnz(), transfers)?,
                transfers,
            )?;
        }
    }
    let fine_entries = levels[0].a.nnz();
    diagnostics.operator_complexity =
        (fine_entries != 0).then(|| diagnostics.operator_entries as f64 / fine_entries as f64);
    tick()?;
    Ok(AmgHierarchy {
        levels,
        mode_count: k,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    #[test]
    fn aggregation_checks_cancellation_in_every_pass() {
        let a = CsrMatrix::from_raw(
            3,
            3,
            alloc::vec![0, 2, 5, 7],
            alloc::vec![0, 1, 0, 1, 2, 1, 2],
            alloc::vec![2.0_f64, -1., -1., 2., -1., -1., 2.],
        )
        .unwrap();
        // Diagonal preparation, seed selection, attachment and residual grouping.
        for stop_at in [1, 4, 7, 10] {
            let mut resources = Resources {
                limits: AmgLimits {
                    max_levels: 4,
                    max_entries: 100,
                    max_setup_bytes: 1024,
                },
                charged: 0,
                accuracy: scioxide_core::Accuracy::Standard,
            };
            let token = scioxide_core::CancellationToken::new();
            let context = ExecutionContext::sequential().with_cancellation(&token);
            let mut calls = 0;
            let error = aggregate(&a, 0.0625, &mut resources, &mut || {
                calls += 1;
                if calls == stop_at {
                    token.cancel();
                }
                context.check()
            })
            .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
            ));
            assert_eq!(calls, stop_at);
        }
    }
    #[test]
    fn combined_entry_limit_reports_complete_paths_and_inputs() {
        let matrix = CsrMatrix::from_raw(
            2,
            2,
            alloc::vec![0, 2, 4],
            alloc::vec![0, 1, 0, 1],
            alloc::vec![1.0_f64;4],
        )
        .unwrap();
        let mut resources = Resources {
            limits: AmgLimits {
                max_levels: 2,
                max_entries: 10,
                max_setup_bytes: usize::MAX,
            },
            charged: 0,
            accuracy: scioxide_core::Accuracy::Standard,
        };
        let error = product(&matrix, &matrix, &mut resources, &mut || Ok(())).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit {
                required: 16,
                limit: 10,
                ..
            }
        ));
    }
}
