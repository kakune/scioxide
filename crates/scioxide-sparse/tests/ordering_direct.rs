//! Ordering defaults, independent mesh fill baselines and large sparse solves.
#![allow(clippy::unwrap_used)]
use scioxide_core::{LinearOperator, SciOxideErrorKind, StructuralPattern, Triangle};
use scioxide_sparse::{
    CsrMatrix, SparseCholesky, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseOrdering,
    SparseQrLimits, SparseQrSymbolic, SparseSymmetricLimits, SparseSymmetricSymbolic,
    approximate_minimum_degree, reverse_cuthill_mckee,
};

include!("data/ordering_meshes.rs");

fn limits(entries: usize) -> SparseSymmetricLimits {
    SparseSymmetricLimits::new(entries, entries, 512 * 1024 * 1024)
}
fn mesh(dimension: usize, width: usize) -> CsrMatrix<f64> {
    let n = width.pow(dimension as u32);
    let mut ptr = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    let mut row = Vec::with_capacity(2 * dimension + 1);
    for i in 0..n {
        row.clear();
        row.push(i);
        let mut stride = 1;
        for _ in 0..dimension {
            let coordinate = i / stride % width;
            if coordinate > 0 {
                row.push(i - stride);
            }
            if coordinate + 1 < width {
                row.push(i + stride);
            }
            stride *= width;
        }
        row.sort_unstable();
        columns.extend_from_slice(&row);
        values.extend(row.iter().map(|&j| {
            if i == j {
                (2 * dimension) as f64 + 1.
            } else {
                -1.
            }
        }));
        ptr.push(columns.len());
    }
    CsrMatrix::from_raw(n, n, ptr, columns, values).unwrap()
}

#[test]
fn mesh_fill_beats_natural_and_rcm_and_tracks_published_amd() {
    for &(dimension, width, reference) in AMD_MESHES {
        let a = mesh(dimension, width);
        let n = a.rows();
        let rcm = reverse_cuthill_mckee(a.structural_pattern()).unwrap();
        let entries = |ordering| {
            n + SparseSymmetricSymbolic::analyze(
                a.structural_pattern(),
                ordering,
                None,
                limits(2 * n * width.pow(dimension as u32 - 1)),
            )
            .unwrap()
            .cholesky_entries()
        };
        let natural = entries(SparseOrdering::Natural);
        let reverse = entries(SparseOrdering::Supplied(&rcm));
        let amd = entries(SparseOrdering::default());
        assert!(
            amd <= natural,
            "{dimension}D {width}: AMD={amd}, natural={natural}"
        );
        assert!(
            amd <= reverse,
            "{dimension}D {width}: AMD={amd}, RCM={reverse}"
        );
        assert!(
            2 * amd <= 3 * reference,
            "{dimension}D {width}: AMD={amd}, SuiteSparse={reference}"
        );
    }
}

#[test]
fn orders_factors_and_solves_more_than_one_hundred_thousand_vertices() {
    const N: usize = 100_003;
    // Multiplication by 37 is a bijection modulo prime N. Construct each sorted
    // CSR row directly, without a quadratic matrix/pattern helper.
    let scramble = |i| i * 37 % N;
    let mut inverse = vec![0; N];
    for i in 0..N {
        inverse[scramble(i)] = i;
    }
    let mut ptr = Vec::with_capacity(N + 1);
    let mut columns = Vec::with_capacity(3 * N - 2);
    let mut values = Vec::with_capacity(3 * N - 2);
    ptr.push(0);
    for (row, &old) in inverse.iter().enumerate() {
        let mut adjacent = [row, row, row];
        let mut len = 1;
        if old > 0 {
            adjacent[len] = scramble(old - 1);
            len += 1;
        }
        if old + 1 < N {
            adjacent[len] = scramble(old + 1);
            len += 1;
        }
        adjacent[..len].sort_unstable();
        columns.extend_from_slice(&adjacent[..len]);
        values.extend(
            adjacent[..len]
                .iter()
                .map(|&col| if col == row { 4. } else { -1. }),
        );
        ptr.push(columns.len());
    }
    let a = CsrMatrix::from_raw(N, N, ptr, columns, values).unwrap();
    let symbolic = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::default(),
        None,
        limits(2 * N),
    )
    .unwrap();
    assert_eq!(symbolic.cholesky_entries(), N - 1);
    let factor = SparseCholesky::factor(symbolic, a.view()).unwrap();
    let exact: Vec<_> = (0..N).map(|i| 1. + (i % 11) as f64 / 16.).collect();
    let mut rhs = vec![0.; N];
    let mut scratch = vec![0.; a.apply_workspace_len().unwrap()];
    a.apply_into(&exact, &mut rhs, &mut scratch).unwrap();
    let solution = factor.solve(&rhs).unwrap();
    let mut residual = vec![0.; N];
    a.apply_into(&solution, &mut residual, &mut scratch)
        .unwrap();
    for ((actual, expected), (ax, b)) in solution.iter().zip(&exact).zip(residual.iter().zip(&rhs))
    {
        assert!((actual - expected).abs() <= 64. * f64::EPSILON * expected.abs());
        assert!((ax - b).abs() <= 64. * f64::EPSILON * b.abs());
    }
}

#[test]
fn solver_choices_defaults_and_supplied_permutations_share_conventions() {
    let a = mesh(2, 4);
    let pattern = a.structural_pattern();
    let order = approximate_minimum_degree(pattern).unwrap();
    assert_ne!(order.new_to_old(), &(0..a.rows()).collect::<Vec<_>>());
    let lu_limits = SparseLuLimits::new(256, 65536);
    for choice in [SparseOrdering::default(), SparseOrdering::Supplied(&order)] {
        let plan = SparseLuSymbolic::analyze(pattern, choice, lu_limits).unwrap();
        assert_eq!(plan.column_permutation(), &order);
        let factor = SparseLu::factor(plan.clone(), a.view()).unwrap();
        let mut rhs = vec![0.; a.rows()];
        a.apply_into(
            &vec![1.; a.rows()],
            &mut rhs,
            &mut vec![0.; a.apply_workspace_len().unwrap()],
        )
        .unwrap();
        let x = factor.solve(&rhs).unwrap();
        assert!(x.iter().all(|v| (v - 1.).abs() <= 1e-13));
        let symmetric =
            SparseSymmetricSymbolic::analyze(pattern, choice, None, limits(256)).unwrap();
        assert_eq!(symmetric.permutation(), &order);
    }
    let natural = SparseLuSymbolic::analyze(pattern, SparseOrdering::Natural, lu_limits).unwrap();
    assert_eq!(
        natural.column_permutation().new_to_old(),
        &(0..a.rows()).collect::<Vec<_>>()
    );
    let mismatch = scioxide_core::Permutation::identity(a.rows() - 1).unwrap();
    assert_eq!(
        SparseLuSymbolic::analyze(pattern, SparseOrdering::Supplied(&mismatch), lu_limits)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let tight = lu_limits.with_max_workspace_bytes(4096);
    assert_eq!(
        SparseLuSymbolic::analyze(pattern, SparseOrdering::default(), tight)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert!(SparseLuSymbolic::analyze(pattern, SparseOrdering::Natural, tight).is_ok());
}

#[test]
fn authoritative_triangle_ignores_the_other_triangles_edges() {
    let lower =
        StructuralPattern::new(4, 4, vec![0, 1, 3, 5, 7], vec![0, 0, 1, 1, 2, 2, 3]).unwrap();
    let extra = StructuralPattern::new(
        4,
        4,
        vec![0, 4, 8, 11, 13],
        vec![0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3],
    )
    .unwrap();
    for triangle in [Triangle::Lower, Triangle::Upper] {
        let transpose_lower = scioxide_sparse::transpose_pattern(lower.view()).unwrap();
        let transpose_extra = scioxide_sparse::transpose_pattern(extra.view()).unwrap();
        let (base, contaminated) = if triangle == Triangle::Lower {
            (lower.view(), extra.view())
        } else {
            (
                transpose_lower.view().pattern(),
                transpose_extra.view().pattern(),
            )
        };
        let first = SparseSymmetricSymbolic::analyze(
            base,
            SparseOrdering::default(),
            Some(triangle),
            limits(16),
        )
        .unwrap();
        let second = SparseSymmetricSymbolic::analyze(
            contaminated,
            SparseOrdering::default(),
            Some(triangle),
            limits(16),
        )
        .unwrap();
        assert_eq!(first.permutation(), second.permutation());
        assert_eq!(first.cholesky_entries(), second.cholesky_entries());
    }
}

#[test]
fn qr_incidence_handles_dense_rows_empty_shapes_and_bad_orders() {
    let limits = SparseQrLimits::new(0, 0, 16 * 1024 * 1024);
    // A single row represents a 20,000-column clique using linear incidence.
    let n = 20_000;
    let row = StructuralPattern::new(1, n, vec![0, n], (0..n).collect()).unwrap();
    let qr = SparseQrSymbolic::analyze(row.view(), SparseOrdering::default(), limits).unwrap();
    assert_eq!(
        qr.column_permutation().new_to_old(),
        &(0..n).collect::<Vec<_>>()
    );
    for (m, n) in [(0, 0), (0, 7), (7, 0)] {
        let p = StructuralPattern::new(m, n, vec![0; m + 1], vec![]).unwrap();
        let qr = SparseQrSymbolic::analyze(p.view(), SparseOrdering::default(), limits).unwrap();
        assert_eq!(
            qr.column_permutation().new_to_old(),
            &(0..n).collect::<Vec<_>>()
        );
    }
    let wrong = scioxide_core::Permutation::identity(2).unwrap();
    assert_eq!(
        SparseQrSymbolic::analyze(row.view(), SparseOrdering::Supplied(&wrong), limits)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn qr_preserves_a_supplied_structural_order() {
    let supplied = scioxide_core::Permutation::new(vec![2, 0, 1]).unwrap();
    let qr_limits = SparseQrLimits::new(16, 16, 65536);
    for (values, expected) in [
        (vec![1., 3., 2.], vec![2, 0, 1]),
        (vec![1.; 3], vec![2, 0, 1]),
    ] {
        let a = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], values).unwrap();
        let plan = SparseQrSymbolic::analyze(
            a.structural_pattern(),
            SparseOrdering::Supplied(&supplied),
            qr_limits,
        )
        .unwrap();
        assert_eq!(plan.column_permutation(), &supplied);
        let factor = scioxide_sparse::SparseQr::factor(plan.clone(), a.view()).unwrap();
        assert_eq!(factor.column_permutation(), expected);
    }
}

#[test]
fn qr_orders_a_rectangular_columns_intersection_star_without_fill() {
    // Row i connects column zero to a different leaf; diagonals and singleton
    // rows contribute no intersections. Repeated incidence paths are harmless.
    let n = 33;
    let mut ptr = vec![0];
    let mut columns = Vec::new();
    for _ in 0..2 {
        for leaf in 1..n {
            columns.extend([0, leaf]);
            ptr.push(columns.len());
        }
    }
    for leaf in 0..n {
        columns.push(leaf);
        ptr.push(columns.len());
    }
    let p = StructuralPattern::new(ptr.len() - 1, n, ptr, columns).unwrap();
    let plan = SparseQrSymbolic::analyze(
        p.view(),
        SparseOrdering::default(),
        SparseQrLimits::new(0, 0, 1 << 20),
    )
    .unwrap();
    let position = plan.column_permutation().old_to_new()[0];
    // Eliminating the hub before the final two variables would create fill.
    assert!(position >= n - 2);
    assert_eq!(
        plan.column_permutation(),
        SparseQrSymbolic::analyze(
            p.view(),
            SparseOrdering::default(),
            SparseQrLimits::new(0, 0, 1 << 20)
        )
        .unwrap()
        .column_permutation()
    );
}

#[test]
fn overlapping_qr_rows_do_not_falsely_defer_sparse_columns_as_dense() {
    let n = 257;
    let scramble = |i| i * 37 % n;
    let mut ptr = vec![0];
    let mut columns = Vec::new();
    for _ in 0..100 {
        for i in 1..n {
            let mut pair = [scramble(i - 1), scramble(i)];
            pair.sort_unstable();
            columns.extend(pair);
            ptr.push(columns.len());
        }
    }
    let pattern = StructuralPattern::new(ptr.len() - 1, n, ptr, columns).unwrap();
    let plan = SparseQrSymbolic::analyze(
        pattern.view(),
        SparseOrdering::default(),
        SparseQrLimits::new(0, 0, 32 << 20),
    )
    .unwrap();
    let mut neighbors = vec![Vec::new(); n];
    for i in 1..n {
        neighbors[scramble(i)].push(scramble(i - 1));
        neighbors[scramble(i - 1)].push(scramble(i));
    }
    let mut active = vec![true; n];
    for &pivot in plan.column_permutation().new_to_old() {
        assert!(neighbors[pivot].iter().filter(|&&i| active[i]).count() <= 1);
        active[pivot] = false;
    }
    assert!(active.iter().all(|&live| !live));
}
