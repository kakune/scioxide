//! Independent structural oracles for graph ordering and derivative scheduling.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, NumericScalar, Permutation, SciOxideError, StructuralPattern, StructuralPatternView,
};
use scioxide_sparse::{
    CsrMatrix, approximate_minimum_degree, derivative_coloring, distance_one_coloring,
    reverse_cuthill_mckee,
};

fn pattern(rows: usize, cols: usize, edge: impl Fn(usize, usize) -> bool) -> StructuralPattern {
    let mut pointers = vec![0];
    let mut indices = Vec::new();
    for row in 0..rows {
        indices.extend((0..cols).filter(|&col| edge(row, col)));
        pointers.push(indices.len());
    }
    StructuralPattern::new(rows, cols, pointers, indices).unwrap()
}
fn adjacency(p: StructuralPatternView<'_>) -> Vec<Vec<bool>> {
    let mut graph = vec![vec![false; p.cols()]; p.rows()];
    for (row, bounds) in p.row_ptr().windows(2).enumerate() {
        for &col in &p.col_indices()[bounds[0]..bounds[1]] {
            if row != col {
                graph[row][col] = true;
                graph[col][row] = true;
            }
        }
    }
    graph
}
fn symbolic_fill(p: StructuralPatternView<'_>, order: &[usize]) -> usize {
    let mut graph = adjacency(p);
    let mut active = vec![true; p.rows()];
    let mut fill = 0;
    for &vertex in order {
        let neighbors: Vec<_> = (0..p.rows())
            .filter(|&v| active[v] && graph[vertex][v])
            .collect();
        for (i, &left) in neighbors.iter().enumerate() {
            for &right in &neighbors[i + 1..] {
                if !graph[left][right] {
                    fill += 1;
                    graph[left][right] = true;
                    graph[right][left] = true;
                }
            }
        }
        active[vertex] = false;
    }
    fill
}
fn check(p: StructuralPatternView<'_>) {
    for order in [
        approximate_minimum_degree(p).unwrap(),
        reverse_cuthill_mckee(p).unwrap(),
    ] {
        let mut sorted = order.new_to_old().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..p.rows()).collect::<Vec<_>>());
        for (new, &old) in order.new_to_old().iter().enumerate() {
            assert_eq!(order.old_to_new()[old], new);
        }
    }
    let order = approximate_minimum_degree(p).unwrap();
    // AMD uses degree bounds, mass elimination and assembly postordering;
    // exact greedy pivot-by-pivot comparisons no longer describe its contract.
    let fill = symbolic_fill(p, order.new_to_old());
    let edges = adjacency(p).iter().flatten().filter(|&&edge| edge).count() / 2;
    let plan = scioxide_sparse::SparseSymmetricSymbolic::analyze(
        p,
        scioxide_sparse::SparseOrdering::Supplied(&order),
        None,
        scioxide_sparse::SparseSymmetricLimits::new(p.rows() * p.rows(), 0, usize::MAX),
    )
    .unwrap();
    assert_eq!(plan.cholesky_entries(), edges + fill);
    assert_eq!(order, approximate_minimum_degree(p).unwrap());
    assert_eq!(
        reverse_cuthill_mckee(p).unwrap(),
        reverse_cuthill_mckee(p).unwrap()
    );
    let graph = adjacency(p);
    let distance = distance_one_coloring(p).unwrap();
    for (row, neighbors) in graph.iter().enumerate() {
        for (col, &edge) in neighbors.iter().enumerate() {
            if edge {
                assert_ne!(distance.colors()[row], distance.colors()[col]);
            }
        }
    }
    check_derivative(p);
}
fn check_derivative(p: StructuralPatternView<'_>) {
    let schedule = derivative_coloring(p).unwrap();
    for bounds in p.row_ptr().windows(2) {
        let colors: Vec<_> = p.col_indices()[bounds[0]..bounds[1]]
            .iter()
            .map(|&col| schedule.colors()[col])
            .collect();
        for (i, color) in colors.iter().enumerate() {
            assert!(!colors[i + 1..].contains(color));
        }
    }
    let mut members: Vec<_> = schedule.groups().flatten().copied().collect();
    members.sort_unstable();
    assert_eq!(members, (0..p.cols()).collect::<Vec<_>>());
    assert_eq!(schedule, derivative_coloring(p).unwrap());
}
#[test]
fn exhaustive_small_graphs_and_unsymmetric_storage() {
    for n in 0..=5 {
        let pairs: Vec<_> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();
        for mask in 0..(1usize << pairs.len()) {
            // Upper-triangular storage independently exercises structural symmetrization.
            let p = pattern(n, n, |i, j| {
                pairs
                    .iter()
                    .enumerate()
                    .any(|(bit, &pair)| pair == (i, j) && mask & (1 << bit) != 0)
            });
            check(p.view());
        }
    }
}
#[test]
fn pathological_and_rectangular_patterns() {
    for n in [0usize, 1, 2, 17, 50] {
        for p in [
            pattern(n, n, |_, _| false),
            pattern(n, n, |i, j| i == j),
            pattern(n, n, |i, j| i.abs_diff(j) == 1),
            pattern(n, n, |i, j| i == 0 || j == 0),
            pattern(n, n, |_, _| true),
            pattern(n, n, |i, j| i / 3 == j / 3),
        ] {
            check(p.view());
        }
    }
    for (rows, cols) in [(0, 7), (7, 0), (3, 9), (9, 3)] {
        let p = pattern(rows, cols, |i, j| (i + j) % 3 != 0);
        check_derivative(p.view());
        assert!(approximate_minimum_degree(p.view()).is_err());
        assert!(reverse_cuthill_mckee(p.view()).is_err());
        assert!(distance_one_coloring(p.view()).is_err());
    }
}
#[test]
fn bandwidth_and_fill_evidence() {
    let n = 30;
    // A path scrambled by a fixed bijection.
    let p = pattern(n, n, |i, j| ((i * 7) % n).abs_diff((j * 7) % n) == 1);
    let bandwidth = |inverse: &[usize]| {
        adjacency(p.view())
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(_, &edge)| edge)
                    .map(move |(j, _)| inverse[i].abs_diff(inverse[j]))
            })
            .max()
            .unwrap_or(0)
    };
    let order = reverse_cuthill_mckee(p.view()).unwrap();
    assert_eq!(bandwidth(order.old_to_new()), 1);
    assert!(bandwidth(&(0..n).collect::<Vec<_>>()) > 1);
    let star = pattern(n, n, |i, j| i != j && (i == 0 || j == 0));
    assert_eq!(
        symbolic_fill(
            star.view(),
            approximate_minimum_degree(star.view())
                .unwrap()
                .new_to_old(),
        ),
        0
    );
    assert!(symbolic_fill(star.view(), &(0..n).collect::<Vec<_>>()) > 0);
}
#[test]
fn permutation_validation_and_transactional_application() {
    assert!(Permutation::new(vec![0, 0]).is_err());
    assert!(Permutation::new(vec![2, 0]).is_err());
    let p = Permutation::new(vec![2, 0, 1]).unwrap();
    let mut out = [0; 3];
    p.apply_into(&[10, 20, 30], &mut out).unwrap();
    assert_eq!(out, [30, 10, 20]);
    let mut original = [0; 3];
    p.inverse_into(&out, &mut original).unwrap();
    assert_eq!(original, [10, 20, 30]);
    assert!(p.apply_into(&[1, 2], &mut out).is_err());
    assert_eq!(out, [30, 10, 20]);
}
fn scalar_adapters<S: NumericScalar>() -> Result<(), SciOxideError> {
    let csr = CsrMatrix::from_raw(3, 3, vec![0, 2, 3, 4], vec![0, 2, 1, 0], vec![S::ZERO; 4])?;
    let borrowed = csr.structural_pattern();
    let owned = borrowed.to_owned()?;
    let csc = csr.to_csc()?;
    let converted = csc.structural_pattern();
    assert_eq!(owned.view(), converted);
    assert_eq!(
        approximate_minimum_degree(borrowed)?,
        approximate_minimum_degree(converted)?
    );
    assert_eq!(
        derivative_coloring(borrowed)?,
        derivative_coloring(owned.view())?
    );
    Ok(())
}
#[test]
fn scalar_independent_owned_and_borrowed_adapters() -> Result<(), SciOxideError> {
    scalar_adapters::<f32>()?;
    scalar_adapters::<f64>()?;
    scalar_adapters::<Complex<f32>>()?;
    scalar_adapters::<Complex<f64>>()
}

#[test]
fn checked_sizes_and_stable_tie_results() {
    let huge = StructuralPatternView::new(0, usize::MAX, &[0], &[]).unwrap();
    assert!(
        matches ! (& (derivative_coloring (huge)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
    );
    let isolated = pattern(5, 5, |_, _| false);
    assert_eq!(
        reverse_cuthill_mckee(isolated.view()).unwrap().new_to_old(),
        &[0, 1, 2, 3, 4]
    );
    assert_eq!(
        approximate_minimum_degree(isolated.view())
            .unwrap()
            .new_to_old(),
        &[0, 1, 2, 3, 4]
    );
    assert_eq!(
        distance_one_coloring(isolated.view()).unwrap().colors(),
        &[0; 5]
    );
    let blocks = pattern(2, 4, |row, col| row == col / 2);
    assert_eq!(
        derivative_coloring(blocks.view()).unwrap().colors(),
        &[0, 1, 0, 1]
    );
}
