//! Shortest paths from vertex 0 of a weighted directed graph.
//! A stored zero is a usable edge; vertex 3 remains unreachable.
use scioxide_core::SciOxideError;
use scioxide_sparse::CsrMatrix;
fn main() -> Result<(), SciOxideError> {
    // Edges 0 → 1 (weight 0), 1 → 2 (weight 2) and 0 → 2 (weight 5).
    let a = CsrMatrix::from_triplets(4, 4, &[0, 1, 0], &[1, 2, 2], &[0.0_f64, 2.0, 5.0])?;
    let paths = scioxide_sparse::shortest_paths(a.view(), 0)?;
    let view = paths.as_view();
    println!("Distances: {:?}", view.distances());
    println!("Path to 2: {:?}", view.path(0, 2)?);
    println!("Path to 3: {:?}", view.path(0, 3)?);
    Ok(())
}
