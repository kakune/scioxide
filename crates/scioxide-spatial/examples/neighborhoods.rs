//! Nearest neighbors and batched queries over plain coordinates.
use scioxide_core::SciOxideError;
use scioxide_spatial::{KdTree, PointSet};
fn main() -> Result<(), SciOxideError> {
    let sites = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 2.0], [5.0, 5.0]];
    let points = PointSet::from_points(&sites)?;
    let tree = KdTree::new(&points)?;
    for neighbor in tree.nearest(&[0.9, 0.1], 2)? {
        println!("point {}: distance {}", neighbor.index, neighbor.distance);
    }
    println!("batch: {:?}", tree.nearest_batch(&points, 2)?);
    Ok(())
}
