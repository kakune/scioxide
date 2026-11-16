//! Shared geometry, interpolation coordinates and bounded dual construction.
use scioxide_spatial::{Delaunay, PointSet, Voronoi};
fn main() -> Result<(), scioxide_core::SciOxideError> {
    let coordinates = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let points = PointSet::from_points(&coordinates)?;
    let geometry = Delaunay::new(&points)?;
    if let Some(location) = geometry.find_simplex(&[0.3, 0.7])? {
        let weights = geometry.barycentric(location.simplex, &[0.3, 0.7])?;
        println!(
            "simplex {:?}, barycentric {weights:?}",
            geometry.simplex(location.simplex)?
        );
    }
    let mesh = geometry.triangle_mesh()?;
    println!("shared triangle mesh has {} cells", mesh.cell_count());
    let dual = Voronoi::new(&points)?;
    println!(
        "square Voronoi vertex {:?}; {} unbounded rays",
        dual.vertices()?,
        dual.rays().len()
    );
    for ridge in dual.ridges() {
        println!(
            "sites {:?} share vertices {:?}",
            ridge.sites(),
            ridge.vertices()
        );
    }
    Ok(())
}
