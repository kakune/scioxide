//! Shared geometry, oriented adjacency and real/complex located data.
use scioxide_core::{Complex64, SciOxideError};
use scioxide_spatial::{
    CellField, Point, SimplexBoundaryTags, TriangleMesh, UnstructuredFaceField, VertexField,
};
fn main() -> Result<(), SciOxideError> {
    let points = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        .into_iter()
        .map(Point::new)
        .collect::<Result<Vec<_>, _>>()?;
    let mesh = TriangleMesh::new(points, vec![[0, 1, 2], [0, 2, 3]])?;
    let temperature = VertexField::<f32, 2>::from_fn(&mesh, |vertex| vertex as f32)?;
    let wave = CellField::<Complex64, 2>::from_fn(&mesh, |_| Complex64::new(1., 0.))?;
    let flux = UnstructuredFaceField::<f64, 2>::from_fn(&mesh, |_| 0.)?;
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|face| (face, 7)).collect())?;
    let area = mesh.cell_measure(0)? + mesh.cell_measure(1)?;
    println!(
        "area={area}; boundary faces={}; vertices={}; cells={}; faces={}",
        tags.len(),
        temperature.len(),
        wave.len(),
        flux.len()
    );
    for (cell, vertices) in mesh.cells().enumerate() {
        println!(
            "cell {cell}: {vertices:?}, neighbors {:?}",
            mesh.cell_neighbors(cell)?.collect::<Vec<_>>()
        );
    }
    Ok(())
}
