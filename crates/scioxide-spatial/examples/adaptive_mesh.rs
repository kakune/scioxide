//! Caller-controlled mesh publication and index-only repartition transfer.
use scioxide_core::SciOxideError;
use scioxide_spatial::{
    AdaptationLimits, AdaptationRequest, AdaptiveMesh, CellField, MeshPartition, Point,
    RepartitionMap, TriangleMesh,
};
fn main() -> Result<(), SciOxideError> {
    let base = TriangleMesh::new(
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
            .map(Point::new)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?,
        vec![[0, 1, 2], [0, 2, 3]],
    )?;
    let limits = AdaptationLimits::new()
        .with_max_depth(8)
        .with_max_cells(1024)
        .with_max_entities(8192)
        .with_max_work(10_000_000)
        .with_max_storage_bytes(32 * 1024 * 1024);
    let mut mesh = AdaptiveMesh::from_simplex(&base, limits)?;
    let candidate = mesh.adapt_scoped(
        AdaptationRequest {
            source: mesh.mesh_token(),
            refine: &[0],
            coarsen: &[],
        },
        scioxide_core::ExecutionContext::sequential(),
    )?;
    candidate
        .map
        .validate(mesh.mesh_token(), candidate.mesh.mesh_token())?;
    println!(
        "{} cells -> {} cells, {} hanging entities",
        mesh.cell_count(),
        candidate.mesh.cell_count(),
        candidate.mesh.hanging_entities().len()
    );
    // A consumer prepares any physical field transfer before committing this candidate.
    mesh = candidate.mesh;
    let field = CellField::<f64, 2>::from_fn(&mesh, |cell| cell as f64)?;
    let source = MeshPartition::balanced(&mesh, 2)?;
    let destination = MeshPartition::new(&mesh, 2, &[1, 0, 1, 0, 1])?;
    let map = RepartitionMap::with_options(
        &source,
        &destination,
        scioxide_spatial::RepartitionOptions::new(),
    )?;
    let input = source
        .cell_order()
        .iter()
        .map(|&cell| field.values()[cell])
        .collect::<Vec<_>>();
    let mut output = vec![0.; mesh.cell_count()];
    let mut staging = vec![0.; mesh.cell_count()];
    map.gather_into(
        &input,
        &mut output,
        &mut staging,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    println!("Repartitioned cell values: {output:?}");
    mesh = mesh
        .adapt_scoped(
            AdaptationRequest {
                source: mesh.mesh_token(),
                refine: &[],
                coarsen: &[0],
            },
            scioxide_core::ExecutionContext::sequential(),
        )?
        .mesh;
    println!(
        "Coarsened back to {} cells at generation {}",
        mesh.cell_count(),
        mesh.mesh_token().generation().get()
    );
    Ok(())
}
