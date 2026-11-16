//! Shared homogeneous shear fixture; its exact pressure and body load are zero.
#![allow(dead_code)]
extern crate alloc;
use alloc::vec::Vec;
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryPoint, ElasticityFormulation, ElasticityKinematics,
    ElasticityOptions, SimplexFemLimits, SimplexFemOptions, SimplexFemOrder, TaggedBoundaries,
};
use scioxide_spatial::{SimplexBoundaryTags, SimplexMesh};

pub fn options<R: RealScalar, const D: usize>(mixed: bool) -> ElasticityOptions<R> {
    ElasticityOptions::new(
        if D == 2 {
            ElasticityKinematics::PlaneStrain
        } else {
            ElasticityKinematics::ThreeDimensional
        },
        if mixed {
            ElasticityFormulation::MixedP2P1
        } else {
            ElasticityFormulation::Displacement(SimplexFemOrder::P2)
        },
    )
    .with_fem(SimplexFemOptions::new().with_limits(SimplexFemLimits::new(
        100_000,
        30_000_000,
        256,
        1_000_000_000,
    )))
}
pub fn boundaries<R: RealScalar, const D: usize>(
    mesh: &SimplexMesh<R, D>,
) -> Result<TaggedBoundaries<R, R, D>, SciOxideError> {
    let tags = SimplexBoundaryTags::new(mesh, mesh.boundary_faces().map(|f| (f, 0)).collect())?;
    TaggedBoundaries::new(
        mesh,
        &tags,
        D,
        vec![BoundaryCondition::dirichlet(
            0,
            BoundaryData::time_invariant(|p: BoundaryPoint<R, D>| {
                Ok(if p.component == 0 {
                    R::try_from_f64_lossy(0.04)? * p.point[1]
                } else {
                    R::ZERO
                })
            }),
        )],
    )
}
pub fn state<const D: usize>(
    nodes: usize,
    unknowns: usize,
    point: impl Fn(usize) -> Result<[f64; D], SciOxideError>,
) -> Result<Vec<f64>, SciOxideError> {
    let mut result = alloc::vec![0.;unknowns];
    for (i, v) in result.iter_mut().take(nodes).enumerate() {
        *v = 0.04 * point(i)?[1];
    }
    Ok(result)
}
