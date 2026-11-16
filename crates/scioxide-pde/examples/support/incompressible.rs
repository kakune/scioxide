//! Independent affine shear/linear-pressure solution on the unit box.
#![allow(dead_code, unreachable_pub)]
extern crate alloc;
use alloc::{vec, vec::Vec};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryPoint, EllipticGauge, IncompressibleOptions,
    SimplexFemLimits, SimplexFemOptions, TaggedBoundaries,
};
use scioxide_spatial::{SimplexBoundaryTags, SimplexMesh};
pub fn options<R: RealScalar, V: NumericScalar<Real = R>>() -> IncompressibleOptions<R, V> {
    IncompressibleOptions::new()
        .with_fem(SimplexFemOptions::new().with_limits(SimplexFemLimits::new(
            100_000,
            30_000_000,
            256,
            1_000_000_000,
        )))
        .with_gauges(vec![EllipticGauge::WeightedMean(V::ZERO)])
}
pub fn boundaries<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
>(
    mesh: &SimplexMesh<R, D>,
    amplitude: V,
) -> Result<TaggedBoundaries<R, V, D>, SciOxideError> {
    let tags = SimplexBoundaryTags::new(mesh, mesh.boundary_faces().map(|f| (f, 0)).collect())?;
    TaggedBoundaries::new(
        mesh,
        &tags,
        D,
        vec![BoundaryCondition::dirichlet(
            0,
            BoundaryData::time_invariant(move |p: BoundaryPoint<R, D>| {
                Ok(if p.component == 0 {
                    amplitude * V::from_real(p.point[1])
                } else {
                    V::ZERO
                })
            }),
        )],
    )
}
pub fn state<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    plan: &scioxide_pde::StokesPlan<R, V, D>,
    amplitude: V,
    pressure: V,
) -> Result<Vec<V>, SciOxideError> {
    let mut x = vec![V::ZERO; plan.unknowns()];
    for (i, v) in x.iter_mut().take(plan.velocity_nodes()).enumerate() {
        *v = amplitude * V::from_real(plan.node_point(i)?[1]);
    }
    for (i, p) in plan.mesh().points().iter().enumerate() {
        x[plan.velocity_unknowns() + i] = pressure * V::from_real(p.as_array()[0] - R::HALF);
    }
    Ok(x)
}
