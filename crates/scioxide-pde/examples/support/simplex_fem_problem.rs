//! Analytic affine coupled equation for tests and the runnable example.
use scioxide_core::SciOxideError;
use scioxide_spatial::SimplexMesh;
pub fn affine<const D: usize>(point: [f64; D], component: usize) -> f64 {
    (component + 1) as f64 * (1. + point.iter().sum::<f64>())
}
/// Coupled anisotropic diffusion with unit reaction for `.0` components.
pub struct Affine(pub usize);
impl<const D: usize> scioxide_pde::SimplexFemCoefficients<f64, f64, D> for Affine {
    fn evaluate(&self, _: [f64; D], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
        let components = self.0;
        let kd = components * D;
        for a in 0..components {
            for b in 0..components {
                c[a * components + b] = if a == b { 1. } else { 0. };
                for axis in 0..D {
                    for direction in 0..D {
                        k[(a * D + axis) * kd + b * D + direction] =
                            (if a == b { 1. } else { 0.1 })
                                * (if axis == direction { 1. } else { 0.125 });
                    }
                }
            }
        }
        Ok(())
    }
}
/// Source of the affine solution for the operator of [`Affine`].
pub fn affine_source<const D: usize>(components: usize) -> scioxide_pde::FieldData<f64, f64, D> {
    scioxide_pde::FieldData::time_invariant(move |p: scioxide_pde::FieldPoint<f64, D>| {
        let _ = components;
        Ok(affine(p.point, p.component))
    })
}
pub fn affine_boundaries<const D: usize>(
    mesh: &SimplexMesh<f64, D>,
    components: usize,
) -> Result<scioxide_pde::TaggedBoundaries<f64, f64, D>, SciOxideError> {
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )?;
    scioxide_pde::TaggedBoundaries::new(
        mesh,
        &tags,
        components,
        vec![scioxide_pde::BoundaryCondition::dirichlet(
            0,
            scioxide_pde::BoundaryData::time_invariant(|p: scioxide_pde::BoundaryPoint<f64, D>| {
                Ok(affine(p.point, p.component))
            }),
        )],
    )
}
pub fn limits() -> scioxide_pde::SimplexFemLimits {
    scioxide_pde::SimplexFemLimits::new(100_000, 30_000_000, 256, 1_000_000_000)
}
