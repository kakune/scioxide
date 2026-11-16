//! Inspect prepared P1 element forms on intervals and solve the
//! variable-coefficient problem `-((1 + x) u')' = -1` with `u = 1 + x` on the
//! boundary, whose exact solution `u = 1 + x` P1 reproduces.
use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, EllipticOptions, EllipticProblem, FieldData,
    SimplexFemCoefficients, SimplexFemGeometry, SimplexFemLimits, SimplexFemOptions,
    SimplexFemOrder, SimplexFemPlan, SpatialDiscretization, TaggedBoundaries,
};
use scioxide_spatial::{IntervalMesh, Point, SimplexBoundaryTags};

struct Model;
impl SimplexFemCoefficients<f64, f64, 1> for Model {
    fn evaluate(
        &self,
        point: [f64; 1],
        diffusion: &mut [f64],
        reaction: &mut [f64],
    ) -> Result<(), SciOxideError> {
        diffusion[0] = 1.0 + point[0];
        reaction[0] = 0.0;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points = (0..=16)
        .map(|i| Point::new([f64::from(i) / 16.0]))
        .collect::<Result<Vec<_>, _>>()?;
    let mesh = IntervalMesh::new(points, (0..16).map(|i| [i, i + 1]).collect())?;
    let tags = SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect())?;
    let boundaries = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![BoundaryCondition::dirichlet(
            0,
            BoundaryData::time_invariant(|p| Ok(1.0 + p.point[0])),
        )],
    )?;
    let limits = SimplexFemLimits::new(1 << 10, 1 << 14, 16, 1 << 24);
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P1,
        &Model,
        &boundaries,
        SimplexFemOptions::new().with_limits(limits),
    )?;
    println!("first element matrix: {:?}", plan.element_matrix(0)?);
    println!("first element mass: {:?}", plan.element_mass(0)?);
    let source = FieldData::constant(-1.0);
    let load = plan.load(&source, 0.0)?;
    println!("load (Dirichlet rows hold values): {:?}", &load[..3]);
    let solution =
        EllipticProblem::new(&plan, &source).solve_with_options(&(EllipticOptions::new()))?;
    println!(
        "termination={:?}, values={:?}",
        solution.report().termination(),
        solution.values()
    );
    Ok(())
}
