//! Fixed simplex and sphere surface integration with borrowed geometry.

use scioxide_core::{Budget, SciOxideError};
use scioxide_integrate::{LebedevRule, SimplexRule};
use scioxide_spatial::{Point, TriangleMesh};

fn main() -> Result<(), SciOxideError> {
    let triangle = SimplexRule::<f64>::new(
        2,
        1,
        scioxide_integrate::GeometricOptions::default().with_max_nodes(Budget::try_from(4)?),
    )?;
    let a = [0.0, 0.0];
    let b = [2.0, 0.0];
    let c = [0.0, 3.0];
    let integral = triangle.integrate(|point: &[f64]| Ok(point[0] + point[1]), &[&a, &b, &c])?;
    println!("triangle integral of x+y: {:?}", integral.0); // 5

    let mesh = TriangleMesh::new(
        vec![Point::new(a)?, Point::new(b)?, Point::new(c)?],
        vec![[0, 1, 2]],
    )?;
    let mut workspace = triangle.workspace(1)?;
    let mut points = vec![0.0; triangle.node_count() * 2];
    let mut weights = vec![0.0; triangle.node_count()];
    let mapped =
        triangle.map_mesh_cell_into(&mesh, 0, &mut points, &mut weights, &mut workspace)?;
    println!("mesh cell area: {}", mapped.measure); // 3

    let sphere = LebedevRule::<f64>::new(
        7,
        scioxide_integrate::GeometricOptions::default().with_max_nodes(Budget::try_from(26)?),
    )?;
    let surface = sphere.integrate(|point: &[f64]| Ok(point[2] * point[2]), &[0.0; 3], 1.0)?;
    println!("sphere integral of z²: {:?}", surface.0); // 4π/3
    Ok(())
}
