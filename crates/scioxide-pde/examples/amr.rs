//! Caller-controlled solve/estimate/mark/refine/transfer loop for localized Poisson.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_pde::*;
use scioxide_spatial::AdaptiveMesh;
#[path = "support/amr.rs"]
mod amr;
#[path = "support/simplex_fem.rs"]
mod meshes;
#[path = "support/amr_poisson.rs"]
mod poisson;
fn main() -> Result<(), SciOxideError> {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "amg".into());
    let mesh =
        AdaptiveMesh::from_simplex(&meshes::triangle_grid::<f64>(4)?, amr::topology_limits())?;
    let mut space = AmrSpace::with_options(
        &mesh,
        AmrOrder::P1,
        AmrOptions::new().with_limits(amr::limits()),
    )?;
    let mut tags = amr::tags(&mesh)?;
    let mut seed = Vec::new();
    println!("family,cycle,cells,dofs,l2_error,energy_error,estimate,effectivity");
    for cycle in 0..7 {
        let (values, indicators) = poisson::solve(&space, &tags, &seed, &backend)?;
        let (l2, energy) = poisson::errors(&space, &values, 4)?;
        println!(
            "adaptive,{cycle},{},{},{l2:.9e},{energy:.9e},{:.9e},{:.9e}",
            space.mesh().cell_count(),
            space.independent_dofs(),
            indicators.global(),
            indicators.global() / energy
        );
        if cycle == 6 {
            break;
        }
        let marks = AmrMarking::new(0.5).mark_scoped(
            space.mesh(),
            &indicators,
            amr::limits(),
            ExecutionContext::sequential_native(),
        )?;
        let candidate = space
            .mesh()
            .adapt_scoped(marks.request(), ExecutionContext::sequential_native())?;
        let next_tags = tags.transfer_with_options(
            space.mesh(),
            &candidate.mesh,
            &candidate.map,
            amr::limits(),
        )?;
        let next = AmrSpace::with_options(
            &candidate.mesh,
            AmrOrder::P1,
            AmrOptions::new().with_limits(amr::limits()),
        )?;
        let mut next_values = vec![0.; next.full_dofs()];
        {
            let transfer =
                AmrNodalTransfer::with_options(&space, &next, &candidate.map, amr::limits())?;
            let mut work = transfer.workspace()?;
            transfer.interpolate_into(
                &values,
                &mut next_values,
                &mut work,
                ExecutionContext::sequential_native(),
            )?;
        }
        // Publish topology, boundary labels and transferred field together, after success.
        space = next;
        tags = next_tags;
        seed = next_values;
    }
    for n in [4, 8, 12, 16] {
        let mesh =
            AdaptiveMesh::from_simplex(&meshes::triangle_grid::<f64>(n)?, amr::topology_limits())?;
        let space = AmrSpace::with_options(
            &mesh,
            AmrOrder::P1,
            AmrOptions::new().with_limits(amr::limits()),
        )?;
        let tags = amr::tags(&mesh)?;
        let (values, indicators) = poisson::solve(&space, &tags, &[], &backend)?;
        let (l2, energy) = poisson::errors(&space, &values, 4)?;
        println!(
            "uniform,{n},{},{},{l2:.9e},{energy:.9e},{:.9e},{:.9e}",
            mesh.cell_count(),
            space.independent_dofs(),
            indicators.global(),
            indicators.global() / energy
        );
    }
    Ok(())
}
