//! Exact matching/flow witnesses, approximate assignment duals, and a graph Laplacian.
use scioxide_core::{LinearOperator, SciOxideError};
use scioxide_sparse::*;
fn main() -> Result<(), SciOxideError> {
    // Costs [[2, 1], [1, ·]]: row 1 can only take column 0.
    let matrix = CsrMatrix::from_triplets(2, 2, &[0, 0, 1], &[0, 1, 0], &[2.0_f64, 1.0, 1.0])?;
    let pattern = matrix.structural_pattern();
    let matching = maximum_bipartite_matching(pattern)?;
    println!(
        "matching cardinality {}, exact witness {}",
        matching.as_view().cardinality(),
        matching.as_view().verify(pattern)
    );
    let assignment = minimum_cost_assignment(&matrix)?;
    let plan = AssignmentPlan::new(pattern)?;
    println!(
        "assignment objective {}, approximate dual diagnostics {:?}",
        assignment.as_view().objective,
        plan.verify(
            matrix.values(),
            AssignmentObjective::Minimize,
            assignment.as_view()
        )?
    );
    let edges = [CapacityEdge {
        source: 0,
        target: 1,
        capacity: 7,
    }];
    let network = MaxFlowPlan::new(2, &edges)?;
    let flow = network.maximum_flow(0, 1)?;
    println!(
        "flow {}, exact cut witness {}",
        flow.as_view().value,
        flow.as_view().verify(&network)?
    );
    let graph = GraphPlan::new(matrix.view(), GraphDirection::Undirected)?;
    let laplacian =
        graph.laplacian::<f64>(LaplacianDegree::Out, LaplacianNormalization::Symmetric)?;
    let mut output = [0.0; 2];
    let mut scratch = [0.0; 2];
    laplacian.apply_into(&[1.0; 2], &mut output, &mut scratch)?;
    println!("normalized Laplacian times ones: {output:?}");
    Ok(())
}
