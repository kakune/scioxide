//! Reuse structural orderings and derivative seed groups independently of values.
use scioxide_core::{SciOxideError, StructuralPatternView};
use scioxide_sparse::{approximate_minimum_degree, derivative_coloring, reverse_cuthill_mckee};

fn main() -> Result<(), SciOxideError> {
    // Upper triangular storage of a four-vertex path; no numerical values needed.
    let pattern = StructuralPatternView::new(4, 4, &[0, 1, 2, 3, 3], &[1, 2, 3])?;
    let rcm = reverse_cuthill_mckee(pattern)?;
    let degree = approximate_minimum_degree(pattern)?;
    println!("RCM new-to-old: {:?}", rcm.new_to_old());
    println!("Minimum degree: {:?}", degree.new_to_old());
    let mut reordered = [0.0; 4];
    rcm.apply_into(&[1.0, 2.0, 3.0, 4.0], &mut reordered)?;

    // A rectangular Jacobian pattern: columns sharing a row need distinct seeds.
    let jacobian = StructuralPatternView::new(2, 4, &[0, 2, 4], &[0, 1, 2, 3])?;
    let schedule = derivative_coloring(jacobian)?;
    schedule.validate_pattern(jacobian)?;
    for (color, columns) in schedule.groups().enumerate() {
        println!("Seed group {color}: {columns:?}");
    }
    Ok(())
}
