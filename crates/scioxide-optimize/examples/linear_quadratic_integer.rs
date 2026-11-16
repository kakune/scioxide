#![allow(
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::uninlined_format_args,
    clippy::similar_names
)]
//! Linear, quadratic, and integer programming demonstration for scioxide-optimize.
//!
//! Demonstrates LP, QP, and MIP problem formulation and solving:
//!
//! 1. **SimplexSolver** — LP with inequality constraints
//! 2. **InteriorPointLp** — Same LP problem, comparing results
//! 3. **QpSolver** — Quadratic program with linear constraints
//! 4. **BranchAndBound** — Mixed-integer LP
//!
//! Run: `cargo run --example linear_quadratic_integer -p scioxide-optimize`

use scioxide_core::{Bounds, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    BranchAndBound, InteriorPointLp, LinearConstraintSense, LinearConstraints, LpProblem,
    MipProblem, QpProblem, QpSolver, SimplexSolver,
};

/// Tolerance for floating-point comparisons.
const TOL: f64 = 0.1;

fn example_lp() -> Result<LpProblem<f64>, SciOxideError> {
    let constraints = LinearConstraints::new(
        Matrix::from_vec(3, 2, vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0])?,
        vec![4.0, 3.0, 3.0],
        vec![LinearConstraintSense::LessEqual; 3],
    )?;
    LpProblem::new(vec![-1.0, -2.0])?
        .with_constraints(constraints)?
        .with_bounds(Bounds::new(vec![(Some(0.0), None); 2])?)
}

// ---------------------------------------------------------------------------
// Part 1: LP with SimplexSolver
// ---------------------------------------------------------------------------

fn part1_simplex() -> Result<(), SciOxideError> {
    println!("=== Part 1: LP with SimplexSolver ===");
    println!();

    // min -x0 - 2*x1
    // s.t. x0 + x1 <= 4
    //      x0 <= 3
    //      x1 <= 3
    //      x0, x1 >= 0
    let problem = example_lp()?;

    let solver = SimplexSolver::<f64>::default();
    let res = solver.solve(&problem)?;

    println!("  min -x0 - 2*x1  s.t. x0+x1<=4, x0<=3, x1<=3, x>=0");
    println!(
        "  result = ({:.4}, {:.4})  obj = {:.4}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "Simplex should converge");
    // Optimal at (1, 3) with value -7
    assert!((res.fun() - (-7.0)).abs() < TOL, "Objective should be -7");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: LP with InteriorPointLp
// ---------------------------------------------------------------------------

fn part2_interior_point_lp() -> Result<(), SciOxideError> {
    println!("=== Part 2: LP with InteriorPointLp ===");
    println!();

    let problem = example_lp()?;

    let solver = InteriorPointLp::<f64>::default();
    let res = solver.solve(&problem)?;

    println!("  Same LP problem with InteriorPointLp:");
    println!(
        "  result = ({:.4}, {:.4})  obj = {:.4}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    assert!(res.is_success(), "InteriorPointLp should converge");
    assert!(
        (res.fun() - (-7.0)).abs() < TOL,
        "Objective should be -7, got {}",
        res.fun()
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: QP with QpSolver
// ---------------------------------------------------------------------------

fn part3_qp() -> Result<(), SciOxideError> {
    println!("=== Part 3: QP with QpSolver ===");
    println!();

    // min 0.5 * x^T Q x + c^T x
    // Q = [[2, 0], [0, 2]] (identity * 2)
    // c = [-2, -4]
    // s.t. x0 + x1 <= 3, x0 >= 0, x1 >= 0
    let constraints = LinearConstraints::new(
        Matrix::from_vec(1, 2, vec![1.0, 1.0])?,
        vec![3.0],
        vec![LinearConstraintSense::LessEqual],
    )?;
    let problem = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0])?,
        vec![-2.0, -4.0],
    )?
    .with_constraints(constraints)?
    .with_bounds(Bounds::new(vec![(Some(0.0), None); 2])?)?;

    let solver = QpSolver::<f64>::default();
    let res = solver.solve(&problem)?;

    println!("  min x0^2 + x1^2 - 2*x0 - 4*x1  s.t. x0+x1<=3, x>=0");
    println!(
        "  result = ({:.6}, {:.6})  obj = {:.6}  iters = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.iterations(),
        res.is_success()
    );

    // Unconstrained optimum at (1, 2), sum=3, so constraint is tight
    assert!(res.is_success(), "QpSolver should converge");
    assert!(
        (res.x()[0] - 1.0).abs() < TOL,
        "x0 should be near 1, got {}",
        res.x()[0]
    );
    assert!(
        (res.x()[1] - 2.0).abs() < TOL,
        "x1 should be near 2, got {}",
        res.x()[1]
    );

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: MIP with BranchAndBound
// ---------------------------------------------------------------------------

fn part4_branch_and_bound() -> Result<(), SciOxideError> {
    println!("=== Part 4: MIP with BranchAndBound ===");
    println!();

    // min -x0 - 2*x1
    // s.t. x0 + x1 <= 4, x0 <= 3, x1 <= 3, x >= 0
    // x0, x1 integer
    let problem = MipProblem::new(example_lp()?, vec![0, 1])?;

    let solver = BranchAndBound::<f64>::default();
    let res = solver.solve(&problem)?;

    println!("  min -x0 - 2*x1 (integer)  s.t. x0+x1<=4, x0<=3, x1<=3");
    println!(
        "  result = ({:.0}, {:.0})  obj = {:.0}  nodes = {}  success = {}",
        res.x()[0],
        res.x()[1],
        res.fun(),
        res.nodes(),
        res.is_success()
    );

    assert!(res.is_success(), "BranchAndBound should converge");
    // Accepted coordinates are rounded and the original objective is recomputed.
    assert!(res.x().iter().all(|value| *value == value.round()));
    assert_eq!(res.fun(), -res.x()[0] - 2.0 * res.x()[1]);
    assert_eq!(res.integrality_violation(), Some(0.0));
    // Integer optimum at (1, 3), value = -7
    assert!(
        (res.fun() - (-7.0)).abs() < TOL,
        "Objective should be -7, got {}",
        res.fun()
    );

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-optimize Linear / Quadratic / Integer Programming");
    println!("=========================================================");
    println!();

    part1_simplex()?;
    part2_interior_point_lp()?;
    part3_qp()?;
    part4_branch_and_bound()?;

    println!("All examples completed successfully.");
    Ok(())
}
