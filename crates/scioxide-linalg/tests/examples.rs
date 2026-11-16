//! Execute every shipped example through the ordinary test harness.

macro_rules! example {
    ($name:ident, $test:ident, $path:literal) => {
        #[path = $path]
        mod $name;

        #[test]
        fn $test() -> Result<(), Box<dyn std::error::Error>> {
            $name::main()?;
            Ok(())
        }
    };
}

example!(
    biorthogonal,
    biorthogonal_runs,
    "../examples/biorthogonal.rs"
);
example!(
    blas_kernels,
    blas_kernels_runs,
    "../examples/blas_kernels.rs"
);
example!(
    dense_basics,
    dense_basics_runs,
    "../examples/dense_basics.rs"
);
example!(
    direct_solvers,
    direct_solvers_runs,
    "../examples/direct_solvers.rs"
);
example!(eigenvalues, eigenvalues_runs, "../examples/eigenvalues.rs");
example!(
    flexible_krylov,
    flexible_krylov_runs,
    "../examples/flexible_krylov.rs"
);
example!(
    generalized_schur,
    generalized_schur_runs,
    "../examples/generalized_schur.rs"
);
example!(gsvd, gsvd_runs, "../examples/gsvd.rs");
example!(
    iterative_solvers,
    iterative_solvers_runs,
    "../examples/iterative_solvers.rs"
);
example!(
    krylov_least_squares,
    krylov_least_squares_runs,
    "../examples/krylov_least_squares.rs"
);
example!(
    ldl_indefinite,
    ldl_indefinite_runs,
    "../examples/ldl_indefinite.rs"
);
example!(
    matrix_actions,
    matrix_actions_runs,
    "../examples/matrix_actions.rs"
);
example!(
    matrix_equations,
    matrix_equations_runs,
    "../examples/matrix_equations.rs"
);
example!(
    matrix_functions,
    matrix_functions_runs,
    "../examples/matrix_functions.rs"
);
example!(
    orthogonal_updates,
    orthogonal_updates_runs,
    "../examples/orthogonal_updates.rs"
);
example!(polar_csd, polar_csd_runs, "../examples/polar_csd.rs");
example!(
    preconditioners,
    preconditioners_runs,
    "../examples/preconditioners.rs"
);
example!(qz, qz_runs, "../examples/qz.rs");
example!(refinement, refinement_runs, "../examples/refinement.rs");
example!(riccati, riccati_runs, "../examples/riccati.rs");
example!(
    sparse_matrix_functions,
    sparse_matrix_functions_runs,
    "../examples/sparse_matrix_functions.rs"
);
example!(
    structured_direct,
    structured_direct_runs,
    "../examples/structured_direct.rs"
);
example!(
    structured_operators,
    structured_operators_runs,
    "../examples/structured_operators.rs"
);
example!(subspaces, subspaces_runs, "../examples/subspaces.rs");

example!(
    workspace_operations,
    workspace_operations_run,
    "../examples/workspace_operations.rs"
);
