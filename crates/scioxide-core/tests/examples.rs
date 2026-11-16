//! Execute caller-owned storage and execution examples.
#[path = "../examples/workspace_operations.rs"]
mod workspace_operations;

#[test]
fn workspace_operations_run() -> Result<(), scioxide_core::SciOxideError> {
    workspace_operations::main()
}
