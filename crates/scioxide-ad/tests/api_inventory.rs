//! The sole authoritative scioxide-ad root census.
//! Cross-crate canonical API checks defer to this inventory.
//!
//! Every crate-root export is imported here as an external consumer, and a
//! test fails when this list and the `pub use` items of `src/lib.rs` differ.
//! Thread-safety of the tape, linearization and batch types and the capacity
//! protocol are asserted at compile time.
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_ad::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    // Traits.
    Real, BaseReal, Mask, TapeValue, BatchFunction, BatchData, ExternalRule, Residual, LoopBody,
    ComplexElementary,
    // Types.
    Lanes, LaneMask, Dual, MultiDual, Jet, Var, Tape, TapeOptions, TapeRequirements, TapeReport,
    Linearization, JacobianOperator, HessianOperator, JacobianMode, KinkPolicy, NonFinitePolicy,
    BatchPlan, BatchOptions, BatchOutputs, BatchWorkspace, BatchRequirements, BatchEstimate,
    BatchReduction, BatchMode, DerivativeMode, CheckpointSchedule, CheckpointAction, BatchDataFn,
    // Functions.
    grad, value_and_grad, jvp, vjp, jacobian, hvp, hessian, jet, dot, gemv_into, gemm_into,
    solve_into, cholesky_into, qr_into, least_squares_into, logsumexp, softmax_into,
    log_softmax_into, implicit_solution_into, checkpointed_loop, batch_mean, batch_mean_and_grad,
    sparse_jacobian, sparse_hessian,
}

/// Names re-exported by the `pub use` statements of a crate root. Each
/// statement re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("\npub use ")
        .skip(1)
        .filter_map(|statement| statement.split_once(';'))
        .flat_map(|(path, _)| {
            path.rsplit_once("::")
                .map_or(path, |(_, names)| names)
                .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
                .split(',')
        })
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn inventory_lists_every_root_export_once() {
    let listed: BTreeSet<_> = INVENTORY.iter().copied().collect();
    assert_eq!(listed.len(), INVENTORY.len(), "duplicate inventory entry");
    assert_eq!(listed.len(), 62);
    let root = root_exports(include_str!("../src/lib.rs"));
    let unlisted: Vec<_> = root.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&root).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "update tests/api_inventory.rs\nunlisted: {unlisted:?}\nstale: {stale:?}"
    );
}

#[test]
fn root_exports_read_single_and_braced_statements() {
    let source =
        "mod a;\npub use a::One;\npub use b::{\n    Two, three,\n};\npub(crate) use c::Hidden;";
    assert_eq!(
        root_exports(source),
        BTreeSet::from(["One", "Two", "three"])
    );
}

// Thread safety and the capacity protocol of the reviewed public surface.
#[allow(clippy::type_complexity)]
const _: fn() = || {
    use scioxide_core::{Capacity, SciOxideError};
    fn send_sync<T: Send + Sync>() {}
    fn sync<T: Sync>() {}
    fn clone<T: Clone>() {}
    send_sync::<Tape<f64>>();
    send_sync::<Tape<Lanes<f32, 8>>>();
    send_sync::<Tape<Dual<Lanes<f64, 4>>>>();
    sync::<Linearization<'static, f64>>();
    send_sync::<BatchPlan<8>>();
    clone::<BatchPlan<8>>();
    send_sync::<BatchWorkspace<f64, 8>>();
    let _: fn(&mut Tape<f64>, &TapeRequirements) -> Result<(), SciOxideError> = Tape::try_reserve;
    let _: fn(&Tape<f64>, &TapeRequirements) -> Result<Capacity, SciOxideError> = Tape::check;
    let _: fn(&Tape<f64>) -> TapeRequirements = Tape::requirements;
    let _: fn(&BatchPlan<8>, usize) -> Result<BatchRequirements, SciOxideError> =
        BatchPlan::workspace_requirements;
    let _: fn(&BatchPlan<8>, usize) -> Result<BatchWorkspace<f64, 8>, SciOxideError> =
        BatchPlan::workspace;
    let _: fn(&mut BatchWorkspace<f64, 8>, &BatchRequirements) -> Result<(), SciOxideError> =
        BatchWorkspace::try_reserve;
    let _: fn(&BatchWorkspace<f64, 8>, &BatchRequirements) -> Result<Capacity, SciOxideError> =
        BatchWorkspace::check;
    // Defaults, sources, seeding and schedules.
    fn default<T: Default>() {}
    fn data<T: BatchData<f64> + ?Sized>() {}
    fn copy_sync<T: Copy + Sync>() {}
    default::<Tape<f64>>();
    default::<Tape<Dual<f64>>>();
    data::<[f64]>();
    data::<Vec<f64>>();
    data::<[f64; 3]>();
    data::<[f64; 0]>();
    type Source =
        BatchDataFn<f64, fn(usize, usize, usize, &mut [f64]) -> Result<(), SciOxideError>>;
    data::<Source>();
    copy_sync::<Source>();
    let _: fn([f64; 3]) -> [MultiDual<f64, 3>; 3] = MultiDual::<f64, 3>::variables;
    let _: fn(usize, &[usize]) -> Result<CheckpointSchedule, SciOxideError> =
        CheckpointSchedule::explicit;
};

// `Var` is neither `Send` nor `Sync`: either marker implementation would make
// the probe below ambiguous.
const _: fn() = || {
    struct Sent;
    struct Shared;
    trait AmbiguousIfSend<A> {
        fn marker() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfSend<Sent> for T {}
    trait AmbiguousIfSync<A> {
        fn marker() {}
    }
    impl<T: ?Sized> AmbiguousIfSync<()> for T {}
    impl<T: ?Sized + Sync> AmbiguousIfSync<Shared> for T {}
    let _ = <Var<'static, f64> as AmbiguousIfSend<_>>::marker;
    let _ = <Var<'static, f64> as AmbiguousIfSync<_>>::marker;
};
