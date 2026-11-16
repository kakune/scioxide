use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    PolynomialRootMethod, PolynomialRootOptions, PolynomialRootWorkspace, SystemRootSolver,
    SystemWorkspace,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! { static FAIL_AFTER: Cell<Option<usize>> = const { Cell::new(None) }; }
struct FailingAllocator;
fn fail() -> bool {
    FAIL_AFTER
        .try_with(|remaining| match remaining.get() {
            Some(0) => {
                remaining.set(None);
                true
            }
            Some(count) => {
                remaining.set(Some(count - 1));
                false
            }
            None => false,
        })
        .unwrap_or(false)
}
// SAFETY: Successful operations delegate unchanged to the system allocator;
// injected failures return null, as permitted by GlobalAlloc.
unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if fail() {
            std::ptr::null_mut()
        } else {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.alloc(layout) }
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if fail() {
            std::ptr::null_mut()
        } else {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.alloc_zeroed(layout) }
        }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if fail() {
            std::ptr::null_mut()
        } else {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.realloc(pointer, layout, size) }
        }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from `System` through this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: FailingAllocator = FailingAllocator;

#[test]
fn failed_system_preparation_can_be_retried() {
    let function = scioxide_rootfind::ResidualFn::new(3, |x: &[f64], out: &mut [f64]| {
        out.copy_from_slice(x);
        Ok(())
    });
    for admitted in [0, 1, 4, 8, 12] {
        let mut workspace = SystemWorkspace::new();
        FAIL_AFTER.set(Some(admitted));
        let outcome = scioxide_rootfind::NewtonSystem::new(
            scioxide_rootfind::NewtonSystemOptions::<f64>::default()
                .with_derivative_strategy(DerivativeStrategy::Central),
        )
        .unwrap()
        .workspace_requirements(&function)
        .and_then(|r| workspace.try_reserve(&r));
        FAIL_AFTER.set(None);
        assert_eq!(
            outcome.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
        scioxide_rootfind::NewtonSystem::new(
            scioxide_rootfind::NewtonSystemOptions::<f64>::default()
                .with_derivative_strategy(DerivativeStrategy::Central),
        )
        .unwrap()
        .workspace_requirements(&function)
        .and_then(|r| workspace.try_reserve(&r))
        .unwrap();
        assert_eq!(workspace.dimension(), 3);
    }
}

#[test]
fn failed_polynomial_preparation_can_be_retried() {
    for admitted in [0, 1, 2, 3] {
        let mut workspace = PolynomialRootWorkspace::<f64>::new();
        FAIL_AFTER.set(Some(admitted));
        let outcome = PolynomialRootWorkspace::workspace_requirements(
            8,
            &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth),
        )
        .and_then(|requirements| workspace.try_reserve(&requirements));
        FAIL_AFTER.set(None);
        assert_eq!(
            outcome.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
        PolynomialRootWorkspace::workspace_requirements(
            8,
            &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth),
        )
        .and_then(|requirements| workspace.try_reserve(&requirements))
        .unwrap();
    }
}

/// Run `call` with every allocation failing; panic if it allocated.
fn without_allocation<T>(call: impl FnOnce() -> T) -> T {
    FAIL_AFTER.set(Some(0));
    let value = call();
    let unspent = FAIL_AFTER.replace(None);
    assert_eq!(unspent, Some(0), "the call allocated");
    value
}

#[test]
fn scalar_solves_execute() {
    use scioxide_core::{Complex, Interval};
    use scioxide_rootfind::{
        Bisection, Brent, Brenth, Chandrupatla, ComplexHalley, ComplexMuller, ComplexNewton,
        ComplexRootSolver, ComplexScalarFn, ComplexSecant, Halley, Illinois, Muller, RegulaFalsi,
        Ridder, SafeguardedHalley, SafeguardedInitialGuess, ScalarFn, ScalarRootInput,
        ScalarRootMethod, ScalarRootOptions, ScalarRootSolver, Secant, Toms748,
    };
    let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0))
        .with_derivative(|x| Ok(2.0 * x))
        .with_second_derivative(|_| Ok(2.0));
    let complex = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)))
        .with_derivative(|z| Ok(z + z))
        .with_second_derivative(|_| Ok(Complex::new(2.0, 0.0)));
    let interval = Interval::new(0.0, 2.0).unwrap();
    let guarded = SafeguardedInitialGuess::new(interval, 1.0).unwrap();
    let options = ScalarRootOptions::default();
    let z0 = Complex::new(0.5, 0.5);
    let real = without_allocation(|| {
        [
            Bisection.solve(&f, interval, &options),
            RegulaFalsi.solve(&f, interval, &options),
            Illinois.solve(&f, interval, &options),
            Ridder.solve(&f, interval, &options),
            Brent.solve(&f, interval, &options),
            Brenth.solve(&f, interval, &options),
            Chandrupatla.solve(&f, interval, &options),
            Toms748.solve(&f, interval, &options),
            Halley.solve(&f, 1.0, &options),
            SafeguardedHalley.solve(&f, guarded, &options),
            Secant.solve(&f, (1.0, 2.0), &options),
            Muller.solve(&f, (1.0, 1.5, 2.0), &options),
            ScalarRootMethod::Brent.solve(&f, ScalarRootInput::Bracket(interval), &options),
        ]
    });
    let complex = without_allocation(|| {
        [
            ComplexNewton.solve(&complex, z0, &options),
            ComplexHalley.solve(&complex, z0, &options),
            ComplexSecant.solve(&complex, (z0, Complex::new(0.4, 0.6)), &options),
            ComplexMuller.solve(
                &complex,
                (
                    Complex::new(-1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                ),
                &options,
            ),
        ]
    });
    for r in real {
        assert!(r.unwrap().is_success());
    }
    for r in complex {
        assert!(r.unwrap().is_success());
    }
}

#[test]
fn finite_difference_scalar_derivatives_execute() {
    use scioxide_core::Interval;
    use scioxide_rootfind::{
        Newton, SafeguardedInitialGuess, SafeguardedNewton, ScalarFn, ScalarRootOptions,
        ScalarRootSolver,
    };
    let calls = Cell::new(0);
    let f = ScalarFn::new(|x: f64| {
        assert!((0.0..=2.0).contains(&x));
        calls.set(calls.get() + 1);
        Ok(x * x - 2.0)
    });
    let initial = SafeguardedInitialGuess::new(Interval::new(0.0, 2.0).unwrap(), 0.0).unwrap();
    let options = ScalarRootOptions::default();
    for strategy in [DerivativeStrategy::Forward, DerivativeStrategy::Central] {
        let (guarded, open) = without_allocation(|| {
            (
                SafeguardedNewton::new(strategy).solve(&f, initial, &options),
                Newton::new(strategy).solve(&f, 1.0, &options),
            )
        });
        assert!(guarded.unwrap().is_success());
        assert!(open.unwrap().is_success());
    }
    assert!(calls.get() > 0);
}

#[test]
fn bracket_search_allocates_nothing() {
    use scioxide_rootfind::{BracketSearchOptions, ScalarFn, find_bracket_with_options};
    let f = ScalarFn::new(|x: f64| Ok(x - 1e6));
    let bracket = without_allocation(|| {
        find_bracket_with_options(
            &f,
            scioxide_rootfind::BracketSearchInput::Guess(0.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &BracketSearchOptions::default(),
        )
    })
    .unwrap();
    assert!(bracket.lower() <= 1e6 && 1e6 <= bracket.upper());
    let options = BracketSearchOptions::default()
        .with_expansion(scioxide_rootfind::BracketExpansion::Multiplicative)
        .with_domain(core::ops::Bound::Excluded(0.0), core::ops::Bound::Unbounded);
    for input in [
        scioxide_rootfind::BracketSearchInput::Guess(1.0),
        scioxide_rootfind::BracketSearchInput::Pair(1.0, 2.0),
    ] {
        let bracket = without_allocation(|| {
            find_bracket_with_options(
                &f,
                input,
                scioxide_rootfind::BracketSearchDirection::Increasing,
                &options,
            )
        })
        .unwrap();
        assert!(bracket.interval().contains(1e6));
    }
}

#[test]
fn special_zero_reservation_is_fallible_and_retryable() {
    let options = scioxide_rootfind::SpecialZeroOptions::<f64>::default();
    for admitted in [0, 1] {
        FAIL_AFTER.set(Some(admitted));
        let result = scioxide_rootfind::SpecialZeroWorkspace::workspace(4, &options);
        FAIL_AFTER.set(None);
        assert_eq!(
            result.err().unwrap().kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
        assert_eq!(
            scioxide_rootfind::SpecialZeroWorkspace::workspace(4, &options)
                .unwrap()
                .capacity(),
            4
        );
    }
}
