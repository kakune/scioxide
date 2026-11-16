//! Allocation failures from shared linear-algebra storage remain typed.

#![allow(clippy::unwrap_used, unsafe_code)]

use scioxide_poly::PolynomialOperationRequest;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr;

use scioxide_core::SciOxideError;
use scioxide_poly::{
    ApproxPolynomial, ExactPolynomial, OrthogonalFamily, PolynomialApproximationPolicy,
};

thread_local! {
    static FAIL_NEXT_ALLOCATION: Cell<bool> = const { Cell::new(false) };
}

struct FailingAllocator;

// SAFETY: successful operations forward the original layout to `System`.
// An explicitly armed allocation on the current test thread returns null, as
// permitted by `GlobalAlloc`.
unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false)) {
            ptr::null_mut()
        } else {
            // SAFETY: `layout` is supplied by the caller under `GlobalAlloc`.
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pair came from a successful `System` allocation above.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: FailingAllocator = FailingAllocator;

fn assert_next_allocation_is_typed<T>(operation: impl FnOnce() -> Result<T, SciOxideError>) {
    FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
    assert!(matches!(
        operation(),
        Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::AllocationFailure { .. })
    ));
}

#[test]
fn golub_welsch_propagates_shared_matrix_allocation_failure() {
    assert_next_allocation_is_typed(|| OrthogonalFamily::<f64>::Legendre.zeros(8));
}

#[test]
fn prepared_polynomial_construction_propagates_allocation_failures() {
    let policy = PolynomialApproximationPolicy::<f64>::default();
    let roots = [1.0_f64, 2.0, 3.0];
    assert_next_allocation_is_typed(|| ApproxPolynomial::from_roots_with_options(&roots, policy));

    // Ordinary owned samples need no allocation; range promotion remains fallible.
    let nodes = vec![-f64::MAX, f64::MAX];
    let values = vec![-f64::MAX, f64::MAX];
    assert_next_allocation_is_typed(|| {
        ApproxPolynomial::interpolate_newton_with_options(nodes, values, policy)
    });
}

#[test]
fn public_basis_kernels_report_allocation_failure() {
    use scioxide_poly::backend::polynomial_basis::barycentric_weights;

    assert_next_allocation_is_typed(|| barycentric_weights(&[0.0_f64, 1.0]));
}

#[path = "support/integer.rs"]
mod integer;
use integer::Integer;

#[test]
fn exact_polynomial_operations_propagate_allocation_failures() {
    let left = ExactPolynomial::new(vec![Integer(1), Integer(2)]).unwrap();
    let right = ExactPolynomial::new(vec![Integer(3), Integer(4)]).unwrap();
    assert_next_allocation_is_typed(|| left.try_add(&right));
}

#[test]
fn orthogonal_preparation_is_fallible_and_warmed_kernels_do_not_allocate() {
    use scioxide_core::LinearOperator;
    use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Orthonormal,
        0.0_f64,
        1.0,
    )
    .unwrap();
    assert_next_allocation_is_typed(|| {
        spec.prepare(8, scioxide_poly::PolynomialExecutionOptions::new())
    });
    let plan = spec
        .prepare(3, scioxide_poly::PolynomialExecutionOptions::new())
        .unwrap();
    assert_next_allocation_is_typed(|| plan.antiderivative_plan());
    assert_next_allocation_is_typed(|| plan.from_monomial_operator());
    assert_next_allocation_is_typed(|| plan.monomial_operator());
    assert_next_allocation_is_typed(|| plan.conversion_operator(&plan));
    let derivative = plan.derivative_operator();
    let primitive = plan.antiderivative_plan().unwrap();
    let coefficients = [1.0, 0.2, 0.3, 0.4];
    let polynomial = ApproxPolynomial::orthogonal_with_plan(
        coefficients.to_vec(),
        spec.prepare(coefficients.len() - 1, Default::default())
            .unwrap(),
        PolynomialApproximationPolicy::default(),
    )
    .unwrap();
    let mut output = [0.0; 3];
    let mut scratch = [0.0; 5];
    let mut primitive_output = [0.0; 5];
    FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
    let scalar = OrthogonalFamily::Jacobi {
        alpha: 0.25,
        beta: 0.75,
    }
    .evaluate(32, 0.5_f64);
    let extended_scalar = OrthogonalFamily::HermiteH.evaluate(200, 0.0_f64);
    let extended_derivative = OrthogonalFamily::HermiteH.evaluate_derivative(201, 0.0_f64);
    let evaluation = plan.evaluate_with_derivative(&coefficients, 0.5);
    let first_many = polynomial
        .evaluate_many_with_derivative(&[0.0, 0.25, 0.5, 0.75, 1.0])
        .try_for_each(|result| result.map(|_| ()));
    let first_only = polynomial
        .evaluate_many_with_derivative(&[0.0, f64::NAN])
        .next();
    let repeated_many = polynomial
        .evaluate_many_with_derivative(&[0.0, 0.25, 0.5, 0.75])
        .try_for_each(|result| result.map(|_| ()));
    let differentiated = derivative.apply_into(&coefficients, &mut output, &mut scratch);
    let integrated = primitive.apply_into(
        &coefficients,
        (0.0, 0.0),
        &mut primitive_output,
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    );
    let no_allocation = FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false));
    assert!(no_allocation);
    assert!(scalar.is_ok());
    assert!(extended_scalar.is_ok());
    assert!(extended_derivative.is_ok());
    assert!(evaluation.is_ok());
    assert!(first_many.is_ok());
    assert!(matches!(first_only, Some(Ok(_))));
    assert!(repeated_many.is_ok());
    assert!(differentiated.is_ok());
    assert!(integrated.is_ok());
    let before = primitive_output;
    assert_eq!(
        primitive.workspace_requirements(()).unwrap().scratch_len(),
        5
    );
    assert!(matches!(
        primitive
            .apply_into(
                &coefficients,
                (0.0, 0.0),
                &mut primitive_output,
                &mut scratch[..4],
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(primitive_output, before);
}

#[test]
fn borrowed_calculus_and_prepared_conversions_never_allocate_even_on_scaled_retry() {
    use scioxide_core::LinearOperator;
    use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
    let plan = OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.0_f64,
        2.0,
    )
    .unwrap()
    .prepare(2, Default::default())
    .unwrap();
    let primitive = plan.antiderivative_plan().unwrap();
    let conversion = plan.monomial_operator().unwrap();
    let from_power = plan.from_monomial_operator().unwrap();
    let mut derivative_output = [0.0; 2];
    let mut primitive_output = [0.0; 4];
    let mut conversion_output = [0.0; 3];
    let mut scratch = [0.0; 4];
    FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
    let derivative = plan.derivative_operator();
    let differentiated = derivative.apply_into(
        &[0.0, f64::from_bits(2), 0.0],
        &mut derivative_output,
        &mut scratch,
    );
    let integrated = primitive.apply_into(
        &[f64::MAX * 0.625, 0.0, f64::MAX * 0.25],
        (0.0, 0.0),
        &mut primitive_output,
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    );
    let converted = conversion.apply_into(
        &[f64::from_bits(1), 0.0, 0.0],
        &mut conversion_output,
        &mut scratch,
    );
    let from_converted = from_power.apply_into(
        &[f64::from_bits(1), 0.0, 0.0],
        &mut conversion_output,
        &mut scratch,
    );
    let untouched = FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false));
    assert!(untouched);
    assert!(differentiated.is_ok());
    assert!(integrated.is_ok());
    assert!(converted.is_ok());
    assert!(from_converted.is_ok());
    assert_eq!(derivative_output[0].to_bits(), 1);
    assert_eq!(conversion_output[0].to_bits(), 1);
}

#[test]
fn root_diagnostics_reservation_failures_are_typed_and_prepared_execution_executes() {
    use scioxide_core::Complex;
    use scioxide_poly::{
        RootDiagnostics, RootDiagnosticsWorkspace, polynomial_root_diagnostics_into,
    };
    let requirements = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2).unwrap();
    let mut output = RootDiagnostics::<f64>::new();
    assert_next_allocation_is_typed(|| output.try_reserve(&requirements));
    output.try_reserve(&requirements).unwrap();
    let mut workspace = RootDiagnosticsWorkspace::new();
    assert_next_allocation_is_typed(|| workspace.try_reserve(&requirements));
    assert_next_allocation_is_typed(|| RootDiagnosticsWorkspace::<f64>::workspace(2));
    workspace.try_reserve(&requirements).unwrap();
    let coefficients = [2.0_f64, -3.0, 1.0];
    let roots = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
    for _ in 0..2 {
        FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
        let result = polynomial_root_diagnostics_into(
            &coefficients,
            &roots,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        );
        let untouched = FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false));
        result.unwrap();
        assert!(untouched, "prepared analysis attempted an allocation");
    }
}

#[test]
fn inconclusive_diagnostics_do_not_allocate_hidden_error_payloads() {
    use scioxide_core::Complex;
    use scioxide_poly::{
        RootDiagnostics, RootDiagnosticsStatus, RootDiagnosticsWorkspace,
        polynomial_root_diagnostics_into,
    };
    let requirements = RootDiagnosticsWorkspace::workspace_requirements(2).unwrap();
    let mut output = RootDiagnostics::new();
    let mut workspace = RootDiagnosticsWorkspace::new();
    output.try_reserve(&requirements).unwrap();
    workspace.try_reserve(&requirements).unwrap();
    for (coefficients, roots) in [
        (
            &[f64::MAX, f64::from_bits(1)][..],
            &[Complex::new(0.0, 0.0)][..],
        ),
        (
            &[-1.0_f64, 0.0, 1.0][..],
            &[Complex::new(-f64::MAX, 0.0), Complex::new(f64::MAX, 0.0)][..],
        ),
        (
            &[1.0_f64, -2.0, 1.0][..],
            &[Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)][..],
        ),
    ] {
        FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
        let result = polynomial_root_diagnostics_into(
            coefficients,
            roots,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        );
        let untouched = FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false));
        result.unwrap();
        assert!(
            untouched,
            "inconclusive result constructed an allocating error"
        );
        assert!(matches!(
            output.status(),
            RootDiagnosticsStatus::Inconclusive(_)
        ));
        assert_eq!(output.clusters().len(), 1);
    }
}

#[test]
fn rational_preparation_failures_are_typed_and_scratch_recovers() {
    use scioxide_poly::{
        ApproxPartialFractionPlan, ApproxPartialFractions, ApproxRationalFunction,
        PartialFractionWorkspace,
    };
    let policy = PolynomialApproximationPolicy::<f64>::default();
    let factor = ApproxPolynomial::monomial_with_options(vec![-1.0, 1.0], policy).unwrap();
    let numerator = ApproxPolynomial::monomial_with_options(vec![1.0], policy).unwrap();
    let denominator = factor.clone();
    assert_next_allocation_is_typed(|| ApproxRationalFunction::new(numerator, denominator));
    assert_next_allocation_is_typed(|| ApproxPartialFractionPlan::new(&factor, &[(&factor, 1)]));
    let plan = ApproxPartialFractionPlan::new(&factor, &[(&factor, 1)]).unwrap();
    let requirements = plan.workspace_requirements(4).unwrap();
    let mut workspace = PartialFractionWorkspace::new();
    let mut output = ApproxPartialFractions::new();
    assert_next_allocation_is_typed(|| workspace.try_reserve(&requirements));
    assert_next_allocation_is_typed(|| output.try_reserve(&requirements));
    assert_next_allocation_is_typed(|| plan.workspace(4));
    assert!(
        plan.decompose_into(
            &[1.0],
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    workspace.try_reserve(&requirements).unwrap();
    output.try_reserve(&requirements).unwrap();
    FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
    let result = plan.decompose_into(
        &[1.0],
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    );
    let no_allocation = FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false));
    assert!(no_allocation);
    assert!(result.is_ok());
}

#[test]
fn fast_polynomial_reservations_propagate_allocation_failures() {
    assert_next_allocation_is_typed(|| {
        scioxide_poly::PolynomialOperationPlan::<f64>::new(&[
            PolynomialOperationRequest::Multiply {
                left_len: 128,
                right_len: 129,
            },
        ])
    });
    let plan = scioxide_poly::PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 128,
            right_len: 129,
        },
    ])
    .unwrap();
    assert_next_allocation_is_typed(|| plan.workspace(plan.requests()));
    let mut workspace = scioxide_poly::PolynomialWorkspace::new();
    let requirements = plan.workspace_requirements(plan.requests()).unwrap();
    assert_next_allocation_is_typed(|| workspace.try_reserve(&requirements));
    type ExactWs = scioxide_poly::ExactPolynomialWorkspace<Integer>;
    assert_next_allocation_is_typed(|| ExactWs::workspace(65));
    let mut exact_workspace = ExactWs::new();
    let requirements = ExactWs::workspace_requirements(65).unwrap();
    assert_next_allocation_is_typed(|| exact_workspace.try_reserve(&requirements));
}

#[test]
fn elimination_storage_reservation_failures_are_typed() {
    let policy = PolynomialApproximationPolicy::default();
    let elimination = scioxide_poly::SylvesterElimination::<f64>::new(3, 2, 0, policy).unwrap();
    assert_next_allocation_is_typed(|| elimination.workspace());
    let discriminant = scioxide_poly::SylvesterDiscriminant::<f64>::new(3, policy).unwrap();
    assert_next_allocation_is_typed(|| discriminant.workspace());
}
