//! Canonical differentiation root API, compiled from an external consumer.
#![allow(unused_imports)]

// adaptive
use scioxide_differentiate::{
    AdaptiveDifference, AdaptiveDifferenceMethod, AdaptiveDifferenceRequirements,
    AdaptiveDifferenceWorkspace, ColoredJacobianPlan, ColoredJacobianRequirements,
    ColoredJacobianWorkspace,
};

// complex
use scioxide_differentiate::{
    ComplexStep, ComplexStepPlan, ComplexStepRequirements, ComplexStepWorkspace,
    RelativeComplexStep,
};

// one-shot functions
use scioxide_differentiate::{
    complex_step_derivative, derivative, gradient, hessian, jacobian, jvp,
    physical_difference_derivative, second_derivative, sparse_jacobian,
};

// fixed
use scioxide_differentiate::{
    FiniteDifference, FiniteDifferenceRequirements, FiniteDifferenceWorkspace,
    FixedCoordinatePolicy,
};

// nodes
use scioxide_differentiate::{
    FiniteDifferenceStencil, NodeDifferenceOptions, PhysicalDifferencePlan,
    PhysicalDifferenceRequirements, PhysicalDifferenceWorkspace, RelativeDifference,
    RelativeDifferenceRequirements, RelativeDifferenceWorkspace,
};

// options
use scioxide_differentiate::{
    DerivativeOperation, DerivativeRequest, DerivativeShape, DifferenceScheme, DifferenceStep,
    ZeroDirectionPolicy,
};

// outcome
use scioxide_differentiate::{
    DerivativeSource, DifferentiationReport, DifferentiationResult, DifferentiationTermination,
};

// sampling
use scioxide_differentiate::{Contextual, Parallel, SampleFunction};

// selection
use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy};

fn shared_value<T: Send + Sync + Clone>() {}
fn copy_value<T: Copy>() {}

#[test]
fn method_values_and_plans_are_shareable_values() {
    shared_value::<FiniteDifference<f64>>();
    shared_value::<AdaptiveDifference<f64>>();
    shared_value::<ComplexStep<f64>>();
    shared_value::<RelativeDifference<f64>>();
    shared_value::<RelativeComplexStep<f64>>();
    shared_value::<ColoredJacobianPlan<'static, f64>>();
    shared_value::<PhysicalDifferencePlan<f64>>();
    shared_value::<ComplexStepPlan<f64>>();
    copy_value::<DerivativeShape>();
    copy_value::<DerivativeRequest<'static, FiniteDifference<f64>>>();
}
