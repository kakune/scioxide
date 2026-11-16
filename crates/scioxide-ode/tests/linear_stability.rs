//! Declared linear stability segments of the stepping methods.
#![allow(clippy::unwrap_used)]

use scioxide_ode::{
    AdamsBashforth, BackwardEuler, Bdf, BuiltinOdeMethod, DormandPrince5, ForwardEuler,
    ImplicitTrapezoid, OdeLinearStability, OdeMethod, RadauIia,
};

fn axes<S: Copy>(stability: OdeLinearStability<S>) -> (Option<S>, Option<S>) {
    let OdeLinearStability::Axes { real, imaginary } = stability else {
        return (None, None);
    };
    assert_ne!((real.is_none(), imaginary.is_none()), (true, true));
    (real, imaginary)
}

fn of<M: OdeMethod<f64>>(method: &M) -> OdeLinearStability<f64> {
    method.linear_stability()
}

#[test]
fn explicit_methods_declare_finite_segments() {
    assert_eq!(axes(of(&ForwardEuler::new())), (Some(2.0), Some(0.0)));
    let (real, imaginary) = axes(OdeMethod::<f32>::linear_stability(
        &DormandPrince5::<f32>::new(),
    ));
    assert!((real.unwrap() - 3.3065).abs() < 1e-6);
    assert!((imaginary.unwrap() - 0.9971).abs() < 1e-6);
}

#[test]
fn implicit_methods_declare_unbounded_or_real_only_segments() {
    let unconditional = OdeLinearStability::<f64>::unconditional();
    assert_eq!(of(&BackwardEuler::new()), unconditional);
    assert_eq!(of(&ImplicitTrapezoid::new()), unconditional);
    assert_eq!(of(&RadauIia::new()), unconditional);
    assert_eq!(of(&Bdf::new(2).unwrap()), unconditional);
    assert_eq!(axes(of(&Bdf::new(5).unwrap())), (None, Some(0.0)));
}

#[test]
fn undeclared_and_forwarded_declarations() {
    assert_eq!(
        of(&AdamsBashforth::new(3).unwrap()),
        OdeLinearStability::Undeclared
    );
    let builtin = BuiltinOdeMethod::from(DormandPrince5::<f64>::new());
    assert_eq!(of(&builtin), of(&DormandPrince5::new()));
}
