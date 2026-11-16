//! Public set/domain contracts, independent of the scalar rounding fixtures.
#![cfg(test)]
use scioxide_core::{ExtendedInterval, Interval, RealScalar};
use scioxide_validated::{DecoratedInterval as D, Decoration};

fn domains<S: RealScalar>() {
    let zero = S::ZERO;
    let one = S::ONE;
    let empty = D::<S>::empty();
    let entire = D::<S>::entire();
    let positive = D::from(scioxide_core::ExtendedInterval::new(Some(one), None).unwrap());
    let negative = positive.neg().unwrap();
    let singleton_zero = D::<S>::point(S::ZERO).unwrap();
    for set in [
        empty,
        entire,
        positive,
        negative,
        singleton_zero,
        D::new(-S::TWO, S::try_from_i64_exact(3).unwrap()).unwrap(),
    ] {
        assert!(set.add(&empty).unwrap().set().is_empty());
        assert!(set.mul(&empty).unwrap().set().is_empty());
        assert!(set.div(&empty).unwrap().set().is_empty());
        assert!(empty.div(&set).unwrap().set().is_empty());
        assert!(set.contains_interval(&empty));
        assert_eq!(set.hull(&empty).set(), set.set());
        assert_eq!(set.intersection(&entire).set(), set.set());
        assert_eq!(set.neg().unwrap().neg().unwrap().set(), set.set());
        assert_eq!(
            set.square().unwrap().set(),
            set.abs().unwrap().square().unwrap().set()
        );
    }
    assert_eq!(
        entire.mul(&singleton_zero).unwrap().set(),
        singleton_zero.set()
    );
    assert_eq!(
        entire.square().unwrap().set(),
        ExtendedInterval::new(Some(zero), None).unwrap()
    );
    assert_eq!(positive.mul(&negative).unwrap().set(), negative.set());
    assert!(
        positive
            .div(&D::new(-S::ONE, S::ONE).unwrap())
            .unwrap()
            .set()
            .is_entire()
    );
    assert_eq!(
        positive
            .div(&D::new(S::ZERO, S::ONE).unwrap())
            .unwrap()
            .set(),
        positive.set()
    );
    assert_eq!(
        positive
            .div(&D::new(-S::ONE, S::ZERO).unwrap())
            .unwrap()
            .set(),
        negative.set()
    );
    assert_eq!(
        singleton_zero.div(&entire).unwrap().set(),
        singleton_zero.set()
    );
    assert_eq!(
        singleton_zero.div(&entire).unwrap().decoration(),
        Decoration::Trv
    );
    assert!(entire.div(&singleton_zero).unwrap().set().is_empty());
    assert_eq!(
        positive.div(&positive).unwrap().set(),
        ExtendedInterval::new(Some(zero), None).unwrap()
    );
    assert_eq!(
        positive.reciprocal().unwrap().set(),
        D::new(S::ZERO, S::ONE).unwrap().set()
    );
    assert_eq!(
        negative.reciprocal().unwrap().set(),
        D::new(-S::ONE, S::ZERO).unwrap().set()
    );
    assert_eq!(
        D::<S>::new(-S::ONE, S::try_from_i64_exact(4).unwrap())
            .unwrap()
            .sqrt()
            .unwrap()
            .set(),
        D::new(S::ZERO, S::TWO).unwrap().set()
    );
    assert_eq!(
        D::<S>::new(-S::ONE, S::try_from_i64_exact(4).unwrap())
            .unwrap()
            .sqrt()
            .unwrap()
            .decoration(),
        Decoration::Trv
    );
    assert!(negative.sqrt().unwrap().set().is_empty());
    assert_eq!(entire.sqrt().unwrap().decoration(), Decoration::Trv);
    assert_eq!(
        D::<S>::new(S::ZERO, S::try_from_i64_exact(4).unwrap())
            .unwrap()
            .sqrt()
            .unwrap()
            .decoration(),
        Decoration::Com
    );
    assert_eq!(
        singleton_zero.mul(&entire).unwrap().decoration(),
        Decoration::Dac
    );
    assert_eq!(entire.restrict(&positive.set()).set(), positive.set());
    for decoration in [
        Decoration::Trv,
        Decoration::Def,
        Decoration::Dac,
        Decoration::Com,
    ] {
        let weak =
            D::with_decoration(D::<S>::new(S::ONE, S::TWO).unwrap().set(), decoration).unwrap();
        assert_eq!(
            weak.add(&D::new(S::ONE, S::TWO).unwrap())
                .unwrap()
                .decoration(),
            decoration
        );
        assert_eq!(
            weak.div(&D::new(S::ZERO, S::TWO).unwrap())
                .unwrap()
                .decoration(),
            Decoration::Trv
        );
    }
    assert!(match D::with_decoration(entire.set(), Decoration::Com) {
        Err(error) => matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ),
        Ok(_) => false,
    });
    assert!(match D::with_decoration(empty.set(), Decoration::Dac) {
        Err(error) => matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ),
        Ok(_) => false,
    });
    assert!(match D::new(one, zero) {
        Err(error) => matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ),
        Ok(_) => false,
    });
    assert!(match D::point(S::INFINITY) {
        Err(error) => matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ),
        Ok(_) => false,
    });
    assert!(match D::point(S::NAN) {
        Err(error) => matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ),
        Ok(_) => false,
    });
    assert!(Interval::<S>::new(zero, S::INFINITY).is_err());
    assert!(Interval::try_from(entire.set()).is_err());
    let zeros = singleton_zero
        .add(&singleton_zero)
        .unwrap()
        .set()
        .bounds()
        .unwrap();
    assert!(zeros.0.unwrap().is_sign_negative());
    assert!(!zeros.1.unwrap().is_sign_negative());
}

#[test]
fn domains_f32() {
    domains::<f32>();
}
#[test]
fn domains_f64() {
    domains::<f64>();
}

fn exact_sign<S: RealScalar>() {
    use core::cmp::Ordering;
    use scioxide_core::RangeAccumulator;

    let tiny = S::ZERO.next_after(S::ONE);
    let mut state = RangeAccumulator::new();
    assert_eq!(state.sign(), Ordering::Equal);
    state.add_product(tiny, tiny).unwrap();
    assert_eq!(state.sign(), Ordering::Greater);
    state.add_product(-tiny, tiny).unwrap();
    assert_eq!(state.sign(), Ordering::Equal);
    state.add_product(-S::MAX, S::MAX).unwrap();
    assert_eq!(state.sign(), Ordering::Less);
}
#[test]
fn unrounded_residual_signs() {
    exact_sign::<f32>();
    exact_sign::<f64>();
}

#[test]
fn error_provenance() {
    use scioxide_core::{SciOxideError, SciOxideErrorKind, argument};
    use scioxide_validated::{
        AffineContext, AffineForm, AffineOptions, PolynomialRootPlan, QuadratureObligation,
        QuadratureVerifier, RootIsolation, ValidatedOptions,
    };
    fn origin(error: SciOxideError) -> scioxide_core::Argument {
        let scioxide_core::SciOxideErrorView::InvalidInput {
            argument: _view_argument,
            reason: _view_reason,
            detail: _view_detail,
        } = error.view()
        else {
            unreachable!("unexpected error kind")
        };
        _view_argument
    }

    let error = D::<f64>::with_decoration(D::entire().set(), Decoration::Com).unwrap_err();
    assert_eq!(origin(error), (argument!("decoration")));

    let options = AffineOptions::new().with_max_symbols(2);
    let context = AffineContext::new(1, options.with_byte_limit(0)).unwrap();
    assert_eq!(origin(context.symbol(1).unwrap_err()), (argument!("index")));
    // An exhausted byte limit is a resource error, which names no argument.
    assert_eq!(
        context.workspace::<f64>().err().unwrap().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    // An allocating convenience reports the prepared entry point it forwards to.
    let context = AffineContext::new(1, options).unwrap();
    let mut workspace = context.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&context, 1.0, &[0.5]).unwrap();
    assert_eq!(
        origin(x.reduce(3, &mut workspace).unwrap_err()),
        (argument!("max_symbols"))
    );

    assert_eq!(
        origin(
            PolynomialRootPlan::new(&[0.0_f64], ValidatedOptions::new())
                .err()
                .unwrap()
        ),
        (argument!("coefficients"))
    );
    assert_eq!(
        origin(RootIsolation::<f64>::new(1, 0).err().unwrap()),
        (argument!("capacity"))
    );
    let oracle = |_, _: &mut [_]| Ok(QuadratureObligation::Unavailable);
    assert_eq!(
        origin(
            QuadratureVerifier::<f64>::new(&[], 1, &oracle, ValidatedOptions::new())
                .err()
                .unwrap()
        ),
        (argument!("pieces"))
    );
}

#[test]
fn value_traits() {
    fn traits<T: Send + Sync + Copy>() {}
    traits::<D<f32>>();
    traits::<D<f64>>();
}

#[test]
fn finite_constructors_and_conversion() {
    use scioxide_core::{InvalidInputReason, SciOxideErrorKind, SciOxideErrorView};
    let zero = D::new(-0.0_f64, 0.0).unwrap();
    let bounds = Interval::try_from(zero).unwrap();
    assert_eq!(bounds.lower().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(bounds.upper().to_bits(), 0.0_f64.to_bits());
    assert_eq!(zero.decoration(), Decoration::Com);
    let point = Interval::try_from(D::point(2.0_f64).unwrap()).unwrap();
    assert!(point.contains(2.0));
    assert_eq!(point.lower().to_bits(), point.upper().to_bits());
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            D::new(value, 1.0).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(
            D::new(0.0, value).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(
            D::point(value).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert!(
        matches!(D::new(2.0, 1.0).unwrap_err().view(), SciOxideErrorView::InvalidInput { argument, .. } if argument == scioxide_core::argument!("lower"))
    );
    for value in [
        D::<f64>::empty(),
        D::from(scioxide_core::ExtendedInterval::new(None, Some(1.0)).unwrap()),
    ] {
        assert!(
            matches!(Interval::try_from(value).unwrap_err().view(), SciOxideErrorView::InvalidInput { argument, reason: InvalidInputReason::OutOfRange, .. } if argument == scioxide_core::argument!("interval"))
        );
    }
}
