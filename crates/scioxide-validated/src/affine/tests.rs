#![allow(clippy::panic)]
use super::*;
use AffineBinary as B;
use AffineUnary as U;
use alloc::{vec, vec::Vec};

use scioxide_core::{Complex, SciOxideErrorKind};

fn context(symbols: usize) -> AffineContext {
    AffineContext::new(
        4,
        AffineOptions::new()
            .with_max_symbols(symbols)
            .with_byte_limit(1 << 20),
    )
    .unwrap()
}
fn kind<T>(result: Result<T, SciOxideError>) -> SciOxideErrorKind {
    result.err().map(|error| error.kind()).unwrap()
}
#[test]
fn options_defaults_and_setters() {
    let options = AffineOptions::default();
    assert_eq!(options, AffineOptions::new());
    assert_eq!(
        (options.max_symbols(), options.byte_limit()),
        (16, usize::MAX)
    );
    let options = options.with_max_symbols(0).with_byte_limit(0);
    assert_eq!((options.max_symbols(), options.byte_limit()), (0, 0));
    let context = AffineContext::new(2, options).unwrap();
    assert_eq!(context.options(), options);
}
#[test]
fn cancellation_and_quadratic_remainder() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 0.0, &[1.0]).unwrap();
    let zero = x.binary(&x, B::Sub, &mut w).unwrap();
    assert_eq!(zero.report().real().bounds(), Some((-0.0, 0.0)));
    let square = x.unary(U::Square, &mut w).unwrap();
    assert_eq!(square.report().real().bounds(), Some((-0.0, 1.0)));
    assert_eq!(
        square
            .binary(&square, B::Sub, &mut w)
            .unwrap()
            .report()
            .real()
            .bounds(),
        Some((-0.0, 0.0))
    );
}
#[test]
fn real_and_complex_exponential() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 1.0, &[0.125]).unwrap();
    let y = x.unary(U::Exp, &mut w).unwrap();
    assert!(y.report().fallback().is_none());
    assert!(y.report().real().contains(0.875_f64.exp()));
    assert!(y.report().real().contains(1.125_f64.exp()));
    let mut cw = c.workspace::<Complex<f64>>().unwrap();
    let z = AffineForm::from_coefficients(&c, Complex::new(0.2, 0.3), &[Complex::new(0.01, 0.02)])
        .unwrap();
    let ez = z.unary(U::Exp, &mut cw).unwrap();
    assert!(ez.report().fallback().is_none());
    for t in [-1.0, 0.0, 1.0] {
        let v = Complex::new(0.2 + 0.01 * t, 0.3 + 0.02 * t).exp();
        assert!(ez.report().real().contains(v.re));
        assert!(ez.report().imaginary().contains(v.im));
    }
}

fn unary_name(name: &str) -> AffineUnary {
    use AffineUnary::*;
    match name {
        "sqrt" => Sqrt,
        "cbrt" => Cbrt,
        "exp" => Exp,
        "exp2" => Exp2,
        "exp10" => Exp10,
        "exp_m1" => ExpM1,
        "ln" => Ln,
        "log2" => Log2,
        "log10" => Log10,
        "ln_1p" => Ln1p,
        "sin" => Sin,
        "cos" => Cos,
        "tan" => Tan,
        "asin" => Asin,
        "acos" => Acos,
        "atan" => Atan,
        "sinh" => Sinh,
        "cosh" => Cosh,
        "tanh" => Tanh,
        "asinh" => Asinh,
        "acosh" => Acosh,
        "atanh" => Atanh,
        "reciprocal" => Reciprocal,
        "powi3" => Powi(3),
        "powi-3" => Powi(-3),
        _ => panic!("unknown operation {name}"),
    }
}
fn replay<T: NumericScalar>(width: &str, complex: &str) {
    let c = context(12);
    let mut w = c.workspace::<T>().unwrap();
    let mut count = 0;
    let mut affine = 0;
    for line in include_str!("../../tests/data/affine.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width || f[1] != complex {
            continue;
        }
        let decode = |s: &str| {
            let bits = u64::from_str_radix(s, 16).unwrap();
            T::Real::try_from_f64_lossy(if width == "32" {
                f64::from(f32::from_bits(bits as u32))
            } else {
                f64::from_bits(bits)
            })
            .unwrap()
        };
        let center = T::from_components(decode(f[3]), decode(f[4]));
        let coefficient = T::from_components(decode(f[5]), decode(f[6]));
        let x = AffineForm::from_coefficients(&c, center, &[coefficient]).unwrap();
        let b = AffineForm::from_coefficients(
            &c,
            T::from_components(decode(f[7]), decode(f[8])),
            &[coefficient],
        )
        .unwrap();
        let result = match f[2] {
            "pow" => x.binary(&b, B::Pow, &mut w),
            "log" => x.binary(&b, B::Log, &mut w),
            "div" => x.binary(&b, B::Div, &mut w),
            "hypot" => x.binary(&b, B::Hypot, &mut w),
            "atan2" => x.binary(&b, B::Atan2, &mut w),
            op => x.unary(unary_name(op), &mut w),
        }
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        let r = result.report();
        assert!(
            r.real().contains(decode(f[10]))
                && r.real().contains(decode(f[11]))
                && r.imaginary().contains(decode(f[12]))
                && r.imaginary().contains(decode(f[13])),
            "{line}: {r:?}"
        );
        if r.fallback().is_none() {
            affine += 1;
        }
        count += 1;
    }
    assert!(count > 500, "{count}");
    assert!(affine > count / 2, "{affine}/{count}");
}
#[test]
fn flint_real_f32() {
    replay::<f32>("32", "0");
}
#[test]
fn flint_real_f64() {
    replay::<f64>("64", "0");
}
#[test]
fn flint_complex_f32() {
    replay::<Complex<f32>>("32", "1");
}
#[test]
fn flint_complex_f64() {
    replay::<Complex<f64>>("64", "1");
}

fn resource_contract<T: NumericScalar>() {
    let c = context(8);
    let mut w = c.workspace::<T>().unwrap();
    let x = AffineForm::from_coefficients(
        &c,
        T::from_real(T::Real::ONE),
        &[T::from_real(T::Real::try_from_f64_lossy(0.125).unwrap())],
    )
    .unwrap();
    let mut state = AffineState::<T>::zero(&c).unwrap();
    let mut buffer = vec![AffineTerm::default(); 8];
    let output = AffineViewMut::new(&mut state, &mut buffer).unwrap();
    x.view().unary_into(U::Exp, output, &mut w).unwrap();
    for operation in [B::Add, B::Mul] {
        x.view()
            .binary_into(
                x.view(),
                operation,
                AffineViewMut::new(&mut state, &mut buffer).unwrap(),
                &mut w,
            )
            .unwrap();
    }
    x.view()
        .reduce_into(
            0,
            AffineViewMut::new(&mut state, &mut buffer).unwrap(),
            &mut w,
        )
        .unwrap();
    x.view()
        .unary_into(
            U::Exp,
            AffineViewMut::new(&mut state, &mut buffer).unwrap(),
            &mut w,
        )
        .unwrap();
    let good = AffineView::new(&state, &buffer).unwrap().report();
    let alien = context(8);
    let mut foreign = alien.workspace::<T>().unwrap();
    assert_eq!(
        kind(x.view().unary_into(
            U::Sin,
            AffineViewMut::new(&mut state, &mut buffer).unwrap(),
            &mut foreign
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(state.report, good);
    w.next = usize::MAX;
    assert_eq!(
        kind(x.view().unary_into(
            U::Exp,
            AffineViewMut::new(&mut state, &mut buffer).unwrap(),
            &mut w
        )),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(state.report, good);
    // Short caller storage is a dimension error, for outputs and borrowed values alike.
    assert_eq!(
        kind(AffineViewMut::new(&mut state, &mut buffer[..7])),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        kind(AffineView::new(&state, &[])),
        SciOxideErrorKind::DimensionMismatch
    );
    let retained = good.retained_symbols();
    assert!(AffineView::new(&state, &buffer[..retained]).is_ok());
    assert_eq!(
        kind(AffineView::new(&state, &buffer[..retained - 1])),
        SciOxideErrorKind::DimensionMismatch
    );
    buffer[0].coefficient = T::from_real(T::Real::MAX);
    assert_eq!(
        kind(AffineView::new(&state, &buffer)),
        SciOxideErrorKind::InvalidInput
    );
    // The byte limit admits exactly the documented scratch payload.
    let needed = c.workspace_bytes::<T>().unwrap();
    assert_eq!(needed, 2 * (2 * 8 + 2) * size_of::<AffineTerm<T>>());
    let options = AffineOptions::new().with_max_symbols(8);
    let small = AffineContext::new(4, options.with_byte_limit(needed - 1)).unwrap();
    assert_eq!(
        kind(small.workspace::<T>()),
        SciOxideErrorKind::ResourceLimit
    );
    let exact = AffineContext::new(4, options.with_byte_limit(needed)).unwrap();
    assert!(exact.workspace::<T>().is_ok());
}
#[test]
fn resources_all_scalars() {
    resource_contract::<f32>();
    resource_contract::<f64>();
    resource_contract::<Complex<f32>>();
    resource_contract::<Complex<f64>>();
}

#[test]
fn unrepresentable_sizes_fail_cleanly() {
    // A tiny byte limit rejects even an empty-symbol workspace's two error slots.
    let tiny = AffineContext::new(
        1,
        AffineOptions::new().with_max_symbols(0).with_byte_limit(1),
    )
    .unwrap();
    assert_eq!(
        kind(tiny.workspace::<f64>()),
        SciOxideErrorKind::ResourceLimit
    );
    // Slot and byte counts that overflow `usize` are size errors, not panics.
    let huge = AffineContext::new(1, AffineOptions::new().with_max_symbols(usize::MAX)).unwrap();
    assert_eq!(
        kind(huge.workspace::<f64>()),
        SciOxideErrorKind::SizeOverflow
    );
    let wide =
        AffineContext::new(1, AffineOptions::new().with_max_symbols(usize::MAX / 4)).unwrap();
    assert_eq!(
        kind(wide.workspace::<f64>()),
        SciOxideErrorKind::SizeOverflow
    );
    // Capacities above `isize::MAX` bytes are refused before the allocator is called,
    // so these failures are deterministic.
    let size = size_of::<AffineTerm<f64>>();
    let over = isize::MAX.unsigned_abs() / size + 1;
    let terms = AffineContext::new(1, AffineOptions::new().with_max_symbols(over)).unwrap();
    assert_eq!(
        kind(AffineForm::<f64>::zero(&terms)),
        SciOxideErrorKind::AllocationFailure
    );
    assert_eq!(
        kind(AffineForm::from_coefficients(&terms, 1.0_f64, &[0.5])),
        SciOxideErrorKind::AllocationFailure
    );
    // Both scratch buffers together exceed `isize::MAX` bytes without overflowing.
    let scratch = AffineContext::new(1, AffineOptions::new().with_max_symbols(over / 4)).unwrap();
    assert_eq!(
        kind(scratch.workspace::<f64>()),
        SciOxideErrorKind::AllocationFailure
    );
}

#[test]
fn reduce_compresses_and_validates_its_limit() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 0.5, &[1.0, -0.5, 0.25, 0.125]).unwrap();
    let reduced = x.reduce(2, &mut w).unwrap();
    let report = reduced.report();
    assert_eq!(report.retained_symbols(), 2);
    assert_eq!(report.discarded_symbols(), 2);
    assert!(report.correlation_lost());
    assert!(report.fallback().is_none());
    assert_eq!(report.status(), EnclosureStatus::Defined);
    assert_eq!(report.real().bounds(), Some((-1.375, 2.375)));
    // The largest coefficients keep their input identities, in source order.
    let kept: Vec<_> = reduced
        .view()
        .terms()
        .iter()
        .map(|t| t.coefficient())
        .collect();
    assert_eq!(kept, [1.0, -0.5]);
    assert_eq!(
        reduced.view().terms()[0].symbol(),
        Some(&c.symbol(0).unwrap())
    );
    assert_eq!(
        reduced.view().terms()[1].symbol(),
        Some(&c.symbol(1).unwrap())
    );
    // Reducing to the context limit copies exactly: no term is lost or generated.
    let copy = x.reduce(8, &mut w).unwrap();
    assert_eq!(copy.report(), x.report());
    assert_eq!(copy.view().terms().len(), 4);
    assert_eq!(w.next, 0);
    // Shared symbols still cancel after a reduction.
    let difference = reduced.binary(&x, B::Sub, &mut w).unwrap();
    assert!(difference.report().real().contains(0.0));
    assert_eq!(kind(x.reduce(9, &mut w)), SciOxideErrorKind::InvalidInput);
    // The borrowed route commits the same reduction into caller storage.
    let mut state = AffineState::<f64>::zero(&c).unwrap();
    let mut terms = vec![AffineTerm::default(); 8];
    let into = x
        .view()
        .reduce_into(
            2,
            AffineViewMut::new(&mut state, &mut terms).unwrap(),
            &mut w,
        )
        .unwrap();
    assert_eq!(into, report);
}

#[test]
fn noise_identity_and_compression() {
    let c = context(8);
    let mut a = c.workspace::<f64>().unwrap();
    let mut b = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 0.0, &[1.0, 0.5, 0.25]).unwrap();
    let y = AffineForm::from_terms(
        &c,
        0.0,
        &[AffineTerm::new(c.symbol(3).unwrap(), 1.0).unwrap()],
    )
    .unwrap();
    assert_eq!(
        x.binary(&y, B::Sub, &mut a)
            .unwrap()
            .report()
            .real()
            .bounds(),
        Some((-2.75, 2.75))
    );
    let s = x.unary(U::Square, &mut a).unwrap();
    let t = x.unary(U::Square, &mut b).unwrap();
    let delta = s.binary(&t, B::Sub, &mut a).unwrap();
    assert!(delta.report().real().bounds().unwrap().1 > 0.0);
    let clone = s.clone();
    assert_eq!(
        s.binary(&clone, B::Sub, &mut a)
            .unwrap()
            .report()
            .real()
            .bounds(),
        Some((-0.0, 0.0))
    );
    for max in [0, 1, 2, 3] {
        let r = x.reduce(max, &mut a).unwrap();
        assert!(r.report().real().contains_interval(&x.report().real()));
        assert!(r.report().retained_symbols() <= max);
        let again = r.reduce(max, &mut a).unwrap();
        assert_eq!(r.report().real(), again.report().real());
        let combined = r.binary(&y, B::Mul, &mut a).unwrap();
        assert!(combined.report().real().contains(-1.75));
        assert!(combined.report().real().contains(1.75));
    }
    let zero = context(0);
    let mut w = zero.workspace::<f64>().unwrap();
    let value = AffineForm::from_coefficients(&zero, 0.0, &[1.0]).unwrap();
    let square = value.unary(U::Square, &mut w).unwrap();
    assert_eq!(square.report().retained_symbols(), 0);
    assert!(square.report().real().contains(1.0));
    let tie = AffineForm::from_coefficients(&c, 0.0, &[1.0, 1.0, 1.0])
        .unwrap()
        .reduce(1, &mut a)
        .unwrap();
    assert_eq!(tie.view().terms()[0].symbol(), Some(&c.symbol(0).unwrap()));
}

#[test]
fn symbol_identity_is_context_origin_and_index() {
    let c = context(8);
    let other = context(8);
    // Clones share identity, which outlives the clone that issued the symbol.
    let alias = c.clone();
    let issued = alias.symbol(1).unwrap();
    drop(alias);
    assert_eq!(c.symbol(1).unwrap(), issued);
    assert_ne!(c.symbol(1).unwrap(), c.symbol(2).unwrap());
    assert_ne!(c.symbol(1).unwrap(), other.symbol(1).unwrap());
    assert_eq!(kind(c.symbol(4)), SciOxideErrorKind::InvalidInput);
    // Generated error symbols are numbered per workspace from zero, never equal to an
    // input symbol of the same index or to another workspace's symbol.
    let mut a = c.workspace::<f64>().unwrap();
    let mut b = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 1.0, &[0.125]).unwrap();
    let ea = x.unary(U::Exp, &mut a).unwrap();
    let eb = x.unary(U::Exp, &mut b).unwrap();
    let generated = |form: &AffineForm<f64>| *form.view().terms()[1].symbol().unwrap();
    assert_eq!(generated(&ea).index, 0);
    assert_eq!(generated(&ea), generated(&ea.clone()));
    assert_ne!(generated(&ea), generated(&eb));
    assert_ne!(generated(&ea), c.symbol(0).unwrap());
    let again = x.unary(U::Exp, &mut a).unwrap();
    assert_eq!(generated(&again).index, 1);
    // A foreign symbol is rejected rather than silently merged.
    let foreign = AffineTerm::new(other.symbol(0).unwrap(), 1.0).unwrap();
    assert_eq!(
        kind(AffineForm::from_terms(&c, 0.0, &[foreign])),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn singularities_empty_unbounded_and_extreme_exponents() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 0.0, &[1.0]).unwrap();
    let reciprocal = x.unary(U::Reciprocal, &mut w).unwrap();
    assert!(reciprocal.report().real().set().is_entire());
    assert!(reciprocal.report().fallback().is_some());
    let root = x.unary(U::Sqrt, &mut w).unwrap();
    assert!(root.report().real().contains(0.0));
    assert!(root.report().real().contains(1.0));
    let negative = AffineForm::constant(&c, -1.0).unwrap();
    assert_eq!(
        negative.unary(U::Ln, &mut w).unwrap().report().status(),
        EnclosureStatus::Undefined
    );
    let zero = AffineForm::constant(&c, 0.0).unwrap();
    let one = AffineForm::constant(&c, 1.0).unwrap();
    for n in [i32::MIN, -3, 0, 1, 2, 3, i32::MAX] {
        assert!(
            one.unary(U::Powi(n), &mut w)
                .unwrap()
                .report()
                .real()
                .contains(1.0)
        );
    }
    assert!(
        zero.unary(U::Powi(0), &mut w)
            .unwrap()
            .report()
            .real()
            .contains(1.0)
    );
    let huge = AffineForm::constant(&c, f64::MAX).unwrap();
    assert!(
        huge.unary(U::Square, &mut w)
            .unwrap()
            .report()
            .fallback()
            .is_some()
    );
    let tiny = AffineForm::constant(&c, f64::from_bits(1)).unwrap();
    let half = AffineForm::constant(&c, 0.5).unwrap();
    let sub = tiny.binary(&half, B::Mul, &mut w).unwrap();
    assert!(sub.report().real().contains(0.0));
    assert!(sub.report().real().contains(f64::from_bits(1)));
    let empty = AffineForm::<f64>::from_enclosure(
        &c,
        DecoratedInterval::empty(),
        DecoratedInterval::point(0.0).unwrap(),
        c.symbol(0).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        empty.unary(U::Sin, &mut w).unwrap().report().status(),
        EnclosureStatus::EmptyInput
    );
    assert_eq!(
        empty.reduce(0, &mut w).unwrap().report().status(),
        EnclosureStatus::EmptyInput
    );
    let mut cw = c.workspace::<Complex<f64>>().unwrap();
    let cut = AffineForm::constant(&c, Complex::new(-1.0, 0.0)).unwrap();
    let log = cut.unary(U::Ln, &mut cw).unwrap();
    assert!(log.report().imaginary().contains(core::f64::consts::PI));
    assert!(log.report().imaginary().contains(-core::f64::consts::PI));
    let sqrt = cut.unary(U::Sqrt, &mut cw).unwrap();
    assert!(sqrt.report().imaginary().contains(-1.0));
    assert!(sqrt.report().imaginary().contains(1.0));
    assert_eq!(
        kind(cut.binary(&cut, B::Atan2, &mut cw)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        kind(cut.binary(&cut, B::Hypot, &mut cw)),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn quadratic_error_rate_and_variable_integer_center_power() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let mut previous = f64::INFINITY;
    for radius in [0.125, 0.0625, 0.03125] {
        let x = AffineForm::from_coefficients(&c, 1.0, &[radius]).unwrap();
        let e = x.unary(U::Exp, &mut w).unwrap();
        let remainder = e
            .view()
            .terms()
            .iter()
            .filter(|t| t.symbol() != Some(&c.symbol(0).unwrap()))
            .map(|t| t.coefficient().abs())
            .sum::<f64>();
        assert!(remainder < previous * 0.3);
        previous = remainder;
    }
    let x = AffineForm::constant(&c, 2.0).unwrap();
    let exponent = AffineForm::from_coefficients(&c, 3.0, &[0.125]).unwrap();
    let pow = x.binary(&exponent, B::Pow, &mut w).unwrap();
    assert!(pow.report().real().contains(2.0_f64.powf(2.875)));
    assert!(pow.report().real().contains(2.0_f64.powf(3.125)));
    // A constant integer exponent takes the integer-power route.
    let three = AffineForm::constant(&c, 3.0).unwrap();
    let base = AffineForm::from_coefficients(&c, -2.0, &[0.125]).unwrap();
    let cube = base.binary(&three, B::Pow, &mut w).unwrap();
    let powi = base.unary(U::Powi(3), &mut w).unwrap();
    assert_eq!(cube.report().real(), powi.report().real());
}

#[test]
fn complex_projection_modulus_and_persistent_provenance() {
    let c = context(8);
    let mut cw = c.workspace::<Complex<f64>>().unwrap();
    let mut rw = c.workspace::<f64>().unwrap();
    let z =
        AffineForm::from_coefficients(&c, Complex::new(3.0, 4.0), &[Complex::new(0.125, -0.125)])
            .unwrap();
    let (re, im) = (
        z.project(AffineProjection::Real).unwrap(),
        z.project(AffineProjection::Imaginary).unwrap(),
    );
    let norm = re.binary(&im, B::Hypot, &mut rw).unwrap();
    let arg = im.binary(&re, B::Atan2, &mut rw).unwrap();
    for e in [-1.0, 0.0, 1.0] {
        let x = 3.0 + e * 0.125;
        let y = 4.0 - e * 0.125;
        assert!(norm.report().real().contains(x.hypot(y)));
        assert!(arg.report().real().contains(y.atan2(x)));
    }
    // The components keep the input symbol, so they recombine without loss.
    assert_eq!(re.view().terms()[0].symbol(), Some(&c.symbol(0).unwrap()));
    assert_eq!(im.view().terms()[0].symbol(), Some(&c.symbol(0).unwrap()));
    assert_eq!(
        re.binary(&re, B::Sub, &mut rw)
            .unwrap()
            .report()
            .real()
            .bounds(),
        Some((-0.0, 0.0))
    );
    let conjugate = z.unary(U::Conj, &mut cw).unwrap();
    let sum = z.binary(&conjugate, B::Add, &mut cw).unwrap();
    assert_eq!(sum.report().imaginary().bounds(), Some((-0.0, 0.0)));
    let reduced = z.reduce(0, &mut cw).unwrap();
    assert!(
        reduced
            .unary(U::Exp, &mut cw)
            .unwrap()
            .report()
            .correlation_lost()
    );
    let cut = AffineForm::constant(&c, Complex::new(-1.0, -0.0))
        .unwrap()
        .unary(U::Ln, &mut cw)
        .unwrap();
    assert_eq!(cut.report().real().decoration(), Decoration::Trv);
    let zero = context(0);
    let overflow = AffineForm::from_coefficients(&zero, 0.0_f64, &[f64::MAX, f64::MAX]).unwrap();
    assert_eq!(
        overflow.report().fallback(),
        Some(AffineFallback::Unbounded)
    );
}

#[test]
fn complex_zero_roots_and_powers() {
    let c = context(8);
    let mut w = c.workspace::<Complex<f64>>().unwrap();
    let zero = AffineForm::constant(&c, Complex::new(0.0, 0.0)).unwrap();
    let root = zero.unary(U::Cbrt, &mut w).unwrap();
    assert!(root.report().real().contains(0.0) && root.report().imaginary().contains(0.0));
    let exponent = AffineForm::constant(&c, Complex::new(2.5, 0.25)).unwrap();
    let result = zero.binary(&exponent, B::Pow, &mut w).unwrap();
    assert!(result.report().real().contains(0.0) && result.report().imaginary().contains(0.0));
    assert_eq!(result.report().status(), EnclosureStatus::Defined);
    let negative = AffineForm::constant(&c, Complex::new(-0.5, 0.0)).unwrap();
    assert_eq!(
        zero.binary(&negative, B::Pow, &mut w)
            .unwrap()
            .report()
            .status(),
        EnclosureStatus::Undefined
    );
}

#[test]
fn canonical_construction_and_borrowed_validation() {
    let c = context(8);
    let terms: Vec<_> = (0..4)
        .rev()
        .map(|i| AffineTerm::new(c.symbol(i).unwrap(), 0.25_f64).unwrap())
        .collect();
    let x = AffineForm::from_terms(&c, 0.0, &terms).unwrap();
    assert!(
        x.view()
            .terms()
            .windows(2)
            .all(|p| p[0].symbol < p[1].symbol)
    );
    let mut unordered = x.view().terms().to_vec();
    unordered.swap(0, 1);
    assert_eq!(
        kind(AffineView::new(&x.state, &unordered)),
        SciOxideErrorKind::InvalidInput
    );
    unordered[1] = unordered[0];
    assert_eq!(
        kind(AffineView::new(&x.state, &unordered)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        kind(AffineForm::from_terms(&c, 0.0, &unordered)),
        SciOxideErrorKind::InvalidInput
    );
    fn is_copy<T: Copy>() {}
    is_copy::<AffineSymbol>();
    is_copy::<AffineTerm<Complex<f64>>>();
    let counter = AtomicUsize::new(usize::MAX);
    assert_eq!(kind(ordinal(&counter)), SciOxideErrorKind::SizeOverflow);
    assert_eq!(counter.load(Ordering::Relaxed), usize::MAX);
    c.data.namespaces.store(usize::MAX, Ordering::Relaxed);
    assert_eq!(kind(c.workspace::<f64>()), SciOxideErrorKind::SizeOverflow);
}

/// Quadratic pair expansion from main, used only as a small numerical oracle.
fn pairwise_product<T: NumericScalar>(a: &AffineForm<T>, b: &AffineForm<T>) -> Box2<T::Real> {
    let mut center = Box2::zero().unwrap();
    let mut error = Box2::zero().unwrap();
    for l in a.view().terms() {
        for r in b.view().terms() {
            let product = Box2::point(l.coefficient)
                .unwrap()
                .mul(Box2::point(r.coefficient).unwrap())
                .unwrap();
            if l.symbol == r.symbol {
                center = center.add(product.scale(0.5).unwrap()).unwrap();
                error = error
                    .add(product.scale(0.5).unwrap().symmetric_hull().unwrap())
                    .unwrap();
            } else {
                error = error.add(product.symmetric_hull().unwrap()).unwrap();
            }
        }
    }
    center.add(error).unwrap()
}

#[test]
fn product_width_matches_pairwise_baseline() {
    let c = context(16);
    let mut w = c.workspace::<f64>().unwrap();
    for (a, b) in [
        ([1.0, -0.5, 0.25, 0.125], [-0.25, 0.5, 0.125, 0.25]),
        ([1.0, 0.5, 0.25, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ([1.0, 0.5, 0.25, 0.125], [1.0, 0.5, 0.25, 0.125]),
    ] {
        let a = AffineForm::from_coefficients(&c, 0.0, &a).unwrap();
        let b = AffineForm::from_coefficients(&c, 0.0, &b).unwrap();
        let old = pairwise_product(&a, &b);
        let result = a.binary(&b, B::Mul, &mut w).unwrap();
        let (lo, hi) = result.report().real().bounds().unwrap();
        let (old_lo, old_hi) = old.re.bounds().unwrap();
        assert!(hi - lo <= old_hi - old_lo);
    }
    let mut cw = c.workspace::<Complex<f64>>().unwrap();
    for coefficient in [Complex::new(0.25, 0.25), Complex::new(-0.5, 0.125)] {
        let x = AffineForm::from_coefficients(&c, Complex::new(0.0, 0.0), &[coefficient]).unwrap();
        let old = pairwise_product(&x, &x);
        let result = x.unary(U::Square, &mut cw).unwrap();
        assert!(old.re.contains_interval(&result.report().real()));
        assert!(old.im.contains_interval(&result.report().imaginary()));
    }
}

#[test]
fn mixed_complex_product_contains_noise_corners() {
    let c = context(16);
    let coefficients = [
        Complex::new(0.25, -0.125),
        Complex::new(-0.5, 0.25),
        Complex::new(0.125, 0.5),
    ];
    let a = AffineForm::from_coefficients(&c, Complex::new(0.5, 0.25), &coefficients).unwrap();
    let b =
        AffineForm::from_coefficients(&c, Complex::new(-0.25, 0.125), &coefficients[..2]).unwrap();
    let mut w = c.workspace::<Complex<f64>>().unwrap();
    let result = a.binary(&b, B::Mul, &mut w).unwrap();
    for bits in 0..8 {
        let mut av = Complex::new(0.5, 0.25);
        let mut bv = Complex::new(-0.25, 0.125);
        for (i, coefficient) in coefficients.iter().enumerate() {
            let noise = if bits & (1 << i) == 0 { -1.0 } else { 1.0 };
            av += *coefficient * noise;
            if i < 2 {
                bv += *coefficient * noise;
            }
        }
        let exact = av * bv;
        assert!(result.report().real().contains(exact.re));
        assert!(result.report().imaginary().contains(exact.im));
    }
    drop(w);
    let mut fresh = c.workspace::<Complex<f64>>().unwrap();
    let zero = result.binary(&result, B::Sub, &mut fresh).unwrap();
    assert_eq!(zero.report().real().bounds(), Some((-0.0, 0.0)));
    assert_eq!(zero.report().imaginary().bounds(), Some((-0.0, 0.0)));
}

#[test]
fn full_union_compresses_with_reused_storage_and_keeps_canonical_ties() {
    let c = AffineContext::new(128, AffineOptions::new().with_max_symbols(64)).unwrap();
    let a = AffineForm::from_coefficients(&c, 0.0_f64, &[0.125; 64]).unwrap();
    let terms: Vec<_> = (64..128)
        .map(|i| AffineTerm::new(c.symbol(i).unwrap(), 0.125).unwrap())
        .collect();
    let b = AffineForm::from_terms(&c, 0.0, &terms).unwrap();
    let mut output = AffineForm::zero(&c).unwrap();
    let mut w = c.workspace::<f64>().unwrap();
    let report = a
        .view()
        .binary_into(b.view(), B::Add, output.view_mut(), &mut w)
        .unwrap();
    assert_eq!(report.retained_symbols(), 64);
    assert_eq!(report.discarded_symbols(), 64);
    assert_eq!(report.real().bounds(), Some((-16.0, 16.0)));
    for (i, term) in output.view().terms().iter().enumerate() {
        assert_eq!(term.symbol, Some(c.symbol(i).unwrap()));
    }
    assert!(AffineView::new(&output.state, &output.terms).is_ok());
}

#[test]
fn generated_terms_insert_before_later_workspace_namespaces() {
    let c = context(16);
    let mut first = c.workspace::<f64>().unwrap();
    let mut later = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 0.5_f64, &[0.125]).unwrap();
    let y = x.unary(U::Exp, &mut later).unwrap();
    let z = y.unary(U::Exp, &mut first).unwrap();
    assert!(
        z.view()
            .terms()
            .windows(2)
            .all(|p| p[0].symbol < p[1].symbol)
    );
    assert!(AffineView::new(&z.state, &z.terms).is_ok());
    assert!(
        z.view()
            .terms()
            .iter()
            .any(|term| term.symbol.is_some_and(|s| s.namespace == first.namespace))
    );
    assert!(
        z.view()
            .terms()
            .iter()
            .any(|term| term.symbol.is_some_and(|s| s.namespace == later.namespace))
    );
}

fn overflowing_forms<S: RealScalar>() {
    for retained in [0, 8] {
        let c = context(retained);
        let mut w = c.workspace::<S>().unwrap();
        for (center, coefficients) in [(S::ZERO, vec![S::MAX, S::MAX]), (S::MAX, vec![S::MAX])] {
            let x = AffineForm::from_coefficients(&c, center, &coefficients).unwrap();
            assert_eq!(x.report().fallback(), Some(AffineFallback::Unbounded));
            assert_eq!(x.report().retained_symbols(), 0);
            let exp = x.unary(U::Exp, &mut w).unwrap();
            assert_eq!(exp.report().fallback(), Some(AffineFallback::Unbounded));
            assert!(
                exp.report()
                    .real()
                    .set()
                    .bounds()
                    .is_some_and(|(_, hi)| hi.is_none())
            );
        }
    }
    let c = context(8);
    let mut w = c.workspace::<S>().unwrap();
    let x = AffineForm::constant(&c, S::MAX).unwrap();
    for result in [
        x.unary(U::Exp, &mut w),
        x.unary(U::Square, &mut w),
        x.binary(&x, B::Add, &mut w),
        x.binary(&x, B::Mul, &mut w),
    ] {
        assert_eq!(
            result.unwrap().report().fallback(),
            Some(AffineFallback::Unbounded)
        );
    }
    let mut cw = c.workspace::<Complex<S>>().unwrap();
    let z = AffineForm::from_coefficients(
        &c,
        Complex::new(S::ZERO, S::ZERO),
        &[Complex::new(S::ZERO, S::MAX), Complex::new(S::ZERO, S::MAX)],
    )
    .unwrap();
    assert_eq!(z.report().fallback(), Some(AffineFallback::Unbounded));
    let z = AffineForm::constant(&c, Complex::new(S::MAX, S::MAX)).unwrap();
    assert_eq!(
        z.binary(&z, B::Add, &mut cw).unwrap().report().fallback(),
        Some(AffineFallback::Unbounded)
    );
    assert_eq!(
        z.unary(U::Square, &mut cw).unwrap().report().fallback(),
        Some(AffineFallback::Unbounded)
    );
    let negative = AffineForm::constant(&c, -S::ONE).unwrap();
    assert_eq!(
        negative.unary(U::Ln, &mut w).unwrap().report().fallback(),
        Some(AffineFallback::DomainOrBranch)
    );
}

#[test]
fn overflowing_constructions_and_operations_report_unbounded() {
    overflowing_forms::<f32>();
    overflowing_forms::<f64>();
}

fn enclosure_construction<S: RealScalar>() {
    let c = context(8);
    let other = context(8);
    let one = S::ONE;
    let two = one + one;
    let real = DecoratedInterval::new(-one, one).unwrap();
    let imaginary = DecoratedInterval::new(one, two).unwrap();
    let zero = DecoratedInterval::point(S::ZERO).unwrap();
    let rs = c.symbol(0).unwrap();
    let is = c.symbol(1).unwrap();
    let foreign = other.symbol(0).unwrap();
    let x = AffineForm::<S>::from_enclosure(&c, real, zero, rs, None).unwrap();
    assert!(x.report().real().contains_interval(&real));
    assert_eq!(x.view().terms()[0].symbol, Some(rs));
    let copy = AffineForm::<S>::from_enclosure(&c, real, zero, rs, None).unwrap();
    let mut w = c.workspace::<S>().unwrap();
    assert_eq!(
        x.binary(&copy, B::Sub, &mut w)
            .unwrap()
            .report()
            .real()
            .bounds(),
        Some((-S::ZERO, S::ZERO))
    );
    for decoration in [
        Decoration::Trv,
        Decoration::Def,
        Decoration::Dac,
        Decoration::Com,
    ] {
        let r = DecoratedInterval::with_decoration(real.set(), decoration).unwrap();
        let i = DecoratedInterval::with_decoration(imaginary.set(), Decoration::Dac).unwrap();
        let z = AffineForm::<Complex<S>>::from_enclosure(&c, r, i, rs, Some(is)).unwrap();
        assert_eq!(z.report().fallback(), None);
        assert!(z.report().real().contains_interval(&r));
        assert!(z.report().imaginary().contains_interval(&i));
        assert_eq!(
            z.report().real().decoration(),
            decoration.min(Decoration::Dac)
        );
        assert_eq!(
            z.report().imaginary().decoration(),
            decoration.min(Decoration::Dac)
        );
        assert_eq!(z.view().terms().len(), 2);
        assert_eq!(z.view().terms()[0].symbol, Some(rs));
        assert_eq!(z.view().terms()[1].symbol, Some(is));
        let x = AffineForm::<S>::from_enclosure(&c, r, zero, rs, None).unwrap();
        assert_eq!(x.report().real().decoration(), decoration);
    }
    for (real_symbol, imaginary_symbol) in [
        (foreign, Some(is)),
        (rs, Some(foreign)),
        (rs, Some(rs)),
        (rs, None),
    ] {
        assert_eq!(
            kind(AffineForm::<Complex<S>>::from_enclosure(
                &c,
                real,
                imaginary,
                real_symbol,
                imaginary_symbol
            )),
            SciOxideErrorKind::InvalidInput
        );
    }
    assert_eq!(
        kind(AffineForm::<S>::from_enclosure(
            &c,
            real,
            imaginary,
            rs,
            Some(is)
        )),
        SciOxideErrorKind::InvalidInput
    );
    let unbounded =
        DecoratedInterval::from(scioxide_core::ExtendedInterval::new(None, None).unwrap());
    let fallback = AffineForm::<S>::from_enclosure(&c, unbounded, zero, rs, None).unwrap();
    assert_eq!(
        fallback.report().fallback(),
        Some(AffineFallback::EnclosureInput)
    );
    assert!(fallback.report().real().set().is_entire());
    let empty = AffineForm::<Complex<S>>::from_enclosure(
        &c,
        real,
        DecoratedInterval::empty(),
        rs,
        Some(is),
    )
    .unwrap();
    assert_eq!(empty.report().status(), EnclosureStatus::EmptyInput);
    assert_eq!(
        empty.report().fallback(),
        Some(AffineFallback::EnclosureInput)
    );
}

#[test]
fn enclosure_rectangles_symbols_and_decorations() {
    enclosure_construction::<f32>();
    enclosure_construction::<f64>();
}

fn fallback_integer_power<S: RealScalar>() {
    let c = context(8);
    let mut w = c.workspace::<S>().unwrap();
    // sqrt(0) cannot admit a finite derivative, but retains the singleton zero.
    let fallback_zero = AffineForm::constant(&c, S::ZERO)
        .unwrap()
        .unary(U::Sqrt, &mut w)
        .unwrap();
    assert!(fallback_zero.report().fallback().is_some());
    for n in [-3, 0, 2, 3] {
        let exponent =
            AffineForm::constant(&c, S::try_from_f64_lossy(f64::from(n)).unwrap()).unwrap();
        let exponent = fallback_zero.binary(&exponent, B::Add, &mut w).unwrap();
        assert!(exponent.report().fallback().is_some());
        for (center, radius) in [(-2.0, 0.125), (0.0, 0.0), (0.0, 0.125)] {
            let x = AffineForm::from_coefficients(
                &c,
                S::try_from_f64_lossy(center).unwrap(),
                &[S::try_from_f64_lossy(radius).unwrap()],
            )
            .unwrap();
            let expected = x.report().real().powi(n).unwrap();
            let result = x.binary(&exponent, B::Pow, &mut w).unwrap();
            assert!(
                result.report().real().contains_interval(&expected.hull()),
                "{n}: {:?}",
                result.report()
            );
            assert!(result.report().fallback().is_some());
        }
    }
    let mut cw = c.workspace::<Complex<S>>().unwrap();
    let zero = AffineForm::constant(&c, Complex::new(S::ZERO, S::ZERO))
        .unwrap()
        .unary(U::Sqrt, &mut cw)
        .unwrap();
    let three = AffineForm::constant(
        &c,
        Complex::new(S::try_from_f64_lossy(3.0).unwrap(), S::ZERO),
    )
    .unwrap();
    let exponent = zero.binary(&three, B::Add, &mut cw).unwrap();
    let base = AffineForm::from_coefficients(
        &c,
        Complex::new(-S::ONE, S::ZERO),
        &[Complex::new(S::try_from_f64_lossy(0.125).unwrap(), S::ZERO)],
    )
    .unwrap();
    let result = base.binary(&exponent, B::Pow, &mut cw).unwrap();
    for value in [-1.125_f64, -1.0, -0.875] {
        assert!(
            result
                .report()
                .real()
                .contains(S::try_from_f64_lossy(value.powi(3)).unwrap())
        );
        assert!(result.report().imaginary().contains(S::ZERO));
    }
}

#[test]
fn fallback_integer_exponents_preserve_nonpositive_base_containment() {
    fallback_integer_power::<f32>();
    fallback_integer_power::<f64>();
}

fn signed_zero_binary<S: RealScalar>() {
    let c = context(8);
    let mut w = c.workspace::<S>().unwrap();
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    let form = |a, b| {
        AffineForm::<S>::from_enclosure(
            &c,
            DecoratedInterval::new(scalar(a), scalar(b)).unwrap(),
            DecoratedInterval::point(S::ZERO).unwrap(),
            c.symbol(0).unwrap(),
            None,
        )
        .unwrap()
    };
    for zero in [-0.0, 0.0] {
        let y = form(zero, 1.0);
        let x = AffineForm::constant(&c, -S::ONE).unwrap();
        let angle = y.binary(&x, B::Atan2, &mut w).unwrap();
        assert!(
            angle
                .report()
                .real()
                .contains(scalar(3.0 * core::f64::consts::FRAC_PI_4))
        );
        assert!(
            angle
                .report()
                .real()
                .contains(scalar(core::f64::consts::PI))
        );
        assert!(!angle.report().real().contains(S::ZERO));
        for (a, b, sign) in [(zero, 1.0, 1.0), (-1.0, zero, -1.0)] {
            let base = form(a, b);
            for exponent in [-3.0, -0.5] {
                if sign < 0.0 && exponent == -0.5 {
                    continue;
                }
                let exponent = AffineForm::constant(&c, scalar(exponent)).unwrap();
                let result = base.binary(&exponent, B::Pow, &mut w).unwrap();
                let expected = if sign > 0.0 {
                    DecoratedInterval::from(
                        scioxide_core::ExtendedInterval::new(Some(S::ONE), None).unwrap(),
                    )
                } else {
                    DecoratedInterval::from(
                        scioxide_core::ExtendedInterval::new(None, Some(-S::ONE)).unwrap(),
                    )
                };
                assert!(result.report().real().contains_interval(&expected));
                assert_eq!(result.report().status(), EnclosureStatus::PartiallyDefined);
            }
        }
    }
}

#[test]
fn signed_zero_binary_boundaries_preserve_containment() {
    signed_zero_binary::<f32>();
    signed_zero_binary::<f64>();
}

fn projection<T: NumericScalar>() {
    let c = context(8);
    let x = AffineForm::from_coefficients(
        &c,
        T::from_components(T::Real::ONE, T::Real::ONE),
        &[T::from_components(T::Real::ONE, -T::Real::ONE)],
    )
    .unwrap();
    for input in [&x, &x.reduce(0, &mut c.workspace::<T>().unwrap()).unwrap()] {
        for projection in [AffineProjection::Real, AffineProjection::Imaginary] {
            let owned = input.project(projection).unwrap();
            let mut output = AffineForm::<T::Real>::zero(&c).unwrap();
            let report = input
                .view()
                .project_into(projection, output.view_mut())
                .unwrap();
            assert_eq!(report, owned.report());
            assert_eq!(report.status(), input.report().status());
            assert_eq!(report.correlation_lost(), input.report().correlation_lost());
            assert_eq!(
                report.real(),
                projection.interval(input.report().real(), input.report().imaginary())
            );
            assert_eq!(
                report.imaginary().bounds(),
                Some((-T::Real::ZERO, T::Real::ZERO))
            );
            for (actual, expected) in output.view().terms().iter().zip(input.view().terms()) {
                assert_eq!(actual.symbol, expected.symbol);
                assert_eq!(actual.coefficient, projection.scalar(expected.coefficient));
            }
            assert!(AffineView::new(&output.state, &output.terms).is_ok());
            let foreign = context(8);
            let alien = AffineForm::<T>::zero(&foreign).unwrap();
            let before = output.report();
            assert_eq!(
                kind(alien.view().project_into(projection, output.view_mut())),
                SciOxideErrorKind::InvalidInput
            );
            assert_eq!(output.report(), before);
            // Test the publication preflight even for privately constructed short views.
            let short = AffineViewMut {
                state: &mut output.state,
                terms: &mut output.terms[..1],
            };
            assert_eq!(
                kind(input.view().project_into(projection, short)),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(output.report(), before);
            assert!(
                output.terms[report.retained_symbols()..]
                    .iter()
                    .all(|term| term.symbol.is_none())
            );
        }
    }
    let huge = AffineForm::from_coefficients(
        &c,
        T::ZERO,
        &[T::from_real(T::Real::MAX), T::from_real(T::Real::MAX)],
    )
    .unwrap();
    assert_eq!(
        huge.project(AffineProjection::Real)
            .unwrap()
            .report()
            .fallback(),
        Some(AffineFallback::Unbounded)
    );
}

#[test]
fn projections_are_transactional_and_allocation_free() {
    projection::<f32>();
    projection::<f64>();
    projection::<Complex<f32>>();
    projection::<Complex<f64>>();
}

#[test]
fn integer_exponents_share_exact_i32_admission() {
    use crate::elementary::integer_exponent;
    for n in [i32::MIN, -3, -1, 0, 1, 3, i32::MAX] {
        assert_eq!(integer_exponent(f64::from(n)), Some(n));
    }
    assert_eq!(integer_exponent(-0.0), Some(0));
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.5,
        -0.5,
        f64::from(i32::MAX) + 1.0,
        f64::from(i32::MIN) - 1.0,
    ] {
        assert_eq!(integer_exponent(value), None);
    }
    let last_f32 = f32::from_bits((i32::MAX as f32).to_bits() - 1);
    assert_eq!(integer_exponent(f64::from(last_f32)), Some(last_f32 as i32));
    assert_eq!(integer_exponent(f64::from(i32::MAX as f32)), None);
    assert_eq!(integer_exponent(f64::from(i32::MIN as f32)), Some(i32::MIN));
}

#[test]
fn singular_derivatives_retain_domain_fallback_reasons() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    for (operation, center, radius) in [
        (U::Asin, 1.0, 0.0),
        (U::Acos, -1.0, 0.0),
        (U::Asin, 0.5, 0.5),
        (U::Acosh, 1.0, 0.0),
        (U::Acosh, 1.5, 0.5),
    ] {
        let x = AffineForm::from_coefficients(&c, center, &[radius]).unwrap();
        let result = x.unary(operation, &mut w).unwrap();
        assert_eq!(
            result.report().fallback(),
            Some(AffineFallback::DomainOrBranch)
        );
    }
}

fn width<S: RealScalar>(x: DecoratedInterval<S>) -> f64 {
    let (lo, hi) = x.bounds().unwrap();
    hi.into_f64() - lo.into_f64()
}
/// Whether an affine range is at most four times as wide as the interval image, up to
/// a few units in the last place.
fn near_image<S: RealScalar>(range: DecoratedInterval<S>, image: DecoratedInterval<S>) -> bool {
    let (lo, hi) = image.bounds().unwrap();
    let magnitude = lo.into_f64().abs().max(hi.into_f64().abs());
    let ulp = S::EPSILON.into_f64() * magnitude + S::MIN_SUBNORMAL.into_f64();
    width(range) <= 4.0 * width(image) + 8.0 * ulp
}
fn real_image<S: RealScalar>(x: DecoratedInterval<S>, op: U) -> DecoratedInterval<S> {
    match op {
        U::Atan => x.atan().unwrap().hull(),
        U::Asinh => x.asinh().unwrap().hull(),
        U::Ln => x.ln().unwrap().hull(),
        U::Sqrt => x.sqrt().unwrap(),
        U::Reciprocal => x.reciprocal().unwrap(),
        U::Powi(n) => x.powi(n).unwrap().hull(),
        _ => panic!("no interval image for {op:?}"),
    }
}
fn extreme_real_remainders<S: RealScalar>(cases: &[(U, f64, f64)], divide: bool) {
    let c = context(8);
    let mut w = c.workspace::<S>().unwrap();
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let one = AffineForm::constant(&c, S::ONE).unwrap();
    for &(op, center, radius) in cases {
        let x = AffineForm::from_coefficients(&c, scalar(center), &[scalar(radius)]).unwrap();
        let result = if divide {
            one.binary(&x, B::Div, &mut w).unwrap().report()
        } else {
            x.unary(op, &mut w).unwrap().report()
        };
        let input = x.report().real();
        let image = real_image(input, op);
        assert_eq!(result.fallback(), None, "{op:?}({center}): {result:?}");
        assert!(
            near_image(result.real(), image),
            "{op:?}({center}): {result:?}"
        );
        let (lo, hi) = input.bounds().unwrap();
        for point in [lo, scalar(center), hi] {
            let exact = real_image(DecoratedInterval::point(point).unwrap(), op);
            assert!(
                result.real().contains_interval(&exact),
                "{op:?}({point:?}): {result:?}"
            );
        }
    }
}
#[test]
fn extreme_magnitude_remainders_stay_near_the_interval_image() {
    extreme_real_remainders::<f32>(
        &[
            (U::Atan, 1e15, 1e12),
            (U::Reciprocal, 1e18, 1e15),
            (U::Reciprocal, 1e30, 1e27),
            (U::Ln, 1e25, 1e22),
            (U::Sqrt, 1e30, 1e27),
        ],
        false,
    );
    extreme_real_remainders::<f32>(&[(U::Reciprocal, 1e30, 1e27)], true);
    extreme_real_remainders::<f64>(
        &[
            (U::Atan, 1e120, 1e117),
            (U::Asinh, 1e120, 1e117),
            (U::Powi(-3), 1e120, 1e117),
            (U::Ln, 1e200, 1e197),
            (U::Sqrt, 1e250, 1e247),
        ],
        false,
    );
    extreme_real_remainders::<f64>(&[(U::Reciprocal, 1e200, 1e197)], true);
    // In the ordinary range the Taylor bound is the tighter one and stays in use: the
    // image bound alone would give a width near 3 times that of the image.
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 1.0, &[0.125]).unwrap();
    let ln = x.unary(U::Ln, &mut w).unwrap().report();
    let image = x.report().real().ln().unwrap().hull();
    assert!(width(ln.real()) <= 1.1 * width(image));
}

/// Complex reciprocals of `c ± r` on the real axis against the interval image.
fn scaled_complex_reciprocals<S: RealScalar>(cases: &[(f64, f64)]) {
    let c = context(8);
    let mut w = c.workspace::<Complex<S>>().unwrap();
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let one = AffineForm::constant(&c, Complex::new(S::ONE, S::ZERO)).unwrap();
    for &(center, radius) in cases {
        let x = AffineForm::from_coefficients(
            &c,
            Complex::new(scalar(center), S::ZERO),
            &[Complex::new(scalar(radius), S::ZERO)],
        )
        .unwrap();
        let image = x.report().real().reciprocal().unwrap();
        for result in [
            x.unary(U::Reciprocal, &mut w).unwrap().report(),
            one.binary(&x, B::Div, &mut w).unwrap().report(),
            x.unary(U::Powi(-1), &mut w).unwrap().report(),
        ] {
            assert_eq!(result.fallback(), None, "1/{center}: {result:?}");
            assert!(near_image(result.real(), image), "1/{center}: {result:?}");
            assert!(
                result.real().contains_interval(&image),
                "1/{center}: {result:?}"
            );
            assert!(result.imaginary().contains(S::ZERO));
            assert!(width(result.imaginary()) <= 8.0 * S::MIN_SUBNORMAL.into_f64());
        }
    }
}
#[test]
fn complex_reciprocals_rescale_extreme_moduli() {
    scaled_complex_reciprocals::<f32>(&[(1e30, 1e27), (1e-30, 1e-33), (1.5, 0.125)]);
    scaled_complex_reciprocals::<f64>(&[(1e200, 1e197), (1e-200, 1e-203), (1.5, 0.125)]);
    // Off the real axis: 1/(a(1 + i)) = (1 − i)/(2a).
    let c = context(8);
    let mut w = c.workspace::<Complex<f64>>().unwrap();
    for a in [1e200, 1e-200] {
        let x =
            AffineForm::from_coefficients(&c, Complex::new(a, a), &[Complex::new(a * 1e-3, 0.0)])
                .unwrap();
        let result = x.unary(U::Reciprocal, &mut w).unwrap().report();
        assert_eq!(result.fallback(), None);
        let (lo, hi) = result.real().bounds().unwrap();
        assert!(0.995 * 0.5 / a <= lo && hi <= 1.005 * 0.5 / a, "{result:?}");
        let (lo, hi) = result.imaginary().bounds().unwrap();
        assert!(
            -1.005 * 0.5 / a <= lo && hi <= -0.995 * 0.5 / a,
            "{result:?}"
        );
    }
}

#[test]
fn complex_tangents_and_large_roots_stay_bounded() {
    let c = context(8);
    let mut w = c.workspace::<Complex<f64>>().unwrap();
    // tan(x + iy) → ±i as y → ±∞ and tanh(x + iy) → ±1 as x → ±∞.
    for (op, center, limit) in [
        (U::Tan, Complex::new(0.5, 800.0), Complex::new(0.0, 1.0)),
        (U::Tan, Complex::new(0.5, -800.0), Complex::new(0.0, -1.0)),
        (U::Tanh, Complex::new(800.0, 0.5), Complex::new(1.0, 0.0)),
        (U::Tanh, Complex::new(-800.0, 0.5), Complex::new(-1.0, 0.0)),
    ] {
        for radius in [Complex::new(0.0, 0.0), Complex::new(0.01, 0.01)] {
            let x = AffineForm::from_coefficients(&c, center, &[radius]).unwrap();
            let result = x.unary(op, &mut w).unwrap().report();
            assert_eq!(result.fallback(), None, "{op:?}: {result:?}");
            for (range, value) in [(result.real(), limit.re), (result.imaginary(), limit.im)] {
                let (lo, hi) = range.bounds().unwrap();
                assert!(lo <= value && value <= hi, "{op:?}: {result:?}");
                assert!(hi - lo <= 1e-14, "{op:?}: {result:?}");
            }
        }
    }
    let mut w32 = c.workspace::<Complex<f32>>().unwrap();
    let x = AffineForm::constant(&c, Complex::new(0.5_f32, 100.0)).unwrap();
    let tan = x.unary(U::Tan, &mut w32).unwrap().report();
    assert_eq!(tan.fallback(), None);
    assert!(tan.imaginary().contains(1.0) && width(tan.imaginary()) <= 1e-6);
    assert!(tan.real().contains(0.0) && width(tan.real()) <= 1e-30);
    // Square root and logarithm of (1 + i)·10³⁰⁰ with a relative radius of 10⁻³.
    let (a, r) = (1e300, 1e297);
    let x = AffineForm::from_coefficients(&c, Complex::new(a, a), &[Complex::new(r, 0.0)]).unwrap();
    let root = Complex::new(1.0_f64, 1.0).sqrt() * 1e150;
    let log = Complex::new(
        a.ln() + core::f64::consts::LN_2 / 2.0,
        core::f64::consts::FRAC_PI_4,
    );
    // |f'(z)| r bounds the first-order variation of each component.
    for (op, value, slope) in [
        (U::Sqrt, root, 0.5 / root.abs()),
        (U::Ln, log, 1.0 / (a * core::f64::consts::SQRT_2)),
    ] {
        let result = x.unary(op, &mut w).unwrap().report();
        assert_eq!(result.fallback(), None, "{op:?}: {result:?}");
        assert!(result.real().contains(value.re) && result.imaginary().contains(value.im));
        assert!(
            width(result.real()) <= 8.0 * slope * r,
            "{op:?}: {result:?}"
        );
        assert!(
            width(result.imaginary()) <= 8.0 * slope * r,
            "{op:?}: {result:?}"
        );
    }
}

#[test]
fn complex_atan2_and_hypot_are_rejected_for_every_operand() {
    let c = context(8);
    let mut w = c.workspace::<Complex<f64>>().unwrap();
    let rs = c.symbol(0).unwrap();
    let is = Some(c.symbol(1).unwrap());
    let empty = DecoratedInterval::<f64>::empty();
    let entire = DecoratedInterval::entire();
    let finite =
        AffineForm::from_coefficients(&c, Complex::new(1.0, 0.5), &[Complex::new(0.125, 0.0)])
            .unwrap();
    let hollow = AffineForm::<Complex<f64>>::from_enclosure(&c, empty, empty, rs, is).unwrap();
    let unbounded = AffineForm::<Complex<f64>>::from_enclosure(&c, entire, entire, rs, is).unwrap();
    let mut output = AffineForm::<Complex<f64>>::zero(&c).unwrap();
    let before = output.report();
    for (x, y) in [
        (&hollow, &finite),
        (&finite, &hollow),
        (&hollow, &hollow),
        (&finite, &finite),
        (&unbounded, &finite),
    ] {
        for op in [B::Atan2, B::Hypot] {
            assert_eq!(
                kind(x.binary(y, op, &mut w)),
                SciOxideErrorKind::InvalidInput
            );
            assert_eq!(
                kind(
                    x.view()
                        .binary_into(y.view(), op, output.view_mut(), &mut w)
                ),
                SciOxideErrorKind::InvalidInput
            );
            assert_eq!(output.report(), before);
        }
    }
    // Real forms accept empty operands and report the empty input.
    let mut rw = c.workspace::<f64>().unwrap();
    let zero = DecoratedInterval::point(0.0).unwrap();
    let real = AffineForm::<f64>::from_enclosure(&c, empty, zero, rs, None).unwrap();
    let one = AffineForm::constant(&c, 1.0).unwrap();
    for op in [B::Atan2, B::Hypot] {
        let result = real.binary(&one, op, &mut rw).unwrap().report();
        assert_eq!(result.status(), EnclosureStatus::EmptyInput);
    }
}

#[test]
fn reductions_keep_fallback_diagnostics() {
    let c = context(8);
    let mut w = c.workspace::<f64>().unwrap();
    let x = AffineForm::from_coefficients(&c, 0.0, &[1.0]).unwrap();
    let zero = DecoratedInterval::point(0.0).unwrap();
    let entire = DecoratedInterval::entire();
    let negative = AffineForm::constant(&c, -1.0).unwrap();
    for (value, status, reason) in [
        (
            x.unary(U::Sqrt, &mut w).unwrap(),
            EnclosureStatus::PartiallyDefined,
            AffineFallback::DomainOrBranch,
        ),
        (
            negative.unary(U::Ln, &mut w).unwrap(),
            EnclosureStatus::Undefined,
            AffineFallback::DomainOrBranch,
        ),
        (
            x.unary(U::Reciprocal, &mut w).unwrap(),
            EnclosureStatus::PartiallyDefined,
            AffineFallback::DomainOrBranch,
        ),
        (
            AffineForm::<f64>::from_enclosure(&c, entire, zero, c.symbol(0).unwrap(), None)
                .unwrap(),
            EnclosureStatus::Defined,
            AffineFallback::EnclosureInput,
        ),
        (
            AffineForm::from_coefficients(&c, 0.0, &[f64::MAX, f64::MAX]).unwrap(),
            EnclosureStatus::Defined,
            AffineFallback::Unbounded,
        ),
    ] {
        assert_eq!(value.report().status(), status);
        assert_eq!(value.report().fallback(), Some(reason));
        for max in [0, 8] {
            assert_eq!(value.reduce(max, &mut w).unwrap().report(), value.report());
        }
        // An operation on the enclosure classifies itself and names the enclosure input.
        let negated = value.unary(U::Neg, &mut w).unwrap().report();
        let expected = if reason == AffineFallback::Unbounded {
            reason
        } else {
            AffineFallback::EnclosureInput
        };
        assert_eq!(negated.fallback(), Some(expected));
        if status != EnclosureStatus::Undefined {
            assert_eq!(negated.status(), EnclosureStatus::Defined);
        }
    }
}

#[test]
fn projections_omit_zero_component_coefficients() {
    let c = context(8);
    let z = AffineForm::from_coefficients(
        &c,
        Complex::new(1.0_f64, 1.0),
        &[
            Complex::new(0.25, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.0, 0.125),
        ],
    )
    .unwrap();
    assert_eq!(z.view().terms().len(), 3);
    for (projection, expected, range) in [
        (
            AffineProjection::Real,
            vec![(0, 0.25), (1, 0.5)],
            (0.25, 1.75),
        ),
        (
            AffineProjection::Imaginary,
            vec![(2, 0.125)],
            (0.875, 1.125),
        ),
    ] {
        let p = z.project(projection).unwrap();
        let terms: Vec<_> = p
            .view()
            .terms()
            .iter()
            .map(|t| (t.symbol().unwrap().index, t.coefficient()))
            .collect();
        assert_eq!(terms, expected);
        assert_eq!(p.report().retained_symbols(), expected.len());
        assert_eq!(p.report().real().bounds(), Some(range));
        assert!(AffineView::new(&p.state, &p.terms).is_ok());
    }
    let x = AffineForm::from_coefficients(&c, 1.0_f64, &[0.25, 0.5]).unwrap();
    let imaginary = x.project(AffineProjection::Imaginary).unwrap();
    assert_eq!(imaginary.report().retained_symbols(), 0);
    assert!(imaginary.view().terms().is_empty());
    assert_eq!(imaginary.report().real().bounds(), Some((-0.0, 0.0)));
    assert!(AffineView::new(&imaginary.state, &imaginary.terms).is_ok());
}

fn empty_component_enclosures<S: RealScalar>() {
    let c = context(8);
    let rs = c.symbol(0).unwrap();
    let real = DecoratedInterval::new(-S::ONE, S::ONE).unwrap();
    let zero = DecoratedInterval::point(S::ZERO).unwrap();
    let empty = DecoratedInterval::empty();
    for (re, im) in [(real, empty), (empty, zero), (empty, empty)] {
        let report = AffineForm::<S>::from_enclosure(&c, re, im, rs, None)
            .unwrap()
            .report();
        assert_eq!(report.status(), EnclosureStatus::EmptyInput);
        assert_eq!(report.fallback(), Some(AffineFallback::EnclosureInput));
        assert!(report.real().set().is_empty());
        assert_eq!(report.imaginary().bounds(), Some((-S::ZERO, S::ZERO)));
    }
    // A nonempty imaginary set other than zero remains invalid for real scalars.
    let one = DecoratedInterval::point(S::ONE).unwrap();
    assert_eq!(
        kind(AffineForm::<S>::from_enclosure(&c, empty, one, rs, None)),
        SciOxideErrorKind::InvalidInput
    );
}
#[test]
fn empty_imaginary_enclosures_are_empty_real_inputs() {
    empty_component_enclosures::<f32>();
    empty_component_enclosures::<f64>();
}

#[test]
fn term_construction_skips_unused_slots() {
    let c = context(8);
    let term = AffineTerm::new(c.symbol(0).unwrap(), 0.5_f64).unwrap();
    let x = AffineForm::from_terms(&c, 1.0, &[AffineTerm::default(), term]).unwrap();
    assert_eq!(x.view().terms().len(), 1);
    assert_eq!(x.report().real().bounds(), Some((0.5, 1.5)));
    assert_eq!(
        kind(AffineTerm::new(c.symbol(0).unwrap(), f64::INFINITY)),
        SciOxideErrorKind::InvalidInput
    );
}
