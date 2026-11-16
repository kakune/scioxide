//! One-shot forms: rational and periodic resampling.
use super::{argument_and_reason, error_kind, identical};
use scioxide_core::{
    Complex, ExecutionContext, InvalidInputReason, NdView, NdViewMut, NumericScalar,
    SciOxideErrorKind, StridedLayout, argument,
};
use scioxide_signal::{
    FourierResamplePlan, PolyphasePlan, ResampleAlignment, SignalExecutionOptions, resample,
    resample_periodic,
};

fn real(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (0.37 * k as f64).sin() + 0.25 * (1.9 * k as f64).cos())
        .collect()
}

fn complex(n: usize) -> Vec<Complex<f64>> {
    real(n)
        .into_iter()
        .zip(real(n + 3).into_iter().skip(3))
        .map(|(re, im)| Complex::new(re, -im))
        .collect()
}

/// The polyphase plan with the designed filter and the centered alignment.
fn polyphase<T: NumericScalar>(input: &[T], up: usize, down: usize) -> Vec<T> {
    PolyphasePlan::rational_default(up, down, ResampleAlignment::Centered)
        .unwrap()
        .resample(input)
        .unwrap()
}

/// The Fourier plan on one-dimensional views, as `hilbert` uses its plan.
fn fourier<T: NumericScalar>(input: &[T], output_len: usize) -> Vec<T> {
    let plan =
        FourierResamplePlan::<T>::new(&[input.len()], 0, output_len, SignalExecutionOptions::new())
            .unwrap();
    let mut output = vec![T::ZERO; output_len];
    plan.resample_into(
        NdView::new(input, StridedLayout::new(&[input.len()], &[1]).unwrap()).unwrap(),
        NdViewMut::new(
            &mut output,
            StridedLayout::new(&[output_len], &[1]).unwrap(),
        )
        .unwrap(),
        &mut plan.workspace(()).unwrap(),
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    output
}

#[test]
fn resample_equals_the_centered_polyphase_plan() {
    for (up, down) in [(3, 2), (2, 3), (1, 4), (5, 1), (6, 4), (4, 4)] {
        for n in [0, 1, 7, 64] {
            let x = real(n);
            let y = resample(&x, up, down).unwrap();
            identical(&y, &polyphase(&x, up, down));
            assert_eq!(y.len(), (n * up).div_ceil(down));
            let z = complex(n);
            identical(&resample(&z, up, down).unwrap(), &polyphase(&z, up, down));
        }
    }
}

#[test]
fn resample_edge_cases() {
    assert!(resample::<f64>(&[], 3, 2).unwrap().is_empty());
    assert!(resample::<Complex<f64>>(&[], 3, 2).unwrap().is_empty());
    // Equal rates reduce to the identity filter, which returns the record.
    let x = real(17);
    let y = resample(&x, 5, 5).unwrap();
    identical(&y, &polyphase(&x, 5, 5));
    identical(&y, &x);
    let z = complex(9);
    identical(&resample(&z, 2, 2).unwrap(), &z);
}

#[test]
fn resample_errors() {
    let x = real(8);
    assert_eq!(
        argument_and_reason(resample(&x, 0, 2)),
        (argument!("up"), InvalidInputReason::OutOfRange)
    );
    assert_eq!(
        argument_and_reason(resample(&x, 2, 0)),
        (argument!("down"), InvalidInputReason::OutOfRange)
    );
    assert_eq!(
        error_kind(resample(&[1.0, f64::NAN, 0.0], 3, 2)),
        SciOxideErrorKind::NonFinite
    );
    // The interpolated square wave overshoots its extreme samples.
    let square: Vec<f64> = (0..64)
        .map(|k| if k % 4 < 2 { f64::MAX } else { -f64::MAX })
        .collect();
    assert_eq!(
        error_kind(resample(&square, 2, 1)),
        SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn resample_periodic_equals_the_fourier_plan() {
    for (n, m) in [(8, 12), (9, 4), (7, 7), (1, 5), (6, 1), (16, 33)] {
        let x = real(n);
        identical(&resample_periodic(&x, m).unwrap(), &fourier(&x, m));
        let z = complex(n);
        identical(&resample_periodic(&z, m).unwrap(), &fourier(&z, m));
    }
}

#[test]
fn resample_periodic_errors_name_its_own_arguments() {
    assert_eq!(
        argument_and_reason(resample_periodic::<f64>(&[], 4)),
        (argument!("input"), InvalidInputReason::Empty)
    );
    assert_eq!(
        argument_and_reason(resample_periodic(&real(4), 0)),
        (argument!("output_len"), InvalidInputReason::Empty)
    );
    assert_eq!(
        argument_and_reason(resample_periodic::<Complex<f64>>(&[], 0)),
        (argument!("input"), InvalidInputReason::Empty)
    );
    assert_eq!(
        error_kind(resample_periodic(&[1.0, f64::INFINITY], 4)),
        SciOxideErrorKind::NonFinite
    );
    // The trigonometric interpolant of a square wave overshoots its samples.
    let square: Vec<f64> = (0..8)
        .map(|k| if k < 4 { f64::MAX } else { -f64::MAX })
        .collect();
    assert_eq!(
        error_kind(resample_periodic(&square, 64)),
        SciOxideErrorKind::NumericalInstability
    );
}
