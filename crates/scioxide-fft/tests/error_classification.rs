//! Public error contracts shared by allocating and prepared transforms.

use scioxide_core::{
    Complex, ExecutionContext, InvalidInputReason, NdView, NdViewMut, NumericalIssue,
    SciOxideError, SciOxideErrorKind, SciOxideErrorView, Shape, StridedLayout, Tolerance,
};
use scioxide_fft::*;

fn invalid(error: SciOxideError, expected: InvalidInputReason) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { reason, .. }
        if reason == expected),
        "{error:?}"
    );
}

fn rejected(error: SciOxideError, expected: InvalidInputReason, name: scioxide_core::Argument) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, reason, .. }
        if reason == expected && argument == name),
        "{error:?}: expected {name:?}"
    );
}

fn dimension(error: SciOxideError, expected_len: usize, actual_len: usize) {
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
        if expected == [expected_len] && actual == [actual_len]),
        "{error:?}"
    );
}

fn nd_plan(
    shape: &[usize],
    axes: &[usize],
    layout: FftLayout,
) -> Result<FftNdPlan<f64>, SciOxideError> {
    FftNdPlan::new(FftNdPlanSpec::new(
        shape,
        axes,
        FftDirection::Forward,
        Normalization::Unscaled,
        layout,
    )?)
}

#[test]
fn required_empty_transforms_share_the_empty_reason() {
    let norm = Normalization::Unscaled;
    let (real, complex): (&[f64], &[Complex<f64>]) = (&[], &[]);
    for error in [
        fft::<f64>(&[], norm).unwrap_err(),
        ifft::<f64>(&[], norm).unwrap_err(),
        rfft::<f64>(&[], norm).unwrap_err(),
        ihfft::<f64>(&[], norm).unwrap_err(),
        dct::<f64>(&[], DctType::II, norm).unwrap_err(),
        idct::<f64>(&[], DctType::II, norm).unwrap_err(),
        dst::<f64>(&[], DstType::II, norm).unwrap_err(),
        idst::<f64>(&[], DstType::II, norm).unwrap_err(),
        fftn(complex, &[0], &[0], norm).unwrap_err(),
        ifftn(complex, &[0], &[0], norm).unwrap_err(),
        rfftn(real, &[0], &[0], norm).unwrap_err(),
        ihfftn(real, &[0], &[0], norm).unwrap_err(),
        dctn(real, &[0], &[0], &[DctType::II], norm).unwrap_err(),
        dstn(real, &[0], &[0], &[DstType::II], norm).unwrap_err(),
        HermitianSpectrum::<f64>::from_vec(0, vec![]).unwrap_err(),
        HermitianSpectrum::<f64>::zeros(0).unwrap_err(),
        fht::<f64>(&[], 0.1, 0.5, 0.0, 0.0).unwrap_err(),
        ifht::<f64>(&[], 0.1, 0.5, 0.0, 0.0).unwrap_err(),
        CztPlan::<f64>::new(CztPlanSpec::new(0, 1, CztContour::dft(1).unwrap(), norm)).unwrap_err(),
    ] {
        invalid(error, InvalidInputReason::Empty);
    }
    for transform_layout in [FftLayout::Complex, FftLayout::Real, FftLayout::Hermitian] {
        invalid(
            FftPlan::<f64>::new(FftPlanSpec::new(
                0,
                FftDirection::Forward,
                norm,
                transform_layout,
            ))
            .unwrap_err(),
            InvalidInputReason::Empty,
        );
        invalid(
            nd_plan(&[0], &[0], transform_layout).unwrap_err(),
            InvalidInputReason::Empty,
        );
    }
    invalid(
        dct(&[1.0_f64], DctType::I, norm).unwrap_err(),
        InvalidInputReason::OutOfRange,
    );
    // Empty complementary batches and shifts are valid.
    assert!(nd_plan(&[0, 2], &[1], FftLayout::Complex).is_ok());
    assert!(fftshift_nd(real, &[0], &[0]).unwrap().is_empty());
}

#[test]
fn packed_storage_and_shape_disagreements_are_dimensions() {
    for actual in [0, 2, 4] {
        let bins = vec![Complex::new(0.0_f64, 0.0); actual];
        for error in [
            HermitianSpectrum::from_vec(4, bins.clone()).unwrap_err(),
            HermitianSpectrumNd::from_vec(&[4], &[0], bins).unwrap_err(),
        ] {
            dimension(error, 3, actual);
        }
    }
    let storage = [Complex::new(0.0_f64, 0.0); 12];
    let plan = FftNdPlan::<f64>::new(
        FftNdPlanSpec::new(
            &[2, 4],
            &[1],
            FftDirection::Inverse,
            Normalization::Inverse,
            FftLayout::Real,
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    for (shape, strides) in [
        (&[6][..], &[1][..]),
        (&[3, 3][..], &[3, 1][..]),
        (&[2, 2][..], &[2, 1][..]),
    ] {
        let bins = NdView::new(&storage, StridedLayout::new(shape, strides).unwrap()).unwrap();
        let mut output = [42.0; 8];
        let error = plan
            .execute_into(
                FftNdInput::Complex(bins),
                FftNdOutput::Real(
                    NdViewMut::new(&mut output, StridedLayout::new(&[2, 4], &[4, 1]).unwrap())
                        .unwrap(),
                ),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
            if expected == [2, 3] && actual == shape),
            "{error:?}"
        );
        assert_eq!(output, [42.0; 8]);
    }
    for types in [vec![], vec![RealTransformKind::Dct(DctType::II); 2]] {
        dimension(
            RealTransformNdPlan::<f64>::new(
                RealTransformNdPlanSpec::new(
                    &[4],
                    &[0],
                    &types,
                    FftDirection::Forward,
                    Normalization::Unscaled,
                )
                .unwrap(),
            )
            .unwrap_err(),
            1,
            types.len(),
        );
    }
}

#[test]
fn layout_variants_share_unsupported_configuration() {
    let cx = ExecutionContext::sequential();
    for layout in [FftLayout::Complex, FftLayout::Real, FftLayout::Hermitian] {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let norm = Normalization::Unscaled;
            let plan = FftPlan::<f64>::new(FftPlanSpec::new(4, direction, norm, layout)).unwrap();
            let nd = FftNdPlan::<f64>::new(
                FftNdPlanSpec::new(&[4], &[0], direction, norm, layout).unwrap(),
            )
            .unwrap();
            let mut out = [0.0_f64; 4];
            // A real output with real input cannot match any Fourier layout/direction.
            let a = plan
                .execute_into(
                    FftInput::from(&[0.0_f64; 4][..]),
                    FftOutput::from(&mut out[..]),
                    &mut [],
                    cx,
                )
                .unwrap_err();
            let dense = StridedLayout::new(&[4], &[1]).unwrap();
            let b = nd
                .execute_into(
                    FftNdInput::Real(NdView::new(&[0.0; 4], dense).unwrap()),
                    FftNdOutput::Real(NdViewMut::new(&mut out, dense).unwrap()),
                    &mut [],
                    cx,
                )
                .unwrap_err();
            for error in [a, b] {
                invalid(error, InvalidInputReason::UnsupportedConfiguration);
            }
            if layout != FftLayout::Complex {
                let mut values = [Complex::new(0.0, 0.0); 4];
                for error in [
                    plan.execute_in_place(&mut values, &mut [], cx).unwrap_err(),
                    nd.execute_in_place(NdViewMut::new(&mut values, dense).unwrap(), &mut [], cx)
                        .unwrap_err(),
                ] {
                    invalid(error, InvalidInputReason::UnsupportedConfiguration);
                }
            }
            assert_eq!(out, [0.0; 4]);
        }
    }
}

#[test]
fn axes_share_one_quantity_and_duplicates_are_inconsistent() {
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let input = [Complex::new(1.0_f64, 0.0); 4];
    let view = || NdView::new(&input, layout).unwrap();
    let czt = CztPlan::new(CztPlanSpec::new(
        2,
        2,
        CztContour::<f64>::dft(2).unwrap(),
        Normalization::Unscaled,
    ))
    .unwrap();
    for axis in [2, usize::MAX] {
        let mut output = input;
        for error in [
            Shape::new(&[2, 2]).unwrap().extent(axis).unwrap_err(),
            layout.axis(axis).unwrap_err(),
            fftn(&input, &[2, 2], &[axis], Normalization::Unscaled).unwrap_err(),
            fftshift_nd(&input, &[2, 2], &[axis]).unwrap_err(),
            ifftshift_nd(&input, &[2, 2], &[axis]).unwrap_err(),
            czt.workspace_requirements(AxisWorkspaceRequest::Axis {
                shape: &[2, 2],
                axis,
            })
            .unwrap_err(),
            czt.execute_axis_into(
                view(),
                axis,
                NdViewMut::new(&mut output, layout).unwrap(),
                &mut FftWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap_err(),
        ] {
            assert!(
                matches!(error.view(), SciOxideErrorView::OutOfBounds { collection, index, size }
                if collection == scioxide_core::quantity!("layout_axis") && index == axis && size == 2),
                "{error:?}"
            );
        }
    }
    for error in [
        fftn(&input, &[2, 2], &[0, 0], Normalization::Unscaled).unwrap_err(),
        fftshift_nd(&input, &[2, 2], &[0, 0]).unwrap_err(),
        ifftshift_nd(&input, &[2, 2], &[0, 0]).unwrap_err(),
    ] {
        invalid(error, InvalidInputReason::Inconsistent);
    }
    for error in [
        nd_plan(&[4], &[], FftLayout::Real).unwrap_err(),
        HermitianSpectrumNd::<f64>::from_vec(&[4], &[], vec![]).unwrap_err(),
    ] {
        invalid(error, InvalidInputReason::Empty);
    }
}

#[test]
fn dht_wrapper_and_plan_share_sample_count_errors() {
    let zeros = [
        core::f64::consts::PI,
        2.0 * core::f64::consts::PI,
        3.0 * core::f64::consts::PI,
    ];
    let tolerance = Tolerance::new(1e-12, 1e-12).unwrap();
    for supplied in [&zeros[..0], &zeros[..1]] {
        for error in [
            dht::<f64>(&[], supplied, 0.5, 1.0).unwrap_err(),
            DiscreteHankelPlan::new(
                supplied,
                DiscreteHankelPlanSpec::new(0.5, 1.0, FftDirection::Forward)
                    .with_tolerance(tolerance),
            )
            .unwrap_err(),
        ] {
            invalid(error, InvalidInputReason::Empty);
        }
    }
    let plan = DiscreteHankelPlan::new(
        &zeros,
        DiscreteHankelPlanSpec::new(0.5, 1.0, FftDirection::Forward).with_tolerance(tolerance),
    )
    .unwrap();
    for n in [0, 1, 3] {
        let input = vec![Complex::new(1.0_f64, 0.0); n];
        for error in [
            dht(&vec![1.0; n], &zeros, 0.5, 1.0).unwrap_err(),
            plan.execute_into(
                &input,
                &mut [Complex::new(0.0, 0.0); 2],
                &mut plan.workspace(()).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap_err(),
        ] {
            dimension(error, 2, n);
        }
    }
}

#[test]
fn nufft_geometry_disagreements_are_dimensions() {
    let storage = [0.0_f64; 4];
    let layout = StridedLayout::new(&[2, 1], &[1, 1]).unwrap();
    let points = || NdView::new(&storage, layout).unwrap();
    let spec = || NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6);
    for modes in [&[][..], &[2, 2][..]] {
        for error in [
            NufftPlan::new(
                NufftTransform::Type1 {
                    points: points(),
                    modes,
                },
                spec(),
            )
            .unwrap_err(),
            NufftPlan::new(
                NufftTransform::Type2 {
                    points: points(),
                    modes,
                },
                spec(),
            )
            .unwrap_err(),
        ] {
            dimension(error, 1, modes.len());
        }
    }
    let frequencies = NdView::new(&storage, StridedLayout::new(&[2, 2], &[2, 1]).unwrap()).unwrap();
    dimension(
        NufftPlan::new(
            NufftTransform::Type3 {
                points: points(),
                frequencies,
            },
            spec(),
        )
        .unwrap_err(),
        1,
        2,
    );
    let rank_one = NdView::new(&storage, StridedLayout::new(&[4], &[1]).unwrap()).unwrap();
    dimension(
        NufftPlan::new(
            NufftTransform::Type1 {
                points: rank_one,
                modes: &[2],
            },
            spec(),
        )
        .unwrap_err(),
        2,
        1,
    );
}

#[test]
fn frequency_singletons_need_no_representable_reciprocal() {
    macro_rules! check {
        ($scalar:ty) => {{
            for d in [
                <$scalar>::from_bits(1),
                <$scalar>::MIN_POSITIVE,
                0.5,
                1.0,
                <$scalar>::MAX,
            ] {
                for spacing in [d, -d] {
                    for grid in [fftfreq::<$scalar>, rfftfreq::<$scalar>] {
                        assert_eq!(grid(1, spacing).unwrap(), [0.0]);
                        if d == <$scalar>::from_bits(1) {
                            assert!(matches!(
                                grid(2, spacing).unwrap_err().view(),
                                SciOxideErrorView::NumericalInstability {
                                    issue: NumericalIssue::Overflow,
                                    ..
                                }
                            ));
                        }
                    }
                }
            }
            for grid in [fftfreq::<$scalar>, rfftfreq::<$scalar>] {
                for spacing in [0.0, -0.0] {
                    invalid(
                        grid(1, spacing).unwrap_err(),
                        InvalidInputReason::OutOfRange,
                    );
                }
                for spacing in [<$scalar>::INFINITY, <$scalar>::NEG_INFINITY, <$scalar>::NAN] {
                    assert_eq!(
                        grid(1, spacing).unwrap_err().kind(),
                        SciOxideErrorKind::NonFinite
                    );
                }
            }
        }};
    }
    check!(f32);
    check!(f64);
}

#[test]
fn specialized_spec_fields_use_qualified_argument_paths() {
    let norm = Normalization::Unscaled;
    let direction = FftDirection::Forward;
    let zeros = [core::f64::consts::PI, 2.0 * core::f64::consts::PI];
    let tolerance = Tolerance::new(1e-12, 1e-12).unwrap();
    let zero = Complex::new(0.0_f64, 0.0);
    let nufft = |spec| {
        NufftPlan::new(
            NufftTransform::Type1 {
                points: NdView::new(&[0.0_f64], StridedLayout::new(&[1, 1], &[1, 1]).unwrap())
                    .unwrap(),
                modes: &[2],
            },
            spec,
        )
        .unwrap_err()
    };
    for (error, expected) in [
        (
            CztPlan::new(CztPlanSpec::new(
                0,
                1,
                CztContour::<f64>::dft(2).unwrap(),
                norm,
            ))
            .unwrap_err(),
            scioxide_core::argument!("spec.n"),
        ),
        (
            CztPlan::new(CztPlanSpec::new(
                1,
                0,
                CztContour::<f64>::dft(2).unwrap(),
                norm,
            ))
            .unwrap_err(),
            scioxide_core::argument!("spec.m"),
        ),
        (
            DiscreteHankelPlan::new(
                &zeros,
                DiscreteHankelPlanSpec::new(-0.5, 1.0, direction).with_tolerance(tolerance),
            )
            .unwrap_err(),
            scioxide_core::argument!("spec.order"),
        ),
        (
            DiscreteHankelPlan::new(
                &zeros,
                DiscreteHankelPlanSpec::new(0.5, 0.0, direction).with_tolerance(tolerance),
            )
            .unwrap_err(),
            scioxide_core::argument!("spec.extent"),
        ),
        (
            FftLogPlan::new(FftLogPlanSpec::new(0, 0.1, 0.0, 0.0, zero, direction)).unwrap_err(),
            scioxide_core::argument!("spec.length"),
        ),
        (
            FftLogPlan::new(FftLogPlanSpec::new(2, 0.0, 0.0, 0.0, zero, direction)).unwrap_err(),
            scioxide_core::argument!("spec.log_spacing"),
        ),
        (
            nufft(NufftPlanSpec::new(direction, norm, 1.0)),
            scioxide_core::argument!("spec.accuracy"),
        ),
        (
            nufft(NufftPlanSpec::new(direction, norm, 1e-6).with_oversampling(1.5)),
            scioxide_core::argument!("spec.oversampling"),
        ),
        (
            fftlog_low_ringing_offset(0.0_f64, 0.0, 0.0, 0.0).unwrap_err(),
            scioxide_core::argument!("log_spacing"),
        ),
        (
            CztContour::<f64>::zoom(0.0, 1.0, 0, 2.0, false).unwrap_err(),
            scioxide_core::argument!("m"),
        ),
    ] {
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. }
            if argument == expected),
            "{error:?}: expected {expected:?}"
        );
    }
}

#[test]
fn dht_and_idht_check_samples_and_parameters_before_zeros() {
    use core::f64::consts::PI;
    let zeros: [f64; 4] = core::array::from_fn(|k| (k + 1) as f64 * PI);
    let samples = [1.0, -0.5, 2.0];
    let zeros_argument = scioxide_core::argument!("zeros");
    for transform in [dht::<f64>, idht::<f64>] {
        for supplied in [&zeros[..0], &zeros[..1]] {
            rejected(
                transform(&[], supplied, 0.5, 1.0).unwrap_err(),
                InvalidInputReason::Empty,
                zeros_argument,
            );
        }
        for supplied in [zeros, [f64::NAN, -1.0, 0.0, 0.0]] {
            for n in [0, 2, 4] {
                let error = transform(&vec![1.0; n], &supplied, 0.5, 1.0).unwrap_err();
                dimension(error, 3, n);
            }
        }
        let unordered = [2.0 * PI, PI, 3.0 * PI, 4.0 * PI];
        for supplied in [zeros, unordered] {
            rejected(
                transform(&samples, &supplied, -0.5, 1.0).unwrap_err(),
                InvalidInputReason::OutOfRange,
                scioxide_core::argument!("order"),
            );
            rejected(
                transform(&samples, &supplied, 0.5, 0.0).unwrap_err(),
                InvalidInputReason::OutOfRange,
                scioxide_core::argument!("extent"),
            );
            for (order, extent) in [(f64::NAN, 1.0), (0.5, f64::INFINITY)] {
                let error = transform(&samples, &supplied, order, extent).unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
            }
        }
        let mut moved = zeros;
        moved[2] += 0.01;
        let swapped = [PI, 3.0 * PI, 2.0 * PI, 4.0 * PI];
        for supplied in [moved, swapped] {
            rejected(
                transform(&samples, &supplied, 0.5, 1.0).unwrap_err(),
                InvalidInputReason::OutOfRange,
                zeros_argument,
            );
        }
        let mut perturbed = zeros;
        perturbed[2] *= 1.0 + 1e-10;
        assert_eq!(transform(&samples, &perturbed, 0.5, 1.0).unwrap().len(), 3);
        let error = transform(&[1.0, f64::NAN, 0.0], &zeros, 0.5, 1.0).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    }
}

#[test]
fn one_shots_name_their_own_arguments() {
    let (w, a) = (Complex::new(0.0, -1.0), Complex::new(1.0, 0.0));
    let norm = Normalization::Unscaled;
    let x = [1.0_f64, 2.0];
    let input = scioxide_core::argument!("input");
    let empty = InvalidInputReason::Empty;
    rejected(czt(&[] as &[f64], 3, w, a, norm).unwrap_err(), empty, input);
    rejected(
        czt(&x, 0, w, a, norm).unwrap_err(),
        empty,
        scioxide_core::argument!("m"),
    );
    let zero = Complex::new(0.0, 0.0);
    rejected(
        czt(&[] as &[f64], 3, zero, a, norm).unwrap_err(),
        empty,
        input,
    );
    let error = zoom_fft(&[] as &[Complex<f64>], 0.0, 0.25, 3, 1.0, true, norm).unwrap_err();
    rejected(error, empty, input);

    let bins = [Complex::new(1.0_f64, 0.0); 3];
    let packed = || FftInput::hermitian(4, &bins);
    let points = [0.0_f64, 0.5, 1.0];
    let view = || NdView::new(&points, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
    let direction = FftDirection::Forward;
    for error in [
        czt(packed(), 3, w, a, norm).unwrap_err(),
        zoom_fft(packed(), 0.0, 0.25, 3, 1.0, true, norm).unwrap_err(),
        nufft_type1(view(), packed(), &[4], direction, norm, 1e-6).unwrap_err(),
    ] {
        rejected(error, InvalidInputReason::UnsupportedConfiguration, input);
    }

    let c = [1.0_f64; 3];
    let modes = [1.0_f64; 4];
    for accuracy in [0.0, 1.0, f64::NAN] {
        for error in [
            nufft_type1(view(), &c, &[4], direction, norm, accuracy).unwrap_err(),
            nufft_type2(view(), &modes, &[4], direction, norm, accuracy).unwrap_err(),
            nufft_type3(view(), &c, view(), direction, norm, accuracy).unwrap_err(),
        ] {
            if accuracy.is_nan() {
                assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
            } else {
                let name = scioxide_core::argument!("accuracy");
                rejected(error, InvalidInputReason::OutOfRange, name);
            }
        }
    }
}
