#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;

use scioxide_core::{Capacity, ExecutionContext, SciOxideErrorKind};
use scioxide_signal::{
    Peak, PeakOptions, PeakWorkspace, find_peaks_into, find_peaks_with_options,
    peak_properties_into,
};

fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

fn workspace(max_input_len: usize) -> PeakWorkspace<f64> {
    PeakWorkspace::workspace(max_input_len).unwrap()
}

// Independent all-pairs priority selection. Geometry is checked through the
// separate supplied-maxima API; this reference never calls find_peaks_into.
fn reference(input: &[f64], options: PeakOptions<f64>) -> Vec<Peak<f64>> {
    let mut candidates = Vec::new();
    for i in 1..input.len().saturating_sub(1) {
        if input[i] <= input[i - 1] {
            continue;
        }
        let end = (i..input.len())
            .take_while(|&j| input[j] == input[i])
            .last()
            .unwrap();
        if end + 1 == input.len() || input[end] <= input[end + 1] {
            continue;
        }
        let mut peak = [Peak::default()];
        peak_properties_into(input, &[i], options.relative_height(), &mut peak, native()).unwrap();
        let p = peak[0];
        if options
            .height()
            .is_none_or(|v| v.contains(input[p.index()]))
            && options
                .prominence()
                .is_none_or(|v| v.contains(p.prominence()))
            && options.width().is_none_or(|v| v.contains(p.width()))
            && options
                .plateau_size()
                .is_none_or(|(lo, hi)| (lo..=hi).contains(&(end - i + 1)))
        {
            candidates.push(p);
        }
    }
    candidates.sort_unstable_by(|a, b| {
        input[b.index()]
            .total_cmp(&input[a.index()])
            .then(a.index().cmp(&b.index()))
    });
    let mut selected: Vec<Peak<f64>> = Vec::new();
    for p in candidates {
        if selected
            .iter()
            .all(|q| p.index().abs_diff(q.index()) >= options.distance())
        {
            selected.push(p);
        }
    }
    selected.sort_unstable_by_key(|p| p.index());
    selected
}

fn samples(n: usize) -> Vec<f64> {
    let mut state = 17_u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8)
        })
        .collect()
}

#[test]
fn distance_selection_matches_brute_force_with_filters_and_plateaus() {
    let mut workspace = workspace(129);
    let mut output = vec![Peak::default(); 64];
    for n in 0..130 {
        let input: Vec<_> = samples(n).into_iter().map(|x| x % 7.0).collect();
        for distance in [0, 1, 2, 3, 7, 128, usize::MAX] {
            for filtered in [false, true] {
                let mut options = PeakOptions::default().with_distance(distance);
                if filtered {
                    options = options
                        .with_height(scioxide_core::Interval::new(2.0, 5.0).unwrap())
                        .with_prominence(scioxide_core::Interval::new(1.0, 6.0).unwrap())
                        .with_width(scioxide_core::Interval::new(0.5, 4.0).unwrap())
                        .with_plateau_size((1, 3));
                }
                let expected = reference(&input, options);
                let count = find_peaks_into(&input, options, &mut output, &mut workspace, native())
                    .unwrap();
                assert_eq!(&output[..count], expected, "n={n}, distance={distance}");
            }
        }
    }
}

#[test]
fn many_peaks_reuse_workspace_with_and_without_distance() {
    let input = samples(65_536);
    let mut output = vec![Peak::default(); input.len() / 2];
    let mut workspace = workspace(input.len());
    for distance in [9, 0, 9, 1, usize::MAX] {
        let options = PeakOptions::default().with_distance(distance);
        let count =
            find_peaks_into(&input, options, &mut output, &mut workspace, native()).unwrap();
        assert!(count > 0);
        assert!(
            output[..count]
                .windows(2)
                .all(|p| p[1].index() - p[0].index() >= distance.max(1))
        );
        if distance == usize::MAX {
            assert_eq!(count, 1);
        } else {
            assert!(count > 1000);
        }
    }
}

#[test]
fn capacity_and_invalid_input_errors_preserve_output() {
    let input = [0.0, 3.0, 0.0, 2.0, 0.0];
    let sentinel = Peak::default();
    let mut output = [sentinel];
    let mut workspace = workspace(input.len());
    assert_eq!(
        error_kind(find_peaks_into(
            &input,
            PeakOptions::default(),
            &mut output,
            &mut workspace,
            native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [sentinel]);
    let requirements = PeakWorkspace::<f64>::workspace_requirements(input.len());
    assert_eq!(requirements.records_len(), 2);
    assert_eq!(requirements.tree_len(), 16);
    let mut short = PeakWorkspace::new();
    assert_eq!(short.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    assert_eq!(
        find_peaks_into(
            &input,
            PeakOptions::default(),
            &mut output,
            &mut short,
            native()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [sentinel]);
    short.try_reserve(&requirements).unwrap();
    assert_eq!(short.check(&requirements).unwrap(), Capacity::Sufficient);
    assert_eq!(
        error_kind(find_peaks_into(
            &[0.0, f64::NAN, 0.0],
            PeakOptions::default(),
            &mut output,
            &mut workspace,
            native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, [sentinel]);
    let count = find_peaks_into(
        &input,
        PeakOptions::default().with_distance(3),
        &mut output,
        &mut workspace,
        native(),
    )
    .unwrap();
    assert_eq!(count, 1);
    assert_eq!(output[0].index(), 1);
    assert_eq!(
        find_peaks_into(
            &[],
            PeakOptions::default(),
            &mut [],
            &mut workspace,
            native()
        )
        .unwrap(),
        0
    );
    // Supplied maxima publish only after every index is measured.
    let mut properties = [sentinel; 2];
    assert_eq!(
        error_kind(peak_properties_into(
            &input,
            &[1, 2],
            0.5,
            &mut properties,
            native()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(properties, [sentinel; 2]);
    peak_properties_into(&input, &[1, 3], 0.5, &mut properties, native()).unwrap();
    assert_eq!((properties[0].index(), properties[1].index()), (1, 3));
    assert!((properties[1].prominence() - 2.0).abs() < 1e-12);
}

#[test]
fn tree_geometry_matches_linear_scans_on_ties_combs_and_staircases() {
    // Equal-height peaks never stop a base scan, and a comb above a deep basin
    // makes every contour crossing span the record: both are worst cases for
    // linear scans and must give the scan geometry exactly.
    // Every record is longer than the 512-sample scan budget, so searches
    // reach the extremum trees.
    let alternating: Vec<f64> = (0..2049).map(|i| f64::from(i % 2)).collect();
    let mut comb = vec![0.0];
    comb.extend((0..1199).map(|i| if i % 2 == 0 { 10.0 } else { 9.0 }));
    comb.push(0.0);
    let staircase: Vec<f64> = (0..1500)
        .map(|i| f64::from(i / 2) + if i % 2 == 0 { 0.0 } else { 0.5 })
        .collect();
    let quantized: Vec<f64> = samples(4000).into_iter().map(|x| x % 3.0).collect();
    let mut workspace = workspace(4000);
    let mut output = vec![Peak::default(); 2000];
    for input in [&alternating, &comb, &staircase, &quantized] {
        for relative_height in [0.0, 0.25, 0.5, 1.0] {
            for distance in [0, 3] {
                let options = PeakOptions::default()
                    .with_relative_height(relative_height)
                    .with_distance(distance);
                let count =
                    find_peaks_into(input, options, &mut output, &mut workspace, native()).unwrap();
                assert_eq!(&output[..count], reference(input, options));
            }
        }
    }
}

#[test]
fn equal_height_peaks_of_a_long_record_have_unit_geometry() {
    let input: Vec<f64> = (0..131_072).map(|i| f64::from(i % 2)).collect();
    let mut workspace = workspace(input.len());
    let mut output = vec![Peak::default(); input.len() / 2];
    let count = find_peaks_into(
        &input,
        PeakOptions::default(),
        &mut output,
        &mut workspace,
        native(),
    )
    .unwrap();
    assert_eq!(count, input.len() / 2 - 1);
    for (k, peak) in output[..count].iter().enumerate() {
        let index = 2 * k + 1;
        assert_eq!(peak.index(), index);
        assert_eq!(
            (peak.left_base(), peak.right_base()),
            (index - 1, index + 1)
        );
        assert_eq!(peak.prominence(), 1.0);
        assert_eq!(peak.width(), 1.0);
    }
}

#[test]
fn allocated_workspace_meets_its_requirements_and_serves_with_reused_storage() {
    for n in [0, 1, 7, 1000] {
        let requirements = PeakWorkspace::<f64>::workspace_requirements(n);
        let mut workspace = PeakWorkspace::<f64>::workspace(n).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::Sufficient
        );
        let input = samples(n);
        let mut output = vec![Peak::default(); requirements.records_len()];
        for options in [
            PeakOptions::default(),
            PeakOptions::default().with_distance(5),
        ] {
            let count =
                find_peaks_into(&input, options, &mut output, &mut workspace, native()).unwrap();
            assert_eq!(
                output[..count],
                find_peaks_with_options(&input, options).unwrap()
            );
            assert!(n < 1000 || count > 100);
        }
    }
    // The saturated tree length of the largest request cannot be reserved.
    assert_eq!(
        error_kind(PeakWorkspace::<f64>::workspace(usize::MAX)),
        SciOxideErrorKind::AllocationFailure
    );
}
