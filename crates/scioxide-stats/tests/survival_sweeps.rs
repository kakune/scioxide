#![allow(clippy::unwrap_used, reason = "checked test fixtures")]

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_stats::{
    ConfidenceProbability, SurvivalObservation, SurvivalOptions, kaplan_meier_with_options,
    logrank_test_with_options, nelson_aalen_with_options,
};

fn scalar<S: RealScalar>(text: &str) -> S {
    S::try_from_f64_lossy(text.parse().unwrap()).unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}

fn references<S: RealScalar>() -> Result<(), SciOxideError> {
    let data: Vec<Vec<&str>> = include_str!("data/survival.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect())
        .collect();
    let tolerance = if core::mem::size_of::<S>() == 4 {
        2e-6
    } else {
        2e-13
    };
    for case in [
        "censored",
        "delayed",
        "gaps",
        "censored_group",
        "all_censored",
    ] {
        let groups: [Vec<SurvivalObservation<S>>; 2] = core::array::from_fn(|group| {
            data.iter()
                .filter(|row| {
                    row[0] == "observation"
                        && row[1] == case
                        && row[2].parse::<usize>().unwrap() == group
                })
                .map(|row| {
                    SurvivalObservation::new(
                        (row[3] != "none").then(|| scalar(row[3])),
                        scalar(row[4]),
                        if row[5] == "1" {
                            scioxide_stats::SurvivalOutcome::Event
                        } else {
                            scioxide_stats::SurvivalOutcome::Censored
                        },
                    )
                    .unwrap()
                })
                .collect()
        });
        // Reversals and rotations must preserve every count, estimate and bound.
        for permutation in 0..3 {
            let groups = groups.clone().map(|mut rows| {
                match permutation {
                    1 => rows.reverse(),
                    2 => rows.rotate_left(1),
                    _ => (),
                }
                rows
            });
            for (group, observations) in groups.iter().enumerate() {
                let km = kaplan_meier_with_options(
                    observations,
                    ConfidenceProbability::default(),
                    scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
                )?;
                let na = nelson_aalen_with_options(
                    observations,
                    ConfidenceProbability::default(),
                    scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
                )?;
                let expected: Vec<_> = data
                    .iter()
                    .filter(|row| {
                        row[0] == "curve"
                            && row[1] == case
                            && row[2].parse::<usize>().unwrap() == group
                    })
                    .collect();
                assert_eq!(km.len(), expected.len());
                assert_eq!(na.len(), expected.len());
                for ((km, na), row) in km.iter().zip(&na).zip(expected) {
                    let expected = |index: usize| row[index].parse::<f64>().unwrap();
                    assert_eq!(km.time, scalar(row[3]));
                    assert_eq!(na.time, km.time);
                    assert_eq!(km.at_risk as f64, expected(4));
                    assert_eq!(km.events as f64, expected(5));
                    assert_eq!(km.censored as f64, expected(6));
                    assert_eq!(
                        (na.at_risk, na.events, na.censored),
                        (km.at_risk, km.events, km.censored)
                    );
                    close(km.survival.into_f64(), expected(7), tolerance);
                    if row[8] == "none" {
                        assert!(km.interval.is_none());
                    } else {
                        let interval = km.interval.unwrap();
                        close(interval.lower().into_f64(), expected(8), tolerance);
                        close(interval.upper().into_f64(), expected(9), tolerance);
                    }
                    close(na.cumulative_hazard.into_f64(), expected(10), tolerance);
                    close(na.variance.into_f64(), expected(11), tolerance);
                    close(na.interval.lower().into_f64(), expected(12), tolerance);
                    close(na.interval.upper().into_f64(), expected(13), tolerance);
                }
            }
            if let Some(row) = data
                .iter()
                .find(|row| row[0] == "logrank" && row[1] == case)
            {
                let result = logrank_test_with_options(
                    &groups[0],
                    &groups[1],
                    SurvivalOptions::new().with_max_bytes(4096),
                )?;
                let expected = |index: usize| row[index].parse::<f64>().unwrap();
                close(result.statistic().into_f64(), expected(2), tolerance);
                close(result.p_value(), expected(3), 2e-13);
                assert_eq!(result.observed()[0] as f64, expected(4));
                assert_eq!(result.observed()[1] as f64, expected(5));
                close(result.expected()[0].into_f64(), expected(6), tolerance);
                close(result.expected()[1].into_f64(), expected(7), tolerance);
                close(result.variance().into_f64(), expected(8), tolerance);
                let swapped = logrank_test_with_options(
                    &groups[1],
                    &groups[0],
                    SurvivalOptions::new().with_max_bytes(4096),
                )?;
                close(
                    swapped.statistic().into_f64(),
                    result.statistic().into_f64(),
                    tolerance,
                );
                assert_eq!(
                    swapped.observed(),
                    [result.observed()[1], result.observed()[0]]
                );
            } else {
                assert!(
                    logrank_test_with_options(
                        &groups[0],
                        &groups[1],
                        SurvivalOptions::new().with_max_bytes(4096)
                    )
                    .is_err()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn independent_censored_and_delayed_entry_references_both_precisions() -> Result<(), SciOxideError>
{
    references::<f64>()?;
    references::<f32>()
}

#[test]
fn mixed_cohorts_match_direct_risk_set_enumeration() -> Result<(), SciOxideError> {
    for size in [1, 2, 17, 128, 1024] {
        for tied in [false, true] {
            let observations: Vec<_> = (0..size)
                .map(|i| {
                    let t = if tied { i % 13 } else { i } as f64;
                    let entry = (i % 3 != 0).then_some(t - (i % 7 + 1) as f64);
                    SurvivalObservation::new(
                        entry,
                        t,
                        if i % 5 != 1 {
                            scioxide_stats::SurvivalOutcome::Event
                        } else {
                            scioxide_stats::SurvivalOutcome::Censored
                        },
                    )
                })
                .collect::<Result<_, _>>()?;
            let km = kaplan_meier_with_options(
                &observations,
                ConfidenceProbability::default(),
                scioxide_stats::SurvivalOptions::new().with_max_bytes(usize::MAX),
            )?;
            let na = nelson_aalen_with_options(
                &observations,
                ConfidenceProbability::default(),
                scioxide_stats::SurvivalOptions::new().with_max_bytes(usize::MAX),
            )?;
            let mut survival = 1.0;
            let mut hazard = 0.0;
            let mut variance = 0.0;
            for (km, na) in km.iter().zip(&na) {
                let t = km.time;
                let n = observations
                    .iter()
                    .filter(|row| row.entry().is_none_or(|entry| entry < t) && row.time() >= t)
                    .count();
                let d = observations
                    .iter()
                    .filter(|row| row.time() == t && row.event())
                    .count();
                let c = observations
                    .iter()
                    .filter(|row| row.time() == t && !row.event())
                    .count();
                assert_eq!((km.at_risk, km.events, km.censored), (n, d, c));
                survival *= 1.0 - d as f64 / n as f64;
                hazard += d as f64 / n as f64;
                variance += d as f64 / (n as f64 * n as f64);
                close(km.survival, survival, 2e-13);
                close(na.cumulative_hazard, hazard, 2e-13);
                close(na.variance, variance, 2e-13);
            }
        }
    }
    Ok(())
}

#[test]
fn signed_zero_subnormal_and_extreme_times_need_no_time_arithmetic() -> Result<(), SciOxideError> {
    let tiny = f64::from_bits(1);
    let observations = [
        SurvivalObservation::new(None, -f64::MAX, scioxide_stats::SurvivalOutcome::Censored)?,
        SurvivalObservation::new(None, -0.0, scioxide_stats::SurvivalOutcome::Event)?,
        SurvivalObservation::new(None, 0.0, scioxide_stats::SurvivalOutcome::Censored)?,
        SurvivalObservation::new(Some(-0.0), tiny, scioxide_stats::SurvivalOutcome::Event)?,
        SurvivalObservation::new(
            Some(-f64::MAX),
            tiny,
            scioxide_stats::SurvivalOutcome::Censored,
        )?,
        SurvivalObservation::new(Some(tiny), f64::MAX, scioxide_stats::SurvivalOutcome::Event)?,
    ];
    let km = kaplan_meier_with_options(
        &observations,
        ConfidenceProbability::default(),
        scioxide_stats::SurvivalOptions::new().with_max_bytes(4096),
    )?;
    assert_eq!(km.len(), 3);
    assert_eq!((km[0].at_risk, km[0].events, km[0].censored), (3, 1, 1));
    assert_eq!((km[1].at_risk, km[1].events, km[1].censored), (2, 1, 1));
    assert_eq!((km[2].at_risk, km[2].events, km[2].censored), (1, 1, 0));
    Ok(())
}
