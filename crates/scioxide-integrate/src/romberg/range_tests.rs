use super::*;

#[test]
fn prepared_dyadic_steps_preserve_the_original_rounding() {
    fn check<S: RealScalar>() {
        let factors = Factors::<S>::new().unwrap();
        let lower = S::MIN_POSITIVE / S::EPSILON;
        for span in [
            lower,
            lower + S::MIN_POSITIVE,
            S::MAX,
            S::ONE,
            S::ONE + S::EPSILON,
        ] {
            for level in 1..MAX_LEVELS {
                let denominator = S::from_f64_ieee((1_usize << (level - 1)) as f64);
                let expected = (span * S::HALF) / denominator;
                let actual = span * factors.steps[level];
                assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits());
                if level <= NATIVE_LEVELS && actual >= S::MIN_POSITIVE {
                    let count = 1 << (level - 1);
                    for i in 0..count {
                        let original = actual * S::from_f64_ieee((2 * i + 1) as f64);
                        let canonical = span * factors.canonical_midpoints[count - 1 + i];
                        assert_eq!(
                            original.into_f64().to_bits(),
                            canonical.into_f64().to_bits()
                        );
                    }
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn overflowing_native_totals_replay_original_parts_before_scaling() {
    fn check<S: RealScalar>() -> Result<(), SciOxideError> {
        let sum = PromotingCompensatedSum::from_compensated(
            scioxide_core::CompensatedSum::from_parts(S::MAX, S::MAX)?,
        );
        assert!(sum.native().is_some());
        assert!(native_total(&sum).is_none());
        assert!(native_product(S::HALF, S::INFINITY).is_none());
        assert!(native_product(S::ZERO, S::INFINITY).is_none());
        let geometry = IntervalGeometry::prepare(Interval::new(S::ZERO, S::ONE)?)?.unwrap();
        let [value, floor] = Component::new().initialize(geometry, &sum, &sum)?;
        assert_eq!(value, S::MAX);
        assert!(floor > S::ZERO && floor.is_finite());
        Ok(())
    }
    check::<f32>().unwrap();
    check::<f64>().unwrap();
}

#[test]
fn admitted_diagnostic_changes_match_scaled_arithmetic_through_subnormal_differences() {
    fn check<S: RealScalar>() -> Result<(), SciOxideError> {
        let guard = S::from_f64_ieee(crate::kernel::EXTRAPOLATION_ERROR_GUARD as f64);
        let bound = S::MAX.sqrt() / S::from_f64_ieee(128.0);
        for (value, previous) in [
            (S::ZERO, S::NEG_ZERO),
            (S::MIN_SUBNORMAL, S::ZERO),
            (S::ZERO, -S::MIN_SUBNORMAL),
            (S::MIN_POSITIVE, S::MIN_POSITIVE - S::MIN_SUBNORMAL),
            (S::MIN_POSITIVE + S::MIN_SUBNORMAL, S::MIN_POSITIVE),
            (S::ONE, S::ONE + S::EPSILON),
            (bound, -bound),
        ] {
            let mut component = Component::new();
            component.previous = previous;
            component.previous_lower = previous;
            let output = component.record_diagonals::<true>(
                MIN_CONVERGENCE_LEVEL - 1,
                value,
                value,
                S::ZERO,
            )?;
            let expected = extended_difference(value, previous, guard)?;
            assert_eq!(
                output[1].into_f64().to_bits(),
                expected.into_f64().to_bits()
            );
        }
        Ok(())
    }
    check::<f32>().unwrap();
    check::<f64>().unwrap();
}

#[test]
fn rounded_paired_functionals_keep_the_native_diagnostic_range_bound() {
    fn check<S: RealScalar>() {
        let factors = Factors::<S>::new().unwrap();
        for level in 1..=NATIVE_LEVELS {
            let masses = factors.paired[level][1..=level]
                .iter()
                .fold([S::ZERO; 2], |totals, weights| {
                    core::array::from_fn(|i| totals[i] + weights[i].abs())
                });
            for mass in masses {
                assert!(mass < S::from_f64_ieee(4.0));
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn prepared_witness_windows_gather_the_grid_around_the_witness() {
    fn check<S: RealScalar>() {
        let witness = Witness::<S>::new().unwrap();
        let fraction = witness.fraction.into_f64();
        // Retain every level's captured midpoints by their exact binary64 abscissae.
        let mut retained = [0.0; RETAINED];
        for level in 1..MAX_LEVELS {
            let Capture { lower, upper, slot } = witness.capture(level);
            let panels = (1_u64 << level) as f64;
            for midpoint in lower..upper {
                retained[slot + midpoint - lower] = (2 * midpoint + 1) as f64 / panels;
            }
            if level < WITNESS_LEVEL {
                continue;
            }
            // The window holds the eight consecutive grid abscissae whose middle
            // panel contains the witness.
            let first = (fraction * panels).floor() - 3.0;
            let window = witness.windows[level - WITNESS_LEVEL];
            for (offset, &slot) in window.iter().enumerate() {
                assert_eq!(
                    retained[usize::from(slot)],
                    (first + offset as f64) / panels
                );
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
