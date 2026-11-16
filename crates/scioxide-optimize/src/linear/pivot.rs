//! Pure bounded Harris selection; all distances use the equilibrated frame.
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, checked_float_quotient};

pub(super) struct Step<S> {
    pub row: Option<usize>,
    pub length: S,
    pub upper: bool,
}

/// First pass relaxes feasibility, second pass selects the strongest pivot
/// whose exact step fits that window. Bland uses the exact minimum ratio.
#[allow(clippy::too_many_arguments)]
pub(super) fn primal<S: RealScalar + NumericScalar<Real = S>>(
    values: &[S],
    direction: &[S],
    indices: &[usize],
    bounds: &[Option<S>],
    sign: S,
    entering_bound: Option<S>,
    feasibility: S,
    pivot: S,
    bland: bool,
) -> Result<Option<Step<S>>, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let candidate = |i: usize| {
        let d = direction[i] * sign;
        if d > pivot {
            Some((values[i].max(zero), d, false))
        } else if d < -pivot && indices[i] < bounds.len() {
            bounds[indices[i]].map(|upper| ((upper - values[i]).max(zero), -d, true))
        } else {
            None
        }
    };
    let mut relaxed = entering_bound;
    for i in 0..values.len() {
        if let Some((distance, magnitude, _)) = candidate(i) {
            let step = checked_float_quotient(
                if bland {
                    distance
                } else {
                    scioxide_core::checked_sum(&[distance, feasibility])?
                },
                magnitude,
            )?;
            relaxed = Some(relaxed.map_or(step, |best| best.min(step)));
        }
    }
    let Some(relaxed) = relaxed else {
        return Ok(None);
    };
    let mut chosen: Option<(usize, S, S, bool)> = None;
    for i in 0..values.len() {
        if let Some((distance, magnitude, upper)) = candidate(i) {
            let step = checked_float_quotient(distance, magnitude)?;
            if step <= relaxed
                && chosen.is_none_or(|(row, best, strength, _)| {
                    if bland {
                        step < best || (step == best && indices[i] < indices[row])
                    } else {
                        magnitude > strength || (magnitude == strength && indices[i] < indices[row])
                    }
                })
            {
                chosen = Some((i, step, magnitude, upper));
            }
        }
    }
    Ok(Some(match chosen {
        Some((row, length, _, upper)) => Step {
            row: Some(row),
            length,
            upper,
        },
        None => Step {
            row: None,
            length: relaxed,
            upper: false,
        },
    }))
}

pub(crate) fn progress<S: RealScalar>(
    stalls: &mut usize,
    bland: &mut bool,
    step: S,
    gain: S,
    feasibility: S,
    cost: S,
) {
    if step <= feasibility || gain.abs() <= cost {
        *stalls += 1;
    } else {
        *stalls = 0;
    }
    *bland |= *stalls >= 8;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn harris_avoids_a_tiny_degenerate_pivot_and_bland_uses_exact_ratios() {
        let harris = primal(
            &[0.0_f64, 1.0],
            &[1e-10, 1.0],
            &[0, 1],
            &[None; 2],
            1.0,
            None,
            1e-8,
            1e-14,
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(harris.row, Some(1));
        assert!((harris.length - 1.0).abs() < 1e-14);
        let bland = primal(
            &[0.0_f64, 1.0],
            &[1e-10, 1.0],
            &[0, 1],
            &[None; 2],
            1.0,
            None,
            1e-8,
            1e-14,
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(bland.row, Some(0));
        assert!(bland.length.abs() < 1e-14);
    }

    #[test]
    fn upper_leaving_bounds_entering_flips_and_unbounded_directions() {
        let step = primal(
            &[1.0_f64],
            &[-2.0],
            &[0],
            &[Some(5.0)],
            1.0,
            None,
            1e-8,
            1e-14,
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(step.row, Some(0));
        assert!(step.upper);
        assert!((step.length - 2.0).abs() < 1e-14);
        let flip = primal(
            &[1.0_f64],
            &[-2.0],
            &[0],
            &[Some(5.0)],
            1.0,
            Some(0.5),
            1e-8,
            1e-14,
            false,
        )
        .unwrap()
        .unwrap();
        assert!(flip.row.is_none());
        assert!((flip.length - 0.5).abs() < 1e-14);
        assert!(
            primal(
                &[1.0_f64],
                &[-2.0],
                &[0],
                &[None],
                1.0,
                None,
                1e-8,
                1e-14,
                false
            )
            .unwrap()
            .is_none()
        );
    }
}
