//! Neville's per-component error estimate through its single error form.

use scioxide_core::SciOxideError;
use scioxide_interp::{ExtrapolationPolicy, Interpolant, NevilleInterpolator};

#[test]
fn correction_starts_at_the_nearest_node_in_coordinate_units() -> Result<(), SciOxideError> {
    macro_rules! check {
        ($scalar:ty) => {{
            for (nodes, query, expected) in [
                ([9.0 as $scalar, 11.05], 10.0, 1.0),
                ([-11.05, -9.0], -10.0, 1.0),
                ([9.0, 11.0], 10.0, 1.0),
            ] {
                let model = NevilleInterpolator::new(nodes.to_vec(), nodes.to_vec())?;
                let mut scratch = model.workspace()?;
                let (mut value, mut error) = (0.0, 0.0);
                model.evaluate_with_error_into(
                    [query],
                    ExtrapolationPolicy::Reject,
                    &mut value,
                    &mut error,
                    &mut scratch,
                )?;
                assert!((value - query).abs() < 32.0 * <$scalar>::EPSILON);
                assert!((error - expected).abs() < 32.0 * <$scalar>::EPSILON);
            }
        }};
    }
    check!(f32);
    check!(f64);
    Ok(())
}

#[test]
fn neville_preserves_value_and_error_contract() -> Result<(), SciOxideError> {
    let nodes = [0.0_f64, 1.0, 2.0, 3.0];
    let values = nodes.map(|node| node * node);
    let neville = NevilleInterpolator::new(nodes.to_vec(), values.to_vec())?;
    let mut workspace = neville.workspace()?;
    let mut with_error = |point: f64| {
        let (mut value, mut error) = (0.0, 0.0);
        neville
            .evaluate_with_error_into(
                [point],
                ExtrapolationPolicy::Reject,
                &mut value,
                &mut error,
                &mut workspace,
            )
            .map(|()| (value, error))
    };

    nodes.iter().zip(values).try_for_each(|(&node, expected)| {
        with_error(node).map(|(value, _)| assert!((value - expected).abs() <= 1.0e-12))
    })?;

    let (value, error) = with_error(1.5)?;
    assert!((value - 2.25).abs() <= 1.0e-13);
    assert!(error <= 1.0e-12);

    Ok(())
}
