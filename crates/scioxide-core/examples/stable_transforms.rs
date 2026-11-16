//! Checked logarithmic sums, normalization, and exact phase helpers.
use scioxide_core::{
    Complex64, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, SciOxideError, exprel,
    log_softmax, log_softmax_axis, log_softmax_axis_into, logsumexp, logsumexp_axis,
    logsumexp_axis_into, sin_pi, softmax, softmax_axis, softmax_axis_into, weighted_logsumexp,
    weighted_logsumexp_axis,
};
fn main() -> Result<(), SciOxideError> {
    // Exact binary64 inputs retain the residual of rounded -ln(2).
    let near_unit = [-core::f64::consts::LN_2; 2];
    let tiny_log = logsumexp(&near_unit)?;
    assert_eq!(tiny_log.to_bits(), 0x3c7a_bc9e_3b39_803f);
    assert_eq!(
        weighted_logsumexp(&near_unit, &[1.0; 2])?.log_magnitude(),
        Some(tiny_log)
    );
    println!("near-unit sum: log={tiny_log:e}");
    let logits = [1000.0_f64, 1001.0, 999.0];
    let probabilities = softmax(&logits)?;
    println!(
        "logsumexp={}, probabilities={probabilities:?}",
        logsumexp(&logits)?
    );
    let sum = weighted_logsumexp(&[10000.0, 10000.0, -10000.0], &[1.0, -1.0, 1.0])?;
    println!(
        "cancelled sum: sign={}, log magnitude={:?}",
        sum.sign(),
        sum.log_magnitude()
    );
    assert_eq!(sin_pi(1e30_f64)?, 0.0);
    assert_eq!(exprel(0.0_f64)?, 1.0);
    assert_eq!(softmax(&[0.0_f64, f64::NEG_INFINITY])?, [1.0, 0.0]);
    assert_eq!(
        log_softmax(&[0.0_f64, f64::NEG_INFINITY])?,
        [0.0, f64::NEG_INFINITY]
    );
    assert_eq!(logsumexp::<f64>(&[])?, f64::NEG_INFINITY);
    // Two lanes of length 2 along axis 1, viewed in row-major order.
    let input = [0.0, f64::NEG_INFINITY, 0.0, 0.0];
    let view = NdView::row_major(&input, &[2, 2])?;
    let values = softmax_axis(view, 1)?;
    let (policy, cx) = (ExecutionPolicy::Sequential, ExecutionContext::sequential());
    let mut output = [0.0; 4];
    let mut scratch = output;
    let target = NdViewMut::row_major(&mut output, &[2, 2])?;
    softmax_axis_into(view, 1, policy, target, &mut scratch, cx)?;
    assert_eq!(values, output);
    let target = NdViewMut::row_major(&mut output, &[2, 2])?;
    log_softmax_axis_into(view, 1, policy, target, &mut scratch, cx)?;
    assert_eq!(output[1], f64::NEG_INFINITY);
    assert_eq!(log_softmax_axis(view, 1)?, output);
    let sums = logsumexp_axis(view, 1)?;
    let mut reduced = [0.0; 2];
    let target = NdViewMut::row_major(&mut reduced, &[2])?;
    logsumexp_axis_into(view, 1, policy, target, &mut scratch[..2], cx)?;
    assert_eq!(sums, reduced);
    let weights = NdView::row_major(&[1.0; 4], &[2, 2])?;
    let weighted = weighted_logsumexp_axis(view, weights, 1)?;
    assert_eq!(
        weighted
            .iter()
            .map(|v| v.log_magnitude())
            .collect::<Vec<_>>(),
        sums.iter().copied().map(Some).collect::<Vec<_>>()
    );
    let input = input.map(|re| Complex64::new(re, 0.25));
    let complex = logsumexp_axis(NdView::row_major(&input, &[2, 2])?, 1)?;
    assert_eq!(complex.iter().map(|v| v.re).collect::<Vec<_>>(), sums);
    Ok(())
}
