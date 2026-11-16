//! Rolling median by insertion into a sorted window.
use crate::empirical::quantile_position;
use crate::ranking::finite_total_cmp;
use crate::{QuantileMethod, QuantileProbability, RollingBoundary, RollingWindow};
use scioxide_core::{ExecutionContext, NumericalIssue, RealScalar, SciOxideError};

/// Rolling median of one batch. `read` is called for the initial window and
/// then for the entering and leaving position of each step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<S: RealScalar, R>(
    n: usize,
    window: RollingWindow,
    method: QuantileMethod,
    context: ExecutionContext<'_>,
    scratch: &mut [S],
    output: &mut [R],
    read: impl Fn(usize) -> Result<Option<S>, SciOxideError>,
    project: impl Fn(S) -> R,
) -> Result<(), SciOxideError> {
    let width = window.width()?;
    let read_position = |center, offset| match window.index(center, offset, n) {
        Some(i) => read(i),
        None => Ok((window.boundary == RollingBoundary::ZeroPad).then_some(S::ZERO)),
    };
    let initial = if window.boundary == RollingBoundary::Truncate {
        window.right.saturating_add(1).min(n)
    } else {
        width
    };
    let mut len = 0;
    for offset in 0..initial {
        if offset % 256 == 0 {
            context.check()?;
        }
        let value = if window.boundary == RollingBoundary::Truncate {
            read(offset)?
        } else {
            read_position(0, offset)?
        };
        if let Some(value) = value {
            scratch[len] = value;
            len += 1;
        }
    }
    scratch[..len].sort_unstable_by(finite_total_cmp);
    let probability = QuantileProbability::new(0.5)?;
    let mut position = quantile_position(len, probability, method)?;
    for (center, result) in output.iter_mut().enumerate() {
        context.check()?;
        if center > 0 {
            let removed = read_position(center - 1, 0)?;
            let added = read_position(center, width - 1)?;
            let previous_len = len;
            // Numerical equality alone would conflate the two signs of zero.
            let unchanged = matches!((removed, added), (Some(a), Some(b))
                if a == b && a.is_sign_negative() == b.is_sign_negative());
            if !unchanged {
                if let Some(value) = removed {
                    let mut index = scratch[..len]
                        .iter()
                        .position(|entry| {
                            *entry == value && entry.is_sign_negative() == value.is_sign_negative()
                        })
                        .ok_or_else(|| {
                            SciOxideError::numerical_instability(
                                NumericalIssue::Breakdown,
                                "rolling median lost an outgoing observation",
                            )
                        })?;
                    if let Some(value) = added {
                        // Move only the entries between the outgoing and incoming
                        // positions. Linear search/shifts avoid two bulk moves and
                        // binary-search branches in the common short-window kernel.
                        while index > 0 && finite_total_cmp(&value, &scratch[index - 1]).is_lt() {
                            scratch[index] = scratch[index - 1];
                            index -= 1;
                        }
                        while index + 1 < len
                            && finite_total_cmp(&scratch[index + 1], &value).is_lt()
                        {
                            scratch[index] = scratch[index + 1];
                            index += 1;
                        }
                        scratch[index] = value;
                    } else {
                        scratch.copy_within(index + 1..len, index);
                        len -= 1;
                    }
                } else if let Some(value) = added {
                    let index = scratch[..len]
                        .partition_point(|entry| finite_total_cmp(entry, &value).is_lt());
                    scratch.copy_within(index..len, index + 1);
                    scratch[index] = value;
                    len += 1;
                }
            }
            if len != previous_len {
                position = quantile_position(len, probability, method)?;
            }
        }
        *result = project(position.evaluate(|index| scratch[index])?);
    }
    Ok(())
}
