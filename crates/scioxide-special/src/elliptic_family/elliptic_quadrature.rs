//! Shared fixed-storage, scaled Gauss refinement for defining elliptic integrals.
use crate::support::ELLIPTIC_ERRORS as ERRORS;
use crate::support::EllipticEstimate as Estimate;
use crate::support::Work;
use crate::support::Z;
use scioxide_core::Scaled;
use scioxide_core::SciOxideError;

#[derive(Clone, Copy)]
struct Panel {
    lo: f64,
    hi: f64,
    value: Z,
    error: Scaled<f64>,
}

// A coarse 32-point panel is checked against two children. Nodes are shared
// with the probability kernels; all integrand work uses the same allowance.
fn panel(
    f: &mut impl FnMut(f64, &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError>,
    lo: f64,
    hi: f64,
    work: &mut Work<'_, '_, '_>,
) -> Result<Panel, SciOxideError> {
    let mid = lo + (hi - lo) * 0.5;
    let half = (hi - lo) * 0.5;
    let mut sum = Z::ZERO;
    let mut error = Scaled::ZERO;
    for &(node, weight) in &crate::support::GL32 {
        work.try_step()?;
        let e = f(mid + half * node, work)?;
        let term = e.value.mul(Z::from_value(scioxide_core::Complex::new(
            weight * half,
            0.0,
        ))?)?;
        sum = sum.add(term)?;
        error = error.add(
            term.magnitude()?
                .mul(Scaled::from_value(e.relative + 16.0 * f64::EPSILON)?)?,
        )?;
    }
    Ok(Panel {
        lo,
        hi,
        value: sum,
        error,
    })
}
pub(crate) fn integrate(
    mut f: impl FnMut(f64, &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError>,
    right: f64,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let first = panel(&mut f, 0.0, right, work)?;
    let global_scale = first.value.magnitude()?;
    let mut stack = [first; 64];
    let mut count = 1;
    let mut total = Z::ZERO;
    let mut error = Scaled::ZERO;
    while count > 0 {
        count -= 1;
        let coarse = stack[count];
        let mid = coarse.lo + (coarse.hi - coarse.lo) * 0.5;
        if mid == coarse.lo || mid == coarse.hi {
            return Err(ERRORS.unstable());
        }
        let l = panel(&mut f, coarse.lo, mid, work)?;
        let r = panel(&mut f, mid, coarse.hi, work)?;
        let fine = l.value.add(r.value)?;
        let delta = fine.sub(coarse.value)?.magnitude()?;
        let local_scale = fine.magnitude()?;
        let allocated_scale =
            global_scale.mul(Scaled::from_value((coarse.hi - coarse.lo) / right)?)?;
        let scale = if local_scale.abs_le(allocated_scale) {
            allocated_scale
        } else {
            local_scale
        };
        let threshold = scale.mul(Scaled::from_value(
            (work.target * 0.05).max(32.0 * f64::EPSILON),
        )?)?;
        if delta.abs_le(threshold) {
            total = total.add(fine)?;
            error = error.add(delta)?.add(l.error)?.add(r.error)?;
        } else {
            if count + 2 > stack.len() {
                return Err(ERRORS.unstable());
            }
            stack[count] = r;
            stack[count + 1] = l;
            count += 2;
        }
    }
    let magnitude = total.magnitude()?;
    if magnitude.is_zero() {
        return Err(ERRORS.unstable());
    }
    Ok(Estimate::new(total, error.div(magnitude)?.round()?))
}
