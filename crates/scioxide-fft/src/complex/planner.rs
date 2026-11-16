//! Algorithm selection, costed radix schedules and checked storage bounds.
use super::*;
const SUPPORTED_RADICES: [usize; 5] = [4, 2, 3, 5, 7];
const RADER_PRIME_LIMIT: usize = 1_048_576;
// The memo visits only 2/3/5/7-smooth divisors and doubles when full.
pub(super) fn planner_bytes(n: usize) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("fft_preparation_bytes");
    let (exponents, _) = crate::smooth_len::smooth_factors(n);
    let divisors = exponents.into_iter().try_fold(1, |divisors, exponent| {
        checked_mul(divisors, exponent + 1, q)
    })?;
    checked_add(
        checked_mul(
            (4 * usize::BITS as usize).max(checked_mul(2, divisors, q)?),
            size_of::<RadixChoice>(),
            q,
        )?,
        usize::BITS as usize * size_of::<usize>(),
        q,
    )
}

// The same selection drives construction and its pre-allocation storage bound.
#[derive(Clone, Copy)]
pub(super) enum NativeAlgorithm {
    Smooth,
    Rader,
    Bluestein(usize),
}
impl NativeAlgorithm {
    pub(super) fn select(n: usize, length: Argument) -> Result<Self, SciOxideError> {
        if is_smooth(n) {
            return Ok(Self::Smooth);
        }
        if rader_is_suitable(n, length)? {
            return Ok(Self::Rader);
        }
        Ok(Self::Bluestein(bluestein_len(n, length)?))
    }
    pub(super) fn prepare<S: scioxide_core::RealScalar>(
        self,
        n: usize,
        dispatch: KernelDispatch,
        length: Argument,
    ) -> Result<FftAlgorithm<S>, SciOxideError> {
        match self {
            Self::Smooth => SmoothPlan::new(n, dispatch, length).map(FftAlgorithm::Smooth),
            Self::Rader => RaderPlan::new(n, dispatch, length).map(FftAlgorithm::Rader),
            Self::Bluestein(m) => {
                BluesteinPlan::new(n, m, dispatch, length).map(FftAlgorithm::Bluestein)
            }
        }
    }
    // Smooth twiddles telescope to m-1. Bound stage/root metadata by prime
    // factors; memo capacity is at most max(initial capacity, twice the number
    // of smooth divisors). Include the temporary radix sequence and convolution
    // kernel even where lifetimes do not overlap. This is a tight linear bound,
    // with only O(log(m)) stage and O(divisors(m)) planner overestimation.
    pub(super) fn preparation_bound<S: scioxide_core::RealScalar>(
        self,
        n: usize,
    ) -> Result<usize, SciOxideError> {
        if n.is_power_of_two() {
            return Ok(ComplexCore::<S>::power_two_requirements(n)?.1);
        }
        let q = scioxide_core::quantity!("fft_preparation_bytes");
        let m = match self {
            Self::Smooth => n,
            Self::Rader => n - 1,
            Self::Bluestein(m) => m,
        };
        let (exponents, _) = crate::smooth_len::smooth_factors(m);
        let stages: usize = exponents.iter().sum();
        let roots: usize = crate::smooth_len::SMOOTH_PRIMES
            .into_iter()
            .zip(exponents)
            .map(|(prime, count)| count * crate::kernel::root_count(prime).unwrap_or(0))
            .sum();
        let c = size_of::<Complex<S>>();
        let u = size_of::<usize>();
        let mut bytes = size_of::<ComplexCore<S>>();
        for part in [
            checked_mul(m, u, q)?,
            checked_mul(checked_add(m - 1, roots, q)?, c, q)?,
            checked_mul(stages, size_of::<crate::kernel::Stage<S>>(), q)?,
            planner_bytes(m)?,
        ] {
            bytes = checked_add(bytes, part, q)?;
        }
        let extra = match self {
            Self::Smooth => 0,
            Self::Rader => checked_mul(m, 2 * (u + c), q)?,
            Self::Bluestein(_) => checked_mul(checked_add(n, checked_mul(2, m, q)?, q)?, c, q)?,
        };
        checked_add(bytes, extra, q)
    }
}

#[derive(Debug, Clone, Copy)]
struct RadixChoice {
    remaining: usize,
    cost: u128,
    radix: usize,
}

pub(super) fn best_radix_plan(n: usize, length: Argument) -> Result<Vec<usize>, SciOxideError> {
    if !is_smooth(n) {
        return Err(SciOxideError::invalid_input(
            length,
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "private convolution plan length must be 2/3/5/7-smooth",
        ));
    }
    // Dynamic programming over the remaining cofactor: `memo` keeps the
    // cheapest (radix, cost) for each divisor of n still to be factored, and
    // the radix sequence is then read back from n downward. The initial
    // capacity covers typical lengths; `best_radix_choice` grows it fallibly
    // for lengths with many 2/3/5/7-smooth divisors.
    if n.is_power_of_two() {
        let mut radices = try_with_capacity(
            (n.trailing_zeros() as usize).div_ceil(2),
            scioxide_core::quantity!("fft_radix_sequence"),
        )?;
        let mut rest = n;
        while rest >= 4 {
            radices.push(4);
            rest /= 4;
        }
        if rest == 2 {
            radices.push(2);
        }
        return Ok(radices);
    }
    let mut memo = try_with_capacity(
        4 * usize::BITS as usize,
        scioxide_core::quantity!("fft_planner_memo"),
    )?;
    memo.push(RadixChoice {
        remaining: 1,
        cost: 0,
        radix: 1,
    });
    let _ = best_radix_choice(n, n, &mut memo, length)?;
    let mut radices = try_with_capacity(
        usize::BITS as usize,
        scioxide_core::quantity!("fft_radix_sequence"),
    )?;
    let mut remaining = n;
    while remaining > 1 {
        let choice = memo
            .iter()
            .find(|choice| choice.remaining == remaining)
            .copied()
            .ok_or_else(|| {
                SciOxideError::invalid_input(
                    length,
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "FFT planner did not retain a radix choice",
                )
            })?;
        radices.push(choice.radix);
        remaining /= choice.radix;
    }
    Ok(radices)
}

fn best_radix_choice(
    remaining: usize,
    n: usize,
    memo: &mut Vec<RadixChoice>,
    length: Argument,
) -> Result<RadixChoice, SciOxideError> {
    if let Some(choice) = memo
        .iter()
        .find(|choice| choice.remaining == remaining)
        .copied()
    {
        return Ok(choice);
    }
    // Radices are chosen in stage order, so the stage chosen for `remaining`
    // runs with width n / remaining. Ties keep the earlier candidate (strict
    // `<`), and SUPPORTED_RADICES lists 4 first, so equal-cost plans prefer
    // radix 4 deterministically.
    let width = n / remaining;
    let mut best: Option<RadixChoice> = None;
    for radix in SUPPORTED_RADICES {
        if !remaining.is_multiple_of(radix) {
            continue;
        }
        let tail = best_radix_choice(remaining / radix, n, memo, length)?;
        let cost = tail
            .cost
            .checked_add(stage_cost(n, radix, width, length)?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("fft_planner_cost"))
            })?;
        let candidate = RadixChoice {
            remaining,
            cost,
            radix,
        };
        if best.is_none_or(|current| candidate.cost < current.cost) {
            best = Some(candidate);
        }
    }
    let choice = best.ok_or_else(|| {
        SciOxideError::invalid_input(
            length,
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "FFT planner found no supported radix",
        )
    })?;
    if memo.len() == memo.capacity() {
        memo.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("fft_planner_memo"),
                memo.len() + 1,
            )
        })?;
    }
    memo.push(choice);
    Ok(choice)
}

// Relative model, not a flop count: `arithmetic_weight` is a per-element cost
// of one pass of the radix butterfly (radix 4 covers two factors of 2 for
// less than two radix-2 passes; symmetric 5/7 share conjugate output pairs), the
// twiddle term models table traffic, and the per-block term models loop and
// codelet-dispatch overhead that penalizes many tiny blocks.
pub(super) fn stage_cost(
    n: usize,
    radix: usize,
    width: usize,
    length: Argument,
) -> Result<u128, SciOxideError> {
    let arithmetic_weight = match radix {
        2 => 5_u128,
        3 => 9,
        4 => 8,
        5 => 14,
        7 => 20,
        _ => {
            return Err(SciOxideError::invalid_input(
                length,
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "unsupported FFT radix",
            ));
        }
    };
    let arithmetic = (n as u128).checked_mul(arithmetic_weight).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("fft_planner_cost"))
    })?;
    let twiddles = ((radix - 1) as u128)
        .checked_mul(width as u128)
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("fft_planner_cost"))
        })?;
    let blocks = (n / (radix * width)) as u128;
    arithmetic
        .checked_add(twiddles)
        .and_then(|cost| cost.checked_add(blocks * 8))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("fft_planner_cost")))
}

pub(super) fn is_smooth(n: usize) -> bool {
    n != 0 && crate::smooth_len::smooth_factors(n).1 == 1
}

pub(super) fn rader_candidate(n: usize) -> bool {
    !is_smooth(n) && n <= RADER_PRIME_LIMIT && is_prime(n) && is_smooth(n - 1)
}

pub(super) fn bluestein_len(n: usize, length: Argument) -> Result<usize, SciOxideError> {
    let minimum = checked_mul(
        n,
        2,
        scioxide_core::quantity!("bluestein_convolution_length"),
    )? - 1;
    crate::smooth_len::fast_len(minimum, true, length)
}

pub(super) fn rader_is_suitable(n: usize, length: Argument) -> Result<bool, SciOxideError> {
    // RADER_PRIME_LIMIT bounds the trial-division primality test and the
    // primitive-root search at planning time. A non-smooth n - 1 would need
    // a nested Rader/Bluestein convolution, which is avoided by design.
    if !rader_candidate(n) {
        return Ok(false);
    }
    // Both alternatives are two FFTs of their convolution length plus one
    // pointwise pass; compare them with the same stage-cost model.
    let bluestein_len = bluestein_len(n, length)?;
    let rader_cost = transform_cost(n - 1, length)?
        .checked_mul(2)
        .and_then(|cost| cost.checked_add((n - 1) as u128 * 8))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("rader_planner_cost"))
        })?;
    let bluestein_cost = transform_cost(bluestein_len, length)?
        .checked_mul(2)
        .and_then(|cost| cost.checked_add(bluestein_len as u128 * 8))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("bluestein_planner_cost"))
        })?;
    Ok(rader_cost < bluestein_cost)
}

pub(super) fn transform_cost(n: usize, length: Argument) -> Result<u128, SciOxideError> {
    let radices = best_radix_plan(n, length)?;
    let mut width = 1_usize;
    let mut cost = 0_u128;
    for radix in radices {
        cost = cost
            .checked_add(stage_cost(n, radix, width, length)?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("fft_planner_cost"))
            })?;
        width = checked_mul(width, radix, scioxide_core::quantity!("fft_stage_width"))?;
    }
    Ok(cost)
}

pub(super) fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut divisor = 3_usize;
    while divisor <= n / divisor {
        if n.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}
