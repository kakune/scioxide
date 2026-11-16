//! Rolling means and variances by compensated updates with a rounding-error bound.
use crate::arithmetic::{native, normalized_deviation, shift, shifted_center, sum_mean};
use crate::error_helpers::{invalid, number};
use crate::{ArithmeticStatistic, ArithmeticValue, RollingBoundary, RollingWindow};
use scioxide_core::backend::two_sum;
use scioxide_core::{
    CompensatedSum, ExecutionContext, NumericScalar, RealScalar, SciOxideError, compensated_reduce,
};

// The private rolling state has already admitted sources within +/-64. Even
// differences, squared moduli, usize-count sums and their compensation stay
// normal and finite. Core owns the error-free transform and block reductions;
// this state additionally bounds historical rounding lost by a downdate.
#[derive(Clone, Copy)]
struct WindowSum {
    parts: [f64; 2],
    error: f64,
}

impl WindowSum {
    const ZERO: Self = Self {
        parts: [0.0; 2],
        error: 0.0,
    };

    fn reduced(sum: CompensatedSum<f64>, mass: f64, factor: f64) -> Self {
        Self {
            parts: sum.parts(),
            error: factor * mass,
        }
    }

    #[inline]
    fn add(&mut self, value: f64) {
        let (sum, residual) = two_sum(self.parts[0], value);
        let correction = self.parts[1] + residual;
        // TwoSum retains the leading addition exactly. Only the addition to
        // the correction rounds: its error is <= u/(1-u)*|correction|.
        // EPSILON=2u leaves headroom for the product, and inflation rounds the
        // accumulated nonnegative bound outward (also when it remains zero).
        self.error = (self.error + f64::EPSILON * correction.abs()) * (1.0 + 2.0 * f64::EPSILON);
        self.parts = [sum, correction];
    }

    #[inline]
    fn total(self) -> f64 {
        self.parts[0] + self.parts[1]
    }

    #[inline]
    fn compensated(self) -> Result<CompensatedSum<f64>, SciOxideError> {
        let [hi, lo] = self.parts;
        CompensatedSum::from_parts(hi, lo)
    }

    #[inline]
    fn mean(self, count: f64) -> f64 {
        sum_mean(self.parts, count)
    }
}

// Short windows seed directly with the same bounded update kernel. Larger
// windows amortize core's 256-row block staging and independent reduction lanes.
fn window_sums<const N: usize>(
    values: impl Iterator<Item = [f64; N]>,
    count: usize,
    factor: f64,
) -> Result<[WindowSum; N], SciOxideError> {
    if count <= 32 {
        let mut sums = [WindowSum::ZERO; N];
        for row in values {
            for (sum, value) in sums.iter_mut().zip(row) {
                sum.add(value);
            }
        }
        return Ok(sums);
    }
    let mut maximum = [0.0_f64; N];
    let sums = compensated_reduce(values.inspect(|row| {
        for (maximum, value) in maximum.iter_mut().zip(row) {
            *maximum = maximum.max(value.abs());
        }
    }))?;
    Ok(core::array::from_fn(|i| {
        WindowSum::reduced(sums[i], count as f64 * maximum[i], factor)
    }))
}

#[derive(Clone, Copy)]
struct Moments {
    count: usize,
    center: [[f64; 2]; 2],
    // Raw sums for a mean, sums of deviations from center for dispersion.
    first: [WindowSum; 2],
    squares: WindowSum,
    energy: f64,
}

fn components<T: NumericScalar>(value: T) -> [f64; 2] {
    [
        value.real_component().into_f64(),
        value.imaginary_component().into_f64(),
    ]
}

impl Moments {
    fn empty() -> Self {
        Self {
            count: 0,
            center: [[0.0; 2]; 2],
            first: [WindowSum::ZERO; 2],
            squares: WindowSum::ZERO,
            energy: 0.0,
        }
    }

    fn rebuild<T: NumericScalar>(
        values: &[T],
        second: bool,
    ) -> Result<Option<Self>, SciOxideError> {
        let mut state = Self::empty();
        let Some(&first) = values.first() else {
            return Ok(Some(state));
        };
        state.count = values.len();
        let count = values.len() as f64;
        let admitted = values.iter().fold(true, |admitted, &value| {
            let x = components(value);
            admitted & native(x[0], 64) & (T::IS_REAL || native(x[1], 64))
        });
        // Each four-lane block has at most n additions along a lane, followed
        // by eight part merges per block. Compensated reduction loses at most
        // 4*gamma_r^2 times absolute addend mass, before final rounding; double
        // that bound to cover its evaluation and the mass upper bound n*max.
        let ru = (count + 8.0 * values.len().div_ceil(256) as f64) * (0.5 * f64::EPSILON);
        if !admitted || ru >= 0.125 {
            return Ok(None);
        }
        let gamma = ru / (1.0 - ru);
        let factor = 8.0 * gamma * gamma;
        if !second {
            state.first = if T::IS_REAL {
                let [real] = window_sums(
                    values
                        .iter()
                        .map(|&value| [value.real_component().into_f64()]),
                    values.len(),
                    factor,
                )?;
                [real, WindowSum::ZERO]
            } else {
                window_sums(
                    values.iter().map(|&value| components(value)),
                    values.len(),
                    factor,
                )?
            };
            return Ok(Some(state));
        }
        let origin = components(first);
        state.center = if T::IS_REAL {
            let [high, low] = window_sums(
                values
                    .iter()
                    .map(|&value| shift(value.real_component().into_f64(), origin[0])),
                values.len(),
                factor,
            )?;
            [
                shifted_center(origin[0], [high.compensated()?, low.compensated()?], count)?,
                [0.0; 2],
            ]
        } else {
            let [rh, rl, ih, il] = window_sums(
                values.iter().map(|&value| {
                    let x = components(value);
                    let [rh, rl] = shift(x[0], origin[0]);
                    let [ih, il] = shift(x[1], origin[1]);
                    [rh, rl, ih, il]
                }),
                values.len(),
                factor,
            )?;
            [
                shifted_center(origin[0], [rh.compensated()?, rl.compensated()?], count)?,
                shifted_center(origin[1], [ih.compensated()?, il.compensated()?], count)?,
            ]
        };
        let deltas = values.iter().map(|&value| state.difference(value));
        let (sums, squares) = if T::IS_REAL {
            let [sum, squares] =
                window_sums(deltas.map(|[d, _]| [d, d * d]), values.len(), factor)?;
            ([sum, WindowSum::ZERO], squares)
        } else {
            let [real, imaginary, squares] = window_sums(
                deltas.map(|[r, i]| [r, i, r * r + i * i]),
                values.len(),
                factor,
            )?;
            ([real, imaginary], squares)
        };
        state.first = sums;
        state.squares = squares;
        state.energy = state.squares.total() - state.correction::<T>();
        Ok((state.energy >= 0.0).then_some(state))
    }

    #[inline]
    fn difference<T: NumericScalar>(&self, value: T) -> [f64; 2] {
        let x = components(value);
        [
            normalized_deviation(x[0], self.center[0]),
            if T::IS_REAL {
                0.0
            } else {
                normalized_deviation(x[1], self.center[1])
            },
        ]
    }

    #[inline]
    fn change<T: NumericScalar>(&mut self, value: T, remove: bool, second: bool) -> bool {
        // Outgoing observations were admitted on entry to this fixed-center
        // state. An incoming miss discards the state and replays the window.
        let x = components(value);
        if !remove && !(native(x[0], 64) & (T::IS_REAL || native(x[1], 64))) {
            return false;
        }
        if remove && self.count == 1 {
            *self = Self::empty();
            return true;
        }
        let delta = if second { self.difference(value) } else { x };
        self.count = if remove {
            self.count - 1
        } else {
            self.count + 1
        };
        let sign = if remove { -1.0 } else { 1.0 };
        self.first[0].add(sign * delta[0]);
        if !T::IS_REAL {
            self.first[1].add(sign * delta[1]);
        }
        if second {
            self.squares
                .add(sign * (delta[0] * delta[0] + delta[1] * delta[1]));
        }
        true
    }

    fn correction<T: NumericScalar>(&self) -> f64 {
        let re = self.first[0].total();
        let im = if T::IS_REAL {
            0.0
        } else {
            self.first[1].total()
        };
        (re * re + im * im) / self.count as f64
    }

    // Assess only a complete new window: a large intermediate contraction says
    // nothing about the rounding lost by its compensated state. Rebuilt states
    // are assessed too, before their projection is published.
    fn reliable<T: NumericScalar>(&mut self, second: bool) -> bool {
        const BUDGET: f64 = 64.0 * f64::EPSILON;
        if !second {
            // A complex mean is bounded normwise, so a component that sums to
            // (nearly) zero beside a nonzero one does not force rebuilds.
            let (error, magnitude) = self
                .first
                .iter()
                .take(if T::IS_REAL { 1 } else { 2 })
                .fold((0.0, 0.0), |(error, magnitude), sum| {
                    (error + sum.error, magnitude + sum.total().abs())
                });
            return error <= BUDGET * magnitude;
        }
        if self.count == 0 {
            return true;
        }
        let q = self.squares.total();
        let correction = self.correction::<T>();
        self.energy = q - correction;
        let first_error = self
            .first
            .iter()
            .take(if T::IS_REAL { 1 } else { 2 })
            .map(|sum| (2.0 * sum.total().abs() + sum.error) * sum.error)
            .sum::<f64>()
            / self.count as f64;
        // Propagate stored summation error through |S|^2/n. Eight epsilons of
        // Q+C cover normalized contrasts, squares, the correction and final
        // subtraction (2*sqrt(Q*C) <= Q+C); double for bound evaluation.
        let error = 2.0
            * (self.squares.error
                + first_error
                + 8.0 * f64::EPSILON * (q.abs() + self.squares.error + correction + first_error));
        q >= 0.0 && self.energy >= 0.0 && error <= BUDGET * self.energy
    }

    fn project<T: NumericScalar>(
        &self,
        statistic: ArithmeticStatistic,
    ) -> Result<ArithmeticValue<T>, SciOxideError> {
        let ddof = match statistic {
            ArithmeticStatistic::Variance(d) | ArithmeticStatistic::StandardDeviation(d) => d,
            _ => 0,
        };
        if self.count <= ddof {
            return Err(invalid(
                scioxide_core::argument!("statistic"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient retained degrees of freedom",
            ));
        }
        Ok(match statistic {
            ArithmeticStatistic::Mean => ArithmeticValue::Location(T::from_components(
                number(self.first[0].mean(self.count as f64))?,
                if T::IS_REAL {
                    <T::Real as scioxide_core::AlgebraicScalar>::ZERO
                } else {
                    number(self.first[1].mean(self.count as f64))?
                },
            )),
            ArithmeticStatistic::Variance(_) | ArithmeticStatistic::StandardDeviation(_) => {
                let variance = self.energy / (self.count - ddof) as f64;
                let value = if matches!(statistic, ArithmeticStatistic::StandardDeviation(_)) {
                    RealScalar::sqrt(variance)
                } else {
                    variance
                };
                ArithmeticValue::Dispersion(number(value)?)
            }
            ArithmeticStatistic::Sum => {
                return Err(invalid(
                    scioxide_core::argument!("statistic"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "rolling sums require a batch kernel",
                ));
            }
        })
    }
}

/// Window rebuilds on the current thread, by cause (unit tests only).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rebuilds {
    initial: usize,
    periodic: usize,
    unreliable: usize,
    fallback: usize,
}

#[cfg(test)]
std::thread_local! {
    static REBUILDS: core::cell::Cell<Rebuilds> = const {
        core::cell::Cell::new(Rebuilds {
            initial: 0,
            periodic: 0,
            unreliable: 0,
            fallback: 0,
        })
    };
}

#[cfg(test)]
impl Rebuilds {
    /// Run `operation` and return its result with the rebuilds it performed.
    fn record<T>(operation: impl FnOnce() -> T) -> (T, Self) {
        REBUILDS.set(Self::default());
        let result = operation();
        (result, REBUILDS.replace(Self::default()))
    }

    fn update(update: impl FnOnce(&mut Self)) {
        let mut rebuilds = REBUILDS.get();
        update(&mut rebuilds);
        REBUILDS.set(rebuilds);
    }
}

// Rebuild-counting hooks: their bodies compile only in unit tests, so
// production rolling carries no counter state or argument.
#[inline(always)]
fn count_initial() {
    #[cfg(test)]
    Rebuilds::update(|rebuilds| rebuilds.initial += 1);
}

#[inline(always)]
fn count_periodic() {
    #[cfg(test)]
    Rebuilds::update(|rebuilds| rebuilds.periodic += 1);
}

#[inline(always)]
fn count_unreliable() {
    #[cfg(test)]
    Rebuilds::update(|rebuilds| rebuilds.unreliable += 1);
}

#[inline(always)]
fn count_fallback() {
    #[cfg(test)]
    Rebuilds::update(|rebuilds| rebuilds.fallback += 1);
}

/// Rolling mean, variance or standard deviation of one batch.
///
/// `scratch` holds the gathered window for periodic and error-triggered
/// rebuilds. A rebuilt state passes the same error test as an updated one;
/// otherwise the window value comes from `batch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<T: NumericScalar, R>(
    n: usize,
    window: RollingWindow,
    statistic: ArithmeticStatistic,
    context: ExecutionContext<'_>,
    scratch: &mut [T],
    output: &mut [R],
    read: impl Fn(usize) -> Result<Option<T>, SciOxideError>,
    mut batch: impl FnMut(&[T]) -> Result<ArithmeticValue<T>, SciOxideError>,
    project: impl Fn(ArithmeticValue<T>) -> R,
) -> Result<(), SciOxideError> {
    let width = window.width()?;
    let second = !matches!(statistic, ArithmeticStatistic::Mean);
    let read_position = |center, offset| match window.index(center, offset, n) {
        Some(i) => read(i),
        None => Ok((window.boundary == RollingBoundary::ZeroPad).then_some(T::ZERO)),
    };
    let period = width.min(n).max(1);
    let mut age = period;
    let mut state: Option<Moments> = None;
    for (center, slot) in output.iter_mut().enumerate() {
        // At most 256 entering/leaving observations between polls. Rebuild
        // gathering also polls in blocks; the caller checks before publication.
        if center % 128 == 0 {
            context.check()?;
        }
        let periodic = age >= period;
        if age < period
            && let Some(current) = state.as_mut()
        {
            if let Some(x) = read_position(center - 1, 0)? {
                current.change(x, true, second);
            }
            let valid = match read_position(center, width - 1)? {
                Some(x) => current.change(x, false, second),
                None => true,
            };
            if !valid || !current.reliable::<T>(second) {
                state = None;
            }
            age += 1;
        } else {
            state = None;
        }
        if state.is_none() {
            let mut len = 0;
            let (start, end) = if window.boundary == RollingBoundary::Truncate {
                (
                    center.saturating_sub(window.left),
                    center.saturating_add(window.right).saturating_add(1).min(n),
                )
            } else {
                (0, width)
            };
            for position in start..end {
                if (position - start) % 256 == 0 {
                    context.check()?;
                }
                let value = if window.boundary == RollingBoundary::Truncate {
                    read(position)?
                } else {
                    read_position(center, position)?
                };
                if let Some(value) = value {
                    scratch[len] = value;
                    len += 1;
                }
            }
            state = Moments::rebuild(&scratch[..len], second)?
                .and_then(|mut rebuilt| rebuilt.reliable::<T>(second).then_some(rebuilt));
            age = 0;
            if state.is_none() {
                count_fallback();
                *slot = project(batch(&scratch[..len])?);
                continue;
            }
            if center == 0 {
                count_initial();
            } else if periodic {
                count_periodic();
            } else {
                count_unreliable();
            }
        }
        if let Some(current) = state.as_ref() {
            *slot = project(current.project(statistic)?);
        }
    }
    context.check()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use scioxide_core::{Complex, RangeAccumulator};

    fn rolled<T: NumericScalar>(
        data: &[T],
        window: RollingWindow,
        statistic: ArithmeticStatistic,
    ) -> (Vec<ArithmeticValue<T>>, Rebuilds) {
        let mut scratch = vec![T::ZERO; window.width().unwrap()];
        let mut output = vec![ArithmeticValue::Location(T::ZERO); data.len()];
        let (result, rebuilds) = Rebuilds::record(|| {
            run(
                data.len(),
                window,
                statistic,
                ExecutionContext::sequential(),
                &mut scratch,
                &mut output,
                |i| Ok(Some(data[i])),
                |values| crate::numeric_statistic(values, statistic),
                |x| x,
            )
        });
        result.unwrap();
        (output, rebuilds)
    }

    fn location<T: NumericScalar>(value: ArithmeticValue<T>) -> Option<T> {
        match value {
            ArithmeticValue::Location(value) => Some(value),
            ArithmeticValue::Dispersion(_) => None,
        }
    }

    fn counts<T: NumericScalar>(
        data: &[T],
        half: usize,
        statistic: ArithmeticStatistic,
    ) -> [usize; 4] {
        let window = RollingWindow::new(half, half).with_boundary(RollingBoundary::Repeat);
        let (_, rebuilds) = rolled(data, window, statistic);
        [
            rebuilds.initial,
            rebuilds.periodic,
            rebuilds.unreliable,
            rebuilds.fallback,
        ]
    }

    #[test]
    fn ordinary_rebuild_counts() {
        for n in [4096, 65536] {
            for half in [8, 128] {
                for (label, offset) in [("ordinary", 0.0), ("zero", -0.5), ("offset", 1e12)] {
                    let real: Vec<_> = (0..n)
                        .map(|i| offset + ((i * 7919) % 65537) as f64 / 65536.0)
                        .collect();
                    let complex: Vec<_> = real
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            Complex::new(x, -offset + ((i * 3571) % 65537) as f64 / 65536.0 - 0.5)
                        })
                        .collect();
                    for statistic in [
                        ArithmeticStatistic::Mean,
                        ArithmeticStatistic::Variance(1),
                        ArithmeticStatistic::StandardDeviation(1),
                    ] {
                        let real_counts = counts(&real, half, statistic);
                        let complex_counts = counts(&complex, half, statistic);
                        std::println!(
                            "rebuilds,{label},{n},{half},{statistic:?},real={real_counts:?},complex={complex_counts:?}"
                        );
                        let periodic = (n - 1) / (2 * half + 2);
                        for counts in [real_counts, complex_counts] {
                            assert_eq!(counts[0], 1);
                            assert!(counts[1].abs_diff(periodic) <= 1);
                            // Near-zero projections and endpoint contraction
                            // can require one genuinely ill-conditioned rebuild.
                            // An exactly cancelling dyadic window has no relative
                            // certificate and publishes its batch mean instead.
                            assert!(counts[2] <= 1, "{label}: {counts:?}");
                            assert!(counts[3] <= counts[2], "{label}: {counts:?}");
                        }
                    }
                }
            }
        }
    }

    fn assert_sum_bound(sum: WindowSum, exact: &RangeAccumulator<f64>) {
        let mut difference = exact.clone();
        for part in sum.parts {
            difference.add(-part).unwrap();
        }
        let error = difference.finish().unwrap().abs();
        assert!(
            error <= sum.error,
            "actual error {error:e} exceeds bound {:e}",
            sum.error
        );
    }

    #[test]
    fn compensated_window_bounds_enclose_exact_reductions_and_downdates() {
        let data: Vec<_> = (0..513)
            .map(|i| {
                let magnitude = 2.0_f64.powi((i * 37 % 121) - 60);
                if i % 3 == 0 {
                    -magnitude
                } else {
                    magnitude * 1.25
                }
            })
            .collect();
        // Check both sides of short-window admission and the core block edge.
        for n in [0, 1, 31, 32, 33, 255, 256, 257] {
            let state = Moments::rebuild(&data[..n], false).unwrap().unwrap();
            let mut exact = RangeAccumulator::<f64>::new();
            for &x in &data[..n] {
                exact.add(x).unwrap();
            }
            assert_sum_bound(state.first[0], &exact);
        }
        let mut state = Moments::rebuild(&data, false).unwrap().unwrap();
        let mut exact = RangeAccumulator::<f64>::new();
        for &x in &data {
            exact.add(x).unwrap();
        }
        assert_sum_bound(state.first[0], &exact);
        for (i, &old) in data.iter().cycle().take(2048).enumerate() {
            let new = data[(i * 17 + 5) % data.len()];
            state.first[0].add(-old);
            exact.add(-old).unwrap();
            state.first[0].add(new);
            exact.add(new).unwrap();
            assert_sum_bound(state.first[0], &exact);
        }
        let mut sum = WindowSum::ZERO;
        let mut exact = RangeAccumulator::<f64>::new();
        for &x in data.iter().cycle().take(2048) {
            sum.add(x);
            exact.add(x).unwrap();
            assert_sum_bound(sum, &exact);
        }
    }

    #[test]
    fn outlier_departure_rebuilds_and_zero_windows_remain_exact() {
        let mut data = vec![1e-18; 65];
        // Leave just after a periodic rebuild; the small tail must recover
        // after both the large value and its compensation-scale neighbor leave.
        data[10] = 1e18;
        data[11] = 3.0;
        for statistic in [
            ArithmeticStatistic::Mean,
            ArithmeticStatistic::Variance(1),
            ArithmeticStatistic::StandardDeviation(1),
        ] {
            let counts = counts(&data, 8, statistic);
            assert!(counts[2] > 0, "{statistic:?}: {counts:?}");
            assert_eq!(counts[3], 0);
        }
        for value in [0.0, 2.0] {
            for second in [false, true] {
                let mut state = Moments::rebuild(&[value; 17], second).unwrap().unwrap();
                for _ in 0..128 {
                    assert!(state.change(value, true, second));
                    assert!(state.change(value, false, second));
                    assert!(state.reliable::<f64>(second));
                    if second {
                        assert_eq!(state.energy, 0.0);
                    } else {
                        assert_eq!(state.first[0].mean(17.0), value);
                    }
                }
            }
        }
    }

    #[test]
    fn short_window_means_reproduce_their_slice_means() {
        let large = 2.0_f64.powi(64);
        for tail in [2.0_f64.powi(-60), 2.0_f64.powi(-42)] {
            let data = [large, 3072.0, tail, -(large + 4096.0), 1024.0];
            // Sequentially seeded short windows and block-seeded long ones.
            for half in [2, 3, 15, 16, 40] {
                let window = RollingWindow::new(half, half);
                let (output, rebuilds) = rolled(&data, window, ArithmeticStatistic::Mean);
                for (center, value) in output.into_iter().enumerate() {
                    let actual = location(value).unwrap();
                    let window = &data[center.saturating_sub(half)..(center + half + 1).min(5)];
                    let expected = location(
                        crate::numeric_statistic(window, ArithmeticStatistic::Mean).unwrap(),
                    )
                    .unwrap();
                    assert!(
                        (actual - expected).abs() <= 64.0 * f64::EPSILON * expected.abs(),
                        "half {half}, center {center}: {actual:e} != {expected:e}"
                    );
                }
                // The cancelling window has no relative certificate.
                assert!(rebuilds.fallback > 0, "half {half}: {rebuilds:?}");
            }
        }
        // The full window is the exact rational tail mean.
        let data = [large, 3072.0, 2.0_f64.powi(-60), -(large + 4096.0), 1024.0];
        let (output, _) = rolled(&data, RollingWindow::new(2, 2), ArithmeticStatistic::Mean);
        assert_eq!(location(output[2]), Some(2.0_f64.powi(-60) / 5.0));
    }

    #[test]
    fn complex_means_with_cancelling_components_keep_downdating() {
        let n = 4096;
        let real: Vec<_> = (0..n)
            .map(|i| 1.0 + ((i * 7919) % 65537) as f64 / 65536.0)
            .collect();
        // Every full window's imaginary parts cancel exactly, but their
        // sequential sum carries a nonzero rounding bound.
        let pattern = [0.1, 1e-17, -0.1, -1e-17];
        let cancelling: Vec<_> = real
            .iter()
            .enumerate()
            .map(|(i, &x)| Complex::new(x, pattern[i % 4]))
            .collect();
        let real_valued: Vec<_> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let window = RollingWindow::new(4, 3).with_boundary(RollingBoundary::Repeat);
        let (_, reference) = rolled(&real, window, ArithmeticStatistic::Mean);
        let (_, zero) = rolled(&real_valued, window, ArithmeticStatistic::Mean);
        assert_eq!(zero, reference);
        let (output, rebuilds) = rolled(&cancelling, window, ArithmeticStatistic::Mean);
        assert_eq!(rebuilds.initial, 1);
        assert!(rebuilds.unreliable <= 1, "{rebuilds:?}");
        assert_eq!(rebuilds.fallback, 0);
        for (center, value) in output.into_iter().enumerate().skip(4).take(n - 8) {
            let actual = location(value).unwrap();
            let expected = location(
                crate::numeric_statistic(
                    &cancelling[center - 4..=center + 3],
                    ArithmeticStatistic::Mean,
                )
                .unwrap(),
            )
            .unwrap();
            let error = (actual.re - expected.re).abs() + (actual.im - expected.im).abs();
            assert!(error <= 64.0 * f64::EPSILON * (expected.re.abs() + expected.im.abs()));
        }
    }

    #[test]
    fn extended_range_fallback_does_not_poison_later_native_windows() {
        let mut data = vec![2.0; 65];
        data[9] = 1e300;
        for statistic in [
            ArithmeticStatistic::Mean,
            ArithmeticStatistic::StandardDeviation(1),
        ] {
            let counts = counts(&data, 8, statistic);
            assert!(counts[3] > 0);
            assert!(counts[1] > 0);
        }
    }

    #[test]
    fn execution_polls_bound_advances_and_check_completion() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        struct StopAfter {
            completed: AtomicUsize,
            stop_at: usize,
        }
        impl scioxide_core::Cancellation for StopAfter {
            fn is_cancelled(&self) -> bool {
                self.completed.load(Ordering::Relaxed) >= self.stop_at
            }
        }
        for (n, expected) in [(513, 256), (129, 129)] {
            let stop = StopAfter {
                completed: AtomicUsize::new(0),
                stop_at: 129,
            };
            let data = vec![2.0_f64; n];
            // A wide window avoids periodic rebuild polls during the advances.
            let window = RollingWindow::new(n, n).with_boundary(RollingBoundary::Repeat);
            let mut scratch = vec![0.0; 2 * n + 1];
            let mut output = vec![ArithmeticValue::Location(0.0); n];
            let error = run(
                n,
                window,
                ArithmeticStatistic::Mean,
                ExecutionContext::sequential().with_cancellation(&stop),
                &mut scratch,
                &mut output,
                |i| Ok(Some(data[i])),
                |values| crate::numeric_statistic(values, ArithmeticStatistic::Mean),
                |value| {
                    stop.completed.fetch_add(1, Ordering::Relaxed);
                    value
                },
            )
            .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
            ));
            assert_eq!(stop.completed.load(Ordering::Relaxed), expected);
        }
    }
}
