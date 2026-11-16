//! Quadrature of sampled data: trapezoid, Simpson and Romberg plans.
use crate::kernel::checked_intermediate;
use alloc::vec::Vec;

use scioxide_core::backend::{
    ProductFactor, RealProductSum, checked_mul, try_filled, try_with_capacity, two_sum,
};
use scioxide_core::{
    AffineValue, AffineValueMut, RangeAccumulator, RealScalar, Scaled, SciOxideError,
    check_value_shape, validate_value, write_value,
};

/// Checks that coordinates are finite and strictly increasing or strictly decreasing.
fn validate_coordinates<S: RealScalar>(coordinates: &[S]) -> Result<(), SciOxideError> {
    validate_value(coordinates)?;
    if coordinates.len() > 1 {
        let increasing = coordinates[1] > coordinates[0];
        if coordinates
            .windows(2)
            .any(|pair| pair[0] == pair[1] || (pair[1] > pair[0]) != increasing)
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("coordinates"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "sampled coordinates must be strictly monotone",
            ));
        }
    }
    Ok(())
}

/// Scratch storage for the sampled-data plans [`SampledTrapezoidPlan`],
/// [`SampledSimpsonPlan`] and [`SampledRombergPlan`].
///
/// Obtain it from the plan's `workspace` or `cumulative_workspace` method,
/// for example [`SampledTrapezoidPlan::workspace`].
///
/// It holds the results until all of them are computed, so its size is
/// `outputs · components` values.
#[derive(Debug, Clone)]
pub struct SampledWorkspace<S: RealScalar> {
    candidate: Vec<S>,
    sums: Vec<RealProductSum<S>>,
    components: usize,
}
impl<S: RealScalar> SampledWorkspace<S> {
    /// Creates a workspace for up to `outputs` results with `components` real
    /// components each (1 for `f64`, 2 for a complex value).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `components · outputs` overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub(crate) fn new(components: usize, outputs: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            candidate: try_filled(
                checked_mul(
                    components,
                    outputs,
                    scioxide_core::quantity!("candidate_values"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("candidate_values"),
            )?,
            sums: try_filled(
                components,
                RealProductSum::new(),
                scioxide_core::quantity!("component_sums"),
            )?,
            components,
        })
    }

    /// Returns the number of real components per value.
    pub const fn components(&self) -> usize {
        self.components
    }
}

// Empty `varied` storage denotes repeating, numerically equal prepared weights.
// A uniform-looking floating grid can have unequal spacings and remains irregular.
#[derive(Debug, Clone)]
enum Weights<S: RealScalar> {
    Trapezoid {
        weight: S,
        varied: Vec<S>,
        factor: ProductFactor<S>,
        factors: Vec<ProductFactor<S>>,
    },
    Simpson {
        halves: [[S; 3]; 2],
        varied: Vec<[S; 3]>,
    },
}
#[derive(Debug, Clone)]
struct Panels<S: RealScalar> {
    samples: usize,
    weights: Weights<S>,
}

fn stencil_first(interval: usize, samples: usize) -> usize {
    (interval / 2 * 2).min(samples - 3)
}
fn half_index(interval: usize, samples: usize) -> usize {
    interval - stencil_first(interval, samples)
}

/// Coefficients of a linear form over the stencil coordinates `[a, b, c]`.
type Form = [i32; 3];
/// The spacings `b - a`, `c - b` and `c - a`.
const H1: Form = [-1, 1, 0];
const H2: Form = [0, -1, 1];
const H: Form = [-1, 0, 1];

/// Returns `scale · Π forms(x)` rounded once, in extended exponent range.
/// Products of up to three forms expand into exact products of coordinates, so
/// no spacing is rounded and none overflows; three-form coefficients are ±1.
fn exact_form<S: RealScalar>(
    sum: &mut RangeAccumulator<S>,
    x: [S; 3],
    scale: i32,
    forms: &[Form],
) -> Result<Scaled<S>, SciOxideError> {
    sum.clear();
    let integer = |k: i32| S::try_from_i64_exact(i64::from(k));
    let terms = |form: &Form| x.into_iter().zip(*form);
    match forms {
        [f] => {
            for (xi, fi) in terms(f).filter(|&(_, fi)| fi != 0) {
                sum.add_product(integer(scale * fi)?, xi)?;
            }
        }
        [f, g] => {
            for (xi, fi) in terms(f) {
                for (xj, gj) in terms(g) {
                    let k = scale * fi * gj;
                    if k != 0 {
                        sum.add_triple_product(integer(k)?, xi, xj)?;
                    }
                }
            }
        }
        [f, g, h] => {
            for (xi, fi) in terms(f) {
                for (xj, gj) in terms(g) {
                    for (xl, hl) in terms(h) {
                        let k = scale * fi * gj * hl;
                        for _ in 0..k.unsigned_abs() {
                            if k > 0 {
                                sum.add_triple_product(xi, xj, xl)?;
                            } else {
                                sum.subtract_triple_product(xi, xj, xl)?;
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    sum.finish_scaled()
}

/// A scaled product of linear forms.
type Product = (i32, &'static [Form]);

/// Numerator and denominator of every half-stencil weight: `h1 (2 h1 + 3 h2) / (6 h)`,
/// `h1 (h1 + 3 h2) / (6 h2)` and `-h1^3 / (6 h2 h)` for the first interval, and their
/// mirror images for the second.
const HALF_WEIGHTS: [[(Product, Product); 3]; 2] = [
    [
        ((1, &[H1, [-2, -1, 3]]), (6, &[H])),
        ((1, &[H1, [-1, -2, 3]]), (6, &[H2])),
        ((-1, &[H1, H1, H1]), (6, &[H2, H])),
    ],
    [
        ((-1, &[H2, H2, H2]), (6, &[H1, H])),
        ((1, &[H2, [-3, 2, 1]]), (6, &[H1])),
        ((1, &[H2, [-3, 1, 2]]), (6, &[H])),
    ],
];

/// Weights of half `half` of the quadratic interpolant through the stencil `x`.
/// Numerator and denominator are exact in the coordinates and rounded once each,
/// and their quotient once more, so every weight is within two units in the last
/// place and depends only on the exact spacings: mirrored halves of equal spacings
/// have identical weights.
fn simpson_half<S: RealScalar>(
    sum: &mut RangeAccumulator<S>,
    x: [S; 3],
    half: usize,
) -> Result<[S; 3], SciOxideError> {
    let mut weights = [S::ZERO; 3];
    for (weight, ((scale, numerator), (divisor, denominator))) in
        weights.iter_mut().zip(HALF_WEIGHTS[half])
    {
        let numerator = exact_form(sum, x, scale, numerator)?;
        *weight = numerator
            .div(exact_form(sum, x, divisor, denominator)?)?
            .round()?;
    }
    Ok(weights)
}

impl<S: RealScalar> Panels<S> {
    /// Trapezoid weights for already validated coordinates.
    fn trapezoid(coordinates: &[S]) -> Result<Self, SciOxideError> {
        let mut varied = Vec::new();
        let mut weight = S::ZERO;
        for (index, pair) in coordinates.windows(2).enumerate() {
            // Half the difference can be representable even when the difference overflows.
            let next = scioxide_core::Scaled::from_value(pair[1])?
                .sub(scioxide_core::Scaled::from_value(pair[0])?)?
                .shifted(-1)?
                .round()?;
            if index == 0 {
                weight = next;
            }
            if varied.is_empty() && next != weight {
                varied =
                    try_with_capacity(coordinates.len() - 1, scioxide_core::quantity!("weights"))?;
                varied.resize(index, weight);
            }
            if !varied.is_empty() {
                varied.push(next);
            }
        }
        let factor = ProductFactor::new(weight)?.normalized();
        let mut factors =
            try_with_capacity(varied.len(), scioxide_core::quantity!("weight_factors"))?;
        for &weight in &varied {
            factors.push(ProductFactor::new(weight)?.normalized());
        }
        Ok(Self {
            samples: coordinates.len(),
            weights: Weights::Trapezoid {
                weight,
                varied,
                factor,
                factors,
            },
        })
    }

    fn apply<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
        &self,
        samples: &[V],
        output: &mut [O],
        workspace: &mut SampledWorkspace<S>,
        cumulative: bool,
    ) -> Result<(), SciOxideError> {
        crate::kernel::shape(self.samples, samples.len())?;
        let count = if cumulative { self.samples } else { 1 };
        crate::kernel::shape(count, output.len())?;
        let components = workspace.components;
        let size = checked_mul(
            count,
            components,
            scioxide_core::quantity!("candidate_values"),
        )?;
        if workspace.candidate.len() < size {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient sampled output workspace",
            ));
        }
        if let Some(dimension) = V::FIXED_DIMENSION.filter(|_| !samples.is_empty()) {
            crate::kernel::shape(components, dimension)?;
        } else {
            for value in samples {
                check_value_shape(value, components)?;
            }
        }
        if let Some(dimension) = O::FIXED_DIMENSION.filter(|_| !output.is_empty()) {
            crate::kernel::shape(components, dimension)?;
        } else {
            for value in output.iter() {
                check_value_shape(value, components)?;
            }
        }
        // Every component is checked as it enters the stencil cache, including zero-weight
        // and singleton samples. Publication occurs only after all components succeed.
        if cumulative {
            self.accumulate::<V, true>(samples, workspace)?;
        } else {
            self.accumulate::<V, false>(samples, workspace)?;
        }
        // Shapes were preflighted, and every candidate is zero or a successful
        // finite finish. Stable affine traversal needs no second validation.
        let mut candidate = workspace.candidate[..size].iter();
        for value in output {
            value.for_each_component_mut(|component| {
                if let Some(value) = candidate.next() {
                    *component = *value;
                }
            });
        }
        Ok(())
    }

    fn accumulate<V: AffineValue<Scale = S>, const CUMULATIVE: bool>(
        &self,
        samples: &[V],
        workspace: &mut SampledWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        if samples.is_empty() {
            if !CUMULATIVE {
                workspace.candidate[..workspace.components].fill(S::ZERO);
            }
            return Ok(());
        }
        // Preflight established this equality. Retain static affine dimensions
        // here so scalar and complex loops have constant strides and bounds.
        let d = V::FIXED_DIMENSION.unwrap_or(workspace.components);
        let count = if CUMULATIVE { samples.len() } else { 1 };
        let size = checked_mul(count, d, scioxide_core::quantity!("candidate_values"))?;
        for (component, sum) in workspace.sums[..d].iter_mut().enumerate() {
            sum.clear();
            let mut value = |index: usize| samples[index].component(component);
            if CUMULATIVE && !samples.is_empty() {
                workspace.candidate[component] = S::ZERO;
            }
            let mut emit = PrefixOutput::<S, CUMULATIVE> {
                candidate: &mut workspace.candidate[..size],
                components: d,
                component,
            };
            match &self.weights {
                Weights::Trapezoid {
                    weight,
                    varied,
                    factor,
                    factors,
                } => {
                    if CUMULATIVE {
                        if factors.is_empty() {
                            trapezoid_prefixes(
                                core::iter::repeat_n(*factor, samples.len().saturating_sub(1)),
                                samples.len(),
                                sum,
                                &mut value,
                                &mut emit,
                            )?;
                        } else {
                            trapezoid_prefixes(
                                factors[..samples.len() - 1].iter().copied(),
                                samples.len(),
                                sum,
                                &mut value,
                                &mut emit,
                            )?;
                        }
                    } else if varied.is_empty() {
                        trapezoid_component(
                            core::iter::repeat_n(*weight, samples.len().saturating_sub(1)),
                            samples.len(),
                            sum,
                            &mut value,
                        )?;
                    } else {
                        trapezoid_component(
                            varied.iter().copied(),
                            samples.len(),
                            sum,
                            &mut value,
                        )?;
                    }
                }
                Weights::Simpson { halves, varied } => {
                    if varied.is_empty() {
                        simpson_component(
                            (0..samples.len() - 1).map(|i| halves[half_index(i, samples.len())]),
                            samples.len(),
                            sum,
                            &mut value,
                            &mut emit,
                        )?;
                    } else {
                        simpson_component(
                            varied.iter().copied(),
                            samples.len(),
                            sum,
                            &mut value,
                            &mut emit,
                        )?;
                    }
                }
            }
            if !CUMULATIVE {
                workspace.candidate[component] = sum.finish()?;
            }
        }
        Ok(())
    }
}

struct PrefixOutput<'a, S: RealScalar, const CUMULATIVE: bool> {
    candidate: &'a mut [S],
    components: usize,
    component: usize,
}
impl<S: RealScalar, const CUMULATIVE: bool> PrefixOutput<'_, S, CUMULATIVE> {
    #[inline(always)]
    fn write(&mut self, index: usize, sum: &mut RealProductSum<S>) -> Result<(), SciOxideError> {
        if CUMULATIVE {
            self.candidate[(index + 1) * self.components + self.component] = sum.finish()?;
        }
        Ok(())
    }
}

fn trapezoid_component<S: RealScalar>(
    weights: impl Iterator<Item = S>,
    samples: usize,
    sum: &mut RealProductSum<S>,
    value: &mut impl FnMut(usize) -> Result<S, SciOxideError>,
) -> Result<(), SciOxideError> {
    let mut previous_weight = S::ZERO;
    for (index, weight) in weights.enumerate() {
        let sample = value(index)?;
        let (high, low) = scioxide_core::backend::two_sum(previous_weight, weight);
        if high.is_finite() && low.is_finite() {
            sum.add_product(high, sample)?;
            if low != S::ZERO {
                sum.add_product(low, sample)?;
            }
        } else {
            // Coalescing may overflow even though the original sum is finite.
            sum.add_product(previous_weight, sample)?;
            sum.add_product(weight, sample)?;
        }
        previous_weight = weight;
    }
    if samples != 0 {
        sum.add_product(previous_weight, value(samples - 1)?)?;
    }
    Ok(())
}

fn trapezoid_prefixes<S: RealScalar, const CUMULATIVE: bool>(
    weights: impl Iterator<Item = ProductFactor<S>>,
    samples: usize,
    sum: &mut RealProductSum<S>,
    value: &mut impl FnMut(usize) -> Result<S, SciOxideError>,
    emit: &mut PrefixOutput<'_, S, CUMULATIVE>,
) -> Result<(), SciOxideError> {
    if samples == 0 {
        return Ok(());
    }
    let mut previous = ProductFactor::new(value(0)?)?;
    for (index, weight) in weights.enumerate() {
        let next = ProductFactor::new(value(index + 1)?)?;
        let prefix = sum.add_weighted_pair_prefix(weight, previous, next)?;
        emit.candidate[(index + 1) * emit.components + emit.component] = prefix;
        previous = next;
    }
    Ok(())
}

fn simpson_component<S: RealScalar, const CUMULATIVE: bool>(
    weights: impl Iterator<Item = [S; 3]>,
    samples: usize,
    sum: &mut RealProductSum<S>,
    value: &mut impl FnMut(usize) -> Result<S, SciOxideError>,
    emit: &mut PrefixOutput<'_, S, CUMULATIVE>,
) -> Result<(), SciOxideError> {
    let mut cache = [value(0)?, value(1)?, value(2)?];
    let mut first = 0;
    for (index, weights) in weights.enumerate() {
        let next_first = stencil_first(index, samples);
        while first < next_first {
            cache = [cache[1], cache[2], value(first + 3)?];
            first += 1;
        }
        for (weight, value) in weights.into_iter().zip(cache) {
            sum.add_product(weight, value)?;
        }
        emit.write(index, sum)?;
    }
    Ok(())
}

/// Composite trapezoid rule for samples `yᵢ = f(xᵢ)` on a fixed grid.
///
/// ```text
/// ∫ f dx ≈ Σᵢ (xᵢ₊₁ − xᵢ) (yᵢ + yᵢ₊₁) / 2,   i = 0, …, n − 2,
/// ```
///
/// for `n` coordinates `xᵢ` that are strictly increasing or strictly
/// decreasing; decreasing coordinates give the integral from `x₀` down to
/// `xₙ₋₁`, with the opposite sign. Fewer than two samples give zero. The rule is
/// exact for linear `f`.
///
/// The plan stores the rounded weights `(xᵢ₊₁ − xᵢ)/2`, and a single weight
/// when the spacing is uniform. Each result is the exact sum of the products of
/// weights and samples, rounded once. Construction costs O(n) and applying the
/// plan O(n d) for `d` real components.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{SampledTrapezoidPlan, SampledWorkspace};
///
/// // Trapezoid rule for x² on [0, 1] with h = 0.5: (0 + 2·0.25 + 1)/4 = 0.375
/// let plan = SampledTrapezoidPlan::new(&[0.0_f64, 0.5, 1.0])?;
/// let mut workspace = plan.workspace(1)?;
/// let mut value = 0.0;
/// plan.integrate_into(&[0.0, 0.25, 1.0], &mut value, &mut workspace)?;
/// assert!((value - 0.375).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SampledTrapezoidPlan<S: RealScalar>(Panels<S>);
impl<S: RealScalar> SampledTrapezoidPlan<S> {
    /// Creates the plan for `coordinates`, in the given order.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `InvalidInput` (argument `coordinates`) if the coordinates repeat a
    ///   value or change direction.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(coordinates: &[S]) -> Result<Self, SciOxideError> {
        validate_coordinates(coordinates)?;
        Ok(Self(Panels::trapezoid(coordinates)?))
    }
}

/// Composite Simpson rule for samples `yᵢ = f(xᵢ)` on a fixed, possibly
/// nonuniform grid.
///
/// The rule integrates, over each pair of intervals `[x₂ₖ, x₂ₖ₊₂]`, the
/// quadratic that interpolates the three samples. With an odd number of
/// intervals, the last interval integrates the quadratic through the last
/// three samples. Two samples use the trapezoid rule, and fewer than two give
/// zero. The coordinates must be strictly increasing or strictly decreasing.
/// The rule is exact for quadratic `f`. A cumulative value that ends inside a
/// pair integrates the pair's quadratic up to that sample. Strongly unequal
/// spacing can amplify errors in the samples; the plan only requires finite
/// weights.
///
/// Each weight is the integral of a Lagrange basis polynomial, computed from
/// the exact coordinate differences with at most three roundings, so it is
/// within 2 ulps of its exact value. Equally spaced stencils give mirrored
/// halves identical weights. Each result is the exact sum of the products of
/// weights and samples, rounded once. Construction costs O(n) and applying the
/// plan O(n d) for `d` real components.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{SampledSimpsonPlan, SampledWorkspace};
///
/// // Simpson's rule is exact for x²: ∫₀¹ x² dx = 1/3
/// let plan = SampledSimpsonPlan::new(&[0.0_f64, 0.5, 1.0])?;
/// let mut workspace = plan.workspace(1)?;
/// let mut value = 0.0;
/// plan.integrate_into(&[0.0, 0.25, 1.0], &mut value, &mut workspace)?;
/// assert!((value - 1.0 / 3.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SampledSimpsonPlan<S: RealScalar>(Panels<S>);
impl<S: RealScalar> SampledSimpsonPlan<S> {
    /// Creates the plan for `coordinates`, in the given order.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `InvalidInput` (argument `coordinates`) if the coordinates repeat a
    ///   value or change direction.
    /// - `NumericalInstability` if a weight overflows.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(coordinates: &[S]) -> Result<Self, SciOxideError> {
        validate_coordinates(coordinates)?;
        if coordinates.len() < 3 {
            return Ok(Self(Panels::trapezoid(coordinates)?));
        }
        let mut varied = Vec::new();
        let mut halves = [[S::ZERO; 3]; 2];
        let mut exact = RangeAccumulator::new();
        let spacings = |first: usize| {
            [0, 1].map(|j| two_sum(coordinates[first + j + 1], -coordinates[first + j]))
        };
        let uniform = spacings(0);
        // Each interval integrates the quadratic through a three-point stencil: intervals
        // are paired (stencil starting at the even index), so the two halves of a pair add
        // up to composite non-uniform Simpson on that pair. With an odd number of
        // intervals the last one reuses the final three points. Integrating each interval
        // separately (rather than whole pairs) is what provides per-sample cumulative values.
        for interval in 0..coordinates.len() - 1 {
            let first = stencil_first(interval, coordinates.len());
            let half = interval - first;
            // Weights depend only on the exact spacings, so a repeat of the first
            // stencil's finite spacings reuses its halves without recomputation.
            let repeat = interval >= 2
                && varied.is_empty()
                && spacings(first) == uniform
                && uniform
                    .iter()
                    .all(|(high, low)| high.is_finite() && low.is_finite());
            let weights = if repeat {
                halves[half]
            } else {
                simpson_half(
                    &mut exact,
                    [
                        coordinates[first],
                        coordinates[first + 1],
                        coordinates[first + 2],
                    ],
                    half,
                )?
            };
            if interval < 2 {
                halves[half] = weights;
            }
            if varied.is_empty() && weights != halves[half] {
                varied =
                    try_with_capacity(coordinates.len() - 1, scioxide_core::quantity!("weights"))?;
                varied.extend((0..interval).map(|i| halves[half_index(i, coordinates.len())]));
            }
            if !varied.is_empty() {
                varied.push(weights);
            }
        }
        Ok(Self(Panels {
            samples: coordinates.len(),
            weights: Weights::Simpson { halves, varied },
        }))
    }
}
macro_rules! panel_api {
    ($name:ident) => {
        impl<S: RealScalar> $name<S> {
            /// Returns the number of samples `n` the plan expects.
            pub const fn sample_count(&self) -> usize {
                self.0.samples
            }
            /// Integrates `samples` over the whole grid and writes the result to
            /// `output`.
            ///
            /// The samples are real, complex or vector values with the component
            /// count of the workspace. There is no error estimate.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `samples.len()` is not [`Self::sample_count`],
            ///   or a sample or `output` does not have the workspace's component
            ///   count.
            /// - `InvalidInput` (argument `workspace`) if the workspace is too small.
            /// - `NonFinite` if a sample is NaN or infinite.
            /// - `NumericalInstability` if the result overflows.
            pub fn integrate_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
                &self,
                samples: &[V],
                output: &mut O,
                workspace: &mut SampledWorkspace<S>,
            ) -> Result<(), SciOxideError> {
                self.0
                    .apply(samples, core::slice::from_mut(output), workspace, false)
            }
            /// Writes the cumulative integrals `∫ from x₀ to xᵢ` for every `i` to
            /// `output`.
            ///
            /// `output` has one entry per sample. The first entry is zero and
            /// the last equals the result of [`Self::integrate_into`].
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `samples.len()` or `output.len()` is not
            ///   [`Self::sample_count`], or a value does not have the workspace's
            ///   component count.
            /// - `InvalidInput` (argument `workspace`) if the workspace has room for
            ///   fewer than [`Self::sample_count`] results.
            /// - `NonFinite` if a sample is NaN or infinite.
            /// - `NumericalInstability` if a result overflows.
            pub fn cumulative_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
                &self,
                samples: &[V],
                output: &mut [O],
                workspace: &mut SampledWorkspace<S>,
            ) -> Result<(), SciOxideError> {
                self.0.apply(samples, output, workspace, true)
            }
            fn apply<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
                &self,
                samples: &[V],
                output: &mut [O],
                workspace: &mut SampledWorkspace<S>,
                cumulative: bool,
            ) -> Result<(), SciOxideError> {
                self.0.apply(samples, output, workspace, cumulative)
            }
        }
    };
}
panel_api!(SampledTrapezoidPlan);
panel_api!(SampledSimpsonPlan);

/// Romberg integration of `2ᵏ + 1` equally spaced samples.
///
/// With trapezoid sums `T₀, …, Tₖ` of step sizes `2ᵏh, …, 2h, h`, where `h` is
/// the grid spacing, Richardson extrapolation
///
/// ```text
/// Rⱼ,ₘ = Rⱼ,ₘ₋₁ + (Rⱼ,ₘ₋₁ − Rⱼ₋₁,ₘ₋₁) / (4ᵐ − 1),   Rⱼ,₀ = Tⱼ,
/// ```
///
/// gives the result `Rₖ,ₖ`, which is exact for polynomials of degree `2k + 1`.
/// The plan folds the extrapolation into one weight per sample, computed in
/// working precision. Each result is the exact sum of the products of weights
/// and samples, rounded once. There is no error estimate.
///
/// The cumulative results are the Romberg values of the prefixes that end at
/// the sample indices `1, 2, 4, …, 2ᵏ`; each uses only the samples of its
/// prefix. Construction costs O(n) operations and storage, and applying the
/// plan O(n d) for `d` real components.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{SampledRombergPlan, SampledWorkspace};
///
/// // Samples of x² at x = 0, 0.25, …, 1; ∫₀¹ x² dx = 1/3
/// let plan = SampledRombergPlan::new(5, 0.25_f64)?;
/// let mut workspace = plan.workspace(1)?;
/// let mut value = 0.0;
/// let samples = [0.0, 0.0625, 0.25, 0.5625, 1.0];
/// plan.integrate_into(&samples, &mut value, &mut workspace)?;
/// assert!((value - 1.0 / 3.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
/// Academic Press, 1984.
#[derive(Debug, Clone)]
pub struct SampledRombergPlan<S: RealScalar> {
    samples: usize,
    endpoints: Vec<usize>,
    weights: Vec<Vec<S>>,
}
impl<S: RealScalar> SampledRombergPlan<S> {
    /// Creates the plan for `samples` equally spaced samples with signed
    /// spacing `spacing`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `spacing` is NaN or infinite.
    /// - `InvalidInput` (argument `samples`) if `samples` is not `2ᵏ + 1` for
    ///   some `k ≥ 0`.
    /// - `InvalidInput` (argument `spacing`) if `spacing` is zero.
    /// - `NumericalInstability` if a weight overflows.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(samples: usize, spacing: S) -> Result<Self, SciOxideError> {
        validate_value(&spacing)?;
        for (invalid_value, argument) in [
            (
                samples < 2 || !(samples - 1).is_power_of_two(),
                scioxide_core::argument!("samples"),
            ),
            (spacing == S::ZERO, scioxide_core::argument!("spacing")),
        ] {
            if invalid_value {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Romberg requires 2^k+1 samples and nonzero spacing",
                ));
            }
        }
        let levels = (samples - 1).trailing_zeros() as usize + 1;
        let mut endpoints =
            try_with_capacity(levels, scioxide_core::quantity!("cumulative_indices"))?;
        let mut weights = try_with_capacity(levels, scioxide_core::quantity!("weight_rows"))?;
        for level in 0..levels {
            let end = 1usize << level;
            let mut row = try_filled(end + 1, S::ZERO, scioxide_core::quantity!("weight_row"))?;
            // Prefix rows exist at dyadic endpoints. Coefficients run from the
            // finest to the coarsest trapezoid, as in adaptive Romberg.
            let mut coefficients = try_filled(
                level + 1,
                scioxide_core::Scaled::ZERO,
                scioxide_core::quantity!("richardson_coefficients"),
            )?;
            coefficients[0] = scioxide_core::Scaled::from_value(S::ONE)?;
            for order in 1..=level {
                let power = checked_intermediate(S::try_from_usize_exact(4)?.powi(order as i32))?;
                crate::romberg::richardson_step(&mut coefficients[..=order], power - S::ONE)?;
            }
            for (j, coefficient) in coefficients.into_iter().enumerate() {
                let step = 1usize << j;
                let weight = coefficient
                    .mul(scioxide_core::Scaled::from_value(spacing)?)?
                    .mul(scioxide_core::Scaled::from_value(S::try_from_usize_exact(
                        step,
                    )?)?)?
                    .round()?;
                for index in (0..=end).step_by(step) {
                    row[index] = checked_intermediate(
                        row[index]
                            + weight
                                * if index == 0 || index == end {
                                    S::HALF
                                } else {
                                    S::ONE
                                },
                    )?;
                }
            }
            endpoints.push(end);
            weights.push(row);
        }
        Ok(Self {
            samples,
            endpoints,
            weights,
        })
    }
    /// Returns the sample indices `1, 2, 4, …, 2ᵏ` at which the cumulative
    /// results of [`Self::cumulative_into`] end.
    pub fn cumulative_indices(&self) -> &[usize] {
        &self.endpoints
    }
    /// Returns the number of samples `2ᵏ + 1` the plan expects.
    pub const fn sample_count(&self) -> usize {
        self.samples
    }
    /// Integrates `samples` over the whole grid and writes the result to
    /// `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `samples.len()` is not [`Self::sample_count`],
    ///   or a sample or `output` does not have the workspace's component count.
    /// - `InvalidInput` (argument `workspace`) if the workspace is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if the result overflows.
    pub fn integrate_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
        &self,
        samples: &[V],
        output: &mut O,
        workspace: &mut SampledWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.apply(samples, core::slice::from_mut(output), workspace, false)
    }
    /// Writes the Romberg values of the prefixes ending at
    /// [`Self::cumulative_indices`] to `output`, in that order.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `samples.len()` is not [`Self::sample_count`],
    ///   `output.len()` is not the number of cumulative indices, or a value does
    ///   not have the workspace's component count.
    /// - `InvalidInput` (argument `workspace`) if the workspace has room for
    ///   fewer results than `output` holds.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a result overflows.
    pub fn cumulative_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
        &self,
        samples: &[V],
        output: &mut [O],
        workspace: &mut SampledWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.apply(samples, output, workspace, true)
    }
    fn apply<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
        &self,
        samples: &[V],
        output: &mut [O],
        workspace: &mut SampledWorkspace<S>,
        cumulative: bool,
    ) -> Result<(), SciOxideError> {
        crate::kernel::shape(self.samples, samples.len())?;
        let rows = if cumulative {
            &self.weights[..]
        } else {
            &self.weights[self.weights.len() - 1..]
        };
        crate::kernel::shape(rows.len(), output.len())?;
        let d = workspace.components;
        let size = checked_mul(d, rows.len(), scioxide_core::quantity!("candidate_values"))?;
        if workspace.candidate.len() < size {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient Romberg workspace",
            ));
        }
        for sample in samples {
            check_value_shape(sample, d)?;
            validate_value(sample)?;
        }
        for value in output.iter() {
            check_value_shape(value, d)?;
        }
        for (row_index, weights) in rows.iter().enumerate() {
            for component in 0..d {
                let mut sum = RealProductSum::new();
                for (sample, &weight) in samples.iter().zip(weights) {
                    sum.add_product(weight, sample.component(component)?)?;
                }
                workspace.candidate[row_index * d + component] = sum.finish()?;
            }
        }
        for (index, value) in output.iter_mut().enumerate() {
            write_value(&workspace.candidate[index * d..(index + 1) * d], value)?;
        }
        Ok(())
    }
}

/// Scratch storage for integrating along one axis of an N-dimensional array
/// with `axis_into` or `cumulative_axis_into` of a sampled-data plan.
///
/// Obtain it from the plan's `axis_workspace` or `cumulative_axis_workspace`
/// method, for example [`SampledTrapezoidPlan::axis_workspace`].
///
/// It holds one lane of samples and all results, so its size is
/// O(n + output_values · d + rank) for `n` samples per lane and `d` real
/// components.
#[derive(Debug, Clone)]
pub struct SampledAxisWorkspace<S: RealScalar> {
    lane: Vec<S>,
    lane_output: Vec<S>,
    candidate: Vec<S>,
    coordinates: Vec<usize>,
    scalar: SampledWorkspace<S>,
    components: usize,
}
impl<S: RealScalar> SampledAxisWorkspace<S> {
    /// Creates a workspace for arrays of rank `rank` with `samples` samples
    /// along the integration axis, `output_values` values in the output array,
    /// and `components` real components per value.
    ///
    /// `lane_outputs` is the number of results per lane: 1 for `axis_into`, the
    /// sample count for `cumulative_axis_into` of the trapezoid and Simpson
    /// plans, and [`SampledRombergPlan::cumulative_indices`]`.len()` for
    /// Romberg. `samples`, `lane_outputs` and `rank` must match the call exactly.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `output_values · components` overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub(crate) fn new(
        samples: usize,
        lane_outputs: usize,
        output_values: usize,
        components: usize,
        rank: usize,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            lane: try_filled(samples, S::ZERO, scioxide_core::quantity!("lane_samples"))?,
            lane_output: try_filled(
                lane_outputs,
                S::ZERO,
                scioxide_core::quantity!("lane_outputs"),
            )?,
            candidate: try_filled(
                checked_mul(
                    output_values,
                    components,
                    scioxide_core::quantity!("candidate_values"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("candidate_values"),
            )?,
            coordinates: try_filled(rank, 0, scioxide_core::quantity!("lane_coordinates"))?,
            scalar: SampledWorkspace::new(1, lane_outputs)?,
            components,
        })
    }
}
trait AxisPlan<S: RealScalar> {
    fn samples(&self) -> usize;
    fn prefixes(&self) -> usize;
    fn lane(
        &self,
        samples: &[S],
        output: &mut [S],
        workspace: &mut SampledWorkspace<S>,
        cumulative: bool,
    ) -> Result<(), SciOxideError>;
}
macro_rules! axis_api {
    ($name:ident, $prefix:expr) => {
        impl<S: RealScalar> AxisPlan<S> for $name<S> {
            fn samples(&self) -> usize {
                self.sample_count()
            }
            fn prefixes(&self) -> usize {
                ($prefix)(self)
            }
            fn lane(
                &self,
                samples: &[S],
                output: &mut [S],
                workspace: &mut SampledWorkspace<S>,
                cumulative: bool,
            ) -> Result<(), SciOxideError> {
                let output = if cumulative { output } else { &mut output[..1] };
                self.apply(samples, output, workspace, cumulative)
            }
        }
        impl<S: RealScalar> $name<S> {
            /// Allocates scratch for one integral with the given number of real components.
            ///
            /// Holds one result for `integrate_into`. Use `components = 1` for
            /// real values and `components = 2` for complex values.
            ///
            /// # Errors
            ///
            /// Returns SizeOverflow or AllocationFailure.
            pub fn workspace(
                &self,
                components: usize,
            ) -> Result<SampledWorkspace<S>, SciOxideError> {
                SampledWorkspace::new(components, 1)
            }
            /// Allocates scratch for all cumulative integrals; it also serves integrate_into.
            ///
            /// Holds one result per sample for trapezoid and Simpson plans,
            /// or one per [`SampledRombergPlan::cumulative_indices`] entry for
            /// Romberg. Each result has `components` real components (1 for a
            /// real value, 2 for a complex value).
            ///
            /// # Errors
            ///
            /// Returns SizeOverflow or AllocationFailure.
            pub fn cumulative_workspace(
                &self,
                components: usize,
            ) -> Result<SampledWorkspace<S>, SciOxideError> {
                SampledWorkspace::new(components, self.prefixes().max(1))
            }
            /// Allocates scratch for integrating one axis of an array with the given shape.
            ///
            /// Holds one lane of samples and one result per lane, each with
            /// `components` real components (1 for real values, 2 for complex
            /// values). The shape and axis must match the subsequent call.
            ///
            /// # Errors
            ///
            /// Returns OutOfBounds for an invalid axis, DimensionMismatch for the
            /// wrong sample extent, and SizeOverflow or AllocationFailure.
            pub fn axis_workspace(
                &self,
                shape: scioxide_core::Shape<'_>,
                axis: usize,
                components: usize,
            ) -> Result<SampledAxisWorkspace<S>, SciOxideError> {
                axis_workspace(self, shape, axis, components, 1)
            }
            /// Allocates scratch for cumulative integration along an array axis.
            ///
            /// Holds one lane of samples and all output values. Each lane has
            /// one result per sample for trapezoid and Simpson plans, or one
            /// per [`SampledRombergPlan::cumulative_indices`] entry for Romberg.
            /// The shape, axis and real component count must match the call.
            ///
            /// # Errors
            ///
            /// Returns the errors of [`Self::axis_workspace`].
            pub fn cumulative_axis_workspace(
                &self,
                shape: scioxide_core::Shape<'_>,
                axis: usize,
                components: usize,
            ) -> Result<SampledAxisWorkspace<S>, SciOxideError> {
                axis_workspace(self, shape, axis, components, self.prefixes())
            }

            /// Integrates along `axis` of `samples` and writes the results to
            /// `output`.
            ///
            /// Each lane along `axis` is integrated as by `integrate_into`.
            /// `output` has the extents of `samples` without `axis`, in the same
            /// order.
            ///
            /// # Errors
            ///
            /// - `OutOfBounds` if `axis` is not less than the rank of `samples`.
            /// - `DimensionMismatch` if the extent along `axis` is not the sample
            ///   count, the extents of `output` do not match, a value does not have
            ///   the workspace's component count, or the workspace was created
            ///   for another sample count, lane output count or rank.
            /// - `InvalidInput` (argument `workspace`) if the workspace has room
            ///   for fewer values than `output`.
            /// - `NonFinite` if a sample is NaN or infinite.
            /// - `NumericalInstability` if a result overflows.
            pub fn axis_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
                &self,
                samples: &scioxide_core::NdView<'_, '_, V>,
                axis: usize,
                output: &mut scioxide_core::NdViewMut<'_, '_, O>,
                workspace: &mut SampledAxisWorkspace<S>,
            ) -> Result<(), SciOxideError> {
                apply_axis(self, samples, axis, output, workspace, false)
            }
            /// Computes cumulative integrals along `axis` of `samples` and writes
            /// them to `output`.
            ///
            /// Each lane along `axis` is processed as by `cumulative_into`.
            /// `output` has the extents of `samples`, with the extent along `axis`
            /// replaced by the number of cumulative results.
            ///
            /// # Errors
            ///
            /// - The errors of `axis_into`.
            pub fn cumulative_axis_into<V: AffineValue<Scale = S>, O: AffineValueMut<Scale = S>>(
                &self,
                samples: &scioxide_core::NdView<'_, '_, V>,
                axis: usize,
                output: &mut scioxide_core::NdViewMut<'_, '_, O>,
                workspace: &mut SampledAxisWorkspace<S>,
            ) -> Result<(), SciOxideError> {
                apply_axis(self, samples, axis, output, workspace, true)
            }
        }
    };
}
axis_api!(SampledTrapezoidPlan, |p: &SampledTrapezoidPlan<S>| p
    .sample_count());
axis_api!(SampledSimpsonPlan, |p: &SampledSimpsonPlan<S>| p
    .sample_count());
axis_api!(SampledRombergPlan, |p: &SampledRombergPlan<S>| p
    .cumulative_indices()
    .len());

fn apply_axis<
    S: RealScalar,
    P: AxisPlan<S>,
    V: AffineValue<Scale = S>,
    O: AffineValueMut<Scale = S>,
>(
    plan: &P,
    samples: &scioxide_core::NdView<'_, '_, V>,
    axis: usize,
    output: &mut scioxide_core::NdViewMut<'_, '_, O>,
    workspace: &mut SampledAxisWorkspace<S>,
    cumulative: bool,
) -> Result<(), SciOxideError> {
    let layout = samples.layout();
    let batches = layout.axis(axis)?;
    let extents = layout.shape().extents();
    crate::kernel::shape(plan.samples(), extents[axis])?;
    let outputs = if cumulative { plan.prefixes() } else { 1 };
    let output_layout = output.layout();
    let output_extents = output_layout.shape().extents();
    crate::kernel::shape(
        extents.len() - usize::from(!cumulative),
        output_extents.len(),
    )?;
    let mut target_axis = 0;
    for (index, &extent) in extents.iter().enumerate() {
        if index == axis && !cumulative {
            continue;
        }
        crate::kernel::shape(
            if index == axis { outputs } else { extent },
            output_extents[target_axis],
        )?;
        target_axis += 1;
    }
    crate::kernel::shape(extents.len(), workspace.coordinates.len())?;
    crate::kernel::shape(plan.samples(), workspace.lane.len())?;
    crate::kernel::shape(outputs, workspace.lane_output.len())?;
    let d = workspace.components;
    let size = checked_mul(
        output_layout.shape().len(),
        d,
        scioxide_core::quantity!("candidate_values"),
    )?;
    if workspace.candidate.len() < size {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient axis output scratch",
        ));
    }
    for value in samples.iter() {
        check_value_shape(value, d)?;
    }
    for value in output.as_view().iter() {
        check_value_shape(value, d)?;
    }
    // Each batch is one 1-D lane along `axis`. The batch number is decoded into the
    // row-major coordinates of the other axes, the lane is gathered per component and
    // integrated with the scalar plan, and results are placed at their row-major logical
    // output index (with the prefix index in place of `axis` for cumulative output).
    // Output is published only at the end, in the same logical order `for_each_mut` visits.
    for batch in 0..batches.len() {
        let mut remainder = batch;
        for (index, &extent) in extents.iter().enumerate().rev() {
            if index != axis {
                workspace.coordinates[index] = remainder % extent;
                remainder /= extent;
            }
        }
        for component in 0..d {
            for (index, target) in workspace.lane.iter_mut().enumerate() {
                workspace.coordinates[axis] = index;
                *target = samples.get(&workspace.coordinates)?.component(component)?;
            }
            plan.lane(
                &workspace.lane,
                &mut workspace.lane_output,
                &mut workspace.scalar,
                cumulative,
            )?;
            for (prefix, &value) in workspace.lane_output.iter().enumerate() {
                let mut logical = 0;
                for (index, &extent) in extents.iter().enumerate() {
                    if index == axis {
                        if cumulative {
                            logical = logical * outputs + prefix;
                        }
                    } else {
                        logical = logical * extent + workspace.coordinates[index];
                    }
                }
                workspace.candidate[logical * d + component] = value;
            }
        }
    }
    let mut index = 0;
    output.for_each_mut(|value| {
        value.for_each_component_mut(|component| {
            *component = workspace.candidate[index];
            index += 1;
        });
    });
    Ok(())
}

fn axis_workspace<S: RealScalar>(
    plan: &impl AxisPlan<S>,
    shape: scioxide_core::Shape<'_>,
    axis: usize,
    components: usize,
    prefixes: usize,
) -> Result<SampledAxisWorkspace<S>, SciOxideError> {
    crate::kernel::shape(plan.samples(), shape.extent(axis)?)?;
    let values = shape
        .extents()
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis)
        .try_fold(prefixes, |count, (_, &extent)| {
            checked_mul(count, extent, scioxide_core::quantity!("candidate_values"))
        })?;
    SampledAxisWorkspace::new(plan.samples(), prefixes, values, components, shape.rank())
}

/// Integrates sampled values by the trapezoid rule.
///
/// Coordinates may increase or decrease strictly; reversing them reverses the
/// sign. Uses the piecewise-linear interpolant of the samples. Costs O(nc) operations for
/// n samples and c real components. Polynomial exactness and rounding behavior
/// follow [`SampledTrapezoidPlan`]; no discretization error estimate is returned.
///
/// # Errors
///
/// Rejects mismatched sample and coordinate counts, non-finite values and
/// repeated or nonmonotone coordinates. Returns allocation and arithmetic errors.
///
/// ```
/// let value = scioxide_integrate::trapezoid(&[0.0_f64, 1.0, 2.0], &[0.0, 1.0, 2.0])?;
/// assert!((value - 2.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, Methods of Numerical Integration, 2nd ed. (1984).
pub fn trapezoid<S, V>(samples: &[V], coordinates: &[S]) -> Result<V, SciOxideError>
where
    S: RealScalar,
    V: AffineValueMut<Scale = S> + Copy + Default,
{
    crate::kernel::shape(coordinates.len(), samples.len())?;
    let plan = SampledTrapezoidPlan::new(coordinates)?;
    let mut workspace = plan.workspace(V::default().dimension()?)?;
    let mut value = V::default();
    plan.integrate_into(samples, &mut value, &mut workspace)?;
    Ok(value)
}

/// Integrates sampled values by Simpson's rule.
///
/// Coordinates may increase or decrease strictly; reversing them reverses the
/// sign. Uses quadratic panels with the endpoint treatment of [`SampledSimpsonPlan`]. Costs O(nc) operations for
/// n samples and c real components. Polynomial exactness and rounding behavior
/// follow [`SampledSimpsonPlan`]; no discretization error estimate is returned.
///
/// # Errors
///
/// Rejects mismatched sample and coordinate counts, non-finite values and
/// repeated or nonmonotone coordinates. Returns allocation and arithmetic errors.
///
/// ```
/// let value = scioxide_integrate::simpson(&[0.0_f64, 1.0, 2.0], &[0.0, 1.0, 2.0])?;
/// assert!((value - 2.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, Methods of Numerical Integration, 2nd ed. (1984).
pub fn simpson<S, V>(samples: &[V], coordinates: &[S]) -> Result<V, SciOxideError>
where
    S: RealScalar,
    V: AffineValueMut<Scale = S> + Copy + Default,
{
    crate::kernel::shape(coordinates.len(), samples.len())?;
    let plan = SampledSimpsonPlan::new(coordinates)?;
    let mut workspace = plan.workspace(V::default().dimension()?)?;
    let mut value = V::default();
    plan.integrate_into(samples, &mut value, &mut workspace)?;
    Ok(value)
}

/// Computes every cumulative trapezoidal integral, starting with zero.
///
/// Coordinates may increase or decrease strictly; reversing them reverses the
/// sign. Uses the piecewise-linear interpolant of the samples. Costs O(nc) operations for
/// n samples and c real components. Polynomial exactness and rounding behavior
/// follow [`SampledTrapezoidPlan`]; no discretization error estimate is returned.
///
/// # Errors
///
/// Rejects mismatched sample and coordinate counts, non-finite values and
/// repeated or nonmonotone coordinates. Returns allocation and arithmetic errors.
///
/// ```
/// let value = scioxide_integrate::cumulative_trapezoid(&[0.0_f64, 1.0, 2.0], &[0.0, 1.0, 2.0])?;
/// assert!((value[2] - 2.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, Methods of Numerical Integration, 2nd ed. (1984).
pub fn cumulative_trapezoid<S, V>(samples: &[V], coordinates: &[S]) -> Result<Vec<V>, SciOxideError>
where
    S: RealScalar,
    V: AffineValueMut<Scale = S> + Copy + Default,
{
    crate::kernel::shape(coordinates.len(), samples.len())?;
    let plan = SampledTrapezoidPlan::new(coordinates)?;
    let mut workspace = plan.cumulative_workspace(V::default().dimension()?)?;
    let mut value = try_filled(
        coordinates.len(),
        V::default(),
        scioxide_core::quantity!("cumulative_integrals"),
    )?;
    plan.cumulative_into(samples, &mut value, &mut workspace)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Per-interval panel weights, kept independent of weight compaction and reuse.
    fn reference<S: RealScalar>(
        x: &[S],
        y: &[[S; 2]],
        simpson: bool,
    ) -> Result<Vec<[S; 2]>, SciOxideError> {
        let mut sums = [RangeAccumulator::new(), RangeAccumulator::new()];
        let mut result = alloc::vec![[S::ZERO; 2]; x.len()];
        let mut exact = RangeAccumulator::new();
        for i in 0..x.len().saturating_sub(1) {
            let (first, weights) = if simpson && x.len() >= 3 {
                let first = (i / 2 * 2).min(x.len() - 3);
                let stencil = [x[first], x[first + 1], x[first + 2]];
                (first, simpson_half(&mut exact, stencil, i - first)?)
            } else {
                let weight = scioxide_core::Scaled::from_value(x[i + 1])?
                    .sub(scioxide_core::Scaled::from_value(x[i])?)?
                    .shifted(-1)?
                    .round()?;
                (i, [weight, weight, S::ZERO])
            };
            for (component, sum) in sums.iter_mut().enumerate() {
                for (offset, weight) in weights.into_iter().enumerate() {
                    if weight != S::ZERO {
                        sum.add_product(weight, y[first + offset][component])?;
                    }
                }
                result[i + 1][component] = sum.finish()?;
            }
        }
        Ok(result)
    }

    fn compare<S: RealScalar>() {
        for n in [0, 1, 2, 3, 4, 5, 6, 17, 64, 65, 130] {
            for irregular in [false, true] {
                for reversed in [false, true] {
                    let mut x: Vec<_> = (0..n)
                        .map(|i| {
                            S::from_f64_ieee(if irregular {
                                i as f64 * 0.3 + (i % 3) as f64 * 0.01
                            } else {
                                i as f64 * 0.125
                            })
                        })
                        .collect();
                    if reversed {
                        x.reverse();
                    }
                    let y: Vec<_> = (0..n)
                        .map(|i| {
                            [
                                S::from_f64_ieee((i * 7919 % 65537) as f64 / 32768.0 - 1.0),
                                S::from_f64_ieee((i * 104729 % 65537) as f64 / 32768.0 - 1.0),
                            ]
                        })
                        .collect();
                    let mut workspace = SampledTrapezoidPlan::new(&x)
                        .unwrap()
                        .cumulative_workspace(2)
                        .unwrap();
                    for simpson in [false, true] {
                        let plan = if simpson {
                            SampledSimpsonPlan::new(&x).unwrap().0
                        } else {
                            SampledTrapezoidPlan::new(&x).unwrap().0
                        };
                        let expected = reference(&x, &y, simpson).unwrap();
                        let mut prefixes = alloc::vec![[S::ZERO; 2]; n];
                        plan.apply(&y, &mut prefixes, &mut workspace, true).unwrap();
                        for (actual, expected) in
                            prefixes.iter().flatten().zip(expected.iter().flatten())
                        {
                            assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits());
                        }
                        let mut total = [[S::ZERO; 2]];
                        plan.apply(&y, &mut total, &mut workspace, false).unwrap();
                        assert_eq!(total[0], expected.last().copied().unwrap_or([S::ZERO; 2]));
                    }
                }
            }
        }
    }

    #[test]
    fn totals_and_prefixes_preserve_original_panel_bits() {
        compare::<f32>();
        compare::<f64>();
    }

    #[test]
    fn repeating_weights_use_constant_storage_without_approximating_grids() {
        for n in [3, 4, 5, 6, 65, 66] {
            let x: Vec<_> = (0..n).map(|i| i as f64 * 0.125).collect();
            let trap = SampledTrapezoidPlan::new(&x).unwrap();
            assert!(
                matches!(trap.0.weights, Weights::Trapezoid { varied, .. } if varied.is_empty())
            );
            let simpson = SampledSimpsonPlan::new(&x).unwrap();
            assert!(
                matches!(simpson.0.weights, Weights::Simpson { varied, .. } if varied.is_empty())
            );
        }
        let x: Vec<_> = (0..66).map(|i| i as f64 / 65.0).collect();
        assert!(matches!(SampledTrapezoidPlan::new(&x).unwrap().0.weights,
            Weights::Trapezoid { varied, .. } if varied.len() == 65));
        assert!(matches!(SampledSimpsonPlan::new(&x).unwrap().0.weights,
            Weights::Simpson { varied, .. } if varied.len() == 65));
    }
}
