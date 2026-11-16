//! Brownian bridge construction of Brownian paths from normal variates.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_with_capacity};
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_core::{checked_difference, checked_float_product, checked_sum_of_products};

use crate::workspace::{SamplingRequirements, SamplingWorkspace};

/// One precomputed conditional Brownian-bridge assignment.
#[derive(Debug, Clone, Copy)]
struct BridgeStep<S: RealScalar> {
    output_index: usize,
    left_index: usize,
    right_index: usize,
    normal_index: usize,
    left_weight: S,
    right_weight: S,
    conditional_std_dev: S,
}

/// Brownian bridge construction of a Brownian path on a fixed time grid `t₀ < t₁ < … < tₙ`.
///
/// The plan maps `n` independent standard normal coordinates `z` to the values
/// `W(t₁), …, W(tₙ)` of a Brownian motion with `W(t₀) = 0`. Coordinate `z[0]` sets the end
/// value `W(tₙ) = √(tₙ − t₀) z[0]`. Each further coordinate fills the midpoint index `m` of an
/// interval `(l, r)` whose ends are already known, from the conditional distribution
///
/// ```text
/// W(tₘ) = ((tᵣ − tₘ) W(tₗ) + (tₘ − tₗ) W(tᵣ)) / (tᵣ − tₗ) + √((tₘ − tₗ)(tᵣ − tₘ) / (tᵣ − tₗ)) z.
/// ```
///
/// Intervals are bisected breadth-first and left to right. For an eight-step grid the
/// coordinates fill path indices `8, 4, 2, 6, 1, 3, 5, 7`. This puts most of the path's
/// variance in the leading coordinates, which is where low-discrepancy points are most
/// uniform (Moskowitz and Caflisch 1996). The grid may be irregular, and `n` need not be a power
/// of two. Construction precomputes the interpolation weights and conditional standard deviation
/// of every bisection step in O(n) time and storage, and each path then costs O(n).
///
/// # References
///
/// - B. Moskowitz and R. E. Caflisch, "Smoothness and dimension reduction in quasi-Monte Carlo
///   methods", *Math. Comput. Modelling* 23(8–9), 1996, 37–54.
/// - P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2004, §3.1.
///
/// # Examples
///
/// ```
/// use scioxide_random::BrownianBridgePlan;
/// let plan = BrownianBridgePlan::new(&[0.0, 0.5, 1.0])?;
/// // z[0] sets W(1); z[1] perturbs the midpoint around (W(0) + W(1)) / 2.
/// let path = plan.apply(&[1.0_f64, 0.0])?;
/// assert!((path[0] - 0.5).abs() < 1e-15 && (path[1] - 1.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct BrownianBridgePlan<S: RealScalar> {
    times: Vec<S>,
    path_len: usize,
    terminal_std_dev: S,
    steps: Vec<BridgeStep<S>>,
}

impl<S: RealScalar> BrownianBridgePlan<S> {
    /// Creates a plan for the time grid `times = [t₀, t₁, …, tₙ]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if fewer than two times are given.
    /// - `NonFinite` if a time or a bridge coefficient is NaN or infinite.
    /// - `DomainError` if the times are not strictly increasing.
    /// - `NumericalInstability` if a positive conditional standard deviation underflows.
    /// - `SizeOverflow` if index arithmetic overflows.
    /// - `AllocationFailure` if the plan's storage cannot be allocated.
    pub fn new(times: &[S]) -> Result<Self, SciOxideError> {
        validate_time_grid(times)?;

        let path_len = times.len() - 1;
        let terminal_variance = checked_difference(times[path_len], times[0])?;
        let terminal_std_dev = validated_std_dev(terminal_variance)?;
        let interior_count = path_len - 1;
        let mut steps = try_with_capacity(
            interior_count,
            scioxide_core::quantity!("brownian_bridge_steps"),
        )?;
        let mut intervals = try_with_capacity(
            interior_count,
            scioxide_core::quantity!("brownian_bridge_topology_worklist"),
        )?;

        if interior_count > 0 {
            intervals.push((0, path_len));
        }

        // Breadth-first bisection of the index range. Processing intervals in
        // FIFO order assigns normal coordinates coarse-to-fine: z[0] fixes
        // the terminal value, z[1] the midpoint, and so on. This concentrates
        // the path's variance in the leading coordinates, which is what makes
        // the bridge effective with low-discrepancy inputs. `intervals` doubles
        // as the queue and never holds more than `interior_count` entries.
        let mut cursor = 0;
        while cursor < intervals.len() {
            let (left_index, right_index) = intervals[cursor];
            cursor = checked_add(
                cursor,
                1,
                scioxide_core::quantity!("brownian_bridge_topology_cursor"),
            )?;
            let output_time_index = usize::midpoint(left_index, right_index);
            let normal_index = checked_add(
                steps.len(),
                1,
                scioxide_core::quantity!("brownian_bridge_normal_coordinate_index"),
            )?;
            steps.push(build_bridge_step(
                times,
                left_index,
                output_time_index,
                right_index,
                normal_index,
            )?);

            if output_time_index - left_index > 1 {
                intervals.push((left_index, output_time_index));
            }
            if right_index - output_time_index > 1 {
                intervals.push((output_time_index, right_index));
            }
        }

        let mut owned_times = try_with_capacity(
            times.len(),
            scioxide_core::quantity!("brownian_bridge_time_grid"),
        )?;
        owned_times.extend_from_slice(times);

        Ok(Self {
            times: owned_times,
            path_len,
            terminal_std_dev,
            steps,
        })
    }

    /// Returns the time grid, including `t₀`.
    #[must_use]
    pub fn times(&self) -> &[S] {
        &self.times
    }

    /// Returns the number `n` of normal coordinates per path.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.path_len
    }

    /// Returns the number `n` of path values written per path.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.path_len
    }

    /// Returns the workspace size [`Self::apply_into`] needs: one full path.
    ///
    /// # Errors
    ///
    /// Never fails. The `Result` matches the other `workspace_requirements` methods.
    pub fn workspace_requirements(&self, (): ()) -> Result<SamplingRequirements, SciOxideError> {
        Ok(SamplingRequirements::new(self.path_len))
    }

    /// Returns a workspace reserved for [`Self::apply_into`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(&self, request: ()) -> Result<SamplingWorkspace<S>, SciOxideError> {
        let mut workspace = SamplingWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Returns the path `W(t₁), …, W(tₙ)` built from the normal coordinates `z` in a new `Vec`.
    ///
    /// If `z` is standard normal, the path is centered Gaussian with
    /// `Cov(W(tᵢ), W(tⱼ)) = min(tᵢ, tⱼ) − t₀`. The cost is O(n) and one allocation of `n`
    /// values. The path equals the output of [`Self::apply_into`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `z` does not have length `n`.
    /// - `NonFinite` if a coordinate of `z` is NaN or infinite.
    /// - `AllocationFailure` if the path cannot be allocated.
    /// - `NumericalInstability` if a path value overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BrownianBridgePlan, Distribution, Normal, SeedableRng, SplitMix64};
    /// let times: Vec<f64> = (0..=256).map(|i| f64::from(i) / 256.0).collect();
    /// let normals = Normal::<f64>::standard().sample_n(&mut SplitMix64::from_seed(663), 256)?;
    /// let path = BrownianBridgePlan::new(&times)?.apply(&normals)?;
    /// // The end value is W(1) = √1 · z[0].
    /// assert!((path[255] - normals[0]).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply(&self, z: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.validate_application(z, self.path_len)?;
        let mut path = scioxide_core::backend::try_filled(
            self.path_len,
            S::ZERO,
            scioxide_core::quantity!("brownian_bridge_path"),
        )?;
        self.fill_path(z, &mut path)?;
        Ok(path)
    }

    /// Writes the path `W(t₁), …, W(tₙ)` built from the normal coordinates `z` into `output`.
    ///
    /// If `z` is standard normal, the path is centered Gaussian with
    /// `Cov(W(tᵢ), W(tⱼ)) = min(tᵢ, tⱼ) − t₀`. The path is computed in `workspace` and copied to
    /// `output` at the end, so on error `output` is unchanged. The cost is O(n).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `z` or `output` does not have length `n`.
    /// - `NonFinite` if a coordinate of `z` is NaN or infinite.
    /// - `AllocationFailure` if the workspace must grow and cannot.
    /// - `NumericalInstability` if a path value overflows.
    pub fn apply_into(
        &self,
        z: &[S],
        output: &mut [S],
        workspace: &mut SamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.validate_application(z, output.len())?;
        self.stage_path(z, workspace)?;
        output.copy_from_slice(workspace.as_slice());
        Ok(())
    }

    fn stage_path(
        &self,
        z: &[S],
        workspace: &mut SamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        workspace.stage(&SamplingRequirements::new(self.path_len), S::ZERO)?;
        self.fill_path(z, workspace.as_mut_slice())
    }

    /// Fills `path` (length `n`) from validated coordinates `z`.
    fn fill_path(&self, z: &[S], path: &mut [S]) -> Result<(), SciOxideError> {
        // path[i] stores W(t_{i+1}); W(t_0) = 0 is implicit, which is why the
        // left index is shifted by one and index 0 reads as zero below. Steps
        // are replayed in construction order, so both neighbours of every
        // step are already filled.
        path[self.path_len - 1] = checked_float_product(self.terminal_std_dev, z[0])?;

        for step in &self.steps {
            let left_value = if step.left_index == 0 {
                S::ZERO
            } else {
                path[step.left_index - 1]
            };
            let right_value = path[step.right_index - 1];
            path[step.output_index] = checked_sum_of_products(
                &[
                    step.left_weight,
                    step.right_weight,
                    step.conditional_std_dev,
                ],
                &[left_value, right_value, z[step.normal_index]],
            )?;
        }

        Ok(())
    }

    fn validate_application(&self, z: &[S], output_len: usize) -> Result<(), SciOxideError> {
        if z.len() != self.input_len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.input_len()),
                scioxide_core::Dimensions::one(z.len()),
            ));
        }
        if output_len != self.output_len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.output_len()),
                scioxide_core::Dimensions::one(output_len),
            ));
        }
        if z.iter().any(|value| !value.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "normal coordinates must be finite",
            ));
        }
        Ok(())
    }
}

fn validate_time_grid<S: RealScalar>(times: &[S]) -> Result<(), SciOxideError> {
    if times.len() < 2 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("times"),
            scioxide_core::InvalidInputReason::Empty,
            "Brownian bridge requires at least two time points",
        ));
    }
    if times.iter().any(|time| !time.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "time grid must contain only finite values",
        ));
    }
    if times.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("times"),
            scioxide_core::DomainReason::OutsideDomain,
            "time grid must be strictly increasing",
        ));
    }
    Ok(())
}

fn build_bridge_step<S: RealScalar>(
    times: &[S],
    left_index: usize,
    output_time_index: usize,
    right_index: usize,
    normal_index: usize,
) -> Result<BridgeStep<S>, SciOxideError> {
    let left_span = checked_difference(times[output_time_index], times[left_index])?;
    let right_span = checked_difference(times[right_index], times[output_time_index])?;
    let span = checked_difference(times[right_index], times[left_index])?;

    // Conditional law of W(t) given W(l) and W(r): mean is the linear
    // interpolation ((r - t) W(l) + (t - l) W(r)) / (r - l), variance is
    // (t - l)(r - t) / (r - l).
    let left_weight = right_span / span;
    let right_weight = left_span / span;
    if !left_weight.is_finite() || !right_weight.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "bridge interpolation produced a non-finite weight",
        ));
    }

    let conditional_std_dev = stable_conditional_std_dev(left_span, right_span, span)?;

    Ok(BridgeStep {
        output_index: output_time_index - 1,
        left_index,
        right_index,
        normal_index,
        left_weight,
        right_weight,
        conditional_std_dev,
    })
}

fn stable_conditional_std_dev<S: RealScalar>(
    left_span: S,
    right_span: S,
    span: S,
) -> Result<S, SciOxideError> {
    // Taking square roots before multiplying avoids collapsing a representable
    // standard deviation when the mathematically equivalent variance product
    // is subnormal.
    let conditional_std_dev = left_span.sqrt() * right_span.sqrt() / span.sqrt();
    if !conditional_std_dev.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "Brownian bridge standard deviation must be finite",
        ));
    }
    if conditional_std_dev <= S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "positive Brownian bridge standard deviation is not representable",
        ));
    }
    Ok(conditional_std_dev)
}

fn validated_std_dev<S: RealScalar>(variance: S) -> Result<S, SciOxideError> {
    if !variance.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "Brownian bridge variance must be finite",
        ));
    }
    if variance < S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("times"),
            scioxide_core::DomainReason::OutsideDomain,
            "Brownian bridge variance must be non-negative",
        ));
    }
    let std_dev = variance.sqrt();
    if !std_dev.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "Brownian bridge standard deviation must be finite",
        ));
    }
    Ok(std_dev)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn bridge_order<S: RealScalar>(plan: &BrownianBridgePlan<S>) -> Vec<usize> {
        std::iter::once(plan.path_len)
            .chain(plan.steps.iter().map(|step| step.output_index + 1))
            .collect()
    }

    fn assert_dense_transform_and_covariance<S: RealScalar>(times: &[S], tolerance: S) {
        let plan = BrownianBridgePlan::new(times).unwrap();
        let n = plan.input_len();
        let mut transform = vec![vec![S::ZERO; n]; n];
        let mut workspace = SamplingWorkspace::new();

        for column in 0..n {
            let mut basis = vec![S::ZERO; n];
            basis[column] = S::ONE;
            let mut path = vec![S::ZERO; n];
            plan.apply_into(&basis, &mut path, &mut workspace).unwrap();
            for (row, value) in path.into_iter().enumerate() {
                transform[row][column] = value;
            }
        }

        for row in 0..n {
            for column in 0..n {
                let covariance = transform[row]
                    .iter()
                    .zip(&transform[column])
                    .fold(S::ZERO, |sum, (&left, &right)| sum + left * right);
                let expected = (times[row + 1] - times[0]).min(times[column + 1] - times[0]);
                let scale = S::ONE + expected.abs();
                assert!(
                    (covariance - expected).abs() <= tolerance * scale,
                    "covariance[{row}, {column}] = {covariance:?}, expected {expected:?}"
                );
            }
        }

        let normals: Vec<S> = (0..n)
            .map(|index| {
                let value = ((index * 17 + 5) % 23) as f64 / 11.0 - 1.0;
                S::try_from_f64_lossy(value).unwrap()
            })
            .collect();
        let mut path = vec![S::ZERO; n];
        plan.apply_into(&normals, &mut path, &mut workspace)
            .unwrap();
        for (row, actual) in path.into_iter().enumerate() {
            let expected = transform[row]
                .iter()
                .zip(&normals)
                .fold(S::ZERO, |sum, (&coefficient, &normal)| {
                    sum + coefficient * normal
                });
            let scale = S::ONE + expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance * scale,
                "dense reconstruction[{row}] = {actual:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn brownian_bridge_known_breadth_first_schedules() {
        for (steps, expected) in [
            (1, vec![1]),
            (2, vec![2, 1]),
            (3, vec![3, 1, 2]),
            (4, vec![4, 2, 1, 3]),
            (8, vec![8, 4, 2, 6, 1, 3, 5, 7]),
        ] {
            let times: Vec<f64> = (0..=steps).map(|index| index as f64).collect();
            let plan = BrownianBridgePlan::new(&times).unwrap();
            assert_eq!(bridge_order(&plan), expected);
        }

        let irregular = BrownianBridgePlan::new(&[0.0_f64, 0.1, 0.4, 1.0]).unwrap();
        assert_eq!(bridge_order(&irregular), [3, 1, 2]);
        assert!((irregular.steps[0].left_weight - 0.9).abs() < 1.0e-15);
        assert!((irregular.steps[0].right_weight - 0.1).abs() < 1.0e-15);
        assert!((irregular.steps[0].conditional_std_dev - 0.3).abs() < 1.0e-15);
        assert!((irregular.steps[1].left_weight - (2.0 / 3.0)).abs() < 1.0e-15);
        assert!((irregular.steps[1].right_weight - (1.0 / 3.0)).abs() < 1.0e-15);
        assert!((irregular.steps[1].conditional_std_dev - 0.2_f64.sqrt()).abs() < 1.0e-15);
    }

    #[test]
    fn topology_is_a_bijection_for_regular_and_irregular_lengths_through_257() {
        for path_len in 1..=257 {
            let mut current = -0.75_f64;
            let times: Vec<f64> = (0..=path_len)
                .map(|index| {
                    if index > 0 {
                        current += ((index * 37) % 19 + 1) as f64 / 17.0;
                    }
                    current
                })
                .collect();
            let plan = BrownianBridgePlan::new(&times).unwrap();
            assert_eq!(plan.times(), times);
            assert_eq!(plan.input_len(), path_len);
            assert_eq!(plan.output_len(), path_len);
            assert_eq!(plan.steps.len(), path_len - 1);

            let mut seen_normals = vec![false; path_len];
            let mut seen_outputs = vec![false; path_len];
            seen_normals[0] = true;
            seen_outputs[path_len - 1] = true;
            for step in &plan.steps {
                assert!(step.left_index < step.output_index + 1);
                assert!(step.output_index + 1 < step.right_index);
                assert!(step.right_index <= path_len);
                assert!(step.normal_index < path_len);
                assert!(!seen_normals[step.normal_index]);
                assert!(!seen_outputs[step.output_index]);
                assert!(step.left_weight.is_finite() && step.left_weight >= 0.0);
                assert!(step.right_weight.is_finite() && step.right_weight >= 0.0);
                assert!(step.conditional_std_dev.is_finite() && step.conditional_std_dev >= 0.0);
                seen_normals[step.normal_index] = true;
                seen_outputs[step.output_index] = true;
            }
            assert!(seen_normals.into_iter().all(|seen| seen));
            assert!(seen_outputs.into_iter().all(|seen| seen));
        }
    }

    #[test]
    fn dense_transform_has_brownian_covariance_for_f32_and_f64() {
        assert_dense_transform_and_covariance(
            &[0.0_f64, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            2.0e-12,
        );
        assert_dense_transform_and_covariance(
            &[-1.25_f64, -1.0, -0.2, 0.0, 0.05, 0.9, 2.75],
            2.0e-12,
        );
        assert_dense_transform_and_covariance(
            &[0.0_f32, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            2.0e-4,
        );
        assert_dense_transform_and_covariance(
            &[-1.25_f32, -1.0, -0.2, 0.0, 0.05, 0.9, 2.75],
            2.0e-4,
        );
    }

    #[test]
    fn one_and_two_step_paths_are_deterministic() {
        let one_step = BrownianBridgePlan::new(&[-2.0_f64, 2.0]).unwrap();
        let mut one_output = [0.0];
        let mut workspace = SamplingWorkspace::new();
        one_step
            .apply_into(&[0.5], &mut one_output, &mut workspace)
            .unwrap();
        assert_eq!(one_output, [1.0]);

        let two_step = BrownianBridgePlan::new(&[1.0_f64, 2.0, 5.0]).unwrap();
        let mut two_output = [0.0; 2];
        two_step
            .apply_into(&[0.5, -1.0], &mut two_output, &mut workspace)
            .unwrap();
        let expected = [0.25 - 0.75_f64.sqrt(), 1.0];
        assert!((two_output[0] - expected[0]).abs() < 1.0e-15);
        assert_eq!(two_output[1], expected[1]);
    }

    #[test]
    fn subnormal_time_grids_preserve_interior_normal_coordinates() {
        let f64_min = f64::from_bits(1);
        let f64_plan = BrownianBridgePlan::new(&[0.0_f64, f64_min, 2.0 * f64_min]).unwrap();
        let mut f64_output = [0.0; 2];
        let mut workspace = SamplingWorkspace::new();
        f64_plan
            .apply_into(&[0.0, 1.0], &mut f64_output, &mut workspace)
            .unwrap();
        assert!(f64_output[0].is_finite() && f64_output[0] > 0.0);
        assert_eq!(f64_output[1], 0.0);

        let f32_min = f32::from_bits(1);
        let f32_plan = BrownianBridgePlan::new(&[0.0_f32, f32_min, 2.0 * f32_min]).unwrap();
        let mut f32_output = [0.0; 2];
        let mut f32_workspace = SamplingWorkspace::new();
        f32_plan
            .apply_into(&[0.0, 1.0], &mut f32_output, &mut f32_workspace)
            .unwrap();
        assert!(f32_output[0].is_finite() && f32_output[0] > 0.0);
        assert_eq!(f32_output[1], 0.0);
    }

    #[test]
    fn plan_rejects_invalid_time_grids() {
        for times in [&[][..], &[0.0_f64][..]] {
            let error = BrownianBridgePlan::new(times).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        }
        for times in [
            &[0.0_f64, f64::NAN][..],
            &[f64::NEG_INFINITY, 0.0][..],
            &[0.0, f64::INFINITY][..],
        ] {
            let error = BrownianBridgePlan::new(times).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        for times in [&[0.0_f64, 0.0][..], &[1.0, 0.0][..]] {
            let error = BrownianBridgePlan::new(times).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        }
        let error = BrownianBridgePlan::new(&[-f64::MAX, f64::MAX]).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn application_validation_leaves_output_unchanged() {
        let plan = BrownianBridgePlan::new(&[0.0_f64, 0.25, 0.5, 0.75, 1.0]).unwrap();
        let mut workspace = SamplingWorkspace::new();

        let mut output = [9.0; 4];
        let error = plan
            .apply_into(&[0.0; 3], &mut output, &mut workspace)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(output, [9.0; 4]);

        let mut short_output = [8.0; 3];
        let error = plan
            .apply_into(&[0.0; 4], &mut short_output, &mut workspace)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(short_output, [8.0; 3]);

        let mut long_output = [7.0; 5];
        let error = plan
            .apply_into(&[0.0; 4], &mut long_output, &mut workspace)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(long_output, [7.0; 5]);

        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut normals = [0.0; 4];
            normals[2] = invalid;
            let mut finite_output = [6.0; 4];
            let error = plan
                .apply_into(&normals, &mut finite_output, &mut workspace)
                .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
            assert_eq!(finite_output, [6.0; 4]);
        }
    }

    #[test]
    fn application_never_returns_a_non_finite_path() {
        let plan = BrownianBridgePlan::new(&[0.0_f64, 4.0]).unwrap();
        let mut output = [7.0];
        let mut workspace = SamplingWorkspace::new();
        let error = plan
            .apply_into(&[f64::MAX], &mut output, &mut workspace)
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(output, [7.0]);
    }
}
