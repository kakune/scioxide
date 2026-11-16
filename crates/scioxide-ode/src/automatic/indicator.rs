//! Power-iteration estimates of the spectral radius, kept across steps.
use crate::safety::{finite, scaled_rms_norm, state_difference, state_increment};
use crate::stepper::{StepRequest, admit_provider, rhs};
use crate::stiff::kernel::StageKernel;
use crate::traits::OdeSystem;
use scioxide_core::{
    EvaluationRole, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError, SolveContext,
};
use scioxide_linalg::Vector;

/// Longest spacing of sampled products, in accepted steps.
const MAX_SPACING: usize = 12;

fn checked_rate<S: RealScalar>(image: S, direction: S) -> Result<S, SciOxideError> {
    let rate = image / direction;
    if !rate.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "stiffness rate is not finite",
        ));
    }
    Ok(rate)
}

/// Power iteration on `J = ∂f/∂y` with one product per estimate and a direction
/// that persists across steps. With power ratios `rₖ = ‖Jvₖ‖/‖vₖ‖` and
/// `vₖ₊₁ = Jvₖ/‖Jvₖ‖`, the estimate is `√(rₖ·rₖ₋₁) = (‖J²vₖ₋₁‖/‖vₖ₋₁‖)^½`,
/// which converges to the spectral radius also for a dominant `±λ` pair or a
/// complex-conjugate pair, where a single ratio alternates.
///
/// The active stiff method's cached linearization supplies free products at
/// every base where it is current. Otherwise a sampled product comes from
/// [`OdeSystem::jacobian_product`] or a forward difference of `f` along the
/// direction, at most once per accepted base. Samples are spaced in accepted
/// steps: the spacing starts at one, doubles after each estimate up to
/// [`MAX_SPACING`], and restarts at one when an estimate inside the caller's
/// switching band differs from its predecessor by more than a quarter. Between
/// estimates the last one stands. Trajectory secants are not used: along the
/// slow solution of a stiff problem they underestimate the spectral radius by
/// orders of magnitude, and endpoint secants of a nonautonomous `f` include its
/// time variation.
pub(super) struct Indicator<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    direction: Vector<T>,
    image: Vector<T>,
    perturbed: Vector<T>,
    rhs: Vector<T>,
    scratch: Vector<T>,
    sampled_at: Option<S>,
    /// Accepted steps since the last estimate.
    since: usize,
    /// Accepted steps from one sampled estimate to the next.
    spacing: usize,
    /// Last estimate `√(rₖ·rₖ₋₁)`.
    rate: Option<S>,
    /// Last power ratio `rₖ`.
    ratio: Option<S>,
    /// The system may still supply directional products in this epoch.
    product: bool,
    /// `rhs` holds `f` at this base time, evaluated for a difference product.
    base_at: Option<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for Indicator<S, T>
{
    fn default() -> Self {
        Self {
            direction: Vector::empty(),
            image: Vector::empty(),
            perturbed: Vector::empty(),
            rhs: Vector::empty(),
            scratch: Vector::empty(),
            sampled_at: None,
            since: 0,
            spacing: 1,
            rate: None,
            ratio: None,
            product: true,
            base_at: None,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Indicator<S, T> {
    pub(super) fn fits(&self, n: usize, scratch: usize) -> bool {
        [&self.direction, &self.image, &self.perturbed, &self.rhs]
            .iter()
            .all(|v| v.len() == n)
            && self.scratch.len() == scratch
    }

    pub(super) fn reserve(&mut self, n: usize, scratch: usize) -> Result<(), SciOxideError> {
        for v in [
            &mut self.direction,
            &mut self.image,
            &mut self.perturbed,
            &mut self.rhs,
        ] {
            v.try_resize(n, T::ZERO)?;
        }
        self.scratch.try_resize(scratch, T::ZERO)
    }

    /// Begin an epoch: forget estimates, the direction and unavailable sources.
    pub(super) fn reset(&mut self) {
        self.sampled_at = None;
        self.since = 0;
        self.spacing = 1;
        self.rate = None;
        self.ratio = None;
        self.product = true;
        self.base_at = None;
        self.reset_direction();
    }

    fn reset_direction(&mut self) {
        for (i, v) in self.direction.as_mut_slice().iter_mut().enumerate() {
            *v = T::from_real(if i % 2 == 0 { S::ONE } else { -S::ONE });
        }
    }

    /// Count an accepted step toward the next sample.
    pub(super) fn accept(&mut self) {
        self.since = self.since.saturating_add(1);
    }

    /// `f(time, y)` when a difference product evaluated it at this base.
    pub(super) fn base_derivative(&self, time: S) -> Option<&[T]> {
        (self.base_at == Some(time)).then_some(self.rhs.as_slice())
    }

    /// Advance the power estimate at a new base. A free cached product is
    /// always used; a callback or difference product only once the spacing has
    /// elapsed. `band` tells whether an estimate lies in the switching band.
    /// Provider errors propagate; an unavailable product callback is not
    /// asked again in this epoch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn estimate<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        time: S,
        state: &[T],
        base_rhs: Option<&[T]>,
        kernel: Option<&StageKernel<S, T>>,
        band: impl Fn(S) -> bool,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        if self.sampled_at == Some(time) || state.is_empty() {
            return Ok(self.rate);
        }
        self.sampled_at = Some(time);
        let sample = self.rate.is_none() || self.since >= self.spacing;
        if self.direction.as_slice().iter().all(|&v| v == T::ZERO) {
            self.reset_direction();
        }
        let direction_norm = scaled_rms_norm(self.direction.as_slice())?;
        self.image.as_mut_slice().fill(T::from_real(S::NAN));
        let mut product = false;
        let mut available = match kernel {
            Some(kernel) => kernel.cached_product(
                time,
                state,
                self.direction.as_slice(),
                self.image.as_mut_slice(),
                request,
                cx,
            )?,
            None => false,
        };
        if !available && sample && self.product {
            // Products use the backend's explicitly reserved provider scratch.
            admit_provider(cx, EvaluationRole::LinearOperator, request.trace)?;
            product = request
                .trace
                .provider(system.jacobian_product(
                    time,
                    state,
                    self.direction.as_slice(),
                    self.image.as_mut_slice(),
                    self.scratch.as_mut_slice(),
                    cx,
                ))?
                .is_some();
            self.product = product;
            available = product;
        }
        let ratio = if available {
            finite(
                self.image.as_slice(),
                if product {
                    NonFiniteOrigin::Callback
                } else {
                    NonFiniteOrigin::Intermediate
                },
            )?;
            let image_norm = scaled_rms_norm(self.image.as_slice())?;
            let ratio = checked_rate(image_norm, direction_norm)?;
            self.follow_image(image_norm);
            ratio
        } else if sample {
            self.probe(system, time, state, base_rhs, direction_norm, request, cx)?
        } else {
            return Ok(self.rate);
        };
        let rate = self
            .ratio
            .replace(ratio)
            .map_or(ratio, |old| ratio.sqrt() * old.sqrt());
        let four = S::TWO + S::TWO;
        let changed = self.rate.is_none_or(|old| (rate - old).abs() * four > old);
        self.spacing = if changed && band(rate) {
            1
        } else {
            self.spacing.saturating_mul(2).min(MAX_SPACING)
        };
        self.since = 0;
        self.rate = Some(rate);
        Ok(self.rate)
    }

    fn follow_image(&mut self, image_norm: S) {
        if image_norm > S::ZERO {
            for (direction, &image) in self
                .direction
                .as_mut_slice()
                .iter_mut()
                .zip(self.image.as_slice())
            {
                *direction = image / T::from_real(image_norm);
            }
        }
    }

    /// Forward difference `(f(y + δv) − f(y))/δ` along the direction `v`.
    #[allow(clippy::too_many_arguments)]
    fn probe<P: OdeSystem<S, T> + ?Sized>(
        &mut self,
        system: &P,
        time: S,
        state: &[T],
        base_rhs: Option<&[T]>,
        direction_norm: S,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        let width = S::EPSILON.sqrt() * scaled_rms_norm(state)?.max(S::ONE);
        for ((out, &value), &direction) in self
            .perturbed
            .as_mut_slice()
            .iter_mut()
            .zip(state)
            .zip(self.direction.as_slice())
        {
            *out = state_increment(value, width / direction_norm, direction)?;
        }
        let base = if let Some(base) = base_rhs {
            base
        } else {
            self.rhs.as_mut_slice().fill(T::from_real(S::NAN));
            rhs(
                system,
                time,
                state,
                self.rhs.as_mut_slice(),
                false,
                request,
                cx,
            )?;
            finite(self.rhs.as_slice(), NonFiniteOrigin::Callback)?;
            self.base_at = Some(time);
            self.rhs.as_slice()
        };
        self.image.as_mut_slice().fill(T::from_real(S::NAN));
        rhs(
            system,
            time,
            self.perturbed.as_slice(),
            self.image.as_mut_slice(),
            false,
            request,
            cx,
        )?;
        finite(self.image.as_slice(), NonFiniteOrigin::Callback)?;
        for (image, &base) in self.image.as_mut_slice().iter_mut().zip(base) {
            *image = state_difference(*image, base)?;
        }
        let image_norm = scaled_rms_norm(self.image.as_slice())?;
        let rate = checked_rate(image_norm, width)?;
        self.follow_image(image_norm);
        Ok(rate)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stepper::{StepTrace, testing::with_context};
    use alloc::vec::Vec;
    use core::cell::Cell;
    use scioxide_linalg::MatrixViewMut;

    #[derive(Clone, Copy, PartialEq)]
    enum Source {
        None,
        Analytic,
        Product,
        BadProduct,
        Error,
    }
    /// `y' = -λy`; every product multiplies `λ` by `growth` first.
    struct System {
        source: Source,
        rate: Cell<f64>,
        growth: f64,
        calls: Cell<usize>,
        products: Cell<usize>,
        jacobians: Cell<usize>,
    }
    impl OdeSystem<f64> for System {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            _: f64,
            state: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            self.calls.set(self.calls.get() + 1);
            out[0] = -self.rate.get() * state[0];
            Ok(())
        }
        fn jacobian(
            &self,
            _: f64,
            _: &[f64],
            j: &mut MatrixViewMut<'_, f64>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            self.jacobians.set(self.jacobians.get() + 1);
            j.set(0, 0, -self.rate.get())?;
            Ok(Some(()))
        }
        fn jacobian_product(
            &self,
            _: f64,
            _: &[f64],
            v: &[f64],
            out: &mut [f64],
            scratch: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            self.products.set(self.products.get() + 1);
            match self.source {
                Source::Product => {
                    assert_eq!(scratch.len(), 3);
                    self.rate.set(self.rate.get() * self.growth);
                    out[0] = -self.rate.get() * v[0];
                    Ok(Some(()))
                }
                Source::BadProduct => {
                    out[0] = f64::INFINITY;
                    Ok(Some(()))
                }
                Source::Error => Err(crate::safety::invalid(
                    scioxide_core::argument!("jacobian_product"),
                    "provider failed",
                )),
                Source::None | Source::Analytic => Ok(None),
            }
        }
    }
    fn system(source: Source, growth: f64) -> System {
        System {
            source,
            rate: Cell::new(1.),
            growth,
            calls: Cell::new(0),
            products: Cell::new(0),
            jacobians: Cell::new(0),
        }
    }
    fn indicator(n: usize) -> Indicator<f64, f64> {
        let mut result = Indicator::default();
        result.reserve(n, 3).unwrap();
        result.reset();
        result
    }
    /// The accepted bases among `0..bases` that sampled a product.
    fn sampled_bases(system: &System, band: bool, bases: u32) -> Vec<u32> {
        let mut indicator = indicator(1);
        let trace = StepTrace::default();
        let mut sampled = Vec::new();
        with_context(|cx| {
            for index in 0..bases {
                let before = system.products.get();
                indicator.estimate(
                    system,
                    f64::from(index),
                    &[1.],
                    None,
                    None,
                    |_| band,
                    StepRequest::new(&trace),
                    cx,
                )?;
                if system.products.get() > before {
                    sampled.push(index);
                }
                indicator.accept();
            }
            Ok(())
        })
        .unwrap();
        sampled
    }

    #[test]
    fn spacing_doubles_to_its_cap_and_restarts_on_change_in_the_band() {
        // Stable estimates double the spacing up to twelve; in the band the first
        // estimate, which has no predecessor, counts as a change.
        for (band, bases) in [
            (false, &[0, 2, 6, 14, 26, 38, 50][..]),
            (true, &[0, 1, 3, 7, 15, 27, 39, 51][..]),
        ] {
            let system = system(Source::Product, 1.);
            assert_eq!(sampled_bases(&system, band, 52), bases);
            assert_eq!(system.calls.get(), 0);
        }
        // Estimates that keep changing restart the spacing only inside the band.
        assert_eq!(
            sampled_bases(&system(Source::Product, 2.), true, 6),
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            sampled_bases(&system(Source::Product, 2.), false, 15),
            [0, 2, 6, 14]
        );
    }

    #[test]
    fn difference_probes_follow_the_spacing_reuse_base_derivatives_and_skip_matrices() {
        for (source, endpoint) in [
            (Source::None, None),
            (Source::None, Some(&[-1.][..])),
            (Source::Analytic, None),
        ] {
            let system = system(source, 1.);
            let mut indicator = indicator(1);
            let trace = StepTrace::default();
            with_context(|cx| {
                for index in 0..17 {
                    for _retry in 0..4 {
                        let time = f64::from(index);
                        let rate = indicator.estimate(
                            &system,
                            time,
                            &[1.],
                            endpoint,
                            None,
                            |_| false,
                            StepRequest::new(&trace),
                            cx,
                        )?;
                        assert!((rate.unwrap() - 1.).abs() < 1e-7);
                        let base = indicator.base_derivative(time);
                        if endpoint.is_none() && [0, 2, 6, 14].contains(&index) {
                            assert_eq!(base, Some(&[-1.][..]));
                        } else {
                            assert!(base.is_none());
                        }
                    }
                    indicator.accept();
                }
                // Four probes, each with a base evaluation unless one is supplied.
                let expected = if endpoint.is_some() { 4 } else { 8 };
                assert_eq!(system.calls.get(), expected);
                assert_eq!(
                    cx.counts().own().provider(EvaluationRole::RightHandSide),
                    expected
                );
                // An unavailable product callback is asked once per epoch.
                assert_eq!(system.products.get(), 1);
                assert_eq!(
                    cx.counts().own().provider(EvaluationRole::LinearOperator),
                    1
                );
                Ok(())
            })
            .unwrap();
            assert_eq!(system.jacobians.get(), 0);
            indicator.reset();
            with_context(|cx| {
                indicator.estimate(
                    &system,
                    0.,
                    &[1.],
                    endpoint,
                    None,
                    |_| false,
                    StepRequest::new(&trace),
                    cx,
                )?;
                Ok(())
            })
            .unwrap();
            assert_eq!(system.products.get(), 2);
        }
    }

    /// `x'' = -ω²x` as a first-order system with analytic products.
    struct Oscillator(f64);
    impl OdeSystem<f64> for Oscillator {
        fn dimension(&self) -> usize {
            2
        }
        fn rhs(
            &self,
            _: f64,
            y: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[y[1], -self.0 * self.0 * y[0]]);
            Ok(())
        }
        fn jacobian_product(
            &self,
            _: f64,
            _: &[f64],
            v: &[f64],
            out: &mut [f64],
            _: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            out.copy_from_slice(&[v[1], -self.0 * self.0 * v[0]]);
            Ok(Some(()))
        }
    }

    #[test]
    fn two_power_ratios_resolve_a_purely_imaginary_pair() {
        // Single ratios alternate between about ω²/√2 and √2; their geometric mean is ω.
        let mut indicator = indicator(2);
        let trace = StepTrace::default();
        let rates = with_context(|cx| {
            let mut rates = Vec::new();
            for index in 0..4 {
                let rate = indicator.estimate(
                    &Oscillator(10.),
                    f64::from(index),
                    &[1., 0.],
                    None,
                    None,
                    |_| true,
                    StepRequest::new(&trace),
                    cx,
                )?;
                rates.push(rate.unwrap());
                indicator.accept();
            }
            Ok(rates)
        })
        .unwrap();
        assert!((rates[0] - 5000.5_f64.sqrt()).abs() < 1e-10, "{rates:?}");
        for rate in &rates[1..] {
            assert!((rate - 10.).abs() < 1e-12, "{rates:?}");
        }
    }

    #[test]
    fn invalid_products_propagate_without_probing() {
        for source in [Source::BadProduct, Source::Error] {
            let system = system(source, 1.);
            let mut indicator = indicator(1);
            let trace = StepTrace::default();
            let error = with_context(|cx| {
                indicator
                    .estimate(
                        &system,
                        0.,
                        &[1.],
                        None,
                        None,
                        |_| false,
                        StepRequest::new(&trace),
                        cx,
                    )
                    .map(drop)
            })
            .unwrap_err();
            if source == Source::BadProduct {
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Callback,
                        ..
                    }
                ));
            } else {
                assert!(trace.provider_failed.get());
            }
            assert_eq!(system.calls.get(), 0);
        }
    }
}
