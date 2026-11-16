//! Prepared one- and two-point trigonometric butterflies.
use super::*;

impl<S: scioxide_core::RealScalar> ShortTransform<S> {
    pub(super) fn new(
        n: usize,
        raw_type: KernelType,
        kind: RealTransformKind,
        phases: &[Complex<S>],
        scaling: Scaling<S>,
    ) -> Option<Self> {
        if n <= 2
            && matches!(
                raw_type,
                KernelType::Two | KernelType::Three | KernelType::Four
            )
        {
            Some(Self {
                kernel: if n == 1 {
                    ShortKernel::One
                } else {
                    match raw_type {
                        KernelType::Two => ShortKernel::Two(phases[1].re),
                        KernelType::Three => ShortKernel::Three(S::TWO * phases[1].re),
                        KernelType::Four => ShortKernel::Four(phases[0]),
                        _ => unreachable!(),
                    }
                },
                scaling,
                sine: matches!(kind, RealTransformKind::Dst(_)),
            })
        } else {
            None
        }
    }

    /// Range-scale one or two samples by `2^-exponent`, transform them and
    /// restore the result, rejecting non-finite outputs.
    pub(super) fn scaled(self, input: &[S], exponent: i32) -> Result<[S; 2], SciOxideError> {
        let down = BinaryScale::new(-exponent);
        let restore = BinaryScale::new(exponent);
        let input = [
            down.apply(input[0]),
            input.get(1).map_or(S::ZERO, |&v| down.apply(v)),
        ];
        let result = self.apply(input).map(|v| restore.restore(v));
        if result.iter().any(|v| !v.is_finite()) {
            return Err(crate::validation::real_transform_overflow());
        }
        Ok(result)
    }

    /// Transform every lane of `run` in place. Bounded lanes are already in the
    /// parent's frame; checked lanes are scaled as by [`RealTransformPlan::execute_into`].
    pub(super) fn apply_run(
        self,
        values: &mut [S],
        run: Run,
        mode: LaneMode,
    ) -> Result<(), SciOxideError> {
        for lane in 0..run.count {
            let start = run.first + lane * run.step;
            let input = [
                values[start],
                if run.len == 2 {
                    values[start + run.stride]
                } else {
                    S::ZERO
                },
            ];
            let output = match mode {
                LaneMode::Bounded => self.apply(input),
                LaneMode::Checked => match lane_exponent(&input[..run.len])? {
                    Some(exponent) => self.scaled(&input[..run.len], exponent)?,
                    None => [S::ZERO; 2],
                },
            };
            values[start] = output[0];
            if run.len == 2 {
                values[start + run.stride] = output[1];
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn apply(self, mut input: [S; 2]) -> [S; 2] {
        let s = self.scaling;
        if matches!(self.kernel, ShortKernel::One) {
            return [
                input[0] * s.input_first * s.input_last * s.output_single,
                S::ZERO,
            ];
        }
        if self.sine {
            if matches!(self.kernel, ShortKernel::Two(_)) {
                input[1] = -input[1];
            } else {
                input.swap(0, 1);
            }
        }
        let a = input[0] * s.input_first;
        let b = input[1] * s.input_last;
        let [a, b] = match self.kernel {
            ShortKernel::One => unreachable!(),
            ShortKernel::Two(phase) => [a + b, (a - b) * phase],
            ShortKernel::Three(phase) => {
                let b = b * phase;
                [a + b, a - b]
            }
            ShortKernel::Four(phase) => [a * phase.re - b * phase.im, -a * phase.im - b * phase.re],
        };
        let mut output = [a * s.output_first, b * s.output_last];
        if self.sine {
            if matches!(self.kernel, ShortKernel::Two(_)) {
                output.swap(0, 1);
            } else {
                output[1] = -output[1];
            }
        }
        output
    }
}
