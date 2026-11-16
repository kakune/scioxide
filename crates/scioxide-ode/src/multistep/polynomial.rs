//! Small normalized Newton polynomials for actual, nonuniform step history.
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

pub(crate) const CAPACITY: usize = 8;

/// A bounded Newton polynomial. Extended-range differences avoid intermediate
/// overflow when the final state or integral is representable.
pub(crate) struct Polynomial<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    pub(crate) coefficients: [crate::safety::StageValue<T>; CAPACITY],
    nodes: [S; CAPACITY],
    len: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Polynomial<S, T> {
    pub(crate) fn new(nodes: &[S], value: impl Fn(usize) -> T) -> Result<Self, SciOxideError> {
        let mut result = Self {
            coefficients: [crate::safety::StageValue::zero(); CAPACITY],
            nodes: [S::ZERO; CAPACITY],
            len: nodes.len(),
        };
        result.nodes[..nodes.len()].copy_from_slice(nodes);
        for (i, coefficient) in result.coefficients[..nodes.len()].iter_mut().enumerate() {
            *coefficient = crate::safety::StageValue::from_value(value(i))?;
        }
        for order in 1..nodes.len() {
            for i in (order..nodes.len()).rev() {
                let denominator = crate::safety::StageValue::from_value(nodes[i])?
                    .sub(crate::safety::StageValue::from_value(nodes[i - order])?)?;
                if nodes[i] == nodes[i - order] {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "history nodes coincide at working precision",
                    ));
                }
                result.coefficients[i] = result.coefficients[i]
                    .sub(result.coefficients[i - 1])?
                    .div(denominator)?;
            }
        }
        Ok(result)
    }
    pub(crate) fn value(&self, x: S) -> Result<T, SciOxideError> {
        let mut result = crate::safety::StageValue::zero();
        for i in (0..self.len).rev() {
            let difference = crate::safety::StageValue::from_value(x)?
                .sub(crate::safety::StageValue::from_value(self.nodes[i])?)?;
            result = result.mul(difference)?.add(self.coefficients[i])?;
        }
        result.round()
    }
    /// Differentiate the same Newton polynomial without converting its basis.
    pub(crate) fn derivative(&self, x: S) -> Result<T, SciOxideError> {
        let mut value = crate::safety::StageValue::zero();
        let mut derivative = crate::safety::StageValue::zero();
        for i in (0..self.len).rev() {
            let distance = crate::safety::StageValue::from_value(x)?
                .sub(crate::safety::StageValue::from_value(self.nodes[i])?)?;
            derivative = derivative.mul(distance)?.add(value)?;
            value = value.mul(distance)?.add(self.coefficients[i])?;
        }
        derivative.round()
    }
    pub(crate) fn powers(&self) -> Result<[crate::safety::StageValue<T>; CAPACITY], SciOxideError> {
        let mut powers = [crate::safety::StageValue::zero(); CAPACITY];
        for i in (0..self.len).rev() {
            for j in (1..self.len).rev() {
                powers[j] = powers[j - 1]
                    .sub(powers[j].mul(crate::safety::StageValue::from_value(self.nodes[i])?)?)?;
            }
            powers[0] = self.coefficients[i]
                .sub(powers[0].mul(crate::safety::StageValue::from_value(self.nodes[i])?)?)?;
        }
        Ok(powers)
    }
    pub(crate) fn integral(&self) -> Result<T, SciOxideError> {
        let mut sum = crate::safety::StageValue::zero();
        for (i, coefficient) in self.powers()?.iter().enumerate().take(self.len) {
            sum = sum.add(coefficient.div(crate::safety::StageValue::from_value(
                S::try_from_usize_exact(i + 1)?,
            )?)?)?;
        }
        sum.round()
    }
}

pub(crate) fn coordinate<S: RealScalar>(time: S, start: S, step: S) -> Result<S, SciOxideError> {
    let difference = time - start;
    let native = difference / step;
    if native.is_finite() && (difference.is_normal() || time == start) {
        return Ok(native);
    }
    Scaled::from_value(time)?
        .sub(Scaled::from_value(start)?)?
        .div(Scaled::from_value(step)?)?
        .round()
}

/// Accepts bounded, separated IEEE history nodes for normal-range preparation.
/// At most seven nodes in [-8, 0], separated by 1/4, keep the small Newton
/// denominator products between 2^-14 and 2^23, and their reciprocals between
/// 2^-23 and 2^14. These bounds select an arithmetic path, not an error tolerance.
pub(crate) fn ordinary_geometry<S: RealScalar + NumericScalar<Real = S>>(old: &[S]) -> bool {
    if old.is_empty() || old.len() > 7 {
        return false;
    }
    if !matches!(
        S::kind(),
        scioxide_core::backend::ScalarKind::F32 | scioxide_core::backend::ScalarKind::F64
    ) {
        return false;
    }
    let eight = S::TWO * S::TWO * S::TWO;
    let gap = S::ONE / (S::TWO * S::TWO);
    old[0] == S::ZERO
        && old[old.len() - 1] >= -eight
        && S::ONE - old[0] >= gap
        && old.windows(2).all(|pair| pair[0] - pair[1] >= gap)
}

/// Integral of the monic node product, used in the interpolation remainder.
#[inline]
pub(crate) fn integral_product<S: RealScalar + NumericScalar<Real = S>>(
    nodes: &[S],
) -> Result<S, SciOxideError> {
    let mut powers = [S::ZERO; CAPACITY];
    powers[0] = S::ONE;
    for (degree, &node) in nodes.iter().enumerate() {
        for j in (1..=degree + 1).rev() {
            powers[j] = scioxide_core::fast_mul_add(-node, powers[j], powers[j - 1]);
        }
        powers[0] = -node * powers[0];
    }
    let mut integral = S::ZERO;
    let mut denominator = S::ONE;
    for &coefficient in powers.iter().take(nodes.len() + 1) {
        integral += coefficient / denominator;
        // Degrees are bounded by CAPACITY, so these small integers are exact
        // in every supported scalar without repeated fallible conversions.
        denominator += S::ONE;
    }
    if integral.is_finite() {
        return Ok(integral);
    }
    integral_product_fallback(nodes)
}

#[cold]
fn integral_product_fallback<S: RealScalar + NumericScalar<Real = S>>(
    nodes: &[S],
) -> Result<S, SciOxideError> {
    let mut extended = [S::ZERO; CAPACITY];
    extended[..nodes.len()].copy_from_slice(nodes);
    extended[nodes.len()] = S::ONE;
    // Construct the monic Newton term directly; no extra distinct node needed.
    let mut polynomial = Polynomial::new(&[], |_| S::ZERO)?;
    polynomial.nodes = extended;
    polynomial.len = nodes.len() + 1;
    polynomial.coefficients[nodes.len()] = crate::safety::StageValue::from_value(S::ONE)?;
    polynomial.integral()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_geometry_admits_only_bounded_separated_ieee_nodes() {
        assert!(ordinary_geometry(&[0.0_f64, -0.25, -8.0]));
        assert!(ordinary_geometry(&[0.0_f32, -0.25, -8.0]));
        assert!(!ordinary_geometry::<f64>(&[]));
        assert!(!ordinary_geometry(&[0.0_f64, -0.125]));
        assert!(!ordinary_geometry(&[0.0_f64, -8.000_000_000_000_002]));
        assert!(!ordinary_geometry(&[0.0_f64, f64::NAN]));
    }

    #[test]
    fn replacing_scaled_geometry_preserves_only_the_new_native_row() {
        let mut weights = Weights::new(&[0.0_f64, 1e-310], Functional::Derivative).unwrap();
        assert!((weights.affine(0.0, 1.0, |i| [0.0, 1e-310][i]).unwrap() - 1.0).abs() < 1e-14);
        weights.replace_native([0.5; CAPACITY], 2, 1.0);
        assert!((weights.affine(0.0, 1.0, |i| [1.0, 3.0][i]).unwrap() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn prepared_functionals_reproduce_nonuniform_monomials() {
        let nodes = [1.0_f64, 0.0, -0.5, -2.0, -2.25, -4.0];
        for degree in 0..=5 {
            for (functional, expected) in [
                (Functional::Value(0.3), 0.3_f64.powi(degree as i32)),
                (Functional::Integral, 1.0 / (degree + 1) as f64),
                (Functional::Derivative, degree as f64),
                (Functional::Coefficient(degree), 1.0),
            ] {
                let weights = Weights::new(&nodes[..=degree], functional).unwrap();
                let value = weights
                    .affine(0.0, 1.0, |i| nodes[i].powi(degree as i32))
                    .unwrap();
                assert!(
                    (value - expected).abs() < 2e-11,
                    "degree={degree}, actual={value}, expected={expected}"
                );
            }
        }
        let derivative = Weights::new(&nodes, Functional::Coefficient(5)).unwrap();
        assert_eq!(derivative.affine(0.0, 1.0, |_| 1e300).unwrap(), 0.0);
        assert!(Weights::new(&[1.0, 1.0], Functional::Derivative).is_err());
    }

    #[test]
    fn prepared_weights_retain_exceptional_geometry_and_values() {
        let weights = Weights::new(&[0.0_f64, 1e-310], Functional::Value(1.0)).unwrap();
        assert!((weights.affine(0.0, 1.0, |i| [0.0, 1e-310][i]).unwrap() - 1.0).abs() < 1e-14);
        let weights = Weights::new(&[0.0_f64, 1.0], Functional::Value(0.5)).unwrap();
        assert_eq!(
            weights
                .affine(0.0, 1.0, |i| [-f64::MAX, f64::MAX][i])
                .unwrap(),
            0.0
        );
        let values = [0.0_f64, f64::from_bits(1)];
        let weights = Weights::new(&[0.0, 1.0], Functional::Value(0.5)).unwrap();
        let restored = weights.affine(0.0, f64::MAX, |i| values[i]).unwrap();
        assert!((restored - (f64::from_bits(1) * f64::MAX) * 0.5).abs() < 1e-30);
        let weights = Weights::new(&[0.0_f64, 1.0], Functional::Derivative)
            .unwrap()
            .multiply(f64::from_bits(1))
            .unwrap();
        let result = weights.affine(0.0, 0.5, |i| [0.0, f64::MAX][i]).unwrap();
        assert!((result - (f64::from_bits(1) * f64::MAX) * 0.5).abs() < 1e-30);
    }

    #[test]
    fn centered_blocks_replay_underflow_before_large_step_scaling() {
        let weights = Weights::new(&[0.0_f64, 1.0], Functional::Value(0.5)).unwrap();
        for dimension in [1, 4, 7] {
            let origin = alloc::vec![0.0; dimension];
            let row: Vec<_> = (0..dimension)
                .map(|i| match i % 3 {
                    0 => f64::from_bits(1),
                    1 => f64::from_bits(3),
                    _ => 1.0,
                })
                .collect();
            for step in [f64::MAX, -f64::MAX] {
                let mut output = alloc::vec![0.0; dimension];
                weights
                    .apply(None, step, &[&origin, &row], &mut output)
                    .unwrap();
                for (&actual, &value) in output.iter().zip(&row) {
                    let expected = (value * step) * 0.5;
                    assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs());
                }
            }
        }
    }

    #[test]
    fn nonuniform_polynomial_reproduction() {
        let nodes = [1.0_f64, 0.0, -0.5, -2.0, -2.25, -4.0];
        for degree in 0..=5 {
            let p =
                Polynomial::new(&nodes[..degree + 1], |i| nodes[i].powi(degree as i32)).unwrap();
            assert!((p.integral().unwrap() - 1.0 / (degree + 1) as f64).abs() < 1e-12);
            assert!((p.value(0.3).unwrap() - 0.3_f64.powi(degree as i32)).abs() < 1e-12);
            assert!((p.derivative(1.0).unwrap() - degree as f64).abs() < 1e-11);
            if degree > 0 {
                assert!(
                    (p.derivative(0.3).unwrap() - degree as f64 * 0.3_f64.powi(degree as i32 - 1))
                        .abs()
                        < 1e-11
                );
            }
        }
    }
}

/// Linear functional of a polynomial in normalized time.
#[derive(Clone, Copy)]
pub(crate) enum Functional<S> {
    Value(S),
    Derivative,
    Integral,
    Coefficient(usize),
}

/// Prepared actual-node weights. Native geometry and application are the usual
/// path. Scaled coefficients use fallible storage only for exceptional
/// geometry, keeping ordinary rows small and free of allocation.
#[derive(Debug)]
pub(crate) struct Weights<S: RealScalar> {
    native: [S; CAPACITY],
    extended: Vec<Scaled<S>>,
    len: usize,
    constant: S,
}
impl<S: RealScalar> Weights<S> {
    pub(crate) fn from_native(native: [S; CAPACITY], len: usize, constant: S) -> Self {
        Self {
            native,
            extended: Vec::new(),
            len,
            constant,
        }
    }

    /// Replace a prepared native row without moving or dropping its storage.
    pub(crate) fn replace_native(&mut self, native: [S; CAPACITY], len: usize, constant: S) {
        self.native = native;
        self.extended.clear();
        self.len = len;
        self.constant = constant;
    }

    #[inline]
    pub(crate) fn new(nodes: &[S], functional: Functional<S>) -> Result<Self, SciOxideError> {
        if matches!(functional, Functional::Coefficient(degree) if degree + 1 == nodes.len()) {
            let mut native = [S::ZERO; CAPACITY];
            let mut ordinary = true;
            for (j, &node) in nodes.iter().enumerate() {
                let mut denominator = S::ONE;
                for (k, &other) in nodes.iter().enumerate() {
                    if j == k {
                        continue;
                    }
                    if node == other {
                        return Err(coincident_nodes());
                    }
                    denominator *= node - other;
                }
                ordinary &= denominator.is_normal();
                native[j] = S::ONE / denominator;
                ordinary &= native[j].is_normal();
            }
            let extended = if ordinary {
                Vec::new()
            } else {
                extended_weights(nodes, functional)?
            };
            return Ok(Self {
                native,
                extended,
                len: nodes.len(),
                constant: if nodes.len() == 1 { S::ONE } else { S::ZERO },
            });
        }
        let mut basis = [S::ZERO; CAPACITY];
        basis[0] = S::ONE;
        let mut native = [S::ZERO; CAPACITY];
        for (k, &node) in nodes.iter().enumerate() {
            native[k] = match functional {
                Functional::Value(x) => basis[..=k]
                    .iter()
                    .rev()
                    .fold(S::ZERO, |sum, &v| scioxide_core::fast_mul_add(sum, x, v)),
                Functional::Coefficient(degree) => basis[degree],
                Functional::Integral | Functional::Derivative => {
                    let mut sum = S::ZERO;
                    let mut degree = S::ZERO;
                    for &v in basis.iter().take(k + 1) {
                        sum += match functional {
                            Functional::Derivative => v * degree,
                            _ => v / (degree + S::ONE),
                        };
                        degree += S::ONE;
                    }
                    sum
                }
            };
            if k + 1 < nodes.len() {
                for j in (1..=k + 1).rev() {
                    basis[j] = scioxide_core::fast_mul_add(-node, basis[j], basis[j - 1]);
                }
                basis[0] = -node * basis[0];
            }
        }
        let mut ordinary = true;
        for order in (1..nodes.len()).rev() {
            for i in order..nodes.len() {
                if nodes[i] == nodes[i - order] {
                    return Err(coincident_nodes());
                }
                let difference = nodes[i] - nodes[i - order];
                ordinary &= difference.is_normal();
                let value = native[i] / difference;
                native[i] = value;
                native[i - 1] -= value;
            }
        }
        ordinary &= native
            .iter()
            .all(|v| v.is_finite() && (*v == S::ZERO || v.is_normal()));
        let extended = if ordinary {
            Vec::new()
        } else {
            extended_weights(nodes, functional)?
        };
        let constant = match functional {
            Functional::Value(_) | Functional::Integral | Functional::Coefficient(0) => S::ONE,
            _ => S::ZERO,
        };
        Ok(Self {
            native,
            extended,
            len: nodes.len(),
            constant,
        })
    }

    pub(crate) fn coefficients(&self) -> Result<[S; CAPACITY], SciOxideError> {
        if self.extended.is_empty() {
            return Ok(self.native);
        }
        let mut result = [S::ZERO; CAPACITY];
        for (out, &value) in result.iter_mut().zip(&self.extended) {
            *out = value.round()?;
        }
        Ok(result)
    }

    /// Fold a geometry-only error factor into the weights before applying lanes.
    pub(crate) fn multiply(mut self, factor: S) -> Result<Self, SciOxideError> {
        let old = self.native;
        let mut ordinary = self.extended.is_empty();
        for weight in &mut self.native[..self.len] {
            let original = *weight;
            *weight *= factor;
            ordinary &= weight.is_normal() || original == S::ZERO || factor == S::ZERO;
        }
        if !ordinary {
            if self.extended.is_empty() {
                self.extended = scioxide_core::backend::try_with_capacity(
                    self.len,
                    scioxide_core::quantity!("multistep_extended_weights"),
                )?;
                for &value in &old[..self.len] {
                    self.extended.push(Scaled::from_value(value)?);
                }
            }
            let factor = Scaled::from_value(factor)?;
            for weight in &mut self.extended {
                *weight = weight.mul(factor)?;
            }
        }
        self.constant *= factor;
        Ok(self)
    }

    fn component_native<T: NumericScalar<Real = S>>(
        &self,
        imaginary: bool,
        value: &impl Fn(usize) -> T,
    ) -> S {
        let part = |v: T| {
            if imaginary {
                v.imaginary_component()
            } else {
                v.real_component()
            }
        };
        let base = if self.len == 0 {
            S::ZERO
        } else {
            part(value(0))
        };
        let mut sum = self.constant * base;
        for j in 1..self.len {
            sum = scioxide_core::fast_mul_add(self.native[j], part(value(j)) - base, sum);
        }
        sum
    }

    /// Center on the first sample so constant histories reproduce exactly and
    /// error functionals do not subtract several large rounded state sums.
    pub(crate) fn affine<T: NumericScalar<Real = S>>(
        &self,
        initial: T,
        scale: S,
        value: impl Fn(usize) -> T,
    ) -> Result<T, SciOxideError> {
        let component = |imaginary| {
            let part = |v: T| {
                if imaginary {
                    v.imaginary_component()
                } else {
                    v.real_component()
                }
            };
            let sum = self.component_native(imaginary, &value);
            let result = scioxide_core::fast_mul_add(scale, sum, part(initial));
            if self.extended.is_empty()
                && sum.is_finite()
                && result.is_finite()
                && (scale.abs() <= S::ONE || sum.is_normal())
            {
                Ok(result)
            } else {
                self.component_scaled(imaginary, &value)?
                    .mul(Scaled::from_value(scale)?)?
                    .add(Scaled::from_value(part(initial))?)?
                    .round()
            }
        };
        let real = component(false)?;
        if matches!(
            T::kind(),
            scioxide_core::backend::ScalarKind::F32 | scioxide_core::backend::ScalarKind::F64
        ) {
            Ok(T::from_real(real))
        } else {
            Ok(T::from_components(real, component(true)?))
        }
    }

    /// Stream prepared weights over equally sized sample rows. Exceptional
    /// coordinates replay through the scalar application using original rows.
    #[inline(always)]
    pub(crate) fn apply<T: NumericScalar<Real = S>>(
        &self,
        initial: Option<&[T]>,
        scale: S,
        rows: &[&[T]],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        if !self.extended.is_empty() {
            for (i, out) in output.iter_mut().enumerate() {
                *out = self.affine(initial.map_or(T::ZERO, |v| v[i]), scale, |j| rows[j][i])?;
            }
            return Ok(());
        }
        crate::safety::combine_rows(
            initial,
            scale,
            rows.first().map(|&row| (self.constant, row)),
            self.len.saturating_sub(1),
            |j| (self.native[j + 1], rows[j + 1]),
            output,
            |i, old| self.affine(old, scale, |j| rows[j][i]),
        )
    }

    /// Native coefficient application with original-operand range checks.
    pub(crate) fn native_component<T: NumericScalar<Real = S>>(
        &self,
        imaginary: bool,
        value: impl Fn(usize) -> T,
    ) -> Option<S> {
        use crate::dense_output::{component, native_product};
        if !self.extended.is_empty() {
            return None;
        }
        let base = if self.len == 0 {
            S::ZERO
        } else {
            component(value(0), imaginary)
        };
        let mut sum = native_product(self.constant, base)?;
        for j in 1..self.len {
            let difference = component(value(j), imaginary) - base;
            native_product(self.native[j], difference)?;
            sum = scioxide_core::fast_mul_add(self.native[j], difference, sum);
            if !sum.is_finite() {
                return None;
            }
        }
        Some(sum)
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn component_scaled<T: NumericScalar<Real = S>>(
        &self,
        imaginary: bool,
        value: &impl Fn(usize) -> T,
    ) -> Result<Scaled<S>, SciOxideError> {
        let part = |v: T| {
            if imaginary {
                v.imaginary_component()
            } else {
                v.real_component()
            }
        };
        let base = if self.len == 0 {
            Scaled::ZERO
        } else {
            Scaled::from_value(part(value(0)))?
        };
        let mut sum = scioxide_core::ScaledAccumulator::ZERO;
        sum.add_scaled(base.mul(Scaled::from_value(self.constant)?)?)?;
        for j in 1..self.len {
            let weight = if self.extended.is_empty() {
                Scaled::from_value(self.native[j])?
            } else {
                self.extended[j]
            };
            let difference = Scaled::from_value(part(value(j)))?.sub(base)?;
            sum.add_product(weight, difference)?;
        }
        sum.total()
    }
}

fn coincident_nodes() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "history nodes coincide at working precision",
    )
}

#[cold]
fn extended_weights<S: RealScalar>(
    nodes: &[S],
    functional: Functional<S>,
) -> Result<Vec<Scaled<S>>, SciOxideError> {
    let mut basis = [Scaled::ZERO; CAPACITY];
    basis[0] = Scaled::from_value(S::ONE)?;
    let mut result = [Scaled::ZERO; CAPACITY];
    for (k, &node) in nodes.iter().enumerate() {
        match functional {
            Functional::Value(x) => {
                for &v in basis[..=k].iter().rev() {
                    result[k] = result[k].mul(Scaled::from_value(x)?)?.add(v)?;
                }
            }
            Functional::Coefficient(degree) => result[k] = basis[degree],
            Functional::Integral | Functional::Derivative => {
                for (j, &v) in basis.iter().enumerate().take(k + 1) {
                    let term = match functional {
                        Functional::Derivative => {
                            v.mul(Scaled::from_value(S::try_from_usize_exact(j)?)?)?
                        }
                        _ => v.div(Scaled::from_value(S::try_from_usize_exact(j + 1)?)?)?,
                    };
                    result[k] = result[k].add(term)?;
                }
            }
        }
        if k + 1 < nodes.len() {
            for j in (1..=k + 1).rev() {
                basis[j] = basis[j - 1].sub(basis[j].mul(Scaled::from_value(node)?)?)?;
            }
            basis[0] = basis[0].mul(Scaled::from_value(-node)?)?;
        }
    }
    for order in (1..nodes.len()).rev() {
        for i in order..nodes.len() {
            if nodes[i] == nodes[i - order] {
                return Err(coincident_nodes());
            }
            let difference =
                Scaled::from_value(nodes[i])?.sub(Scaled::from_value(nodes[i - order])?)?;
            let value = result[i].div(difference)?;
            result[i] = value;
            result[i - 1] = result[i - 1].sub(value)?;
        }
    }
    let mut weights = scioxide_core::backend::try_with_capacity(
        nodes.len(),
        scioxide_core::quantity!("multistep_extended_weights"),
    )?;
    weights.extend_from_slice(&result[..nodes.len()]);
    Ok(weights)
}

/// Derivative at one or integral on [0, 1], prepared once from actual nodes.
pub(crate) fn weights<S: RealScalar>(
    nodes: &[S],
    derivative: bool,
) -> Result<[S; CAPACITY], SciOxideError> {
    Weights::new(
        nodes,
        if derivative {
            Functional::Derivative
        } else {
            Functional::Integral
        },
    )?
    .coefficients()
}
