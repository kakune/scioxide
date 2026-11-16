//! Four-point Gauss–Legendre enclosure on a finite interval.
use super::*;

// Adjacent binary64 endpoints from 256-bit FLINT algebraic evaluation.
// Each row contains the node and weight on [0, 1].
const RULE: [[u64; 4]; 4] = [
    [
        0x3fb1c6490c2719eb,
        0x3fb1c6490c2719ec,
        0x3fc64340f7e7b66b,
        0x3fc64340f7e7b66c,
    ],
    [
        0x3fd51ee013116101,
        0x3fd51ee013116102,
        0x3fd4de5f840c24ca,
        0x3fd4de5f840c24cb,
    ],
    [
        0x3fe5708ff6774f7f,
        0x3fe5708ff6774f80,
        0x3fd4de5f840c24ca,
        0x3fd4de5f840c24cb,
    ],
    [
        0x3fedc736de7b1cc2,
        0x3fedc736de7b1cc3,
        0x3fc64340f7e7b66b,
        0x3fc64340f7e7b66c,
    ],
];

/// Convert both endpoints outward, including binary32 conversion.
pub(super) fn bracket<R: RealScalar>(lo: u64, hi: u64) -> Result<D<R>> {
    let lo = crate::rectangle::constant(f64::from_bits(lo))?;
    let hi = crate::rectangle::constant(f64::from_bits(hi))?;
    Ok(D::from(ExtendedInterval::new(
        lo.bounds().map(|(a, _)| a),
        hi.bounds().map(|(_, b)| b),
    )?))
}

impl<S: NumericScalar> QuadratureVerifier<'_, S> {
    pub(super) fn gauss_bound(
        &self,
        panel: QuadraturePanel<S::Real>,
        out: &mut [IntervalBox<S::Real>],
        scratch: &mut [IntervalBox<S::Real>],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<QuadratureInconclusive>> {
        let (a, b) = finite_endpoints(panel.domain, UNBOUNDED_PANEL)?;
        let h = D::point(b)?.sub_admitted(&D::point(a)?)?;
        if !h.set().is_finite() {
            return Ok(Some(QuadratureInconclusive::Unbounded));
        }
        let values = &mut scratch[..self.components];
        out.fill(zero()?);
        for [nl, nh, wl, wh] in RULE {
            let node = D::point(a)?.add_admitted(&h.mul_admitted(&bracket(nl, nh)?)?)?;
            // The exact mapped node belongs to the panel, even when directed
            // arithmetic widens its enclosure across a boundary.
            let domain = node.set().intersection(&panel.domain);
            if let Some(reason) =
                self.query(panel, domain, QuadratureQuery::Value, values, context)?
            {
                return Ok(Some(reason));
            }
            let weight = Box2::real(h.mul_admitted(&bracket(wl, wh)?)?)?;
            for (total, &value) in out.iter_mut().zip(values.iter()) {
                *total = pack(unpack(*total).add(unpack(value).mul(weight)?)?);
            }
        }
        if let Some(reason) = self.query(
            panel,
            panel.domain,
            QuadratureQuery::EighthDerivative,
            values,
            context,
        )? {
            return Ok(Some(reason));
        }
        // DLMF 3.5.19, transformed to [a,b]:
        // (4!)^4 h^9 / (9 (8!)^3) = h^9 / 1778112000.
        let h4 = h.square_admitted()?.square_admitted()?;
        let remainder = h4
            .square_admitted()?
            .mul_admitted(&h)?
            .div_admitted(&crate::rectangle::constant(1778112000.0)?)?;
        if !remainder.set().is_finite() {
            return Ok(Some(QuadratureInconclusive::Unbounded));
        }
        let remainder = Box2::real(remainder)?;
        for (total, &derivative) in out.iter_mut().zip(values.iter()) {
            *total = pack(unpack(*total).add(unpack(derivative).mul(remainder)?)?);
        }
        Ok(out
            .iter()
            .any(|&value| !unpack(value).finite())
            .then_some(QuadratureInconclusive::Unbounded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constants_match_independent_flint_endpoints() {
        let mut checked = 0;
        for line in include_str!("../../tests/data/quadrature-gauss.txt")
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            let (name, index) = fields[0].split_at(fields[0].len() - 1);
            let offset = match name {
                "node" => 0,
                "weight" => 2,
                _ => continue,
            };
            checked += 1;
            let index: usize = index.parse().unwrap();
            let lo = u64::from_str_radix(fields[2], 16).unwrap();
            let hi = u64::from_str_radix(fields[3], 16).unwrap();
            if fields[1] == "64" {
                assert_eq!(
                    bracket::<f64>(RULE[index][offset], RULE[index][offset + 1])
                        .unwrap()
                        .bounds(),
                    Some((f64::from_bits(lo), f64::from_bits(hi)))
                );
            } else {
                assert_eq!(
                    bracket::<f32>(RULE[index][offset], RULE[index][offset + 1])
                        .unwrap()
                        .bounds(),
                    Some((f32::from_bits(lo as u32), f32::from_bits(hi as u32)))
                );
            }
        }
        assert_eq!(checked, 16);
    }
}
