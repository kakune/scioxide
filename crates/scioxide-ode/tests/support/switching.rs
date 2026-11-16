//! IVPtestset systems for automatic-method work/accuracy regressions.
use scioxide_core::{SciOxideError, SolveContext};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::OdeSystem;

#[derive(Debug, Clone, Copy)]
pub enum Problem {
    Arenstorf,
    Pleiades,
    Robertson,
    VanDerPol,
    Orego,
}

impl Problem {
    pub const ALL: [Self; 5] = [
        Self::Arenstorf,
        Self::Pleiades,
        Self::Robertson,
        Self::VanDerPol,
        Self::Orego,
    ];

    pub fn initial(self) -> Vec<f64> {
        match self {
            Self::Arenstorf => vec![0.994, 0., 0., -2.0015851063790825],
            Self::Pleiades => vec![
                3., 3., -1., -3., 2., -2., 2., 3., -3., 2., 0., 0., -4., 4., 0., 0., 0., 0., 0.,
                1.75, -1.5, 0., 0., 0., -1.25, 1., 0., 0.,
            ],
            Self::Robertson => vec![1., 0., 0.],
            Self::VanDerPol => vec![2., 0.],
            Self::Orego => vec![1., 2., 3.],
        }
    }

    pub fn end(self) -> f64 {
        match self {
            Self::Arenstorf => 17.065216560157963,
            Self::Pleiades => 3.,
            Self::Robertson => 1e11,
            Self::VanDerPol => 2.,
            Self::Orego => 360.,
        }
    }

    pub fn tolerance(self) -> (f64, f64) {
        match self {
            Self::Arenstorf | Self::Pleiades => (1e-9, 1e-9),
            _ => (1e-10, 1e-6),
        }
    }

    #[allow(clippy::unwrap_used)]
    pub fn reference(self) -> Vec<f64> {
        match self {
            Self::Arenstorf => self.initial(),
            Self::Pleiades => include_str!("../data/pleiades.tsv")
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect(),
            Self::Robertson => vec![
                2.083340149701255e-8,
                8.333360770334713e-14,
                0.999999979166505,
            ],
            Self::VanDerPol => vec![1.706_167_732_170_483, -0.8928097010247975],
            Self::Orego => vec![
                1.000_814_870_318_523,
                1_228.178_521_549_917,
                132.0554942846706,
            ],
        }
    }
}

impl OdeSystem<f64> for Problem {
    fn dimension(&self) -> usize {
        match self {
            Self::Arenstorf => 4,
            Self::Pleiades => 28,
            Self::VanDerPol => 2,
            _ => 3,
        }
    }

    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Arenstorf => {
                let mu = 0.012277471;
                let a = ((y[0] + mu).powi(2) + y[1].powi(2)).powf(-1.5);
                let b = ((y[0] - 1. + mu).powi(2) + y[1].powi(2)).powf(-1.5);
                out.copy_from_slice(&[
                    y[2],
                    y[3],
                    y[0] + 2. * y[3] - (1. - mu) * (y[0] + mu) * a - mu * (y[0] - 1. + mu) * b,
                    y[1] - 2. * y[2] - ((1. - mu) * a + mu * b) * y[1],
                ]);
            }
            Self::Pleiades => {
                out[..14].copy_from_slice(&y[14..]);
                out[14..].fill(0.);
                for i in 0..7 {
                    for k in 0..7 {
                        if i == k {
                            continue;
                        }
                        let dx = y[k] - y[i];
                        let dy = y[7 + k] - y[7 + i];
                        let weight = (k + 1) as f64 * (dx * dx + dy * dy).powf(-1.5);
                        out[14 + i] += weight * dx;
                        out[21 + i] += weight * dy;
                    }
                }
            }
            Self::Robertson => {
                let a = 0.04 * y[0] - 1e4 * y[1] * y[2];
                let b = 3e7 * y[1] * y[1];
                out.copy_from_slice(&[-a, a - b, b]);
            }
            Self::VanDerPol => {
                out.copy_from_slice(&[y[1], ((1. - y[0] * y[0]) * y[1] - y[0]) / 1e-6])
            }
            Self::Orego => out.copy_from_slice(&[
                77.27 * (y[1] + y[0] * (1. - 8.375e-6 * y[0] - y[1])),
                (y[2] - (1. + y[0]) * y[1]) / 77.27,
                0.161 * (y[0] - y[2]),
            ]),
        }
        Ok(())
    }

    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..self.dimension() {
            for col in 0..self.dimension() {
                j.set(row, col, 0.)?;
            }
        }
        match self {
            Self::Arenstorf => {
                j.set(0, 2, 1.)?;
                j.set(1, 3, 1.)?;
                j.set(2, 3, 2.)?;
                j.set(3, 2, -2.)?;
                let mu = 0.012277471;
                let (mut xx, mut xy, mut yy) = (1., 0., 1.);
                for (mass, dx) in [(1. - mu, y[0] + mu), (mu, y[0] - 1. + mu)] {
                    let r2 = dx * dx + y[1] * y[1];
                    let weight = mass * r2.powf(-1.5);
                    xx -= weight * (1. - 3. * dx * dx / r2);
                    xy += weight * 3. * dx * y[1] / r2;
                    yy -= weight * (1. - 3. * y[1] * y[1] / r2);
                }
                j.set(2, 0, xx)?;
                j.set(2, 1, xy)?;
                j.set(3, 0, xy)?;
                j.set(3, 1, yy)?;
            }
            Self::Pleiades => {
                for i in 0..14 {
                    j.set(i, 14 + i, 1.)?;
                }
                for i in 0..7 {
                    for k in 0..7 {
                        if i == k {
                            continue;
                        }
                        let dx = y[k] - y[i];
                        let dy = y[7 + k] - y[7 + i];
                        let r2 = dx * dx + dy * dy;
                        let weight = (k + 1) as f64 * r2.powf(-1.5);
                        for (row, col, value) in [
                            (14 + i, k, weight * (1. - 3. * dx * dx / r2)),
                            (14 + i, 7 + k, -3. * weight * dx * dy / r2),
                            (21 + i, k, -3. * weight * dx * dy / r2),
                            (21 + i, 7 + k, weight * (1. - 3. * dy * dy / r2)),
                        ] {
                            j.set(row, col, value)?;
                            let own = if col < 7 { i } else { 7 + i };
                            j.set(row, own, j.get(row, own)? - value)?;
                        }
                    }
                }
            }
            Self::Robertson => {
                for (r, c, v) in [
                    (0, 0, -0.04),
                    (0, 1, 1e4 * y[2]),
                    (0, 2, 1e4 * y[1]),
                    (1, 0, 0.04),
                    (1, 1, -1e4 * y[2] - 6e7 * y[1]),
                    (1, 2, -1e4 * y[1]),
                    (2, 1, 6e7 * y[1]),
                ] {
                    j.set(r, c, v)?;
                }
            }
            Self::VanDerPol => {
                j.set(0, 1, 1.)?;
                j.set(1, 0, (-2. * y[0] * y[1] - 1.) / 1e-6)?;
                j.set(1, 1, (1. - y[0] * y[0]) / 1e-6)?;
            }
            Self::Orego => {
                for (r, c, v) in [
                    (0, 0, 77.27 * (1. - 2. * 8.375e-6 * y[0] - y[1])),
                    (0, 1, 77.27 * (1. - y[0])),
                    (1, 0, -y[1] / 77.27),
                    (1, 1, -(1. + y[0]) / 77.27),
                    (1, 2, 1. / 77.27),
                    (2, 0, 0.161),
                    (2, 2, -0.161),
                ] {
                    j.set(r, c, v)?;
                }
            }
        }
        Ok(Some(()))
    }
}
