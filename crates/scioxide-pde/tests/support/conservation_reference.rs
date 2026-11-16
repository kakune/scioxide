//! Independent exact ideal-gas Riemann oracle (test use only).
//! Toro's shock Hugoniot and isentropic rarefaction relations, gamma=1.4.
//! A scalar bisection of the pressure matching equation is independent of
//! production HLL/HLLE/HLLC approximations. No comparator output is the oracle.

#[derive(Clone, Copy)]
pub struct Primitive {
    pub rho: f64,
    pub velocity: f64,
    pub pressure: f64,
}
impl Primitive {
    pub fn conserved(self) -> [f64; 3] {
        [
            self.rho,
            self.rho * self.velocity,
            self.pressure / 0.4 + 0.5 * self.rho * self.velocity * self.velocity,
        ]
    }
    fn sound(self) -> f64 {
        (1.4 * self.pressure / self.rho).sqrt()
    }
    fn pressure_curve(self, p: f64) -> f64 {
        if p > self.pressure {
            (p - self.pressure) * (2.0 / (2.4 * self.rho) / (p + self.pressure / 6.0)).sqrt()
        } else {
            5.0 * self.sound() * ((p / self.pressure).powf(1.0 / 7.0) - 1.0)
        }
    }
}
pub struct ExactRiemann {
    left: Primitive,
    right: Primitive,
    pub pressure: f64,
    pub velocity: f64,
}
impl ExactRiemann {
    pub fn new(left: Primitive, right: Primitive) -> Self {
        let residual =
            |p| left.pressure_curve(p) + right.pressure_curve(p) + right.velocity - left.velocity;
        assert!(residual(0.0) < 0.0, "exact oracle excludes vacuum");
        let mut low = 0.0;
        let mut high = left.pressure.max(right.pressure);
        for _ in 0..128 {
            if residual(high) >= 0.0 {
                break;
            }
            high *= 2.0;
        }
        assert!(residual(high) >= 0.0);
        for _ in 0..128 {
            let mid = (low + high) * 0.5;
            if residual(mid) > 0.0 {
                high = mid;
            } else {
                low = mid;
            }
        }
        let pressure = (low + high) * 0.5;
        let velocity = 0.5
            * (left.velocity + right.velocity + right.pressure_curve(pressure)
                - left.pressure_curve(pressure));
        Self {
            left,
            right,
            pressure,
            velocity,
        }
    }
    pub fn sample(&self, speed: f64) -> Primitive {
        let (state, sign) = if speed <= self.velocity {
            (self.left, -1.0)
        } else {
            (self.right, 1.0)
        };
        let q = self.pressure / state.pressure;
        let star_rho = if q > 1.0 {
            state.rho * (q + 1.0 / 6.0) / (q / 6.0 + 1.0)
        } else {
            state.rho * q.powf(1.0 / 1.4)
        };
        let star = Primitive {
            rho: star_rho,
            velocity: self.velocity,
            pressure: self.pressure,
        };
        if q > 1.0 {
            let shock = state.velocity + sign * state.sound() * (6.0 / 7.0 * q + 1.0 / 7.0).sqrt();
            if sign * (speed - shock) >= 0.0 {
                state
            } else {
                star
            }
        } else {
            let head = state.velocity + sign * state.sound();
            let tail = self.velocity + sign * state.sound() * q.powf(1.0 / 7.0);
            if sign * (speed - head) >= 0.0 {
                state
            } else if sign * (speed - tail) <= 0.0 {
                star
            } else {
                let velocity = 2.0 / 2.4 * (-sign * state.sound() + 0.2 * state.velocity + speed);
                let sound = 2.0 / 2.4 * (state.sound() + sign * 0.2 * (speed - state.velocity));
                let ratio = sound / state.sound();
                Primitive {
                    rho: state.rho * ratio.powf(5.0),
                    velocity,
                    pressure: state.pressure * ratio.powf(7.0),
                }
            }
        }
    }
    pub fn cell_average(&self, left: f64, right: f64, time: f64) -> [f64; 3] {
        // Composite midpoint integration; 128 samples keep quadrature error well
        // below the finite-volume acceptance threshold even at discontinuities.
        let mut average = [0.; 3];
        for i in 0..128 {
            let x = left + (right - left) * (i as f64 + 0.5) / 128.;
            let u = self.sample((x - 0.5) / time).conserved();
            for j in 0..3 {
                average[j] += u[j] / 128.;
            }
        }
        average
    }
}
