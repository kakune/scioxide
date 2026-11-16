//! One owner for elementary routing, guards and width adaptations.

macro_rules! registry {
    ($emit:ident, $ty:ident, $module:ident, $next_after:path, $scalbn:path) => {
        $emit! { $ty, $module;
            sqrt(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::sqrt_rounded(value as f64, <$ty>::MANTISSA_DIGITS) as $ty }
            }
            floor(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::floor(value as f64) as $ty }
            }
            ceil(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::ceil(value as f64) as $ty }
            }
            round(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::round(value as f64) as $ty }
            }
            trunc(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::trunc(value as f64) as $ty }
            }
            fract(value: $ty) -> $ty [native] {
                guard {  }
                owned { (value as f64 - crate::math::trunc(value as f64)) as $ty }
            }
            signum(value: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            recip(value: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            mul_add(value: $ty, factor: $ty, addend: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::fused::$module(value, factor, addend) }
            }
            powi(value: $ty, exponent: i32) -> $ty [owned] {
                guard {  }
                // Binary64 represents every i32 exactly, including odd powers
                // beyond binary32's integer lattice. Native powi instead has
                // multiplication-count error, exceeding our four-ULP envelope.
                owned { crate::math::dispatch::binary64::powf(value as f64, f64::from(exponent)) as $ty }
            }
            powf(value: $ty, exponent: $ty) -> $ty [native] {
                guard { if exponent == 0.0 || value == 1.0 { return 1.0; } }
                owned { crate::math::elementary::pow(value as f64, exponent as f64) as $ty }
            }
            cbrt(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::cbrt(value as f64) as $ty }
            }
            hypot(value: $ty, other: $ty) -> $ty [native] {
                guard { if value.is_infinite() || other.is_infinite() { return <$ty>::INFINITY; } }
                owned { crate::math::elementary::hypot(value as f64, other as f64) as $ty }
            }
            exp(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::exp(value as f64) as $ty }
            }
            exp2(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::exp2(value as f64) as $ty }
            }
            exp_m1(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::expm1(value as f64) as $ty }
            }
            ln(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::log(value as f64) as $ty }
            }
            log2(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::log2(value as f64) as $ty }
            }
            log10(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::log10(value as f64) as $ty }
            }
            ln_1p(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::log1p(value as f64) as $ty }
            }
            sin(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::sincos(value as f64).0 as $ty }
            }
            cos(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::sincos(value as f64).1 as $ty }
            }
            tan(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::tan(value as f64) as $ty }
            }
            asin(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::asin(value as f64) as $ty }
            }
            acos(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::acos(value as f64) as $ty }
            }
            atan(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::atan(value as f64) as $ty }
            }
            sinh(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::sinh(value as f64) as $ty }
            }
            cosh(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::cosh(value as f64) as $ty }
            }
            tanh(value: $ty) -> $ty [native] {
                guard {  }
                owned { crate::math::elementary::tanh(value as f64) as $ty }
            }
            exp10(value: $ty) -> $ty [owned] {
                guard {  }
                owned { crate::math::dispatch::binary64::powf(10.0, value as f64) as $ty }
            }
            log(value: $ty, base: $ty) -> $ty [native] {
                guard {  }
                owned { (crate::math::elementary::log(value as f64) / crate::math::elementary::log(base as f64)) as $ty }
            }
            max(value: $ty, other: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            min(value: $ty, other: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            clamp(value: $ty, minimum: $ty, maximum: $ty) -> $ty [intrinsic] {
                guard { if minimum.is_nan() || maximum.is_nan() || minimum > maximum { return <$ty>::NAN; } }
                owned {  }
            }
            copysign(value: $ty, sign: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            atan2(value: $ty, other: $ty) -> $ty [native] {
                guard { if let Some(boundary) = crate::math::atan2_special(value as f64, other as f64) { return boundary as $ty; } }
                owned { crate::math::elementary::atan2(value as f64, other as f64) as $ty }
            }
            sin_cos(value: $ty) -> ($ty, $ty) [native] {
                guard {  }
                owned { { let (sin, cos) = crate::math::elementary::sincos(value as f64); (sin as $ty, cos as $ty) } }
            }
            asinh(value: $ty) -> $ty [owned] {
                guard { }
                owned { crate::math::elementary::asinh(value as f64) as $ty }
            }
            acosh(value: $ty) -> $ty [native] {
                guard {
                    if value < 1.0 { return <$ty>::NAN; }
                    // Preserve the small correction at the branch point;
                    // native log(x+sqrt(x*x-1)) can lose millions of ULPs.
                    if value < 2.0 {
                        let delta = value as f64 - 1.0;
                        return crate::math::dispatch::binary64::ln_1p(delta
                            + crate::math::dispatch::binary64::sqrt(delta * (delta + 2.0))) as $ty;
                    }
                    if value >= 67_108_864.0 {
                        return (crate::math::dispatch::binary64::ln(value as f64)
                            + core::f64::consts::LN_2) as $ty;
                    }
                }
                owned { crate::math::elementary::acosh(value as f64) as $ty }
            }
            atanh(value: $ty) -> $ty [owned] {
                guard {  }
                owned { crate::math::elementary::atanh(value as f64) as $ty }
            }
            to_degrees(value: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            to_radians(value: $ty) -> $ty [intrinsic] {
                guard {  }
                owned {  }
            }
            next_after(value: $ty, toward: $ty) -> $ty [owned] {
                guard {  }
                owned { $next_after(value, toward) }
            }
            scalbn(value: $ty, exponent: i32) -> $ty [owned] {
                guard {  }
                owned { $scalbn(value, exponent) }
            }
        }
    };
}
pub(crate) use registry;

macro_rules! route {
    (native, $ty:ident, $name:ident, ($($arg:ident),*), $owned:block) => {{
        #[cfg(feature = "std")]
        { <$ty>::$name($($arg),*) }
        #[cfg(not(feature = "std"))]
        $owned
    }};
    (owned, $ty:ident, $name:ident, ($($arg:ident),*), $owned:block) => { $owned };
    (intrinsic, $ty:ident, $name:ident, ($($arg:ident),*), $owned:block) => { <$ty>::$name($($arg),*) };
}

macro_rules! functions {
    ($ty:ident, $module:ident; $(
        $name:ident($($arg:ident: $argty:ty),*) -> $ret:ty [$mode:ident] {
            guard $guard:block owned $owned:block
        }
    )*) => {
        pub(crate) mod $module {
            $(
                #[inline]
                pub(crate) fn $name($($arg: $argty),*) -> $ret {
                    $guard
                    route!($mode, $ty, $name, ($($arg),*), $owned)
                }
            )*
        }
    };
}
registry!(
    functions,
    f32,
    binary32,
    crate::math::next_after_f32,
    crate::math::scalbn_f32
);
registry!(
    functions,
    f64,
    binary64,
    crate::math::next_after_f64,
    crate::math::scalbn_f64
);

macro_rules! scalar_methods {
    ($ty:ident, $module:ident; $(
        $name:ident($value:ident: $value_ty:ty $(, $arg:ident: $argty:ty)*) -> $ret:ty [$mode:ident] {
            guard $guard:block owned $owned:block
        }
    )*) => { $(
        #[inline]
        fn $name(self $(, $arg: $argty)*) -> $ret {
            crate::math::dispatch::$module::$name(self $(, $arg)*)
        }
    )* };
}
pub(crate) use scalar_methods;

// Lane routes: each entry names its branch-free kernel, the domain in which the
// kernel is used, and the route for the other lanes, the scalar `binary64`
// function unless stated. Every lane runs the kernel; only a call with an
// outside lane takes the cold patch.
macro_rules! lane_routes {
    ($(
        $(#[$doc:meta])*
        $name:ident($($arg:ident),+) {
            kernel: $kernel:path,
            inside: $inside:expr,
            outside: $outside:path $(,)?
        }
    )*) => { $(
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<const W: usize>($($arg: [f64; W]),+) -> [f64; W] {
            #[cold]
            #[inline(never)]
            fn patch<const W: usize>(results: &mut [f64; W], $($arg: &[f64; W]),+) {
                for (lane, result) in results.iter_mut().enumerate() {
                    $(let $arg = $arg[lane];)+
                    let inside: bool = $inside;
                    if !inside {
                        *result = $outside($($arg),+);
                    }
                }
            }
            let mut results = [0.0; W];
            let mut outside = false;
            for (lane, result) in results.iter_mut().enumerate() {
                $(let $arg = $arg[lane];)+
                *result = $kernel($($arg),+);
                let inside: bool = $inside;
                outside |= !inside;
            }
            if outside {
                patch(&mut results, $(&$arg),+);
            }
            results
        }
    )* };
}

/// `eˣ` of a lane beyond the one-step range of the kernel: the scalar route for
/// NaN, and the kernel with two-step scaling and limits otherwise.
fn exp_outside(x: f64) -> f64 {
    if x.is_nan() {
        binary64::exp(x)
    } else {
        crate::math::lanes::exp::exp_full(x)
    }
}

lane_routes! {
    /// Computes `eˣ` in every lane.
    ///
    /// The kernel rounds `k = x/ln 2` to an integer by adding `1.5·2⁵²`,
    /// reduces `r = x − k ln 2` with a two-constant Cody–Waite split of `ln 2`
    /// and evaluates `eʳ = 1 + (r + r²·P(r))` with the degree-13 Taylor
    /// polynomial (Estrin's scheme). For `|x| ≤ 708` the result is normal and
    /// `k` is added to its exponent field: about 40 arithmetic and bit
    /// operations per lane and no division. A call with a lane beyond `±708`
    /// takes the cold path, which multiplies that lane's `eʳ` by `2^⌊k/2⌋` and
    /// then `2^⌈k/2⌉`, so a subnormal result is rounded once, and returns `+∞`
    /// above the overflow threshold `709.78…` and `+0` below `−745.13…`. Both
    /// forms give the same bits where both apply. Results tested against MPFR
    /// are within one ulp of the correctly rounded value.
    ///
    /// Domain: every lane except NaN; a NaN lane returns the scalar
    /// [`RealScalar::exp`](crate::RealScalar::exp). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_exp;
    ///
    /// let x = [-745.0, -1.0, 0.5, 700.0];
    /// for (y, x) in lanes_exp(x).into_iter().zip(x) {
    ///     assert!((y - x.exp()).abs() <= 4.0 * f64::EPSILON * x.exp());
    /// }
    /// assert_eq!(lanes_exp([f64::NEG_INFINITY, 710.0]), [0.0, f64::INFINITY]);
    /// ```
    ///
    /// # References
    ///
    /// - W. J. Cody and W. Waite, *Software Manual for the Elementary
    ///   Functions*, Prentice-Hall, 1980, ch. 6.
    lanes_exp(x) {
        kernel: crate::math::lanes::exp::exp,
        inside: x.abs() <= crate::math::lanes::exp::NORMAL_LIMIT,
        outside: exp_outside,
    }
    /// Computes `eˣ − 1` in every lane.
    ///
    /// The kernel is fdlibm's `expm1` with every case evaluated and the result
    /// selected: `k = round(x/ln 2)` by the `1.5·2⁵²` shift, a Cody–Waite
    /// reduction with its rounding correction, the rational approximation of
    /// `eʳ − 1` on `|r| ≤ ln 2/2` and the reconstructions for `k = 0`, `±1`,
    /// `2…56` and larger `|k|`. It costs about 70 arithmetic and bit
    /// operations and one division per lane. Arguments beyond `±746` select
    /// `+∞` or `−1`; `±0` and subnormal arguments return themselves. Results
    /// tested against MPFR are within one ulp of the correctly rounded value.
    ///
    /// Domain: every lane except NaN; a NaN lane returns the scalar
    /// [`RealScalar::exp_m1`](crate::RealScalar::exp_m1). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_exp_m1;
    ///
    /// let x = [-40.0, -1e-10, 1e-300, 3.0];
    /// for (y, x) in lanes_exp_m1(x).into_iter().zip(x) {
    ///     assert!((y - x.exp_m1()).abs() <= 4.0 * f64::EPSILON * x.exp_m1().abs());
    /// }
    /// ```
    ///
    /// # References
    ///
    /// - K. C. Ng, `s_expm1.c`, Sun Microsystems fdlibm 5.3, 1993.
    lanes_exp_m1(x) {
        kernel: crate::math::lanes::exp::exp_m1,
        inside: !x.is_nan(),
        outside: binary64::exp_m1,
    }
    /// Computes the natural logarithm `ln x` in every lane.
    ///
    /// The kernel takes `x = 2ᵏ·m` with `m ∈ [√½, √2)` from the exponent and
    /// significand bits, sets `f = m − 1` and `s = f/(2 + f)`, and evaluates
    /// fdlibm's even polynomial for `ln(1 + f) = 2 atanh s` plus `k ln 2`. It
    /// costs about 30 arithmetic and bit operations and one division per lane.
    /// Results tested against MPFR are within one ulp of the correctly rounded
    /// value.
    ///
    /// Domain: positive normal finite `x`. Other lanes (zeros, subnormals,
    /// negative values, `+∞` and NaN) return the scalar
    /// [`RealScalar::ln`](crate::RealScalar::ln). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_ln;
    ///
    /// let x = [0.5, 1.0 + 1e-12, 3.0, 1e300];
    /// for (y, x) in lanes_ln(x).into_iter().zip(x) {
    ///     assert!((y - x.ln()).abs() <= 4.0 * f64::EPSILON * x.ln().abs());
    /// }
    /// let special = lanes_ln([0.0, -1.0]);
    /// assert!(special[0] == f64::NEG_INFINITY && special[1].is_nan());
    /// ```
    ///
    /// # References
    ///
    /// - K. C. Ng, `e_log.c`, Sun Microsystems fdlibm 5.3, 1993.
    lanes_ln(x) {
        kernel: crate::math::lanes::log::ln,
        inside: (f64::MIN_POSITIVE..=f64::MAX).contains(&x),
        outside: binary64::ln,
    }
    /// Computes `ln(1 + x)` in every lane.
    ///
    /// The kernel is fdlibm's `log1p` with every case evaluated and the result
    /// selected: `1 + x` is reduced like [`lanes_ln`] with a correction for
    /// its rounding error, `1 + x ∈ [√½, √2)` uses `f = x` directly, and
    /// `|x| < 2⁻⁵³` returns `x`, which keeps signed zeros. It costs about 45
    /// arithmetic and bit operations and two divisions per lane. Results tested
    /// against MPFR are within one ulp of the correctly rounded value.
    ///
    /// Domain: finite `x > −1`. Other lanes (`x ≤ −1`, `+∞` and NaN) return the
    /// scalar [`RealScalar::ln_1p`](crate::RealScalar::ln_1p). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_ln_1p;
    ///
    /// let x = [-0.75, -1e-12, 0.25, 1e10];
    /// for (y, x) in lanes_ln_1p(x).into_iter().zip(x) {
    ///     assert!((y - x.ln_1p()).abs() <= 4.0 * f64::EPSILON * x.ln_1p().abs());
    /// }
    /// ```
    ///
    /// # References
    ///
    /// - K. C. Ng, `s_log1p.c`, Sun Microsystems fdlibm 5.3, 1993.
    lanes_ln_1p(x) {
        kernel: crate::math::lanes::log::ln_1p,
        inside: (x > -1.0) & (x <= f64::MAX),
        outside: binary64::ln_1p,
    }
    /// Computes `sin x` in every lane.
    ///
    /// The kernel rounds `n = 2x/π` to an integer by adding `1.5·2⁵²`, reduces
    /// `y = x − n·π/2` with fdlibm's three-constant Cody–Waite split of `π/2`
    /// (exact products and two-sums for `|x| ≤ 2²⁰`, about 150 bits of `π/2`),
    /// evaluates fdlibm's sine and cosine kernels on `y` and selects one with
    /// the sign of quadrant `n mod 4` by bit masks. It costs about 55
    /// arithmetic and bit operations per lane and no division. Results tested
    /// against MPFR are within one ulp of the correctly rounded value.
    ///
    /// Domain: `|x| ≤ 2²⁰`. Other lanes (larger arguments, infinities and NaN)
    /// return the scalar [`RealScalar::sin`](crate::RealScalar::sin). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_sin;
    ///
    /// let x = [-0.0, 0.5, 3.0, 1e5];
    /// let y = lanes_sin(x);
    /// for (&y, x) in y.iter().zip(x) {
    ///     assert!((y - x.sin()).abs() <= 4.0 * f64::EPSILON * x.sin().abs());
    /// }
    /// assert_eq!(y[0].to_bits(), (-0.0_f64).to_bits());
    /// ```
    ///
    /// # References
    ///
    /// - W. J. Cody and W. Waite, *Software Manual for the Elementary
    ///   Functions*, Prentice-Hall, 1980, ch. 8.
    /// - K. C. Ng, `e_rem_pio2.c`, `k_sin.c` and `k_cos.c`, Sun Microsystems
    ///   fdlibm 5.3, 1993.
    lanes_sin(x) {
        kernel: crate::math::lanes::trig::sin,
        inside: x.abs() <= crate::math::lanes::trig::LIMIT,
        outside: binary64::sin,
    }
    /// Computes `cos x` in every lane.
    ///
    /// The kernel is that of [`lanes_sin`] with the quadrant advanced by one,
    /// `cos x = sin(x + π/2)`, at the same cost and accuracy.
    ///
    /// Domain: `|x| ≤ 2²⁰`. Other lanes (larger arguments, infinities and NaN)
    /// return the scalar [`RealScalar::cos`](crate::RealScalar::cos). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_cos;
    ///
    /// let x = [0.0, 0.5, 3.0, 1e5];
    /// for (y, x) in lanes_cos(x).into_iter().zip(x) {
    ///     assert!((y - x.cos()).abs() <= 4.0 * f64::EPSILON * x.cos().abs());
    /// }
    /// ```
    lanes_cos(x) {
        kernel: crate::math::lanes::trig::cos,
        inside: x.abs() <= crate::math::lanes::trig::LIMIT,
        outside: binary64::cos,
    }
    /// Computes `tanh x` in every lane.
    ///
    /// The kernel follows fdlibm's case split on `a = |x|`, evaluated without
    /// branches: `e = expm1(2a)` with `1 − 2/(e + 2)` for `a > ln(3)/2`,
    /// `e/(e + 2)` for `a > ln(5/3)/2`, and `e = expm1(−2a)` with `−e/(e + 2)`
    /// below, where `expm1` is the kernel of [`lanes_exp_m1`]; `a` is clamped
    /// to 20, where `tanh` rounds to one, and the sign of `x` is copied. It
    /// costs about 80 arithmetic and bit operations and two divisions per
    /// lane. Results tested against MPFR are within two ulps of the correctly
    /// rounded value.
    ///
    /// Domain: every lane except NaN; a NaN lane returns the scalar
    /// [`RealScalar::tanh`](crate::RealScalar::tanh). The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_tanh;
    ///
    /// let x = [-3.0, 1e-8, 0.4, 0.9];
    /// for (y, x) in lanes_tanh(x).into_iter().zip(x) {
    ///     assert!((y - x.tanh()).abs() <= 4.0 * f64::EPSILON * x.tanh().abs());
    /// }
    /// assert_eq!(lanes_tanh([f64::NEG_INFINITY]), [-1.0]);
    /// ```
    ///
    /// # References
    ///
    /// - K. C. Ng, `s_tanh.c`, Sun Microsystems fdlibm 5.3, 1993.
    lanes_tanh(x) {
        kernel: crate::math::lanes::exp::tanh,
        inside: !x.is_nan(),
        outside: binary64::tanh,
    }
    /// Computes `aᵇ` in every lane, for `a` in `base` and `b` in `exponent`.
    ///
    /// The kernel evaluates `exp(b·ln a)`. `ln a` is formed as a double-length
    /// value `h + l` with relative error of about `2⁻⁶³`: `a = 2ᵏ·m` with
    /// `m ∈ [√½, √2)`, `s = (m − 1)/(m + 1)` with an exact Dekker remainder and
    /// `ln a = k ln 2 + 2s + (2/3)s³ + s⁵·U(s²)` with the first three terms in
    /// double-length arithmetic. `b·(h + l)` uses Dekker's product with
    /// Veltkamp splitting, and its double-length sum enters the reduction of
    /// [`lanes_exp`], so the result keeps the exponential's accuracy up to
    /// overflow and through gradual underflow; products beyond `±746` select
    /// `+∞` or `+0`. It costs about 170 arithmetic operations and one division
    /// per lane. Results tested against MPFR are within one ulp of the
    /// correctly rounded value.
    ///
    /// Domain: positive normal finite `a` with `|b| ≤ 2⁵¹²`. Other lanes (zero,
    /// subnormal, negative, infinite or NaN bases, and huge, infinite or NaN
    /// exponents) return the scalar
    /// [`RealScalar::powf`](crate::RealScalar::powf), with its rules for zero
    /// exponents, unit bases and negative bases with integer exponents. The
    /// [lane-kernel rules](crate::backend#lane-kernels) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::lanes_powf;
    ///
    /// let (a, b) = ([0.5, 2.0, 1.0 + 1e-15, -2.0], [-3.5, 0.75, 1e16, 3.0]);
    /// for ((y, a), b) in lanes_powf(a, b).into_iter().zip(a).zip(b) {
    ///     assert!((y - a.powf(b)).abs() <= 4.0 * f64::EPSILON * a.powf(b).abs());
    /// }
    /// ```
    ///
    /// # References
    ///
    /// - T. J. Dekker, "A floating-point technique for extending the available
    ///   precision", *Numer. Math.* 18 (1971), 224–242.
    /// - J.-M. Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed.,
    ///   Birkhäuser, 2018, ch. 4 and 10.
    lanes_powf(base, exponent) {
        kernel: crate::math::lanes::pow::powf,
        inside: (f64::MIN_POSITIVE..=f64::MAX).contains(&base)
            & (exponent.abs() <= crate::math::lanes::pow::EXPONENT_LIMIT),
        outside: binary64::powf,
    }
}
