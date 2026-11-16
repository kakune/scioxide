# Third-party notices

## Cephes scaled order-zero Bessel tables

The Chebyshev tables in `src/bessel_airy/bessel_i0.rs` adapt Cephes 2.8 `i0.c`, Copyright 1984, 1987, 2000 Stephen L. Moshier, from SciPy xsf's `include/xsf/cephes/i0.h` (C++ translation by SciPy developers, 2024): <https://github.com/scipy/xsf/blob/main/include/xsf/cephes/i0.h>.
The original Cephes permission is described below.
SciPy's distribution uses the following BSD 3-Clause notice:

Copyright (c) 2024, SciPy

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Cephes error-function coefficient tables

The rational coefficient tables used by `erf` below one in `src/error_probability/error_fn.rs` are adapted from Stephen L. Moshier's Cephes Mathematical Library `ndtr.c`, distributed by Netlib: <https://www.netlib.org/cephes/doubldoc.html>.

The Netlib distribution states that the material may be used freely and is provided without support or guarantee: <https://www.netlib.org/cephes/readme>.

The direct gamma rational on [1, 2] is the recurrence-shifted Cephes `gamma.c` approximation on [2, 3], also attributed to Stephen L. Moshier.

## Cephes complete elliptic coefficient tables

The complete K/E polynomial tables in `src/elliptic_family/complete.rs` are adapted from Stephen L. Moshier's Cephes `ellpk.c` and `ellpe.c`, as published in SciPy xsf's `include/xsf/cephes/ellpk.h` and `ellpe.h`: <https://github.com/scipy/xsf/tree/main/include/xsf/cephes>.
Copyright 1984, 1987, 1989 Stephen L. Moshier; C++ translation by SciPy developers (2024).
The original Cephes permission notice is linked above.

## Boost.Math coefficient tables

The low-order Bessel rational tables in `src/bessel_airy/bessel_low_tables.rs` adapt Boost.Math 1.90.0 `detail/bessel_{j0,j1,y0,y1}.hpp`, Copyright (c) 2006 Xiaogang Zhang, under the Boost Software License 1.0 reproduced below: <https://github.com/boostorg/math/tree/boost-1.90.0/include/boost/math/special_functions/detail>.
Root offsets and the local ODE polynomial tables are independently computed with mpmath 1.3.0; no Boost binary or dependency is included.

The piecewise rational coefficient tables and region boundaries used by `src/error_probability/error_fn.rs` are adapted from Boost.Math's `special_functions/detail/erf_inv.hpp`.
The retained ranges include `18 <= sqrt(-ln(q)) < 44`, covering all binary64 subnormal complementary probabilities; the unreachable range above 44 is omitted.

The Temme transition coefficient tables used by `src/gamma_beta/incomplete_gamma.rs` are adapted from Boost.Math's `special_functions/detail/igamma_large.hpp`, whose primary reference is N. M. Temme, *The Asymptotic Expansion of the Incomplete Gamma Functions*, SIAM Journal on Mathematical Analysis 10(4), 1979.

The `g = 6.0246800407767296`, thirteen-term Lanczos rational, zero-factored log-gamma rational approximations, and root-factored digamma rational in `src/support/gamma_real.rs` are adapted from Boost.Math 1.89.0 `lanczos.hpp`, `detail/lgamma_small.hpp` and `digamma.hpp`: <https://www.boost.org/doc/libs/release/libs/math/doc/html/math_toolkit/lanczos.html>.

The initial inverse-gamma and inverse-beta approximations in `src/gamma_beta/probability_tails.rs` adapt equation 31 of DiDonato and Morris (1986) and the first Temme (1992) correction and small-eta inversion from Boost.Math 1.89.0 `detail/igamma_inverse.hpp` and `detail/ibeta_inverse.hpp`.

Copyright John Maddock 2006, Paul A. Bristow 2007, and Matt Borland 2024.

Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization obtaining a copy of the software and accompanying documentation covered by this license (the "Software") to use, reproduce, display, distribute, execute, and transmit the Software, and to prepare derivative works of the Software, and to permit third-parties to whom the Software is furnished to do so, all subject to the following:

The copyright notices in the Software and this entire statement, including the above license grant, this restriction and the following disclaimer, must be included in all copies of the Software, in whole or in part, and all derivative works of the Software, unless such copies or derivative works are solely in the form of machine-executable object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT.
IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Faddeeva probability kernels

`src/error_probability/faddeeva.rs` adapts the Algorithm 916 sums and continued-fraction region selection from Steven G. Johnson's Faddeeva implementation, as maintained at https://github.com/scipy/xsf/blob/main/include/xsf/faddeeva.h (retrieved 2026-09-09).
Primary algorithms: Zaghloul and Ali, ACM TOMS 38(2), Algorithm 916 (2011), and Poppe and Wijers, ACM TOMS 16(1), 38–46 (1990).

Copyright (c) 2012 Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Owen T uses the symmetry identities and Gaussian-quadrature method T5 of Patefield and Tandy, *Fast and Accurate Calculation of Owen's T Function*, JSS 5(5), 2000, https://doi.org/10.18637/jss.v005.i05.
Quadrature nodes are independently generated with mpmath; no Boost implementation is copied.

## Lambert, omega, dilogarithm and Lerch formulas

The Lambert/polylog implementation is independently written from the formulas of Corless et al. (1996), Maximon (2003), Erdelyi et al. (1953), and DLMF sections 4.13, 25.12 and 25.14.
Wright omega's branch convention follows Lawrence, Corless and Jeffrey (2012); its source code is not copied.
Existing Lanczos and Bernoulli coefficients retain their original provenance.
The 32-point Gauss rule is generated by `generate_coefficients.py support`.
Offline references use mpmath 1.3.0 under its BSD license; no Python code or external library is linked into production.

## Hypergeometric qualification

The hypergeometric recurrences, U integral and connection formulas follow DLMF Chapters 13, 15 and 16 and Johansson, *Computing hypergeometric functions rigorously*, ACM TOMS 45(3), 2019 (DOI 10.1145/3328732).
Production kernels are original Rust implementations; existing gamma coefficients remain attributed above.
Offline reference generation uses python-flint 0.9.0 / FLINT-Arb ball arithmetic.
No FLINT source or binary is included in published dependencies.

## Complex Lambert W initialization

The principal-sheet region selector in `src/zeta_polylog/lambert.rs` adapts `_lambertw_approx_hybrid` from mpmath 1.3.0, with signed-zero conjugation and SciOxide branch-point refinement.
Source: <https://github.com/mpmath/mpmath/blob/1.3.0/mpmath/functions/functions.py>.
The logarithmic expansion follows Corless et al., *On the Lambert W function*, Advances in Computational Mathematics 5 (1996), equation (4.19).

Copyright (c) 2005-2021 Fredrik Johansson and mpmath contributors

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  a.
  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  b.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  c.
  Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Real exponential and trigonometric integrals

`src/integrals/real_coefficients.rs` and its rational E1/Ei region formulas adapt Boost.Math 1.90.0 `special_functions/expint.hpp` (binary64 tables), Copyright John Maddock 2007 and Matt Borland 2024, under the Boost Software License 1.0 reproduced above.
Source: <https://github.com/boostorg/math/blob/boost-1.90.0/include/boost/math/special_functions/expint.hpp>.

The Si/Ci central Chebyshev tables and their low components are independently computed from the defining integrals at 200 decimal digits.
The Si/Ci auxiliary rational tables adapt Cephes `sici.c`, Copyright 1984, 1987, 1989 Stephen L. Moshier, maintained in SciPy XSF <https://github.com/scipy/xsf/blob/main/include/xsf/cephes/sici.h>.
The positive En convergents follow DLMF 8.19.17 and Stephen L. Moshier's Cephes `expn.c` (Copyright 1985), as maintained in SciPy's XSF <https://github.com/scipy/xsf/blob/main/include/xsf/cephes/expn.h>.
The independent mpmath fixture generator uses neither production tables nor production evaluation code.

## Cephes original redistribution notice

The Cephes-derived tables identified above and below retain the following additional notice from XSF's [LICENSES_bundled.txt](https://github.com/scipy/xsf/blob/main/LICENSES_bundled.txt).
It applies in addition to the SciPy translation notice; the Netlib permission link is not a replacement for these terms.

```text
Name: Cephes
Files: include/xsf/cephes/*
License: BSD-3-Clause
  Distributed under 3-clause BSD license with permission from the author,
  see https://lists.debian.org/debian-legal/2004/12/msg00295.html

  Cephes Math Library Release 2.8:  June, 2000
  Copyright 1984, 1995, 2000 by Stephen L. Moshier

  This software is derived from the Cephes Math Library and is
  incorporated herein by permission of the author.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
