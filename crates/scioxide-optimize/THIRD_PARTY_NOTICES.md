# Third-party notices

The derivative-free trust-region LP kernels are adapted from PRIMA’s COBYLA through Basin (https://github.com/jolars/basin).

MIT License

Copyright (c) 2026 Johan Larsson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 (https://github.com/libprima/prima)
================================================================================

Basin's Powell-family derivative-free solvers --- COBYLA, NEWUOA, BOBYQA, and LINCOA (src/solver/{cobyla,newuoa,bobyqa,lincoa}{,.rs}) --- are derived from PRIMA, Zaikun Zhang's modern-Fortran reference implementation of M. J. D. Powell's methods.
PRIMA is the authoritative source for the exact formulas and was used as the cross-validation oracle for these solvers.
PRIMA is distributed under the BSD 3-Clause License, whose terms require that the following notice be retained:

    BSD 3-Clause License

    Copyright (c) 2020--2023, Zaikun ZHANG ( https://www.zhangzk.net )

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

## Hock–Schittkowski problem definitions

`tests/data/hs.json` contains independently encoded mathematical problems and independently computed solver results.
Its input definitions and published comparison values were checked against CUTEst/SIF commit `52453b216581d7801cabb6f892b5fce8ecff488a`, problems HS1, HS6, HS14, HS21, HS35, HS51, HS71, HS100 and HS106.
The original source identifiers and hashes remain in the fixture.
The corresponding [SIF redistribution notice](https://github.com/ralna/SIF/blob/52453b216581d7801cabb6f892b5fce8ecff488a/LICENSE) is retained below for the input material; SciOxide's original encoding and computed outputs are MIT OR Apache-2.0.

```text
The CUTEst set of optimization test problems expressed in Standard Input
Format (SIF).
Copyright (C) 2000 Nick Gould, Dominique Orban and Philippe Toint.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Nick I. M. Gould <nick.gould@stfc.ac.uk>
Computational Mathematics Theme
Rutherford Appleton Laboratory
Chilton, Oxfordshire, UK

Dominique Orban <Dominique.Orban@polymtl.ca>
Ecole Polytechnique de Montreal
Department of Mathematics and Industrial Engineering
Montreal QC, Canada

Philippe L. Toint <Philippe.Toint@unamur.be>
Department of Mathematics
Univeristy of Namur
Namur, Belgium.

Licence: BSD 2.0 (3-Clause)
```

## MINPACK and mathematical reference computations

`tests/data/mgh.json` contain mathematical benchmark definitions and independently computed numerical reference results.
The MINPACK input problems and starts are attributed to Jorge J. Moré, Burton S. Garbow and Kenneth E. Hillstrom, Argonne National Laboratory.
No MINPACK Fortran source, compiled library or generator executable is shipped in this crate.
For the associated input material we retain the complete [MINPACK notice](https://netlib.org/minpack/disclaimer), including its acknowledgment:

```text
Minpack Copyright Notice (1999) University of Chicago.  All rights reserved

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above
copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials
provided with the distribution.

3. The end-user documentation included with the
redistribution, if any, must include the following
acknowledgment:

   "This product includes software developed by the
   University of Chicago, as Operator of Argonne National
   Laboratory.

Alternately, this acknowledgment may appear in the software
itself, if and wherever such third-party acknowledgments
normally appear.

4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS"
WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDER, THE
UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND
THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE
OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY
OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF
THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4)
DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION
UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL
BE CORRECTED.

5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT
HOLDER, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF
ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF
ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER
SUCH LIABILITY IS ASSERTED ON THE BASIS OF CONTRACT, TORT
(INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR OTHERWISE,
EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
POSSIBILITY OF SUCH LOSS OR DAMAGES.
```

## Original reference outputs

`tests/data/mgh.json` also includes numerical evaluations made by running ACM CALGO/TOMS 566 separately as an offline reference tool, compared against independently encoded mathematical expressions.
No CALGO source code or compiled object is distributed in this crate.
Numerical output is distinct from the reference program; this notice grants no permission to redistribute CALGO itself.
See [Moré, Garbow and Hillstrom (1981)](https://doi.org/10.1145/355934.355936) for the mathematical problem collection.

The executable residual expressions in `models` are reproducible with [`scripts/generators/mgh_models.py`](https://github.com/kakune/scioxide/blob/main/scripts/generators/mgh_models.py), which constructs 52 models from mathematical formulas without reading CALGO or MINPACK source.
The generator, pinned dependencies, exact comparison command and scope are recorded in the repository's [`DATA_PROVENANCE.md`](https://github.com/kakune/scioxide/blob/main/DATA_PROVENANCE.md#reproducing-the-mgh-problem-definitions) and the fixture's `model_generation` field.
Numerical oracle results and work counts are separate reference data and are not regenerated by this script.
The tabulated MINPACK input observations retain the notice above; the Gaussian observations are computed from the normal density rather than copied from CALGO.

`tests/data/programming.json` consists entirely of original manufactured LP/QP/MIP/QAP instances, with analytic or exhaustively enumerated optima.
It is generated by `scripts/generators/programming_fixtures.py` in the source repository using only the Python standard library and no downloaded problem data.
Its original data and encoding are Copyright (c) 2026 kakune, MIT OR Apache-2.0.
