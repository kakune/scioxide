# Third-party notices

## Hairer/Wanner DOP853 coefficients

The stage, weight, error-estimate and continuous-extension coefficient tables in `src/explicit_rk/dormand_prince853.rs` use Hairer and Wanner's DOP853 distribution.
The original source is <https://www.unige.ch/~hairer/prog/nonstiff/dop853.f>.
A versioned comparison copy is [SciPy v1.15.3's dop853.f](https://github.com/scipy/scipy/blob/v1.15.3/scipy/integrate/dop/dop853.f).
This attribution covers the tables; it does not claim that the SciOxide stage runner is a translation of that Fortran program.

The original permission is BSD-2-Clause.
The author's license URL is <https://www.unige.ch/~hairer/prog/licence.txt>.
The text below is reproduced from the explicitly labelled "Original DOP853 License" preserved in the [DOP853 maintainer's LICENSE](https://github.com/jacobwilliams/dop853/blob/master/LICENSE).
Only the damaged quotation marks around "AS IS" in that copy are normalized.
This is the original Hairer notice, not the separate license for Jacob Williams's modern Fortran changes, which are not incorporated here.
Keep this notice with source and binary distributions containing these tables.

```text
Copyright (c) 2004, Ernst Hairer

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

