// Independently computed from the defining series/integrals at 200 decimal digits.
// Chebyshev coefficients use c[0]/2 + sum(c[k] T_k(t)); see the consuming function
// for its coordinate and normalization. Rounded once to binary64.
pub(super) const ACLAUS: [f64; 15] = [
    2.1426943637666884,
    0.07233242812212579,
    0.0010164247502115115,
    3.245250328531645e-05,
    1.3331518757147239e-06,
    6.213240591653388e-08,
    3.13004135336801e-09,
    1.66357230558645e-10,
    9.19659293475523e-12,
    5.240046162580153e-13,
    3.058039524933366e-14,
    1.8196919730321366e-15,
    1.1003982833382514e-16,
    6.745177599309576e-18,
    4.1827846554445156e-19,
];
