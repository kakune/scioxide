//! Constrained optimization by linear approximation (COBYLA).

super::derivative_free::solver!(
    Cobyla,
    CobylaOptions,
    CobylaWorkspace,
    CobylaWorkspaceRequirements,
    CobylaDiagnostics,
    Linear,
    "Minimizes a nonlinear program without derivatives by constrained \
optimization by linear approximation (COBYLA).\n\n\
```text\n\
minimize f(x)  subject to  h(x) = 0,  g(x) ≤ 0,  l ≤ x ≤ u\n\
```\n\n\
The method uses only values of `f` and of the constraints. It fits linear \
models of the objective and of every constraint row to `d + 1` points that \
form a simplex, where `d` is the number of variables that the bounds do not \
fix. Each step first minimizes the largest linearized constraint violation \
within a trust region of radius `Δ`, then minimizes the linearized objective \
without increasing that violation. Steps are accepted with the \
merit function `f(x) + σ v(x)`, where `v` is the largest scaled constraint \
violation and the penalty `σ` grows as needed. Geometry steps keep the simplex \
well conditioned, and the radius decreases from the initial to the final \
radius in [`CobylaOptions::with_radii`].\n\n\
Every evaluation lies inside the intersected box of the problem's \
[`Constraints`](crate::Constraints). The solve converges when the radius has \
reached the final radius, the interpolation points pass the rank test, and the \
model is feasible and stationary to within that resolution. Because no \
gradient is available, the result reports no stationarity measure.\n\n\
# References\n\n\
M. J. D. Powell, \"A direct search optimization method that models the \
objective and constraint functions by linear interpolation\", in *Advances in \
Optimization and Numerical Analysis*, Kluwer, 1994, 51–67."
);
