//! Constrained optimization by quadratic approximation (COBYQA).

super::derivative_free::solver!(
    Cobyqa,
    CobyqaOptions,
    CobyqaWorkspace,
    CobyqaWorkspaceRequirements,
    CobyqaDiagnostics,
    Quadratic,
    "Minimizes a nonlinear program without derivatives by constrained \
optimization by quadratic approximation (COBYQA).\n\n\
```text\n\
minimize f(x)  subject to  h(x) = 0,  g(x) ≤ 0,  l ≤ x ≤ u\n\
```\n\n\
The method uses only values of `f` and of the constraints. It fits quadratic \
models of the objective and of every constraint row to `2d + 1` points, where \
`d` is the number of variables that the bounds do not fix, and updates them by \
minimizing the change of the Hessian in the Frobenius norm. Each step is a \
trust-region SQP step split into a normal step, which reduces the linearized \
constraint violation, and a tangential step, which reduces the model of the \
Lagrangian (Byrd–Omojokun). Steps are accepted with the merit function \
`f(x) + σ v(x)`, where `v` is the largest scaled constraint violation and the \
penalty `σ` grows as needed. Geometry steps keep the interpolation points well \
poised, and the radius decreases from the initial to the final radius in \
[`CobyqaOptions::with_radii`].\n\n\
Every evaluation lies inside the intersected box of the problem's \
[`Constraints`](crate::Constraints). The convergence test and the reported \
quantities are those of [`Cobyla`](crate::Cobyla).\n\n\
# References\n\n\
- T. M. Ragonneau, *Model-Based Derivative-Free Optimization Methods and \
Software*, PhD thesis, The Hong Kong Polytechnic University, 2022.\n\
- M. J. D. Powell, \"Least Frobenius norm updating of quadratic models that \
satisfy interpolation conditions\", *Math. Program.* 100(1), 183–215, 2004."
);
