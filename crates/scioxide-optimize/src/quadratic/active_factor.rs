//! Native active coordinates, bordered free-Hessian factors and general-row QR updates.
#![allow(clippy::needless_range_loop)]
use super::model::{Model, Normal, Row};
use crate::context::SolveSession;
use alloc::vec::Vec;
use scioxide_core::backend::{standard, try_copy_slice, try_filled, try_with_capacity};
use scioxide_core::{EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, HermitianInput, LinalgExecutionOptions,
    LinalgWorkspace, Matrix,
};

#[derive(Clone, Copy)]
enum Active<S> {
    Coordinate { variable: usize, sign: S },
    General { slot: usize },
}
pub(super) struct ActiveFactor<S: RealScalar + NumericScalar<Real = S>> {
    h: Matrix<S>,
    original: Option<CholeskyDecomposition<S>>,
    free: Vec<usize>,
    // Upper Cholesky factor of H[free,free], retained across bound changes.
    upper: Matrix<S>,
    // Contiguous rows are the projected general-row basis vectors (Qᵀ).
    q: Matrix<S>,
    r: Matrix<S>,
    normals: Matrix<S>,
    active: Vec<Active<S>>,
    slots: Vec<usize>,
    general: Vec<(usize, usize)>,
    transformed: Vec<S>,
    coordinates: Vec<S>,
    work: Vec<S>,
    multipliers: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> ActiveFactor<S> {
    pub(super) fn new(
        h: &Matrix<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<Self, SciOxideError> {
        let n = h.rows();
        let mut original = CholeskyDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        let attempt = session.invoke(EvaluationRole::Factorization, || {
            CholeskyDecomposition::factor_into(
                h,
                HermitianInput::Exact,
                CholeskyPivoting::None,
                &LinalgExecutionOptions::new(),
                &mut original,
                &mut workspace,
                session.execution()?,
            )
        });
        let original = crate::dense::positive_definite(attempt)?.then_some(original);
        let mut result = Self {
            h: Matrix::from_vec(
                n,
                n,
                try_copy_slice(h.as_slice(), scioxide_core::quantity!("qp_factor_hessian"))?,
            )?,
            original,
            free: try_with_capacity(n, scioxide_core::quantity!("qp_free_variables"))?,
            upper: Matrix::zeros(n, n)?,
            q: Matrix::zeros(n, n)?,
            r: Matrix::zeros(n, n)?,
            normals: Matrix::zeros(n, n)?,
            active: try_with_capacity(n, scioxide_core::quantity!("qp_active_normals"))?,
            slots: try_with_capacity(n, scioxide_core::quantity!("qp_normal_slots"))?,
            general: try_with_capacity(n, scioxide_core::quantity!("qp_general_order"))?,
            transformed: try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("qp_transformed_normal"),
            )?,
            coordinates: try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("qp_normal_coordinates"),
            )?,
            work: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_factor_work"))?,
            multipliers: try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("qp_general_multipliers"),
            )?,
        };
        result.reset();
        Ok(result)
    }
    pub(super) fn positive_definite(&self) -> bool {
        self.original.is_some()
    }
    fn reset(&mut self) {
        let n = self.h.rows();
        self.free.clear();
        self.free.extend(0..n);
        self.active.clear();
        self.general.clear();
        self.slots.clear();
        self.slots.extend((0..n).rev());
        self.upper.as_mut_slice().fill(S::ZERO);
        if let Some(original) = &self.original {
            for i in 0..n {
                for j in i..n {
                    self.upper.as_mut_slice()[i * n + j] =
                        original.l_matrix().as_slice()[j * n + i];
                }
            }
        }
        self.identity();
    }
    fn identity(&mut self) {
        let n = self.h.rows();
        self.q.as_mut_slice().fill(S::ZERO);
        self.r.as_mut_slice().fill(S::ZERO);
        for i in 0..self.free.len() {
            self.q.as_mut_slice()[i * n + i] = S::ONE;
        }
    }
    fn forward(&mut self) -> Result<(), SciOxideError> {
        if self.original.is_some() {
            let n = self.h.rows();
            let k = self.free.len();
            let start = self.transformed[..k]
                .iter()
                .position(|&v| v != S::ZERO)
                .unwrap_or(k);
            for i in start..k {
                let rhs = self.transformed[i];
                let products =
                    (0..i).map(|j| (self.upper.as_slice()[j * n + i], self.transformed[j]));
                self.transformed[i] = standard::finish_substitution(
                    rhs,
                    self.upper.as_slice()[i * n + i],
                    rhs - standard::dot(products.clone()),
                    products,
                    None,
                )?;
            }
        }
        crate::utils::validate_finite(&self.transformed[..self.free.len()])
    }
    fn rotate_q(&mut self, i: usize, j: usize, c: S, s: S) {
        let n = self.h.rows();
        let k = self.free.len();
        let (top, bottom) = self.q.as_mut_slice().split_at_mut(j * n);
        for (x, y) in top[i * n..i * n + k].iter_mut().zip(&mut bottom[..k]) {
            let (a, b) = (*x, *y);
            *x = c * a + s * b;
            *y = c * b - s * a;
        }
    }
    fn add_general(
        &mut self,
        slot: usize,
        rank: scioxide_core::RankThreshold<S>,
    ) -> Result<bool, SciOxideError> {
        let n = self.h.rows();
        let k = self.free.len();
        let m = self.general.len();
        if m == k {
            return Ok(false);
        }
        for (j, &variable) in self.free.iter().enumerate() {
            self.transformed[j] = self.normals.as_slice()[slot * n + variable];
        }
        self.forward()?;
        for j in 0..k {
            self.coordinates[j] = standard::dot(
                self.q.as_slice()[j * n..j * n + k]
                    .iter()
                    .zip(&self.transformed[..k])
                    .map(|(&a, &b)| (a, b)),
            );
        }
        crate::utils::validate_finite(&self.coordinates[..k])?;
        if crate::utils::norm2(&self.coordinates[m..k])?
            <= rank.threshold(crate::utils::norm2(&self.coordinates[..k])?)?
        {
            return Ok(false);
        }
        for j in (m + 1..k).rev() {
            let (a, b) = (self.coordinates[j - 1], self.coordinates[j]);
            let norm = a.hypot(b);
            if norm != S::ZERO {
                self.coordinates[j - 1] = norm;
                self.coordinates[j] = S::ZERO;
                self.rotate_q(j - 1, j, a / norm, b / norm);
            }
        }
        for i in 0..=m {
            self.r.as_mut_slice()[i * n + m] = self.coordinates[i];
        }
        Ok(true)
    }
    fn rebuild_general(
        &mut self,
        rank: scioxide_core::RankThreshold<S>,
    ) -> Result<bool, SciOxideError> {
        self.identity();
        self.general.clear();
        for position in 0..self.active.len() {
            if let Active::General { slot } = self.active[position] {
                if !self.add_general(slot, rank)? {
                    return Ok(false);
                }
                self.general.push((position, slot));
            }
        }
        Ok(true)
    }
    fn remove_free(&mut self, position: usize) {
        let n = self.h.rows();
        let k = self.free.len();
        if self.original.is_some() {
            for i in 0..k {
                for j in position..k - 1 {
                    self.upper.as_mut_slice()[i * n + j] = self.upper.as_slice()[i * n + j + 1];
                }
                self.upper.as_mut_slice()[i * n + k - 1] = S::ZERO;
            }
            for j in position..k - 1 {
                let (a, b) = (
                    self.upper.as_slice()[j * n + j],
                    self.upper.as_slice()[(j + 1) * n + j],
                );
                let norm = a.hypot(b);
                if norm != S::ZERO {
                    let (c, s) = (a / norm, b / norm);
                    for col in j..k - 1 {
                        let (a, b) = (
                            self.upper.as_slice()[j * n + col],
                            self.upper.as_slice()[(j + 1) * n + col],
                        );
                        self.upper.as_mut_slice()[j * n + col] = c * a + s * b;
                        self.upper.as_mut_slice()[(j + 1) * n + col] = c * b - s * a;
                    }
                }
                self.upper.as_mut_slice()[(j + 1) * n + j] = S::ZERO;
            }
        }
        self.free.remove(position);
    }
    fn add_free(&mut self, variable: usize) -> Result<(), SciOxideError> {
        let n = self.h.rows();
        let k = self.free.len();
        if self.original.is_some() {
            for i in 0..k {
                self.transformed[i] = self.h.as_slice()[self.free[i] * n + variable];
            }
            self.forward()?;
            let squared = standard::dot(self.transformed[..k].iter().map(|&x| (x, x)));
            let pivot = scioxide_core::checked_difference(
                self.h.as_slice()[variable * n + variable],
                squared,
            )?;
            if pivot <= S::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "free-Hessian Cholesky border lost positive definiteness",
                ));
            }
            for i in 0..k {
                self.upper.as_mut_slice()[i * n + k] = self.transformed[i];
                self.upper.as_mut_slice()[k * n + i] = S::ZERO;
            }
            self.upper.as_mut_slice()[k * n + k] = pivot.sqrt();
        }
        self.free.push(variable);
        Ok(())
    }
    /// Native coordinate changes update only the free-Hessian factor. General
    /// normals use QR insertion/deletion; their projected basis is refreshed
    /// when the free coordinate space changes.
    pub(super) fn insert(
        &mut self,
        row: &Row<S>,
        rank: scioxide_core::RankThreshold<S>,
    ) -> Result<bool, SciOxideError> {
        let n = self.h.rows();
        match &row.normal {
            Normal::Coordinate { variable, sign } => {
                let Some(position) = self.free.iter().position(|v| v == variable) else {
                    return Ok(false);
                };
                if self.free.len() <= self.general.len() {
                    return Ok(false);
                }
                self.remove_free(position);
                if !self.rebuild_general(rank)? {
                    self.add_free(*variable)?;
                    if !self.rebuild_general(rank)? {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Breakdown,
                            "restoring a dependent bound lost general-row rank",
                        ));
                    }
                    return Ok(false);
                }
                self.active.push(Active::Coordinate {
                    variable: *variable,
                    sign: *sign,
                });
            }
            Normal::General(a) => {
                let Some(&slot) = self.slots.last() else {
                    return Ok(false);
                };
                self.normals.as_mut_slice()[slot * n..(slot + 1) * n].copy_from_slice(a);
                if !self.add_general(slot, rank)? {
                    return Ok(false);
                }
                self.slots.pop();
                self.general.push((self.active.len(), slot));
                self.active.push(Active::General { slot });
            }
        }
        Ok(true)
    }
    pub(super) fn delete(
        &mut self,
        position: usize,
        rank: scioxide_core::RankThreshold<S>,
    ) -> Result<(), SciOxideError> {
        match self.active.remove(position) {
            Active::Coordinate { variable, .. } => {
                self.add_free(variable)?;
                if !self.rebuild_general(rank)? {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "releasing a bound lost general-row rank",
                    ));
                }
            }
            Active::General { slot } => {
                let n = self.h.rows();
                let m = self.general.len();
                let column = self
                    .general
                    .iter()
                    .position(|&(_, s)| s == slot)
                    .ok_or_else(|| SciOxideError::singular(None))?;
                self.general.remove(column);
                self.slots.push(slot);
                for i in 0..m {
                    for j in column..m - 1 {
                        self.r.as_mut_slice()[i * n + j] = self.r.as_slice()[i * n + j + 1];
                    }
                    self.r.as_mut_slice()[i * n + m - 1] = S::ZERO;
                }
                for j in column..m - 1 {
                    let (a, b) = (
                        self.r.as_slice()[j * n + j],
                        self.r.as_slice()[(j + 1) * n + j],
                    );
                    let norm = a.hypot(b);
                    if norm != S::ZERO {
                        let (c, s) = (a / norm, b / norm);
                        for col in j..m - 1 {
                            let (a, b) = (
                                self.r.as_slice()[j * n + col],
                                self.r.as_slice()[(j + 1) * n + col],
                            );
                            self.r.as_mut_slice()[j * n + col] = c * a + s * b;
                            self.r.as_mut_slice()[(j + 1) * n + col] = c * b - s * a;
                        }
                        self.rotate_q(j, j + 1, c, s);
                    }
                    self.r.as_mut_slice()[(j + 1) * n + j] = S::ZERO;
                }
                for (p, _) in &mut self.general {
                    if *p > position {
                        *p -= 1;
                    }
                }
            }
        }
        Ok(())
    }
    /// Initialize a candidate set by eliminating all native active coordinates
    /// before preparing the reduced factor. Dependence restores the empty set.
    pub(super) fn initialize(
        &mut self,
        model: &Model<S>,
        indices: &[usize],
        rank: scioxide_core::RankThreshold<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let n = self.h.rows();
        if indices.len() > n {
            return Ok(false);
        }
        for (position, &index) in indices.iter().enumerate() {
            if let Normal::Coordinate { variable, .. } = model.rows[index].normal
                && indices[..position].iter().any(|&i|matches!(model.rows[i].normal,Normal::Coordinate{variable:v,..}if v==variable)) {
                return Ok(false);
            }
        }
        self.free.clear();
        self.free.extend((0..n).filter(|&v| {
            !indices.iter().any(
                |&i| matches!(model.rows[i].normal,Normal::Coordinate{variable,..}if variable==v),
            )
        }));
        let k = self.free.len();
        let general = indices
            .iter()
            .filter(|&&i| matches!(model.rows[i].normal, Normal::General(_)))
            .count();
        if general > k {
            self.reset();
            return Ok(false);
        }
        if self.original.is_some() && k != n && k != 0 {
            let mut h = Matrix::zeros(k, k)?;
            for (i, &a) in self.free.iter().enumerate() {
                for (j, &b) in self.free.iter().enumerate() {
                    h.view_mut().set(i, j, self.h.as_slice()[a * n + b])?;
                }
            }
            let mut lower = CholeskyDecomposition::empty();
            let mut workspace = LinalgWorkspace::new();
            session.invoke(EvaluationRole::Factorization, || {
                CholeskyDecomposition::factor_into(
                    &h,
                    HermitianInput::Exact,
                    CholeskyPivoting::None,
                    &LinalgExecutionOptions::new(),
                    &mut lower,
                    &mut workspace,
                    session.execution()?,
                )
            })?;
            self.upper.as_mut_slice().fill(S::ZERO);
            for i in 0..k {
                for j in i..k {
                    self.upper.as_mut_slice()[i * n + j] = lower.l_matrix().as_slice()[j * k + i];
                }
            }
        }
        self.identity();
        for &i in indices {
            match &model.rows[i].normal {
                Normal::Coordinate { variable, sign } => self.active.push(Active::Coordinate {
                    variable: *variable,
                    sign: *sign,
                }),
                Normal::General(a) => {
                    let slot = self
                        .slots
                        .pop()
                        .ok_or_else(|| SciOxideError::singular(None))?;
                    self.normals.as_mut_slice()[slot * n..(slot + 1) * n].copy_from_slice(a);
                    if !self.add_general(slot, rank)? {
                        self.reset();
                        return Ok(false);
                    }
                    self.general.push((self.active.len(), slot));
                    self.active.push(Active::General { slot });
                }
            }
        }
        Ok(true)
    }
    pub(super) fn solve(
        &mut self,
        g: &[S],
        p: &mut [S],
        lambda: &mut [S],
    ) -> Result<(), SciOxideError> {
        let n = self.h.rows();
        let k = self.free.len();
        let m = self.general.len();
        for (i, &variable) in self.free.iter().enumerate() {
            self.transformed[i] = g[variable];
        }
        self.forward()?;
        for j in 0..k {
            self.coordinates[j] = standard::dot(
                self.q.as_slice()[j * n..j * n + k]
                    .iter()
                    .zip(&self.transformed[..k])
                    .map(|(&a, &b)| (a, b)),
            );
        }
        self.work[..k].fill(S::ZERO);
        for j in m..k {
            for (value, &coefficient) in self.work[..k]
                .iter_mut()
                .zip(&self.q.as_slice()[j * n..j * n + k])
            {
                *value -= coefficient * self.coordinates[j];
            }
        }
        if self.original.is_some() {
            for i in (0..k).rev() {
                let rhs = self.work[i];
                let products = (i + 1..k).map(|j| (self.upper.as_slice()[i * n + j], self.work[j]));
                self.work[i] = standard::finish_substitution(
                    rhs,
                    self.upper.as_slice()[i * n + i],
                    rhs - standard::dot(products.clone()),
                    products,
                    None,
                )?;
            }
        }
        p.fill(S::ZERO);
        for (&variable, &value) in self.free.iter().zip(&self.work) {
            p[variable] = value;
        }
        for i in (0..m).rev() {
            let rhs = -self.coordinates[i];
            let products = (i + 1..m).map(|j| (self.r.as_slice()[i * n + j], self.multipliers[j]));
            self.multipliers[i] = standard::finish_substitution(
                rhs,
                self.r.as_slice()[i * n + i],
                rhs - standard::dot(products.clone()),
                products,
                None,
            )?;
        }
        for (j, &(position, _)) in self.general.iter().enumerate() {
            lambda[position] = self.multipliers[j];
        }
        for (position, active) in self.active.iter().enumerate() {
            if let Active::Coordinate { variable, sign } = *active {
                let residual = crate::utils::compensated_products(|| {
                    core::iter::once((g[variable], S::ONE))
                        .chain(
                            self.free
                                .iter()
                                .map(|&j| (self.h.as_slice()[variable * n + j], p[j])),
                        )
                        .chain(self.general.iter().map(|&(position, slot)| {
                            (
                                self.normals.as_slice()[slot * n + variable],
                                lambda[position],
                            )
                        }))
                })?;
                lambda[position] = -residual / sign;
            }
        }
        crate::utils::validate_finite(p)?;
        crate::utils::validate_finite(&lambda[..self.active.len()])
    }
}
