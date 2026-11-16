use super::*;
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StokesPlan<R, V, D> {
    /// Sample divergence, total flux, discrete continuity and gauge defects.
    /// `length` defines Reynolds number; `step` defines Courant/diffusion numbers.
    /// Scratch needs `LinearOperator::apply_workspace_len()` scalars. No output state is
    /// modified.
    /// # Errors
    /// Rejects invalid states, scratch, nonpositive length, negative step or
    /// unrepresentable arithmetic. Numbers are diagnostics, not stability bounds.
    pub fn diagnostics(
        &self,
        x: &[V],
        length: R,
        step: R,
        scratch: &mut [V],
    ) -> Result<IncompressibleDiagnostics<R>, SciOxideError> {
        self.check(x, x, scratch, self.work())?;
        if !length.is_finite() || length <= R::ZERO || !step.is_finite() || step < R::ZERO {
            return Err(invalid(
                if !length.is_finite() || length <= R::ZERO {
                    scioxide_core::argument!("length")
                } else {
                    scioxide_core::argument!("step")
                },
                "invalid flow diagnostic length or step",
            ));
        }
        scratch[..self.unknowns()].fill(V::ZERO);
        let mut divergence = R::ZERO;
        let mut integrated = V::ZERO;
        let mut flux = V::ZERO;
        let mut speed = R::ZERO;
        let mut courant = R::ZERO;
        let mut diffusion = R::ZERO;
        let nu = finite(
            self.viscosity / self.density,
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;
        for s in &self.samples.volume {
            let (u, g) = self.values(s, x, false)?;
            let div = (0..D).fold(V::ZERO, |v, a| v + g[a][a]);
            divergence = finite(
                divergence + s.weight * div.norm_sqr(),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            integrated = finite(
                integrated + V::from_real(s.weight) * div,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            let velocity = finite(
                u.iter().fold(R::ZERO, |v, &u| v + u.norm_sqr()),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?
            .sqrt();
            speed = speed.max(velocity);
            let h = self.lengths[s.cell];
            courant = courant.max(finite(
                velocity * step / h,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?);
            diffusion = diffusion.max(finite(
                nu * step / h / h,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?);
            for i in 0..=D {
                let row = self.velocity_unknowns() + self.space.dofs(s.cell)[i];
                scratch[row] = finite(
                    scratch[row] - V::from_real(s.weight * self.linear(s)[i]) * div,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        for s in &self.samples.facets {
            let (u, _) = self.values(s, x, false)?;
            let normal = u
                .iter()
                .zip(&s.normal)
                .fold(V::ZERO, |v, (&u, &n)| v + u * V::from_real(n));
            flux = finite(
                flux + normal * V::from_real(s.weight),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
        let weak = finite(
            scratch[self.velocity_unknowns()..self.physical()]
                .iter()
                .fold(R::ZERO, |s, &v| s + v.norm_sqr()),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?
        .sqrt();
        for (v, &c) in self.component.iter().enumerate() {
            if let Some(k) = self.gauge_index[c] {
                let row = self.physical() + k;
                scratch[row] = finite(
                    scratch[row]
                        + V::from_real(self.gauge_weights[v]) * x[self.velocity_unknowns() + v],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        let gauge = self
            .gauge_targets
            .iter()
            .enumerate()
            .try_fold(R::ZERO, |s, (k, &v)| {
                Ok::<_, SciOxideError>(
                    s.max(
                        finite(
                            scratch[self.physical() + k] - v,
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?
                        .abs(),
                    ),
                )
            })?;
        Ok(IncompressibleDiagnostics {
            divergence_l2: divergence.max(R::ZERO).sqrt(),
            integrated_divergence: integrated.abs(),
            boundary_flux: flux.abs(),
            weak_divergence: weak,
            gauge_defect: gauge,
            reynolds: finite(
                speed * length / nu,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?,
            courant,
            diffusion_number: diffusion,
        })
    }
}
