//! Block drivers and the evaluation entry points of [`BatchPlan`].

use core::num::NonZeroUsize;

use scioxide_core::backend::PartitionSlot;
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, InvalidInputReason, NonFiniteOrigin, PlanLimits, RealScalar,
    SciOxideError,
};

use super::reduce;
use super::workspace::{BlockStats, MAX_WORKERS, Worker};
use super::{
    BatchData, BatchEstimate, BatchFunction, BatchMode, BatchOutputs, BatchPlan, BatchReduction,
    BatchWorkspace,
};
use crate::{DerivativeMode, Dual, KinkPolicy, Lanes, MultiDual, NonFinitePolicy, Var};

/// What a block computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Value,
    Forward(usize),
    Reverse,
    Jvp,
    Hvp,
    HvpForward(usize),
}

/// The tangent count of `active` values, one to four.
const fn tangents(active: usize) -> usize {
    match active {
        0 | 1 => 1,
        2 => 2,
        3 => 3,
        _ => 4,
    }
}

/// The borrowed arguments of one evaluation, shared by every worker.
pub(crate) struct Run<'a, S, F: ?Sized, A: ?Sized, D: ?Sized> {
    function: &'a F,
    parameters: &'a [S],
    inputs: &'a A,
    data: &'a D,
    parameter_direction: &'a [S],
    input_direction: Option<&'a A>,
    kind: Kind,
    items: usize,
    block: usize,
    widths: [usize; 3],
    stage_inputs: bool,
    check: bool,
    moments: bool,
}

/// The slices one block writes.
struct Out<'a, S> {
    stats: &'a mut BlockStats<S>,
    gradient: &'a mut [S],
    values: &'a mut [S],
    input_gradients: &'a mut [S],
    derivatives: &'a mut [S],
}

/// The blocks, worker and output slices of one partition.
struct Job<'a, S: RealScalar, const W: usize> {
    first: usize,
    worker: &'a mut Worker<S, W>,
    stats: &'a mut [BlockStats<S>],
    gradients: &'a mut [S],
    values: &'a mut [S],
    input_gradients: &'a mut [S],
    derivatives: &'a mut [S],
}

/// Stack storage on a cache-line boundary. The packs of the active inputs are
/// written and read with full-width loads and stores once per group; aligned,
/// none of those straddles two cache lines.
#[repr(align(64))]
struct Aligned<T>(T);

fn non_finite(origin: NonFiniteOrigin) -> SciOxideError {
    SciOxideError::non_finite(origin, "a batch value is not finite")
}

#[inline]
fn count<S: RealScalar>(n: usize) -> S {
    S::from_f64_ieee(n as f64)
}

/// Fills `targets`, one pack per value, with the values of the items
/// `start..start + valid`; missing lanes repeat the last item.
///
/// A full group hands the source a whole pack, so its length is a constant and a
/// source with adjacent values copies with full-width stores.
#[inline(always)]
fn load<'t, S, A, const W: usize>(
    source: &A,
    start: usize,
    valid: usize,
    targets: impl ExactSizeIterator<Item = &'t mut Lanes<S, W>>,
    check: bool,
) -> Result<(), SciOxideError>
where
    S: RealScalar + 't,
    A: BatchData<S> + ?Sized,
{
    let width = targets.len();
    for (column, target) in targets.enumerate() {
        let lanes = target.as_mut_array();
        if valid == W {
            source.fill(start, width, column, lanes)?;
        } else {
            pad(source, start, width, column, lanes, valid)?;
        }
        if check && lanes.iter().any(|value| !value.is_finite()) {
            return Err(non_finite(NonFiniteOrigin::Input));
        }
    }
    Ok(())
}

/// Fills the first `valid` lanes of the partial last group and repeats the last
/// of them in the others.
#[cold]
#[inline(never)]
fn pad<S, A, const W: usize>(
    source: &A,
    start: usize,
    width: usize,
    column: usize,
    lanes: &mut [S; W],
    valid: usize,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    A: BatchData<S> + ?Sized,
{
    let (head, tail) = lanes.split_at_mut(valid.min(W));
    source.fill(start, width, column, head)?;
    if let Some(&last) = head.last() {
        tail.fill(last);
    }
    Ok(())
}

/// Adds the values of the first `valid` lanes, in item order, and stages them.
#[inline(always)]
fn push_values<S: RealScalar, const W: usize>(
    stats: &mut BlockStats<S>,
    values: &mut [S],
    offset: usize,
    lanes: &Lanes<S, W>,
    valid: usize,
) {
    let lanes = lanes.as_array();
    stats.count += valid;
    if valid == W {
        for &x in lanes {
            stats.sum += x;
        }
    } else {
        for &x in &lanes[..valid] {
            stats.sum += x;
        }
    }
    let targets = values.get_mut(offset..offset + valid);
    match targets.map(<&mut [S; W]>::try_from) {
        Some(Ok(targets)) => *targets = *lanes,
        Some(Err(_)) => {
            for (target, &x) in values[offset..offset + valid].iter_mut().zip(lanes) {
                *target = x;
            }
        }
        None => {}
    }
}

/// Writes lane `l` of column `k` to element `k` of row `l`, for `K` columns.
#[inline(always)]
fn stage<S: RealScalar, const W: usize, const K: usize>(rows: &mut [S], columns: &[Lanes<S, W>]) {
    let (rows, _) = rows.as_chunks_mut::<K>();
    let Ok(columns) = <&[Lanes<S, W>; K]>::try_from(columns) else {
        return;
    };
    if let Ok(rows) = <&mut [[S; K]; W]>::try_from(&mut *rows) {
        for (lane, row) in rows.iter_mut().enumerate() {
            for (target, column) in row.iter_mut().zip(columns) {
                *target = column.as_array()[lane];
            }
        }
    } else {
        for (lane, row) in rows.iter_mut().enumerate() {
            for (target, column) in row.iter_mut().zip(columns) {
                *target = column.as_array()[lane];
            }
        }
    }
}

/// Sets the block mean and the sum of squared deviations from the staged values.
///
/// Item `i` goes to accumulator `i mod 8` and the accumulators are added
/// pairwise, so the order is fixed and the eight sums run in parallel.
fn moments<S: RealScalar>(stats: &mut BlockStats<S>, values: &[S]) {
    if stats.count == 0 {
        return;
    }
    let mean = stats.sum / count::<S>(stats.count);
    stats.mean = mean;
    let values = &values[..stats.count];
    let mut sums = [S::ZERO; 8];
    let chunks = values.chunks_exact(8);
    let rest = chunks.remainder();
    for chunk in chunks {
        for (sum, &x) in sums.iter_mut().zip(chunk) {
            *sum += (x - mean) * (x - mean);
        }
    }
    for (sum, &x) in sums.iter_mut().zip(rest) {
        *sum += (x - mean) * (x - mean);
    }
    let [a, b, c, d, e, f, g, h] = sums;
    stats.m2 = ((a + b) + (c + d)) + ((e + f) + (g + h));
}

/// The unit tangents `e_k` of `N` directions; `k ≥ N` gives zero.
fn unit<S: RealScalar, const W: usize, const N: usize>(k: usize) -> [Lanes<S, W>; N] {
    core::array::from_fn(|d| Lanes::splat(if d == k { S::ONE } else { S::ZERO }))
}

fn seed<S: RealScalar, const W: usize>(valid: usize) -> Lanes<S, W> {
    Lanes::from_fn(|lane| if lane < valid { S::ONE } else { S::ZERO })
}

/// The groups of a block: `(first item, valid lanes)`.
fn groups<const W: usize>(first: usize, last: usize) -> impl Iterator<Item = (usize, usize)> {
    (first..last)
        .step_by(W)
        .map(move |start| (start, (last - start).min(W)))
}

impl<S, F, A, D> Run<'_, S, F, A, D>
where
    S: RealScalar,
    F: BatchFunction<S> + ?Sized,
    A: BatchData<S> + ?Sized,
    D: BatchData<S> + ?Sized,
{
    fn block<const W: usize>(
        &self,
        block: usize,
        worker: &mut Worker<S, W>,
        out: Out<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let first = block * self.block;
        let last = (first + self.block).min(self.items);
        match self.kind {
            Kind::Value => self.value_block(first, last, worker, out),
            Kind::Reverse | Kind::Hvp => self.reverse_block(first, last, worker, out, context),
            Kind::Forward(1) => self.forward_block::<W, 1>(first, last, worker, out),
            Kind::Forward(2) => self.forward_block::<W, 2>(first, last, worker, out),
            Kind::Forward(3) => self.forward_block::<W, 3>(first, last, worker, out),
            Kind::Forward(_) => self.forward_block::<W, 4>(first, last, worker, out),
            Kind::HvpForward(1) => self.second_block::<W, 1>(first, last, worker, out),
            Kind::HvpForward(2) => self.second_block::<W, 2>(first, last, worker, out),
            Kind::HvpForward(3) => self.second_block::<W, 3>(first, last, worker, out),
            Kind::HvpForward(_) => self.second_block::<W, 4>(first, last, worker, out),
            Kind::Jvp => self.jvp_block(first, last, worker, out),
        }
    }

    /// Completes the statistics of a block: the mean and the second moment are
    /// needed only for the standard error of a mean.
    fn close(&self, stats: &mut BlockStats<S>, values: &[S]) {
        if self.moments {
            moments(stats, values);
        }
    }

    fn value_block<const W: usize>(
        &self,
        first: usize,
        last: usize,
        worker: &mut Worker<S, W>,
        out: Out<'_, S>,
    ) -> Result<(), SciOxideError> {
        let [np, ni, nd] = self.widths;
        let (parameters, data, inputs) = (
            &mut worker.parameters[..np],
            &mut worker.data[..nd],
            &mut worker.inputs[..ni],
        );
        for (lane, &p) in parameters.iter_mut().zip(self.parameters) {
            *lane = Lanes::splat(p);
        }
        let shared = self.function.prepare(parameters)?;
        *out.stats = BlockStats::empty();
        for (start, valid) in groups::<W>(first, last) {
            load(self.data, start, valid, data.iter_mut(), false)?;
            load(self.inputs, start, valid, inputs.iter_mut(), self.check)?;
            let y = self.function.item(&shared, inputs, data)?;
            push_values(out.stats, out.values, start - first, &y, valid);
        }
        self.close(out.stats, out.values);
        Ok(())
    }

    /// Forward mode with `N` tangents: parameters carry unit tangents in `prepare`,
    /// inputs in `item`.
    fn forward_block<const W: usize, const N: usize>(
        &self,
        first: usize,
        last: usize,
        worker: &mut Worker<S, W>,
        out: Out<'_, S>,
    ) -> Result<(), SciOxideError> {
        let [np, ni, nd] = self.widths;
        let zero = Lanes::splat(S::ZERO);
        let mut parameters = [MultiDual::new(zero, [zero; N]); 4];
        for (k, (target, &p)) in parameters.iter_mut().zip(self.parameters).enumerate() {
            *target = MultiDual::new(Lanes::splat(p), unit(k));
        }
        let shared = self.function.prepare(&parameters[..np])?;
        let mut inputs: Aligned<[_; 4]> =
            Aligned(core::array::from_fn(|j| MultiDual::new(zero, unit(np + j))));
        let (inputs, data) = (&mut inputs.0[..ni], &mut worker.data[..nd]);
        *out.stats = BlockStats::empty();
        out.gradient.fill(S::ZERO);
        for (start, valid) in groups::<W>(first, last) {
            load(self.data, start, valid, data.iter_mut(), false)?;
            let values = inputs.iter_mut().map(MultiDual::value_mut);
            load(self.inputs, start, valid, values, self.check)?;
            let result = self.function.item(&shared, inputs, data);
            let Ok(y) = &result else {
                return result.map(drop);
            };
            let (value, tangents) = y.parts();
            push_values(out.stats, out.values, start - first, value, valid);
            self.push_gradients(
                out.gradient,
                out.input_gradients,
                start - first,
                tangents,
                valid,
            );
        }
        self.close(out.stats, out.values);
        Ok(())
    }

    /// Hessian-vector products in forward mode: [`Self::forward_block`] with every
    /// value also a dual in the direction `(u, w)`.
    fn second_block<const W: usize, const N: usize>(
        &self,
        first: usize,
        last: usize,
        worker: &mut Worker<S, W>,
        out: Out<'_, S>,
    ) -> Result<(), SciOxideError> {
        let [np, ni, nd] = self.widths;
        let zero = Lanes::splat(S::ZERO);
        let lift = |value: Lanes<S, W>, tangents: [Lanes<S, W>; N], direction: Lanes<S, W>| {
            Dual::new(
                MultiDual::new(value, tangents),
                MultiDual::new(direction, [zero; N]),
            )
        };
        let mut parameters = [lift(zero, [zero; N], zero); 4];
        for (k, (target, &p)) in parameters.iter_mut().zip(self.parameters).enumerate() {
            let u = self.parameter_direction.get(k).copied().unwrap_or(S::ZERO);
            *target = lift(Lanes::splat(p), unit(k), Lanes::splat(u));
        }
        let shared = self.function.prepare(&parameters[..np])?;
        let mut inputs: Aligned<[_; 4]> =
            Aligned(core::array::from_fn(|j| lift(zero, unit(np + j), zero)));
        let (inputs, data) = (&mut inputs.0[..ni], &mut worker.data[..nd]);
        *out.stats = BlockStats::empty();
        out.gradient.fill(S::ZERO);
        for (start, valid) in groups::<W>(first, last) {
            load(self.data, start, valid, data.iter_mut(), false)?;
            let values = inputs.iter_mut().map(|x| x.parts_mut().0.value_mut());
            load(self.inputs, start, valid, values, self.check)?;
            if let Some(direction) = self.input_direction {
                let directions = inputs.iter_mut().map(|x| x.parts_mut().1.value_mut());
                load(direction, start, valid, directions, self.check)?;
            }
            let result = self.function.item(&shared, inputs, data);
            let Ok(y) = &result else {
                return result.map(drop);
            };
            let (value, derivative) = y.parts();
            push_values(out.stats, out.values, start - first, value.parts().0, valid);
            self.push_gradients(
                out.gradient,
                out.input_gradients,
                start - first,
                derivative.parts().1,
                valid,
            );
        }
        self.close(out.stats, out.values);
        Ok(())
    }

    /// Adds the parameter tangents of the first `valid` lanes in item order and
    /// stages the input tangents.
    #[inline(always)]
    fn push_gradients<const W: usize, const N: usize>(
        &self,
        gradient: &mut [S],
        input_gradients: &mut [S],
        offset: usize,
        tangents: &[Lanes<S, W>; N],
        valid: usize,
    ) {
        let [np, ni, _] = self.widths;
        for (target, tangent) in gradient.iter_mut().zip(tangents).take(np) {
            for &value in tangent.as_array().iter().take(valid) {
                *target += value;
            }
        }
        if self.stage_inputs && ni > 0 {
            let rows = &mut input_gradients[offset * ni..(offset + valid) * ni];
            let columns = &tangents[np..np + ni];
            match ni {
                1 => stage::<S, W, 1>(rows, columns),
                2 => stage::<S, W, 2>(rows, columns),
                3 => stage::<S, W, 3>(rows, columns),
                4 => stage::<S, W, 4>(rows, columns),
                _ => {
                    for (lane, row) in rows.chunks_exact_mut(ni).enumerate() {
                        for (target, column) in row.iter_mut().zip(columns) {
                            *target = column.as_array()[lane];
                        }
                    }
                }
            }
        }
    }

    fn jvp_block<const W: usize>(
        &self,
        first: usize,
        last: usize,
        worker: &mut Worker<S, W>,
        out: Out<'_, S>,
    ) -> Result<(), SciOxideError> {
        let [np, ni, nd] = self.widths;
        let data = &mut worker.data[..nd];
        let (parameters, items) = worker.duals[..np + ni].split_at_mut(np);
        for ((dual, &p), &u) in parameters
            .iter_mut()
            .zip(self.parameters)
            .zip(self.parameter_direction)
        {
            *dual = Dual::new(Lanes::splat(p), Lanes::splat(u));
        }
        let shared = self.function.prepare(parameters)?;
        if self.input_direction.is_none() {
            for dual in items.iter_mut() {
                *dual.parts_mut().1 = Lanes::splat(S::ZERO);
            }
        }
        *out.stats = BlockStats::empty();
        for (start, valid) in groups::<W>(first, last) {
            load(self.data, start, valid, data.iter_mut(), false)?;
            let values = items.iter_mut().map(|x| x.parts_mut().0);
            load(self.inputs, start, valid, values, self.check)?;
            if let Some(direction) = self.input_direction {
                let tangents = items.iter_mut().map(|x| x.parts_mut().1);
                load(direction, start, valid, tangents, self.check)?;
            }
            let y = self.function.item(&shared, items, data)?;
            let (value, derivative) = y.parts();
            push_values(out.stats, out.values, start - first, value, valid);
            for (lane, &d) in derivative.as_array().iter().enumerate().take(valid) {
                out.stats.derivative += d;
                if let Some(target) = out.derivatives.get_mut(start - first + lane) {
                    *target = d;
                }
            }
        }
        self.close(out.stats, out.values);
        Ok(())
    }

    fn reverse_block<const W: usize>(
        &self,
        first: usize,
        last: usize,
        worker: &mut Worker<S, W>,
        out: Out<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let [_, ni, nd] = self.widths;
        let Worker {
            data,
            inputs,
            directions,
            tape,
            second,
            ..
        } = worker;
        let (data, inputs, directions) =
            (&mut data[..nd], &mut inputs[..ni], &mut directions[..ni]);
        let lanes = Lanes::splat;
        if self.kind == Kind::Hvp {
            let zero = lanes(S::ZERO);
            let direction = self.parameter_direction;
            let parameter = |k: usize, p: S| Dual::new(lanes(p), lanes(direction[k]));
            let input = |x: Lanes<S, W>, w: Lanes<S, W>| Dual::new(x, w);
            self.tape_block(
                first,
                last,
                data,
                inputs,
                directions,
                second,
                out,
                context,
                parameter,
                input,
                |seed| Dual::new(seed, zero),
                Dual::value,
                Dual::derivative,
            )
        } else {
            self.tape_block(
                first,
                last,
                data,
                inputs,
                directions,
                tape,
                out,
                context,
                |_, p| lanes(p),
                |x, _| x,
                |seed| seed,
                |y| y,
                |g| g,
            )
        }
    }

    /// Records a block on a tape of `V`: the parameters once, then each group,
    /// swept down to the mark; the prefix is swept once at the end.
    #[allow(clippy::too_many_arguments)]
    fn tape_block<V, const W: usize>(
        &self,
        first: usize,
        last: usize,
        data: &mut [Lanes<S, W>],
        input_lanes: &mut [Lanes<S, W>],
        direction_lanes: &mut [Lanes<S, W>],
        tape: &mut crate::Tape<V>,
        out: Out<'_, S>,
        context: ExecutionContext<'_>,
        parameter: impl Fn(usize, S) -> V,
        input: impl Fn(Lanes<S, W>, Lanes<S, W>) -> V,
        seed_of: impl Fn(Lanes<S, W>) -> V,
        value_of: impl Fn(V) -> Lanes<S, W>,
        adjoint_of: impl Fn(V) -> Lanes<S, W>,
    ) -> Result<(), SciOxideError>
    where
        V: crate::TapeValue<Scalar = S> + crate::Real<S, Base = Lanes<S, W>>,
    {
        let [np, ni, _] = self.widths;
        let Out {
            stats,
            gradient,
            values,
            input_gradients,
            ..
        } = out;
        tape.session(np + ni, |session, (parameters, inputs)| {
            *stats = BlockStats::empty();
            parameters.clear();
            session.set_valid_lanes(1);
            parameters.extend(
                self.parameters
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| session.input(parameter(k, p))),
            );
            let shared = self.function.prepare(parameters)?;
            // After an overflow the recording continues without sweeps, so the
            // peak lengths of every group are known before the tape grows.
            let mut overflow = session.overflowed();
            let mark = session.mark();
            for (start, valid) in groups::<W>(first, last) {
                load(self.data, start, valid, data.iter_mut(), false)?;
                load(
                    self.inputs,
                    start,
                    valid,
                    input_lanes.iter_mut(),
                    self.check,
                )?;
                if let Some(direction) = self.input_direction {
                    let targets = direction_lanes.iter_mut();
                    load(direction, start, valid, targets, self.check)?;
                }
                inputs.clear();
                inputs.extend(
                    input_lanes
                        .iter()
                        .zip(direction_lanes.iter())
                        .map(|(&lane, &tangent)| session.input(input(lane, tangent))),
                );
                session.set_valid_lanes(valid);
                let y: Var<'_, V> = self.function.item(&shared, inputs, data)?;
                overflow |= session.overflowed();
                push_values(stats, values, start - first, &value_of(y.value()), valid);
                if overflow {
                    session.rewind(mark);
                    continue;
                }
                session.seed(y, seed_of(seed(valid)));
                session.sweep(mark, context)?;
                for (j, &input) in inputs.iter().enumerate() {
                    let adjoint = adjoint_of(session.take_adjoint(input));
                    if self.stage_inputs {
                        for (lane, &g) in adjoint.as_array().iter().enumerate().take(valid) {
                            input_gradients[(start - first + lane) * ni + j] = g;
                        }
                    }
                }
                session.rewind(mark);
            }
            if overflow {
                return Ok(None);
            }
            session.sweep_between(mark, Default::default(), context)?;
            for (target, &parameter) in gradient.iter_mut().zip(parameters.iter()) {
                let adjoint = adjoint_of(session.take_adjoint(parameter));
                *target = adjoint
                    .as_array()
                    .iter()
                    .fold(S::NEG_ZERO, |sum, &lane| sum + lane);
            }
            stats.kinks = session.kinks();
            self.close(stats, values);
            Ok(Some(()))
        })
    }
}

impl<S: RealScalar, const W: usize> Job<'_, S, W> {
    fn run<F, A, D>(
        self,
        run: &Run<'_, S, F, A, D>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        F: BatchFunction<S> + ?Sized,
        A: BatchData<S> + ?Sized,
        D: BatchData<S> + ?Sized,
    {
        let [np, ni, _] = run.widths;
        let mut values = self.values;
        let mut input_gradients = self.input_gradients;
        let mut derivatives = self.derivatives;
        for (k, (stats, gradient)) in self
            .stats
            .iter_mut()
            .zip(self.gradients.chunks_exact_mut(np.max(1)))
            .enumerate()
        {
            let block = self.first + k;
            let items = (run.items - block * run.block).min(run.block);
            let (v, rest) = core::mem::take(&mut values).split_at_mut(items);
            values = rest;
            let (g, rest) = core::mem::take(&mut input_gradients).split_at_mut(items * ni);
            input_gradients = rest;
            let (d, rest) = core::mem::take(&mut derivatives).split_at_mut(items);
            derivatives = rest;
            let out = Out {
                stats,
                gradient: &mut gradient[..np],
                values: v,
                input_gradients: g,
                derivatives: d,
            };
            run.block(block, self.worker, out, context)?;
        }
        Ok(())
    }
}

impl<const W: usize> BatchPlan<W> {
    fn kind(&self, gradient: bool) -> Result<Kind, SciOxideError> {
        let active = scioxide_core::backend::checked_add(
            self.parameters(),
            self.inputs(),
            scioxide_core::quantity!("batch_workspace"),
        )?;
        if !gradient || active == 0 {
            return Ok(Kind::Value);
        }
        let forward = match (
            self.options().kink_policy(),
            self.options().derivative_mode(),
        ) {
            (KinkPolicy::Error, _) | (_, DerivativeMode::Reverse) => false,
            (_, DerivativeMode::Auto) => active <= 4,
            (_, DerivativeMode::Forward) => {
                if active > 4 {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("options.derivative_mode"),
                        InvalidInputReason::UnsupportedConfiguration,
                        "forward mode carries at most four tangents",
                    ));
                }
                true
            }
        };
        Ok(if forward {
            Kind::Forward(tangents(active))
        } else {
            Kind::Reverse
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute<S, F, A, D>(
        &self,
        run: &Run<'_, S, F, A, D>,
        workspace: &mut BatchWorkspace<S, W>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        S: RealScalar,
        F: BatchFunction<S> + ?Sized,
        A: BatchData<S> + ?Sized,
        D: BatchData<S> + ?Sized,
    {
        context.check()?;
        let workers = workspace.workers().max(1);
        let requirements = self.workspace_requirements(workers)?;
        if workspace.check(&requirements)? == scioxide_core::Capacity::NeedsGrowth {
            workspace.try_reserve(&requirements)?;
        }
        // The workspace outlives the plan that built it, so the plan's kink policy is
        // applied here; a session under `KinkPolicy::Error` then fails its block.
        let (kink, limit) = (self.options().kink_policy(), self.options().byte_limit());
        for worker in &mut workspace.workers {
            worker.tape.set_batch_options(kink, limit);
            worker.second.set_batch_options(kink, limit);
        }
        let blocks = self.blocks();
        let [np, ni, _] = run.widths;
        let BatchWorkspace {
            workers: states,
            blocks: stats,
            gradients,
            values,
            input_gradients,
            derivatives,
            ..
        } = workspace;
        let stats = &mut stats[..blocks];
        let gradients = &mut gradients[..blocks * np.max(1)];
        let values = &mut values[..self.items()];
        let input_gradients = &mut input_gradients[..self.items() * ni];
        let derivatives = &mut derivatives[..self.items()];
        let policy = self.options().execution_policy();
        if policy == ExecutionPolicy::Sequential || blocks < 2 || states.len() < 2 {
            let job = Job {
                first: 0,
                worker: &mut states[0],
                stats,
                gradients,
                values,
                input_gradients,
                derivatives,
            };
            return job.run(run, context);
        }
        let limits = PlanLimits {
            auto_threshold: NonZeroUsize::MIN.saturating_add(1),
            work_per_item: NonZeroUsize::new(self.block().saturating_mul(64))
                .unwrap_or(NonZeroUsize::MIN),
            worker_limit: NonZeroUsize::new(states.len().min(MAX_WORKERS))
                .unwrap_or(NonZeroUsize::MIN),
        };
        let plan = context.plan(policy, blocks, limits);
        let mut slots: [PartitionSlot<Job<'_, S, W>>; MAX_WORKERS] =
            core::array::from_fn(|_| PartitionSlot::empty());
        let mut states = &mut states[..];
        let mut stats = stats;
        let mut gradients = gradients;
        let mut values = values;
        let mut input_gradients = input_gradients;
        let mut derivatives = derivatives;
        for partition in plan.partitions() {
            let range = partition.range();
            let first_item = range.start * self.block();
            let items = (range.end * self.block()).min(self.items()) - first_item;
            let (worker, rest) =
                core::mem::take(&mut states)
                    .split_first_mut()
                    .ok_or_else(|| {
                        SciOxideError::accounting_invariant("more partitions than workers", None)
                    })?;
            states = rest;
            let (s, rest) = core::mem::take(&mut stats).split_at_mut(range.len());
            stats = rest;
            let (g, rest) = core::mem::take(&mut gradients).split_at_mut(range.len() * np.max(1));
            gradients = rest;
            let (v, rest) = core::mem::take(&mut values).split_at_mut(items);
            values = rest;
            let (i, rest) = core::mem::take(&mut input_gradients).split_at_mut(items * ni);
            input_gradients = rest;
            let (d, rest) = core::mem::take(&mut derivatives).split_at_mut(items);
            derivatives = rest;
            slots[partition.worker_index()] = PartitionSlot::new(Job {
                first: range.start,
                worker,
                stats: s,
                gradients: g,
                values: v,
                input_gradients: i,
                derivatives: d,
            });
        }
        context.execute_partitioned(
            plan,
            &slots[..plan.worker_count()],
            &|_, job: Job<'_, S, W>, context| job.run(run, context),
        )
    }

    /// Reduces the block results and writes the outputs.
    fn finish<S: RealScalar>(
        &self,
        kind: Kind,
        workspace: &mut BatchWorkspace<S, W>,
        outputs: BatchOutputs<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<BatchEstimate<S>, SciOxideError> {
        let blocks = self.blocks();
        let np = self.parameters();
        let n = self.items();
        let mean = self.reduction() == BatchReduction::Mean;
        let scale = |value: S| if mean { value / count::<S>(n) } else { value };
        // Gradient errors by batch means need the block gradients before reduction.
        // `check_outputs` admitted errors only over two or more blocks.
        let error = outputs.parameter_gradient_error.is_some();
        let width = np.max(1);
        let (gradients, scratch) = (
            &workspace.gradients[..blocks * width],
            &mut workspace.scratch,
        );
        scratch[..blocks * width].copy_from_slice(gradients);
        reduce::vectors(&mut scratch[..blocks * width], blocks, width);
        if blocks == 0 {
            // No block staged a gradient: the total over no items is +0.
            <[S]>::fill(&mut scratch[..width], S::ZERO);
        }
        let total = reduce::stats(&mut workspace.blocks[..blocks]);
        let value = scale(total.sum);
        let standard_error =
            (mean && n >= 2).then(|| (total.m2 / count::<S>(n - 1) / count::<S>(n)).sqrt());
        let derivative = (kind == Kind::Jvp).then(|| scale(total.derivative));
        if error {
            let (totals, errors) = workspace.scratch.split_at_mut(blocks * width);
            for (k, target) in errors[..np].iter_mut().enumerate() {
                let overall = totals[k] / count::<S>(n);
                let mut sum = S::ZERO;
                for block in 0..blocks {
                    let items = (n - block * self.block()).min(self.block());
                    let block_mean = workspace.gradients[block * width + k] / count::<S>(items);
                    let delta = block_mean - overall;
                    sum += count::<S>(items) * delta * delta;
                }
                let se = (sum / (count::<S>(blocks - 1) * count::<S>(n))).sqrt();
                *target = if mean { se } else { se * count::<S>(n) };
            }
        }
        if self.options().non_finite_policy() == NonFinitePolicy::Error {
            // Value and jvp blocks write no gradients; the scratch holds stale ones.
            let gradients = !matches!(kind, Kind::Value | Kind::Jvp);
            let errors = &workspace.scratch[blocks * width..blocks * width + np];
            let finite = value.is_finite()
                && derivative.is_none_or(|d| d.is_finite())
                && standard_error.is_none_or(|e| e.is_finite())
                && (!error || errors.iter().all(|e| e.is_finite()))
                && (!gradients || workspace.scratch[..np].iter().all(|g| g.is_finite()))
                && workspace.values[..n].iter().all(|v| v.is_finite())
                && (outputs.input_gradient.is_none()
                    || workspace.input_gradients[..n * self.inputs()]
                        .iter()
                        .all(|g| g.is_finite()));
            if !finite {
                return Err(non_finite(NonFiniteOrigin::Output));
            }
        }
        context.check()?;
        if let Some(target) = outputs.values {
            target.copy_from_slice(&workspace.values[..n]);
        }
        if let Some(target) = outputs.parameter_gradient {
            for (t, &g) in target.iter_mut().zip(&workspace.scratch[..np]) {
                *t = scale(g);
            }
        }
        if let Some(target) = outputs.parameter_gradient_error {
            target.copy_from_slice(&workspace.scratch[blocks * width..blocks * width + np]);
        }
        if let Some(target) = outputs.input_gradient {
            target.copy_from_slice(&workspace.input_gradients[..n * self.inputs()]);
        }
        Ok(BatchEstimate {
            value,
            standard_error,
            derivative,
            items: n,
            blocks,
            mode: match kind {
                Kind::Value => BatchMode::Value,
                Kind::Jvp => BatchMode::Forward { tangents: 1 },
                Kind::Forward(n) => BatchMode::Forward { tangents: n },
                Kind::HvpForward(n) => BatchMode::Forward { tangents: 2 * n },
                Kind::Reverse | Kind::Hvp => BatchMode::Reverse,
            },
            kinks: matches!(kind, Kind::Reverse | Kind::Hvp).then_some(total.kinks),
        })
    }

    fn check_outputs<S: RealScalar>(
        &self,
        outputs: &BatchOutputs<'_, S>,
    ) -> Result<(), SciOxideError> {
        let check = |slice: &Option<&mut [S]>, len: usize| {
            slice
                .as_ref()
                .map_or(Ok(()), |s| crate::support::check_len(s.len(), len))
        };
        check(&outputs.values, self.items())?;
        check(&outputs.parameter_gradient, self.parameters())?;
        check(&outputs.parameter_gradient_error, self.parameters())?;
        let matrix = self.items().checked_mul(self.inputs()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("outputs.input_gradient"))
        })?;
        check(&outputs.input_gradient, matrix)?;
        if outputs.parameter_gradient_error.is_some() && self.blocks() < 2 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("outputs.parameter_gradient_error"),
                InvalidInputReason::OutOfRange,
                "batch-means errors need at least two blocks",
            ));
        }
        Ok(())
    }

    fn check_parameters<S: RealScalar>(&self, values: &[S]) -> Result<(), SciOxideError> {
        if self.options().non_finite_policy() == NonFinitePolicy::Error
            && values.iter().any(|v| !v.is_finite())
        {
            return Err(non_finite(NonFiniteOrigin::Input));
        }
        Ok(())
    }

    /// Evaluates the reduced value and, when requested, gradients and errors.
    ///
    /// Value mode runs when no gradient is requested; otherwise forward or reverse
    /// mode as described on [`BatchPlan`]. Outputs are staged in the workspace and
    /// written after every block succeeded; `context.check()` runs before execution,
    /// once per block and before publication.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `parameters` or an output has the wrong length.
    /// - `InvalidInput` for gradient errors over fewer than two blocks, or for
    ///   [`DerivativeMode::Forward`] with more than four active values.
    /// - `NonFinite` under [`NonFinitePolicy::Error`] (origin `Input` for
    ///   parameters and inputs, `Output` for results).
    /// - `DomainError` under [`KinkPolicy::Error`] when an active argument sits on a
    ///   kink.
    /// - `SizeOverflow` if a workspace size or the active count overflows;
    ///   `AllocationFailure` if the workspace cannot grow.
    /// - The errors of the function and the data sources, unchanged; tape errors;
    ///   `ExecutionStopped` if `context` is cancelled.
    ///
    /// Every output is unchanged on error.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_into<S, F, A, D>(
        &self,
        function: &F,
        parameters: &[S],
        inputs: &A,
        data: &D,
        outputs: BatchOutputs<'_, S>,
        workspace: &mut BatchWorkspace<S, W>,
        context: ExecutionContext<'_>,
    ) -> Result<BatchEstimate<S>, SciOxideError>
    where
        S: RealScalar,
        F: BatchFunction<S>,
        A: BatchData<S> + ?Sized,
        D: BatchData<S> + ?Sized,
    {
        crate::support::check_len(parameters.len(), self.parameters())?;
        self.check_outputs(&outputs)?;
        self.check_parameters(parameters)?;
        let kind = self.kind(outputs.wants_gradient())?;
        let run = Run {
            function,
            parameters,
            inputs,
            data,
            parameter_direction: &[],
            input_direction: None,
            kind,
            items: self.items(),
            block: self.block(),
            widths: [self.parameters(), self.inputs(), self.data()],
            stage_inputs: outputs.input_gradient.is_some(),
            check: self.options().non_finite_policy() == NonFinitePolicy::Error,
            moments: self.reduction() == BatchReduction::Mean,
        };
        self.execute(&run, workspace, context)?;
        self.finish(kind, workspace, outputs, context)
    }

    /// Evaluates the directional derivative at `(θ + εu, aᵢ + εwᵢ)` with
    /// `Dual<Lanes<S, W>>`: the derivative of each item goes to `derivatives`
    /// and the reduced one to [`BatchEstimate::derivative`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `parameters`, `parameter_direction` or
    ///   `derivatives` has the wrong length.
    /// - The errors of [`BatchPlan::evaluate_into`], except `DomainError`: duals
    ///   count no kinks, so [`KinkPolicy::Error`] does not apply.
    #[allow(clippy::too_many_arguments)]
    pub fn jvp_into<S, F, A, D>(
        &self,
        function: &F,
        parameters: &[S],
        inputs: &A,
        data: &D,
        parameter_direction: &[S],
        input_direction: &A,
        derivatives: &mut [S],
        workspace: &mut BatchWorkspace<S, W>,
        context: ExecutionContext<'_>,
    ) -> Result<BatchEstimate<S>, SciOxideError>
    where
        S: RealScalar,
        F: BatchFunction<S>,
        A: BatchData<S> + ?Sized,
        D: BatchData<S> + ?Sized,
    {
        crate::support::check_len(parameters.len(), self.parameters())?;
        crate::support::check_len(parameter_direction.len(), self.parameters())?;
        crate::support::check_len(derivatives.len(), self.items())?;
        self.check_parameters(parameters)?;
        self.check_parameters(parameter_direction)?;
        let run = Run {
            function,
            parameters,
            inputs,
            data,
            parameter_direction,
            input_direction: Some(input_direction),
            kind: Kind::Jvp,
            items: self.items(),
            block: self.block(),
            widths: [self.parameters(), self.inputs(), self.data()],
            stage_inputs: false,
            check: self.options().non_finite_policy() == NonFinitePolicy::Error,
            moments: self.reduction() == BatchReduction::Mean,
        };
        self.execute(&run, workspace, context)?;
        let estimate = self.finish(Kind::Jvp, workspace, BatchOutputs::new(), context)?;
        if self.options().non_finite_policy() == NonFinitePolicy::Error
            && workspace.derivatives[..self.items()]
                .iter()
                .any(|d| !d.is_finite())
        {
            return Err(non_finite(NonFiniteOrigin::Output));
        }
        derivatives.copy_from_slice(&workspace.derivatives[..self.items()]);
        Ok(estimate)
    }

    /// Evaluates `∂/∂ε` of the reduced parameter gradient and of the per-item input
    /// gradients at `(θ + εu, aᵢ + εwᵢ)`. `parameter_product` differentiates the
    /// reduced parameter gradient; `input_product` differentiates each item's
    /// unscaled input gradient, as in [`BatchOutputs::with_input_gradient`]. For
    /// a mean over `n` items, divide `input_product` by `n` to obtain the input
    /// part of the Hessian-vector product of the reduced value. Where
    /// [`BatchPlan::evaluate_into`] runs forward mode, the items run on
    /// `Dual<MultiDual<Lanes<S, W>, N>>` (forward over forward); otherwise on
    /// `Var<Dual<Lanes<S, W>>>` (forward over reverse).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `parameters`, `parameter_direction`,
    ///   `parameter_product` (parameters) or `input_product` (items × inputs) has
    ///   the wrong length.
    /// - The errors of [`BatchPlan::evaluate_into`].
    #[allow(clippy::too_many_arguments)]
    pub fn hvp_into<S, F, A, D>(
        &self,
        function: &F,
        parameters: &[S],
        inputs: &A,
        data: &D,
        parameter_direction: &[S],
        input_direction: &A,
        parameter_product: &mut [S],
        input_product: &mut [S],
        workspace: &mut BatchWorkspace<S, W>,
        context: ExecutionContext<'_>,
    ) -> Result<BatchEstimate<S>, SciOxideError>
    where
        S: RealScalar,
        F: BatchFunction<S>,
        A: BatchData<S> + ?Sized,
        D: BatchData<S> + ?Sized,
    {
        crate::support::check_len(parameters.len(), self.parameters())?;
        crate::support::check_len(parameter_direction.len(), self.parameters())?;
        self.check_parameters(parameters)?;
        self.check_parameters(parameter_direction)?;
        let outputs = BatchOutputs::new()
            .with_parameter_gradient(parameter_product)
            .with_input_gradient(input_product);
        self.check_outputs(&outputs)?;
        let run = Run {
            function,
            parameters,
            inputs,
            data,
            parameter_direction,
            input_direction: Some(input_direction),
            kind: match self.kind(true)? {
                Kind::Forward(n) => Kind::HvpForward(n),
                _ => Kind::Hvp,
            },
            items: self.items(),
            block: self.block(),
            widths: [self.parameters(), self.inputs(), self.data()],
            stage_inputs: true,
            check: self.options().non_finite_policy() == NonFinitePolicy::Error,
            moments: self.reduction() == BatchReduction::Mean,
        };
        self.execute(&run, workspace, context)?;
        self.finish(run.kind, workspace, outputs, context)
    }
}
