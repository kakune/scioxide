//! Contiguous copy runs for padding, boundary extension and cropping.
use alloc::vec::Vec;
use scioxide_core::SciOxideError;

#[derive(Clone, Debug)]
pub(super) struct CopyRun {
    source: usize,
    target: usize,
    len: usize,
    step: i8,
}

pub(super) fn prepare(
    len: usize,
    map: impl Fn(usize) -> Option<usize>,
) -> Result<Vec<CopyRun>, SciOxideError> {
    let mut runs: Vec<CopyRun> = Vec::new();
    for target in 0..len {
        let Some(source) = map(target) else { continue };
        if let Some(run) = runs.last_mut()
            && target == run.target + run.len
        {
            let previous = match run.step {
                -1 => run.source - (run.len - 1),
                0 => run.source,
                _ => run.source + run.len - 1,
            };
            let step = if source == previous {
                Some(0)
            } else if source.checked_add(1) == Some(previous) {
                Some(-1)
            } else if previous.checked_add(1) == Some(source) {
                Some(1)
            } else {
                None
            };
            if let Some(step) = step
                && (run.len == 1 || run.step == step)
            {
                run.step = step;
                run.len += 1;
                continue;
            }
        }
        runs.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("convolution_copy_runs"),
                len,
            )
        })?;
        runs.push(CopyRun {
            source,
            target,
            len: 1,
            step: 1,
        });
    }
    Ok(runs)
}

pub(super) fn copy<T: Copy, U: Copy>(
    runs: &[CopyRun],
    source: &[T],
    target: &mut [U],
    map: impl Fn(T) -> U,
) {
    for run in runs {
        let target = &mut target[run.target..run.target + run.len];
        match run.step {
            -1 => target
                .iter_mut()
                .zip(source[run.source + 1 - run.len..=run.source].iter().rev())
                .for_each(|(out, &value)| *out = map(value)),
            0 => target.fill(map(source[run.source])),
            _ => target
                .iter_mut()
                .zip(&source[run.source..run.source + run.len])
                .for_each(|(out, &value)| *out = map(value)),
        }
    }
}
