//! Recorder faults that the public API cannot provoke.
#![allow(clippy::unwrap_used, clippy::panic)]

use alloc::vec;
use alloc::vec::Vec;

use scioxide_core::{ExecutionContext, InvalidInputReason, SciOxideErrorView};

use super::recorder::{MIXED, Recorder};
use super::{Tape, TapeOptions, Var};
use crate::Real;

#[test]
fn operands_from_two_recorders_are_flagged() {
    let (mut w1, mut f1, mut s1, mut c1, mut x1) = (
        vec![0; 8],
        vec![0.0; 8],
        vec![0.0; 8],
        Vec::new(),
        vec![0.0; 4],
    );
    let (mut w2, mut f2, mut s2, mut c2, mut x2) = (
        vec![0; 8],
        vec![0.0; 8],
        vec![0.0; 8],
        Vec::new(),
        vec![0.0; 4],
    );
    let first = Recorder::<f64>::new(&mut w1, &mut f1, &mut s1, &mut c1, &mut x1, 1 << 28);
    let second = Recorder::<f64>::new(&mut w2, &mut f2, &mut s2, &mut c2, &mut x2, 1 << 28);
    let a = first.input(1.0);
    let b = second.input(2.0);
    for mixed in [a + b, a * b, a / b, a - b, a.hypot(b)] {
        assert!(mixed.value().is_nan());
        assert!(!mixed.is_active());
    }
    let combined = Var::combine(&[a, b], 3.0, &[1.0, 1.0]).unwrap();
    assert!(combined.value().is_nan());
    assert_eq!(first.faults() & MIXED, MIXED);
    assert_eq!(second.faults() & MIXED, MIXED);
}

#[test]
fn a_mixing_fault_fails_every_transform_with_invalid_input() {
    // Lifetimes keep the public API from mixing recordings, so the fault is raised
    // directly on the recording.
    let flag = |x: &Var<'_, f64>| {
        if let Some(r) = x.recorder() {
            r.fault(MIXED);
        }
    };
    let check = |error: scioxide_core::SciOxideError| match error.view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => {
            assert_eq!(argument.as_str(), "function");
            assert_eq!(reason, InvalidInputReason::Inconsistent);
        }
        other => panic!("{other:?}"),
    };
    let cx = ExecutionContext::sequential();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut gradient = [5.0];
    check(
        tape.gradient_into(
            |x| {
                flag(&x[0]);
                Ok(x[0].sin())
            },
            &[0.5],
            &mut gradient,
            cx,
        )
        .unwrap_err(),
    );
    assert_eq!(gradient, [5.0]);
    let linearized = tape.linearize(
        |x, out| {
            flag(&x[0]);
            out.push(x[0].sin());
            Ok(())
        },
        &[0.5],
        cx,
    );
    check(linearized.err().unwrap());
    // The tape records the next function normally.
    tape.gradient_into(|x| Ok(x[0].sin()), &[0.5], &mut gradient, cx)
        .unwrap();
    assert_eq!(gradient, [0.5_f64.cos()]);
}

#[test]
fn the_default_word_limit_is_two_to_the_28() {
    assert_eq!(TapeOptions::new().word_limit, 1 << 28);
}

#[test]
fn the_word_limit_is_a_resource_limit() {
    let mut tape = Tape::<f64>::new(TapeOptions::new().with_word_limit(16));
    let mut gradient = [5.0; 12];
    let error = tape
        .gradient_into(
            |x| Ok(x.iter().map(|&v| v.sin()).sum()),
            &[0.5; 12],
            &mut gradient,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    match error.view() {
        SciOxideErrorView::ResourceLimit {
            resource,
            required,
            limit,
        } => {
            assert_eq!(resource.as_str(), "tape_words");
            assert_eq!(limit, 16);
            assert!(required > 16);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(gradient, [5.0; 12]);
}

#[test]
fn cancellation_is_polled_inside_long_extended_statements() {
    use super::call::Call;
    use super::recorder::{Columns, Counts, Frozen};
    use core::cell::Cell;
    use scioxide_core::CancellationToken;
    struct CancelAfterReads<'a> {
        columns: Frozen<'a, f64>,
        token: CancellationToken,
        reads: Cell<usize>,
    }
    impl Columns<f64> for CancelAfterReads<'_> {
        fn word(&self, index: usize) -> u64 {
            let reads = self.reads.get() + 1;
            self.reads.set(reads);
            if reads == 8 {
                self.token.cancel();
            }
            self.columns.word(index)
        }
        fn full(&self, index: usize) -> f64 {
            self.columns.full(index)
        }
        fn scalar(&self, index: usize) -> f64 {
            self.columns.scalar(index)
        }
        fn with_call<R>(&self, index: usize, f: impl FnOnce(Option<&Call<f64>>) -> R) -> R {
            self.columns.with_call(index, f)
        }
    }
    let mut tape = Tape::<f64>::default();
    tape.record(&[1.0], &|x, y| {
        let values = vec![x[0]; 4096];
        y.push(crate::dot(&values, &values)?);
        Ok(())
    })
    .unwrap();
    for reverse in [false, true] {
        let columns = CancelAfterReads {
            columns: tape.frozen(),
            token: CancellationToken::new(),
            reads: Cell::new(0),
        };
        let context = ExecutionContext::sequential().with_cancellation(&columns.token);
        let mut work = vec![1.0; tape.counts.words];
        let outcome = if reverse {
            super::sweep::reverse(&columns, tape.counts, 0, &mut work, &mut [], context)
        } else {
            super::sweep::tangent(
                &columns,
                Counts::default(),
                tape.counts.words,
                &mut work,
                &mut [],
                context,
            )
        };
        assert!(matches!(
            outcome.unwrap_err().view(),
            SciOxideErrorView::ExecutionStopped { .. }
        ));
        assert!(
            columns.reads.get() <= 8 + 1024,
            "{} reads",
            columns.reads.get()
        );
    }
}
