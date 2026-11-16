//! Independent mathematical reference data, evaluated with test-local second jets.
use core::ops::{Add, Mul, Neg, Sub};
use scioxide_core::RealScalar;
use serde::Deserialize;

#[derive(Clone, Copy)]
pub struct Jet<S> {
    pub value: S,
    pub first: S,
    pub second: S,
    pub mixed: S,
}
impl<S: RealScalar> Jet<S> {
    pub(super) fn constant(value: S) -> Self {
        Self {
            value,
            first: S::ZERO,
            second: S::ZERO,
            mixed: S::ZERO,
        }
    }
    fn unary(self, value: S, derivative: S, curvature: S) -> Self {
        Self {
            value,
            first: derivative * self.first,
            second: derivative * self.second,
            mixed: derivative * self.mixed + curvature * self.first * self.second,
        }
    }
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(S::ONE);
        }
        if n == 1 {
            return self;
        }
        let p = S::try_from_f64_lossy(n as f64).unwrap();
        self.unary(
            self.value.powi(n),
            p * self.value.powi(n - 1),
            p * (p - S::ONE) * self.value.powi(n - 2),
        )
    }
    fn powf(self, p: S) -> Self {
        self.unary(
            self.value.powf(p),
            p * self.value.powf(p - S::ONE),
            p * (p - S::ONE) * self.value.powf(p - S::ONE - S::ONE),
        )
    }
    fn exp(self) -> Self {
        let v = self.value.exp();
        self.unary(v, v, v)
    }
    fn ln(self) -> Self {
        self.unary(
            self.value.ln(),
            S::ONE / self.value,
            -S::ONE / (self.value * self.value),
        )
    }
    fn atan2(self, x: Self) -> Self {
        let r = x.value * x.value + self.value * self.value;
        let fy = x.value / r;
        let fx = -self.value / r;
        let cross = (self.value * self.value - x.value * x.value) / (r * r);
        let yy = -(S::ONE + S::ONE) * x.value * self.value / (r * r);
        Self {
            value: self.value.atan2(x.value),
            first: fy * self.first + fx * x.first,
            second: fy * self.second + fx * x.second,
            mixed: fy * self.mixed + fx * x.mixed + yy * self.first * self.second
                - yy * x.first * x.second
                + cross * (self.first * x.second + self.second * x.first),
        }
    }
}
impl<S: RealScalar> Add for Jet<S> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            value: self.value + b.value,
            first: self.first + b.first,
            second: self.second + b.second,
            mixed: self.mixed + b.mixed,
        }
    }
}
impl<S: RealScalar> Neg for Jet<S> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            first: -self.first,
            second: -self.second,
            mixed: -self.mixed,
        }
    }
}
impl<S: RealScalar> Sub for Jet<S> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + (-b)
    }
}
impl<S: RealScalar> Mul for Jet<S> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            value: self.value * b.value,
            first: self.first * b.value + self.value * b.first,
            second: self.second * b.value + self.value * b.second,
            mixed: self.mixed * b.value
                + self.first * b.second
                + self.second * b.first
                + self.value * b.mixed,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Token {
    Constant(f64),
    Operation(String),
}
#[derive(Clone, Copy)]
enum Instruction<S> {
    Constant(S),
    Variable(usize),
    Add,
    Multiply,
    PowI(i32),
    PowF(S),
    Power,
    Sin,
    Cos,
    Exp,
    Ln,
    Abs,
    Atan2,
    Less,
    And,
    Select,
}
pub struct Program<S> {
    instructions: Vec<Instruction<S>>,
}
impl<S: RealScalar> Program<S> {
    pub fn new(tokens: &[Token], dimension: usize) -> Self {
        let mut depth = 0_usize;
        let instructions = tokens
            .iter()
            .map(|token| {
                let (instruction, arity) = match token {
                    Token::Constant(v) => {
                        (Instruction::Constant(S::try_from_f64_lossy(*v).unwrap()), 0)
                    }
                    Token::Operation(op) => match op.as_str() {
                        "+" => (Instruction::Add, 2),
                        "*" => (Instruction::Multiply, 2),
                        "pow" => (Instruction::Power, 2),
                        "sin" => (Instruction::Sin, 1),
                        "cos" => (Instruction::Cos, 1),
                        "exp" => (Instruction::Exp, 1),
                        "ln" => (Instruction::Ln, 1),
                        "abs" => (Instruction::Abs, 1),
                        "atan2" => (Instruction::Atan2, 2),
                        "<" => (Instruction::Less, 2),
                        "and" => (Instruction::And, 2),
                        "select" => (Instruction::Select, 3),
                        _ if op.starts_with("x:") => {
                            let i = op[2..].parse::<usize>().unwrap();
                            assert!(i < dimension);
                            (Instruction::Variable(i), 0)
                        }
                        _ if op.starts_with("powi:") => {
                            (Instruction::PowI(op[5..].parse().unwrap()), 1)
                        }
                        _ if op.starts_with("powf:") => (
                            Instruction::PowF(
                                S::try_from_f64_lossy(op[5..].parse().unwrap()).unwrap(),
                            ),
                            1,
                        ),
                        _ => panic!("unknown reference operation {op}"),
                    },
                };
                assert!(depth >= arity);
                depth = depth - arity + 1;
                assert!(depth <= 128);
                instruction
            })
            .collect();
        assert_eq!(depth, 1);
        Self { instructions }
    }
    pub fn evaluate(&self, x: &[S], first: Option<usize>, second: Option<usize>) -> Jet<S> {
        let mut stack = [Jet::constant(S::ZERO); 128];
        let mut depth = 0;
        for &instruction in &self.instructions {
            match instruction {
                Instruction::Constant(value) => {
                    stack[depth] = Jet::constant(value);
                    depth += 1;
                }
                Instruction::Variable(i) => {
                    stack[depth] = Jet {
                        value: x[i],
                        first: if first == Some(i) { S::ONE } else { S::ZERO },
                        second: if second == Some(i) { S::ONE } else { S::ZERO },
                        mixed: S::ZERO,
                    };
                    depth += 1;
                }
                Instruction::Select => {
                    depth -= 2;
                    let condition = stack[depth - 1].value;
                    stack[depth - 1] = if condition != S::ZERO {
                        stack[depth]
                    } else {
                        stack[depth + 1]
                    };
                }
                Instruction::Add
                | Instruction::Multiply
                | Instruction::Power
                | Instruction::Atan2
                | Instruction::Less
                | Instruction::And => {
                    depth -= 1;
                    let a = stack[depth - 1];
                    let b = stack[depth];
                    stack[depth - 1] = match instruction {
                        Instruction::Add => a + b,
                        Instruction::Multiply => a * b,
                        Instruction::Power => (b * a.ln()).exp(),
                        Instruction::Atan2 => a.atan2(b),
                        Instruction::Less => {
                            Jet::constant(if a.value < b.value { S::ONE } else { S::ZERO })
                        }
                        Instruction::And => {
                            Jet::constant(if a.value != S::ZERO && b.value != S::ZERO {
                                S::ONE
                            } else {
                                S::ZERO
                            })
                        }
                        _ => unreachable!(),
                    };
                }
                _ => {
                    let a = stack[depth - 1];
                    stack[depth - 1] = match instruction {
                        Instruction::PowI(n) => a.powi(n),
                        Instruction::PowF(p) => a.powf(p),
                        Instruction::Sin => a.unary(a.value.sin(), a.value.cos(), -a.value.sin()),
                        Instruction::Cos => a.unary(a.value.cos(), -a.value.sin(), -a.value.cos()),
                        Instruction::Exp => a.exp(),
                        Instruction::Ln => a.ln(),
                        Instruction::Abs => {
                            if a.value < S::ZERO {
                                -a
                            } else {
                                a
                            }
                        }
                        _ => unreachable!(),
                    };
                }
            }
        }
        stack[0]
    }
}
