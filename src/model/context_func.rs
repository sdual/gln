use std::convert::TryInto;

use ndarray::Array;
use ndarray::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub trait ContextFunction {
    fn indicator_func(&self, side_info: &[f32]) -> Vec<bool>;
}

pub struct HalfSpaceContext {
    feature_dim: usize,
    context_dim: usize,
    context_maps: Vec<Vec<f32>>,
    context_bias: Vec<f32>,
}

impl HalfSpaceContext {
    pub fn new(context_dim: usize, feature_dim: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let context_maps: Vec<Vec<f32>> = (0..context_dim).into_iter().map(
            |_| normal.sample_iter(&mut thread_rng()).take(feature_dim).collect::<Vec<f32>>()
        ).collect();

        let context_bias: Vec<f32> = normal.sample_iter(&mut thread_rng()).take(context_dim).collect();

        HalfSpaceContext {
            feature_dim,
            context_dim,
            context_maps,
            context_bias,
        }
    }
}

impl ContextFunction for HalfSpaceContext {
    fn indicator_func(&self, side_info: &[f32]) -> Vec<bool> {
        let mut results = Vec::with_capacity(self.context_dim);
        // split space by x . v  > b
        for row_index in 0..self.context_dim {
            let mut value = 0.0;
            for col_index in 0..self.feature_dim {
                value += self.context_maps[row_index][col_index] * side_info[col_index];
            }
            results.push(value > self.context_bias[row_index]);
        }
        results
    }
}

pub struct SkipGramContext {}

impl ContextFunction for SkipGramContext {
    fn indicator_func(&self, side_info: &[f32]) -> Vec<bool> {
        todo!()
    }
}

#[test]
fn test_new_half_space_context() {
    // このテストは後々いらない。初期化がうまく出来ているかチェックするためだけ。
    let actual = HalfSpaceContext::new(2, 3);
}

#[test]
fn test_half_space_indicator_func() {
    let side_info = vec![1.2, 1.5, 0.9];
    let context_dim = 4;
    let feature_dim = 3;
    let half_space_context = HalfSpaceContext::new(context_dim, feature_dim);
    // TODO: テストできないので、乱数の生成はシードを固定できるようにする。
    let actual = half_space_context.indicator_func(&side_info);
}
