use crate::utils::math;

pub trait OnlineGradient {
    fn calculate_grad(&self, xs: &Vec<f32>, target: i32, weights: &Vec<f32>, index: usize) -> f32;
}

pub struct LogGeometricMixingGradient;

impl LogGeometricMixingGradient {
    pub fn new() -> Self {
        LogGeometricMixingGradient
    }
}

impl OnlineGradient for LogGeometricMixingGradient {
    fn calculate_grad(&self, inputs: &Vec<f32>, target: i32, weights: &Vec<f32>, index: usize) -> f32 {
        (math::geometric_mixing(inputs, weights) - target as f32) * math::logit(inputs[index])
    }
}

#[cfg(test)]
mod test {
    use crate::optimize::grad::{LogGeometricMixingGradient, OnlineGradient};

    #[test]
    fn test_log_geometric_mixing_gradient() {
        let grad = LogGeometricMixingGradient::new();
        let xs = vec![0.1, 0.4, 0.6];
        let target = 1;
        let weights = vec![0.2, 1.6, 0.7];
        let index: usize = 1;
        let actual = grad.calculate_grad(&xs, target, &weights, index);
        assert_eq!(actual, 0.28013876);
    }
}
