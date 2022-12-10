use crate::utils::math;

pub trait OnlineGradient {
    fn calculate_grad(
        &self,
        xs: &Vec<f32>,
        target: i32,
        weights: &Vec<f32>,
        index: usize,
        clipping_value: f32,
        weight: f32,
    ) -> f32;
}

pub struct LogGeometricMixingGradient {
    reg_param: f32,
}

impl LogGeometricMixingGradient {
    pub fn new(reg_param: f32) -> Self {
        LogGeometricMixingGradient {
            reg_param: reg_param,
        }
    }
}

impl OnlineGradient for LogGeometricMixingGradient {
    fn calculate_grad(
        &self,
        inputs: &Vec<f32>,
        target: i32,
        weights: &Vec<f32>,
        index: usize,
        clipping_value: f32,
        weight: f32,
    ) -> f32 {
        if target == 1 {
            weight * (math::geometric_mixing(inputs, weights, clipping_value) - target as f32)
                * math::logit(inputs[index]) + self.reg_param * weights[index]
        } else {
            (math::geometric_mixing(inputs, weights, clipping_value) - target as f32)
                * math::logit(inputs[index]) + self.reg_param * weights[index]
        }
    }
}

// #[cfg(test)]
// mod test {
//     use crate::optimize::grad::{LogGeometricMixingGradient, OnlineGradient};

//     #[test]
//     fn test_log_geometric_mixing_gradient() {
//         let grad = LogGeometricMixingGradient::new();
//         let xs = vec![0.1, 0.4, 0.6];
//         let target = 1;
//         let weights = vec![0.2, 1.6, 0.7];
//         let index: usize = 1;
//         let actual = grad.calculate_grad(&xs, target, &weights, index);
//         assert_eq!(actual, 0.28013876);
//     }
// }
