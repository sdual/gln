use crate::model::config::LayerConfig;
use crate::model::context_func::{ContextFunction, HalfSpaceContext, SkipGramContext};
use crate::model::gate::{Gate, initialize_balanced_weights};
use crate::optimize::grad::{LogGeometricMixingGradient, OnlineGradient};
use crate::optimize::optimizer::OnlineGradientDecent;
use crate::utils::math::{clip, logit, sigmoid};

pub struct Neuron<C: ContextFunction> {
    gate: Gate<C>,
    optimizer: OnlineGradientDecent,
    gradient: LogGeometricMixingGradient,
    pred_clipping_value: f32,
}

impl Neuron<HalfSpaceContext> {
    pub fn with_half_space_context(input_dim: usize,
                                   context_dim: usize,
                                   feature_dim: usize) -> Neuron<HalfSpaceContext> {
        let config = LayerConfig::with_default_value();
        Neuron {
            gate: Gate::<HalfSpaceContext>::new(
                input_dim,
                context_dim,
                feature_dim,
                initialize_balanced_weights,
            ),
            optimizer: OnlineGradientDecent::new(config.learning_rate, config.pred_clipping_value),
            gradient: LogGeometricMixingGradient::new(),
            pred_clipping_value: config.pred_clipping_value,
        }
    }
}

impl Neuron<SkipGramContext> {
    pub fn with_skip_gram_context(input_dim: usize,
                                  context_dim: usize,
                                  feature_dim: usize) {
        todo!()
    }
}

impl<C: ContextFunction> Neuron<C> {
    pub fn predict_and_update_weights(&mut self, features: &Vec<f32>, inputs: &Vec<f32>, target: i32) -> f32 {
        let (current_weights, context_index) = self.gate.select_weights(features);
        let mut logit_sum = 0.0_f32;
        for (weight, input) in current_weights.iter().zip(inputs) {
            logit_sum += weight * logit(clip(*input, self.pred_clipping_value));
        };
        let prediction = clip(sigmoid(logit_sum), self.pred_clipping_value);

        self.update_weights(inputs, target, &current_weights, context_index);
        prediction
    }

    fn update_weights(&mut self, inputs: &Vec<f32>, target: i32,
                      current_weights: &Vec<f32>,
                      context_index: usize) {
        let mut updated_weights = Vec::with_capacity(inputs.len());

        for (weight_index, _) in current_weights.iter().enumerate() {
            let grad = self.gradient.calculate_grad(inputs, target, current_weights, weight_index);
            let updated_weight = self.optimizer.update(current_weights[weight_index], grad);
            updated_weights.push(updated_weight);
        }

        self.gate.update_weights(context_index, updated_weights);
    }
}

#[cfg(test)]
mod test {
    use crate::model::context_func::HalfSpaceContext;
    use crate::model::neuron::Neuron;

    #[test]
    fn test_update_weights() {
        let neuron = Neuron::with_half_space_context(5, 10, 100);
    }

    #[test]
    fn test_predict_and_update_weights() {
        let mut neuron = Neuron::with_half_space_context(5, 2, 5);

        let previous_weights = &neuron.gate.weights;
        for v1 in previous_weights {
            for v2 in v1 {
                println!("{}", v2);
            }
        }

        let features = vec![1.4, 1.3, 2.4, 3.1, 5.4];
        let inputs = vec![0.4, 0.3, 0.4, 0.9, 0.4];
        let actual = neuron.predict_and_update_weights(&features, &inputs, 1);
        assert_eq!(actual, 0.50667614);
        let actual_weight = &neuron.gate.weights;

        println!("-----");

        for v1 in actual_weight {
            for v2 in v1 {
                println!("{}", v2);
            }
        }
    }

    // #[test]
    // fn test_update_weights() {}
}
