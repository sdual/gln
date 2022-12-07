use nalgebra::DVector;

use crate::model::config::LayerConfig;
use crate::model::context_func::{ContextFunction, HalfSpaceContext, SkipGramContext};
use crate::model::gate::{initialize_balanced_weights, Gate};
use crate::optimize::grad::{LogGeometricMixingGradient, OnlineGradient};
use crate::optimize::optimizer::OnlineGradientDecent;
use crate::utils::data_type::ContextIndex;
use crate::utils::math::{clip_hypercube, clip_prob, geometric_mixing};

pub struct Neuron<C: ContextFunction> {
    gate: Gate<C>,
    optimizer: OnlineGradientDecent,
    gradient: LogGeometricMixingGradient,
    pred_clipping_value: f32,
    weight_clipping_value: f32,
}

pub struct NeuronTrainHistory {
    prediction: f32,
    loss: f32,
}

impl Neuron<HalfSpaceContext> {
    pub fn with_half_space_context(
        input_dim: usize,
        context_dim: usize,
        feature_dim: usize,
        learning_rate: f32,
    ) -> Neuron<HalfSpaceContext> {
        let config = LayerConfig::with_default_value();
        Neuron {
            gate: Gate::<HalfSpaceContext>::new(
                input_dim,
                context_dim,
                feature_dim,
                initialize_balanced_weights,
            ),
            optimizer: OnlineGradientDecent::new(learning_rate),
            gradient: LogGeometricMixingGradient::new(),
            pred_clipping_value: config.pred_clipping_value,
            weight_clipping_value: config.weight_clipping_value,
        }
    }
}

impl Neuron<SkipGramContext> {
    pub fn with_skip_gram_context(_input_dim: usize, _context_dim: usize, _feature_dim: usize) {
        todo!()
    }
}

impl<C: ContextFunction> Neuron<C> {
    pub fn predict_by_context_index(&self, context_index: ContextIndex, inputs: &Vec<f32>) -> f32 {
        let current_weights = self.gate.get_weights(context_index);
        // let mut logit_sum: f32 = 0.0;
        // for (weight, input) in current_weights.iter().zip(inputs) {
        //     logit_sum += weight * logit(clip_prob(*input, self.pred_clipping_value));
        // }

        let prediction = clip_prob(
            geometric_mixing(inputs, &current_weights, self.pred_clipping_value),
            self.pred_clipping_value,
        );

        // let loss = geometric_mixing_loss(target, prediction);
        // NeuronTrainHistory { prediction, loss }
        prediction
    }

    pub fn update_weights(&mut self, inputs: &Vec<f32>, target: i32, context_index: ContextIndex) {
        let mut updated_weights = Vec::with_capacity(inputs.len());
        let current_weights = self.gate.get_weights(context_index);

        for (weight_index, _) in current_weights.iter().enumerate() {
            let grad = self.gradient.calculate_grad(
                inputs,
                target,
                &current_weights,
                weight_index,
                self.pred_clipping_value,
            );

            let updated_weight = self.optimizer.update(current_weights[weight_index], grad);
            updated_weights.push(clip_hypercube(updated_weight, self.weight_clipping_value));
        }

        self.gate.update_weights(context_index, updated_weights);
    }

    pub fn get_current_weights(&self, features: &DVector<f32>) -> (Vec<f32>, usize) {
        let (current_weights, context_index) = self.gate.select_weights(features);
        (current_weights, context_index)
    }
}

// #[cfg(test)]
// mod test {
//     use crate::model::context_func::HalfSpaceContext;
//     use crate::model::neuron::Neuron;

//     #[test]
//     fn test_update_weights() {
//         let neuron = Neuron::with_half_space_context(5, 10, 100);
//     }

//     #[test]
//     fn test_predict_and_update_weights() {
//         let mut neuron = Neuron::with_half_space_context(5, 2, 5);

//         let previous_weights = &neuron.gate.weights;
//         for v1 in previous_weights {
//             for v2 in v1 {
//                 println!("{}", v2);
//             }
//         }

//         let features = vec![1.4, 1.3, 2.4, 3.1, 5.4];
//         let inputs = vec![0.4, 0.3, 0.4, 0.9, 0.4];
//         let actual = neuron.predict_and_update_weights(&features, &inputs, 1);
//         assert_eq!(actual, 0.50667614);
//         let actual_weight = &neuron.gate.weights;

//         println!("-----");

//         for v1 in actual_weight {
//             for v2 in v1 {
//                 println!("{}", v2);
//             }
//         }
//     }

//     // #[test]
//     // fn test_update_weights() {}
// }
