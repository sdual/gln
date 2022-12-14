use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use crate::model::context_func::HalfSpaceContext;
use crate::model::neuron::Neuron;
use crate::utils::data_type::{ContextIndex, NeuronId};
use crate::utils::math::{clip_prob, logit};

pub struct LayerPrediction {
    pub predictions: DMatrix<f32>,
    pub context_index_map: Option<HashMap<NeuronId, usize>>,
}

pub struct Layer {
    neurons: Vec<Neuron<HalfSpaceContext>>,
    num_neurons: usize,
    input_dim: usize,
}

pub struct LayerTrainHistory {
    neuron_losses: Vec<f32>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron<HalfSpaceContext>>, input_dim: usize) -> Self {
        let num_neurons = neurons.len();
        Layer {
            neurons,
            num_neurons,
            input_dim,
        }
    }

    pub fn with_neuron_num(
        neuron_num: usize,
        input_dim: usize,
        context_dim: usize,
        feature_dim: usize,
        learning_rate: f32,
        weight_clipping_value: f32,
        negative_weight: f32,
        reg_param: f32,
    ) -> Self {
        let neurons: Vec<Neuron<HalfSpaceContext>> = (0usize..neuron_num)
            .map(|_| {
                Neuron::with_half_space_context(
                    input_dim,
                    context_dim,
                    feature_dim,
                    learning_rate,
                    weight_clipping_value,
                    negative_weight,
                    reg_param,
                )
            })
            .collect();

        Self::new(neurons, input_dim)
    }

    pub fn train(
        &mut self,
        context_index_map: &HashMap<NeuronId, ContextIndex>,
        inputs: &Vec<f32>,
        target: i32,
    ) {
        for neuron_id in 0usize..self.num_neurons {
            self.neurons[neuron_id].update_weights(inputs, target, context_index_map[&neuron_id]);
        }
    }

    pub fn predict_by_context_index(
        &self,
        context_index_map: &HashMap<NeuronId, ContextIndex>,
        inputs: &Vec<f32>,
    ) -> Vec<f32> {
        let mut probabilities = Vec::with_capacity(self.num_neurons);
        for neuron_id in 0usize..self.num_neurons {
            let probability = self.neurons[neuron_id]
                .predict_by_context_index(context_index_map[&neuron_id], inputs);
            probabilities.push(probability);
        }
        probabilities
    }

    pub fn calculate_next_weight_matrix(
        &self,
        features: &DVector<f32>,
        previous_vector: &DMatrix<f32>,
    ) -> LayerPrediction {
        let mut weight_vec = Vec::new();
        let mut context_index_map = HashMap::new();
        for (neuron_id, neuron) in self.neurons.iter().enumerate() {
            let (weights, context_index) = neuron.get_current_weights(features);
            weight_vec.append(&mut weights.clone());
            context_index_map.insert(neuron_id, context_index);
        }

        let weight_matrix = DMatrix::from_row_slice(self.num_neurons, self.input_dim, &weight_vec);

        LayerPrediction {
            predictions: (weight_matrix * previous_vector),
            context_index_map: Some(context_index_map),
        }
    }
}

pub struct BaseLayer {
    pred_clipping_value: f32,
    feature_dim: usize,
}

impl BaseLayer {
    pub fn new(pred_clipping_value: f32, feature_dim: usize) -> Self {
        BaseLayer {
            pred_clipping_value,
            feature_dim,
        }
    }

    pub fn predict(&self, features: &DVector<f32>) -> Vec<f32> {
        // let max_value = features.max();
        // let min_value = features.min();

        // let base_predictions = features
        //     .iter()
        //     .map(|value| (value - min_value) / (max_value - min_value))
        //     .map(|value| clip_prob(value, self.pred_clipping_value))
        //     .collect::<Vec<f32>>();
        self.normalize(features)
    }

    pub fn predict_with_logits(&self, features: &DVector<f32>) -> LayerPrediction {
        let max_value = features.max();
        let min_value = features.min();

        let base_predictions = if max_value != min_value {
            features
                .iter()
                .map(|value| (value - min_value) / (max_value - min_value))
                .map(|value| logit(clip_prob(value, self.pred_clipping_value)))
                .collect::<Vec<f32>>()
        } else {
            let feature_len = features.len();
            features
                .iter()
                .map(|value| value / feature_len as f32)
                .map(|value| logit(clip_prob(value, self.pred_clipping_value)))
                .collect::<Vec<f32>>()
        };

        LayerPrediction {
            predictions: DMatrix::from_row_slice(self.feature_dim, 1, &base_predictions),
            context_index_map: None,
        }
    }

    fn normalize(&self, features: &DVector<f32>) -> Vec<f32> {
        let max_value = features.max();
        let min_value = features.min();

        if max_value != min_value {
            features
                .iter()
                .map(|value| (value - min_value) / (max_value - min_value))
                .map(|value| clip_prob(value, self.pred_clipping_value))
                .collect::<Vec<f32>>()
        } else if max_value != 0.0 {
            features
                .iter()
                .map(|_| max_value as f32)
                .map(|value| clip_prob(value, self.pred_clipping_value))
                .collect::<Vec<f32>>()
        } else {
            features
                .iter()
                .map(|_| 0.0 as f32)
                .map(|value| clip_prob(value, self.pred_clipping_value))
                .collect::<Vec<f32>>()
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::model::layer::BaseLayer;

    #[test]
    fn test_base_layer_predifct() {
        let features = vec![1.0, 5.0, 4.0, 4.0];
        let base_layer = BaseLayer::new(0.01, features.len());
        let feature_vec = DVector::from_vec(features);
        let actual = base_layer.predict(&feature_vec);

        let expected = vec![0.01, 0.99, 0.75, 0.75];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_predict_with_logit_all_zero() {
        let features = vec![0.0, 0.0, 0.0, 0.0];
        let base_layer = BaseLayer::new(0.01, features.len());
        let feature_vec = DVector::from_vec(features);

        let actual = base_layer.predict_with_logits(&feature_vec);

        let expected = DMatrix::from_row_slice(4, 1, &vec![-4.59512, -4.59512, -4.59512, -4.59512]);
        assert_eq!(actual.predictions, expected);
    }
}
