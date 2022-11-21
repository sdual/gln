use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use crate::model::context_func::HalfSpaceContext;
use crate::model::neuron::Neuron;
use crate::utils::data_type::{ContextIndex, NeuronId};
use crate::utils::math::{clip, logit};

pub struct LayerPrediction {
    pub predictions: DMatrix<f32>,
    pub context_index_map: Option<HashMap<NeuronId, usize>>,
}

pub struct Layer {
    neurons: Vec<Neuron<HalfSpaceContext>>,
    num_neurons: usize,
    input_dim: usize,
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
    ) -> Self {
        let neurons: Vec<Neuron<HalfSpaceContext>> = (0usize..neuron_num)
            .map(|_| Neuron::with_half_space_context(input_dim, context_dim, feature_dim))
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
            self.neurons[neuron_id].update_weights(
                inputs,
                target,
                *context_index_map.get(&(neuron_id as NeuronId)).unwrap(),
            );
        }
    }

    pub fn predict_by_context_index(
        &self,
        context_index_map: &HashMap<NeuronId, ContextIndex>,
        inputs: &Vec<f32>,
    ) -> Vec<f32> {
        let mut probabilities = Vec::with_capacity(self.num_neurons);
        for (neuron_id, context_index) in context_index_map {
            let probability =
                self.neurons[*neuron_id].predict_by_context_index(*context_index, inputs);
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
        for (neuron_index, neuron) in self.neurons.iter().enumerate() {
            let (weights, context_index) = neuron.get_current_weights(features);
            weight_vec.append(&mut weights.clone());
            context_index_map.insert(neuron_index, context_index);
        }

        let weight_matrix = DMatrix::from_vec(self.num_neurons, self.input_dim, weight_vec);

        LayerPrediction {
            predictions: (weight_matrix * previous_vector),
            context_index_map: Some(context_index_map),
        }
    }
}

pub struct BaseLayerPrediction {
    pub logits: DMatrix<f32>,
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
        let max_value = features.max();
        let min_value = features.min();

        let base_predictions = features
            .iter()
            .map(|value| (value - min_value) / (max_value - min_value))
            .map(|value| clip(value, self.pred_clipping_value))
            .collect::<Vec<f32>>();
        base_predictions
    }

    pub fn predict_with_logits(&self, features: &DVector<f32>) -> LayerPrediction {
        let max_value = features.max();
        let min_value = features.min();

        let base_predictions = features
            .iter()
            .map(|value| (value - min_value) / (max_value - min_value))
            .map(|value| logit(clip(value, self.pred_clipping_value)))
            .collect::<Vec<f32>>();

        LayerPrediction {
            predictions: DMatrix::from_vec(self.feature_dim, 1, base_predictions),
            context_index_map: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::layer::BaseLayer;

    #[test]
    fn test_base_layer_predict() {
        let features = vec![1.0, 5.0, 4.0, 4.0];
        let base_layer = BaseLayer {
            pred_clipping_value: 0.01,
        };
        let actual = base_layer.predict(&features);

        let expected = vec![-4.59512, 4.595121, 1.0986123, 1.0986123];
        assert_eq!(actual, expected);
    }
}
