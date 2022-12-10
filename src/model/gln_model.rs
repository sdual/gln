use nalgebra::DVector;

use crate::model::config::LayerConfig;
use crate::model::layer::{BaseLayer, Layer};
use crate::utils::data_type::{ContextIndex, LayerId, NeuronId};
use crate::utils::math::{geometric_mixing_loss, sigmoid};
use std::collections::HashMap;

pub struct GLN {
    layers: Vec<Layer>,
    base_layer: BaseLayer,
    num_layers: usize,
}

pub struct GLNPrediction {
    pub probability: f32,
    pub context_index_map: HashMap<LayerId, HashMap<NeuronId, ContextIndex>>,
}

pub struct GLNTrainHistory {
    pub loss_histories: HashMap<LayerId, HashMap<NeuronId, f32>>,
}

pub struct PredictFitResult {
    pub prediction: f32,
    pub loss_histories: HashMap<LayerId, HashMap<NeuronId, f32>>,
}

impl GLN {
    pub fn new(
        neuron_nums: Vec<usize>,
        context_dim: usize,
        feature_dim: usize,
        learning_rate: f32,
        weight_clipping_value: f32,
        grad_weight: f32,
        reg_param: f32,
    ) -> Self {
        let mut layers = Vec::with_capacity(neuron_nums.len());
        let first_layer = Layer::with_neuron_num(
            neuron_nums[0],
            feature_dim,
            context_dim,
            feature_dim,
            learning_rate,
            weight_clipping_value,
            grad_weight,
            reg_param,
        );
        layers.push(first_layer);

        let num_layers = neuron_nums.len();
        for layer_index in (0..num_layers).skip(1) {
            let input_dim = neuron_nums[layer_index - 1] as usize;
            let layer = Layer::with_neuron_num(
                neuron_nums[layer_index],
                input_dim,
                context_dim,
                feature_dim,
                learning_rate,
                weight_clipping_value,
                grad_weight,
                reg_param,
            );
            layers.push(layer);
        }

        let config = LayerConfig::with_default_value();

        GLN {
            layers,
            base_layer: BaseLayer::new(config.pred_clipping_value, feature_dim),
            num_layers,
        }
    }

    pub fn predict_fit(&mut self, features: &DVector<f32>, target: i32) -> PredictFitResult {
        let pred = self.predict(features);
        let train_history = self.train(features, target, &pred.context_index_map);

        PredictFitResult {
            prediction: pred.probability,
            loss_histories: train_history.loss_histories,
        }
    }

    pub fn train(
        &mut self,
        features: &DVector<f32>,
        target: i32,
        context_index_map: &HashMap<LayerId, HashMap<NeuronId, ContextIndex>>,
    ) -> GLNTrainHistory {
        let mut inputs = self.base_layer.predict(features);
        let mut loss_history = HashMap::new();

        for layer_id in 0usize..self.num_layers {
            let layer_context_index_map = &context_index_map[&layer_id];
            let inputs_tmp =
                self.layers[layer_id].predict_by_context_index(&layer_context_index_map, &inputs);
            loss_history.insert(layer_id, self.calculate_layer_losses(&inputs_tmp, target));

            self.layers[layer_id].train(&layer_context_index_map, &inputs, target);
            inputs = inputs_tmp;
        }

        GLNTrainHistory {
            loss_histories: loss_history,
        }
    }

    pub fn calculate_layer_losses(
        &self,
        predictions: &Vec<f32>,
        target: i32,
    ) -> HashMap<NeuronId, f32> {
        predictions
            .iter()
            .enumerate()
            .map(|(neuron_id, pred)| (neuron_id, geometric_mixing_loss(target, *pred)))
            .collect()
    }

    pub fn predict(&self, features: &DVector<f32>) -> GLNPrediction {
        let mut layer_prediction = self.base_layer.predict_with_logits(features);
        let mut layer_context_index_map = HashMap::new();

        for layer_index in 0..self.num_layers {
            layer_prediction = self.layers[layer_index]
                .calculate_next_weight_matrix(features, &layer_prediction.predictions);
            layer_context_index_map
                .insert(layer_index, layer_prediction.context_index_map.unwrap());
        }

        if let Some(&pred) = layer_prediction.predictions.get(0) {
            GLNPrediction {
                probability: sigmoid(pred),
                context_index_map: layer_context_index_map,
            }
        } else {
            panic!("prediction value is not found. `predictions` vector is empty.");
        }
    }
}
