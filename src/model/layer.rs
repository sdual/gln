use crate::model::config::LayerConfig;
use crate::model::context_func::HalfSpaceContext;
use crate::model::neuron::Neuron;
use crate::utils::math::{clip, logit, max, min};

pub struct Layer {
    pred_clipping_value: f32,
    weight_clipping_value: f32,
    neurons: Vec<Neuron<HalfSpaceContext>>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron<HalfSpaceContext>>) -> Self {
        let config = LayerConfig::with_default_value();
        Layer {
            pred_clipping_value: config.pred_clipping_value,
            weight_clipping_value: config.weight_clipping_value,
            neurons: neurons,
        }
    }

    pub fn with_neuron_num(neuron_num: i32, input_dim: usize, context_dim: usize, feature_dim: usize) -> Self {
        let neurons = (0..neuron_num)
            .map(|_| Neuron::with_half_space_context(input_dim, context_dim, feature_dim)).collect();
        Self::new(neurons)
    }

    pub fn add_neuron(&mut self, neuron: Neuron<HalfSpaceContext>) {
        self.neurons.push(neuron)
    }

    pub fn predict_by_all_neurons(&mut self, features: &Vec<f32>, target: i32, inputs: &Vec<f32>) -> Vec<f32> {
        let mut predictions = Vec::with_capacity(self.neurons.len());
        for neuron in &mut self.neurons {
            let pred = clip(neuron.predict_and_update_weights(features, inputs, target), self.pred_clipping_value);
            predictions.push(pred);
        }
        predictions
    }
}

pub struct BaseLayer {
    pred_clipping_value: f32,
}

impl BaseLayer {
    pub fn new(pred_clipping_value: f32) -> Self {
        BaseLayer {
            pred_clipping_value,
        }
    }

    pub fn predict(&self, features: &Vec<f32>) -> Vec<f32> {
        let max_value: f32 = max(features);
        let min_value: f32 = min(features);
        features.iter()
            .map(|value| (value - min_value) / (max_value - min_value))
            .map(|value|clip(value, self.pred_clipping_value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::model::layer::BaseLayer;

    #[test]
    fn test_base_layer_predict() {
        let features = vec![1.0, 5.0, 4.0, 4.0];
        let base_layer = BaseLayer {
            pred_clipping_value: 0.01
        };
        let actual = base_layer.predict(&features);

        let expected = vec![-4.59512, 4.595121, 1.0986123, 1.0986123];
        assert_eq!(actual, expected);
    }
}
