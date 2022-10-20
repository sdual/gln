use crate::model::config::LayerConfig;
use crate::model::context_func::HalfSpaceContext;
use crate::model::neuron::Neuron;
use crate::utils::math::{max, min};

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

    pub fn add_neuron(&mut self, neuron: Neuron<HalfSpaceContext>) {
        self.neurons.push(neuron)
    }

    pub fn predict_by_all_neurons(&mut self, features: &Vec<f32>, target: i32, inputs: &Vec<f32>) -> Vec<f32> {
        let mut predictions = Vec::with_capacity(self.neurons.len());
        for neuron in &mut self.neurons {
            let pred = neuron.predict_and_update_weights(features, inputs, target);
            predictions.push(pred);
        }
        predictions
    }
}

pub struct BaseLayer;

impl BaseLayer {
    pub fn predict(&self, features: &Vec<f32>) -> Vec<f32> {
        let max_value: f32 = max(features);
        let min_value: f32 = min(features);
        features.iter().map(|value| (value - min_value) / (max_value - min_value)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::model::layer::BaseLayer;

    #[test]
    fn test_base_layer_predict() {
        let features = vec![1.0, 5.0, 4.0, 4.0];
        let base_layer = BaseLayer;
        let actual = base_layer.predict(&features);

        let expected = vec![0.0, 1.0, 0.75, 0.75];
        assert_eq!(actual, expected);
    }
}
