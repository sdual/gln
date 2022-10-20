use crate::model::layer::{BaseLayer, Layer};

struct GLN {
    layers: Vec<Layer>,
    base_layer: BaseLayer,
}

impl GLN {
    pub fn new(neuron_nums: Vec<i32>, context_dim: usize, feature_dim: usize) -> Self {
        let num_layers = neuron_nums.len();
        let mut layers = Vec::with_capacity(neuron_nums.len());
        let first_layer = Layer::with_neuron_num(neuron_nums[0], feature_dim, context_dim, feature_dim);
        layers.push(first_layer);
        for layer_index in (0..num_layers).skip(1) {
            let input_dim = neuron_nums[layer_index - 1] as usize;
            let layer = Layer::with_neuron_num(neuron_nums[layer_index], input_dim, context_dim, feature_dim);
            layers.push(layer);
        }

        GLN {
            layers,
            base_layer: BaseLayer,
        }
    }

    pub fn predict(&mut self, features: &Vec<f32>, target: i32) -> f32 {
        let mut predictions = self.base_layer.predict(features);
        for layer in &mut self.layers {
            predictions = layer.predict_by_all_neurons(features, target, &predictions);
        }

        // TODO: 安全に取り出す
        predictions[0]
    }
}
