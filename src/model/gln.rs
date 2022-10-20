use crate::model::layer::{BaseLayer, Layer};

struct GLN {
    layers: Vec<Layer>,
    base_layer: BaseLayer,
}

impl GLN {
    pub fn new(layers: Vec<Layer>, base_layer: BaseLayer) -> Self {
        GLN {
            layers,
            base_layer,
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
