use crate::model::layer::Layer;

struct GLN {
    layers: Vec<Layer>,
}

impl GLN {
    pub fn new(layers: Vec<Layer>) -> Self {
        GLN {
            layers
        }
    }

    pub fn predict(&mut self, features: &Vec<f32>, target: i32) {
        // for layer in &mut self.layers {
        //     layer.predict_by_all_neurons()
        // }
        todo!()
    }

}
