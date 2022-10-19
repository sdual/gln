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
}
