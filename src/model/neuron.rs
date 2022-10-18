use crate::model::config::NeuronConfig;
use crate::model::context_func::{ContextFunction, HalfSpaceContext};
use crate::model::gate::{Gate, initialize_balanced_weights};

pub struct Neuron<C: ContextFunction> {
    pred_clipping_value: f32,
    weight_clipping_value: f32,
    learning_late: f32,
    gate: Gate<C>,
}

impl Neuron<HalfSpaceContext> {
    pub fn new(input_dim: usize,
               context_dim: usize,
               feature_dim: usize) -> Neuron<HalfSpaceContext> {
        let config = NeuronConfig::with_default_value();
        Neuron {
            pred_clipping_value: config.pred_clipping_value,
            weight_clipping_value: config.weight_clipping_value,
            learning_late: config.learning_late,
            gate: Gate::<HalfSpaceContext>::new(
                input_dim,
                context_dim,
                feature_dim,
                initialize_balanced_weights,
            ),
        }
    }
}

impl<C: ContextFunction> Neuron<C> {
    pub fn update_weights(&self, side_info: &Vec<f32>, inputs: &Vec<f32>) {
        todo!()
    }
}
