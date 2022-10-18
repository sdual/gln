pub struct NeuronConfig {
    pub pred_clipping_value: f32,
    pub weight_clipping_value: f32,
    pub learning_late: f32,
}

impl NeuronConfig {
    pub fn with_default_value() -> Self {
        NeuronConfig {
            pred_clipping_value: 1e-3,
            weight_clipping_value: 5.0,
            learning_late: 1e-3,
        }
    }
}
