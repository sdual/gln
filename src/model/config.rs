pub struct LayerConfig {
    pub pred_clipping_value: f32,
    pub weight_clipping_value: f32,
    pub learning_rate: f32,
}

impl LayerConfig {
    pub fn with_default_value() -> Self {
        LayerConfig {
            pred_clipping_value: 1e-1,
            weight_clipping_value: 5.0,
            learning_rate: 1e-3,
        }
    }
}
