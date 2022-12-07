pub struct LayerConfig {
    pub pred_clipping_value: f32,
}

impl LayerConfig {
    pub fn with_default_value() -> Self {
        LayerConfig {
            pred_clipping_value: 1e-3,
        }
    }
}
