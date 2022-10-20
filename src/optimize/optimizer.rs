pub struct OnlineGradientDecent {
    learning_rate: f32,
    pred_clipping_value: f32,
}

impl OnlineGradientDecent {
    pub fn new(learning_rate: f32, pred_clipping_value: f32) -> Self {
        OnlineGradientDecent {
            learning_rate: learning_rate,
            pred_clipping_value: pred_clipping_value,
        }
    }

    pub fn update(&self, weight: f32, grad: f32) -> f32 {
        weight - self.learning_rate * grad
    }
}
