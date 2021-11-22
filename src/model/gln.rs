pub struct GLN {
    layer_sizes: Vec<usize>,
    input_size: usize,
    num_classes: usize,
    context_map_size: usize,
    bias: bool,
    context_bias: bool,
    learning_rage: f32,
    pred_clipping: f32,
    weight_clipping: f32,
}

pub struct Linear {
    size: usize,
    input_size: usize,
    context_size: usize,
    context_map_size: usize,
    num_classes: usize,
    learning_rate: f32,
    pred_clipping: f32,
    weight_clipping: f32,
    bias: bool,
    context_bias: bool,
}

impl GLN {
    pub fn predict(&self, input: Vec<Vec<f32>>, target: Vec<f32>) -> Vec<f32> {}
}

impl Linear {
    pub fn predict(&self, context: Vec<f32>, target: Vec<f32>) -> f32 {
        let distances =
    }
}
