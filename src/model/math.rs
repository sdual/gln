fn logit(p: f32) -> f32 {
    (p / 1.0 - p).ln()
}

fn sigmoid(x: f32) -> f32 {
    1.0 as f32 / (1.0 as f32 + (-x).exp())
}