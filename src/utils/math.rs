pub fn logit(value: f32) -> f32 {
    (value / (1.0 - value)).ln()
}

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn geometric_mixing(probabilities: &Vec<f32>, weights: &Vec<f32>) -> f32 {
    let weight_multiplied_logits = weights.iter()
        .zip(probabilities).map(|(w, p)| w * logit(*p)).sum();
    sigmoid(weight_multiplied_logits)
}

#[test]
fn test_logit() {
    let actual = logit(0.3);
    assert_eq!(actual, -0.8472978);
}

#[test]
fn test_sigmoid() {
    let actual = sigmoid(1.0);
    assert_eq!(actual, 0.7310586);
}

#[test]
fn test_logit_sigmoid() {
    let actual = logit(sigmoid(2.0));
    let abs_difference = (actual - 2.0).abs();
    assert!(abs_difference <= 0.000001);
}


#[test]
fn test_geometric_mixing() {
    let probabilities = vec![0.3, 0.2, 0.7];
    let weights = vec![3.0, 5.0, 2.0];

    let actual = geometric_mixing(&probabilities, &weights);
    assert_eq!(actual, 0.00041835158);
}
