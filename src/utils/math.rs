fn logit(value: &f32) -> f32 {
    (value / (1.0 - value)).ln()
}

fn sigmoid(value: &f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[test]
fn test_logit() {
    let actual = logit(&0.3);
    assert_eq!(actual, -0.8472978);
}

#[test]
fn test_sigmoid() {
    let actual = sigmoid(&1.0);
    assert_eq!(actual, 0.7310586);
}

#[test]
fn test_logit_sigmoid() {
    let actual = logit(&sigmoid(&2.0));
    let abs_difference = (actual - 2.0).abs();
    assert!(abs_difference <= 0.000001);
}
