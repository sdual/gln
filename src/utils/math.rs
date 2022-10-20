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

pub fn clip(value: f32, epsilon: f32) -> f32 {
    if value == 1.0 {
        value - epsilon
    } else if value == 0.0 {
        value + epsilon
    } else {
        value
    }
}

pub fn max(values: &Vec<f32>) -> f32 {
    values.iter().fold(0.0 / 0.0, |m, v| v.max(m))
}

pub fn min(values: &Vec<f32>) -> f32 {
    values.iter().fold(0.0 / 0.0, |m, v| v.min(m))
}

#[cfg(test)]
mod tests {
    use crate::utils::math::{clip, geometric_mixing, logit, sigmoid};

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

    #[test]
    fn test_clip() {
        let epsilon = 0.01;

        let value1 = 1.0_f32;
        let value2 = 0.0_f32;
        let value3 = 0.3_f32;

        let actual1 = clip(value1, epsilon);
        let actual2 = clip(value2, epsilon);
        let actual3 = clip(value3, epsilon);

        assert_eq!(actual1, 1.0_f32 - epsilon);
        assert_eq!(actual2, 0.0_f32 + epsilon);
        assert_eq!(actual3, 0.3_f32);
    }
}
