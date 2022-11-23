use std::collections::HashMap;

use gln::model::gln;
use nalgebra::DVector;

#[test]
fn test_gln_predict() {
    let neuron_nums = vec![3, 2, 1];
    let context_dim = 5;
    let learning_rate = 0.1;
    let feature_dim = 3;

    let feature_vec = DVector::from_vec(vec![0.2, 0.3, 0.1]);

    let gln = gln::GLN::new(neuron_nums, context_dim, feature_dim, learning_rate);
    let pred = gln.predict(&feature_vec);

    assert_eq!(pred.probability, 0.5000011);
}

#[test]
fn test_gln_train() {
    let neuron_nums = vec![3, 2, 1];
    let context_dim = 5;
    let learning_rate = 0.1;
    let feature_dim = 3;

    let feature_vec = DVector::from_vec(vec![0.2, 0.3, 0.1]);
    let target = 1;

    let mut gln = gln::GLN::new(neuron_nums, context_dim, feature_dim, learning_rate);
    let pred = gln.predict(&feature_vec);
    let train_history = gln.train(&feature_vec, target, &pred.context_index_map);

    let expected = HashMap::from([
        (
            0,
            HashMap::from([(0, 0.69314504), (1, 0.69314504), (2, 0.69314504)]),
        ),
        (1, HashMap::from([(0, 0.69314504), (1, 0.69314504)])),
        (2, HashMap::from([(0, 0.69314504)])),
    ]);

    assert_eq!(train_history.loss_histories, expected);
}

#[test]
fn test_gln_predict_fit() {
    let neuron_nums = vec![3, 2, 1];
    let context_dim = 5;
    let learning_rate = 0.1;
    let feature_dim = 3;

    let feature_vec = DVector::from_vec(vec![0.2, 0.3, 0.1]);
    let target = 1;

    let mut gln = gln::GLN::new(neuron_nums, context_dim, feature_dim, learning_rate);
    let predict_fit_result = gln.predict_fit(&feature_vec, target);

    let expected_loss = HashMap::from([
        (
            0,
            HashMap::from([(0, 0.69314504), (1, 0.69314504), (2, 0.69314504)]),
        ),
        (1, HashMap::from([(0, 0.69314504), (1, 0.69314504)])),
        (2, HashMap::from([(0, 0.69314504)])),
    ]);

    assert_eq!(predict_fit_result.loss_histories, expected_loss);
    assert_eq!(predict_fit_result.prediction, 0.5000011);
}
