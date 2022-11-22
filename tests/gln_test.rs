use gln::model::gln;

#[test]
fn gln_test() {
    let neuron_nums = vec![3, 2, 1];
    let context_dim = 5;
    let learning_rate = 0.1;
    let feature_dim = 3;

    let feature_vec = DVector::from_vec(vec![0.2, 0.3, 0.1]);
    let label = 1;

    let mut gln = gln::GLN::new(neuron_nums, context_dim, feature_dim, learning_rate);
    let pred = gln.predict(feature_vec);

    println!("prediction: {}", pred.probability);
}
