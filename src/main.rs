use nalgebra::{DMatrix, DVector};

use crate::model::gln::GLN;

use crate::io::csv::DataFrame;
use crate::utils::math::accuracy;

mod io;
mod model;
mod optimize;
mod utils;

fn main() {
    let mnist_38_df = DataFrame::read_csv(
        "/Users/qtk/git/python/mnist-data/notebook/mnist_38.csv",
        true,
    );

    let neuron_nums = vec![20, 20, 1];
    let context_dim = 5;

    let mut gln = GLN::new(neuron_nums, context_dim, mnist_38_df.features[0].len());

    let mut predictions = Vec::new();
    for (features, label) in (&mnist_38_df.features).iter().zip(&mnist_38_df.labels) {
        let feature_vec = DVector::from_vec(features.clone());
        let mut output = gln.predict(&feature_vec);
        predictions.push(output.probability);
        _ = gln.train(&feature_vec, *label, &mut output.context_index_map);

        println!("preidction: {}", output.probability);
    }

    let acc = accuracy(&predictions, &mnist_38_df.labels);
    println!("accuracy: {}", acc);
}
