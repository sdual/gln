use crate::model::gln::GLN;

use std::process;
use crate::io::csv::DataFrame;
use crate::utils::math::accuracy;

mod model;
mod utils;
mod optimize;
mod io;

fn main() {
    let mnist_38_df = DataFrame::read_csv("/Users/qtk/git/python/mnist-data/notebook/mnist_38.csv", true);


    let neuron_nums = vec![8, 8, 1];
    let context_dim = 4;

    let mut gln = GLN::new(neuron_nums, context_dim, mnist_38_df.features[0].len());

    let mut predictions = Vec::new();
    for (features, label) in (&mnist_38_df.features).iter().zip(&mnist_38_df.labels) {
        let output = gln.predict(features, *label);
        predictions.push(output);
        println!("{}", output);
    }

    let acc = accuracy(&predictions, &mnist_38_df.labels);
    println!("accuracy: {}", acc);
}
