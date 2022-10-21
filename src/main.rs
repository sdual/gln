use crate::model::gln::GLN;

use std::process;
use crate::io::csv::DataFrame;

mod model;
mod utils;
mod optimize;
mod io;

fn main() {
    let mnist_38_df = DataFrame::read_csv("/Users/qtk/git/python/mnist-data/notebook/mnist_38.csv", true);


    let neuron_nums = vec![4, 4, 1];
    let context_dim = 4;

    let mut gln = GLN::new(neuron_nums, context_dim, mnist_38_df.features[0].len());

    for (features, label) in mnist_38_df.features.iter().zip(mnist_38_df.labels) {
        let output = gln.predict(features, label);
        println!("{}", output);
    }
}
