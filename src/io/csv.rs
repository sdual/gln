extern crate csv;
extern crate rustc_serialize;

use std::collections::HashMap;
use std::error::Error;

use csv::StringRecord;

#[derive(Debug)]
pub struct DataFrame {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<i32>,
}

impl DataFrame {
    pub fn new() -> DataFrame {
        DataFrame {
            features: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn read_csv(filepath: &str, has_headers: bool) -> DataFrame {
        // Open file
        let file = std::fs::File::open(filepath).unwrap();
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .from_reader(file);

        let mut data_frame = DataFrame::new();

        // push all the records
        for result in rdr.records().into_iter() {
            let record = result.unwrap();
            data_frame.push(&record);
        }
        return data_frame;
    }

    fn push(&mut self, row: &csv::StringRecord) {
        let feature_dim = 784;
        let mut row_vec = Vec::with_capacity(feature_dim);
        for index in 0..feature_dim {
            row_vec.push(row[index].parse().unwrap());
        }
        self.features.push(row_vec);
        self.labels.push(row[feature_dim].parse().unwrap());
    }
}
