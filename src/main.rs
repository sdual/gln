extern crate core;
extern crate alloc;

mod model;
mod utils;
mod optimize;

fn main() {
    let packed = "hoge";
    let spaced = "fuga";

    assert!(packed != spaced);
}
