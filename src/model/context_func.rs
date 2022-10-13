pub trait ContextFunction<const N: usize> {
    fn indicator_func(&self, side_effects: &Vec<f32>) -> [bool; N];
}


pub struct HalfSpaceContext {}

impl<const N: usize> ContextFunction<N> for HalfSpaceContext {
    fn indicator_func(&self, side_effects: &Vec<f32>) -> [bool; N] {
        todo!()
    }
}

pub struct SkipGramContext {}

impl<const N: usize> ContextFunction<N> for SkipGramContext {
    fn indicator_func(&self, side_effects: &Vec<f32>) -> [bool; N] {
        todo!()
    }
}
