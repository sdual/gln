pub trait ContextFunction {
    fn indicator_func(&self, side_effects: &Vec<f32>) -> Vec<bool>;
}

pub struct HalfSpaceContext {}

impl ContextFunction for HalfSpaceContext {
    fn indicator_func(&self, side_effects: &Vec<f32>) -> Vec<bool> {
        todo!()
    }
}

pub struct SkipGramContext {}

impl ContextFunction for SkipGramContext {
    fn indicator_func(&self, side_effects: &Vec<f32>) -> Vec<bool> {
        todo!()
    }
}
