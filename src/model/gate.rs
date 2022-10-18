use mockall::mock;

use crate::model::context_func::ContextFunction;
use crate::model::context_func::HalfSpaceContext;

pub struct Gate<'a> {
    input_dim: usize,
    weights: Vec<Vec<f32>>,
    context_func: &'a dyn ContextFunction,
}

impl<'a> Gate<'a> {
    pub fn with_half_space_context(
        input_dim: usize, context_dim: usize) -> Self {
        Gate {
            input_dim: input_dim,
            weights: vec![vec![]],
            context_func: &HalfSpaceContext {},
        }
    }

    pub fn select_weights(&self, side_effects: &Vec<f32>) -> &Vec<f32> {
        let indicator = Self::transform_contexts_to_weight_indicator(
            self.context_func.indicator_func(side_effects)
        );
        &self.weights[indicator]
    }

    fn transform_contexts_to_weight_indicator(contexts: Vec<bool>) -> usize {
        let mut weight_indicator: i32 = 0;
        for (index, bit) in contexts.iter().enumerate() {
            weight_indicator += 2_i32.pow(index as u32) * (*bit as i32)
        }
        weight_indicator as usize
    }
}

#[test]
fn test_transform_bits_to_weight_indicator() {
    let contexts = vec![true, false, true, true];
    let actual = Gate::transform_contexts_to_weight_indicator(contexts);
    assert_eq!(actual, 13);
}

mock! {
    pub ContextFunctionM {}

    impl ContextFunction for ContextFunctionM {
        fn indicator_func(&self, side_effects: &Vec<f32>) -> Vec<bool>;
    }
}

#[test]
fn test_select_weights() {
    let mut mock_context_func = MockContextFunctionM::new();
    mock_context_func.expect_indicator_func()
        .returning(|side_effects| vec![false, true]);
    let gate = Gate {
        input_dim: 4,
        weights: vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6], vec![0.7, 0.8]],
        context_func: &mock_context_func,
    };

    let side_effects = vec![0.1, 0.2, 0.2, 0.9];
    let actual = gate.select_weights(&side_effects);

    assert_eq!(*actual, vec![0.5, 0.6]);
}
