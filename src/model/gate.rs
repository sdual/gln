use mockall::mock;

use crate::model::context_func::ContextFunction;

pub struct Gate<'a, const N: usize, const M: usize> {
    input_dim: usize,
    weights: [Vec<f32>; M],
    context_func: &'a dyn ContextFunction<N>,
}

impl<'a, const N: usize, const M: usize> Gate<'a, N, M> {
    pub fn get_weights(&self, side_effects: &Vec<f32>) -> &Vec<f32> {
        let indicator = Self::transform_contexts_to_weight_indicator(
            self.context_func.indicator_func(side_effects)
        );
        &self.weights[indicator]
    }

    fn transform_contexts_to_weight_indicator(contexts: [bool; N]) -> usize {
        let mut weight_indicator: i32 = 0;
        for (index, bit) in contexts.iter().enumerate() {
            weight_indicator += 2_i32.pow(index as u32) * (*bit as i32)
        }
        weight_indicator as usize
    }
}

#[test]
fn test_transform_bits_to_weight_indicator() {
    let contexts = [true, false, true, true];
    let actual = Gate::<4, 1>::transform_contexts_to_weight_indicator(contexts);
    assert_eq!(actual, 13);
}

mock! {
    pub ContextFunctionM {}

    impl ContextFunction<2> for ContextFunctionM {
        fn indicator_func(&self, side_effects: &Vec<f32>) -> [bool; 2];
    }
}

#[test]
fn test_get_weights() {
    let mut mock_context_func = MockContextFunctionM::new();
    mock_context_func.expect_indicator_func()
        .returning(|side_effects| [false, true]);
    let gate = Gate {
        input_dim: 4,
        weights: [vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6], vec![0.7, 0.8]],
        context_func: &mock_context_func,
    };

    let side_effects = vec![0.1, 0.2, 0.2, 0.9];
    let actual = gate.get_weights(&side_effects);

    assert_eq!(*actual, vec![0.5, 0.6]);
}
