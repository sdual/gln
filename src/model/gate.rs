use nalgebra::DVector;

use crate::model::context_func::ContextFunction;
use crate::model::context_func::HalfSpaceContext;
use crate::utils::data_type::ContextIndex;

pub struct Gate<C: ContextFunction> {
    weights: Vec<Vec<f32>>,
    context_func: C,
}

impl Gate<HalfSpaceContext> {
    pub fn new<F>(
        input_dim: usize,
        context_dim: usize,
        feature_dim: usize,
        weight_init_func: F,
    ) -> Gate<HalfSpaceContext>
    where
        F: Fn(usize, usize) -> Vec<Vec<f32>>,
    {
        Gate {
            weights: weight_init_func(input_dim, context_dim),
            context_func: HalfSpaceContext::new(context_dim, feature_dim),
        }
    }
}

impl<C: ContextFunction> Gate<C> {
    pub fn select_weights(&self, side_info: &DVector<f32>) -> (Vec<f32>, usize) {
        let indicator = Self::transform_contexts_to_weight_indicator(
            self.context_func.indicator_func(side_info.as_slice()),
        );
        (self.weights[indicator].clone(), indicator)
    }

    pub fn update_weights(&mut self, context_index: usize, weights: Vec<f32>) {
        self.weights[context_index] = weights;
    }

    fn transform_contexts_to_weight_indicator(contexts: Vec<bool>) -> usize {
        let mut weight_indicator: i32 = 0;
        // Transform a binary number context to a decimal number.
        // This decimal number specifies the index of the weight to be used.
        for (index, bit) in contexts.iter().enumerate() {
            weight_indicator += 2_i32.pow(index as u32) * (*bit as i32)
        }
        weight_indicator as usize
    }

    pub fn get_weights(&self, context_index: ContextIndex) -> Vec<f32> {
        self.weights[context_index].clone()
    }
}

pub fn initialize_balanced_weights(input_dim: usize, context_dim: usize) -> Vec<Vec<f32>> {
    let init_value: f32 = 1.0 / (input_dim as f32);
    (0..2_i32.pow(context_dim as u32))
        .into_iter()
        .map(|_| {
            (0..input_dim)
                .into_iter()
                .map(|_| init_value)
                .collect::<Vec<f32>>()
        })
        .collect()
}

// #[cfg(test)]
// mod test {
//     use mockall::mock;

//     use crate::model::context_func::ContextFunction;
//     use crate::model::context_func::HalfSpaceContext;
//     use crate::model::gate::{initialize_balanced_weights, Gate};

//     #[test]
//     fn test_transform_bits_to_weight_indicator() {
//         let contexts = vec![true, false, true, true];
//         let actual = Gate::<HalfSpaceContext>::transform_contexts_to_weight_indicator(contexts);
//         assert_eq!(actual, 13);
//     }

//     mock! {
//         pub ContextFunctionM {}

//         impl ContextFunction for ContextFunctionM {
//             fn indicator_func(&self, side_info: &[f32]) -> Vec<bool>;
//         }
//     }

//     #[test]
//     fn test_select_weights() {
//         let mut mock_context_func = MockContextFunctionM::new();
//         mock_context_func
//             .expect_indicator_func()
//             .returning(|side_info| vec![false, true]);
//         let gate = Gate {
//             input_dim: 4,
//             weights: vec![
//                 vec![0.1, 0.2],
//                 vec![0.3, 0.4],
//                 vec![0.5, 0.6],
//                 vec![0.7, 0.8],
//             ],
//             context_func: mock_context_func,
//         };

//         let side_info = vec![0.1, 0.2, 0.2, 0.9];
//         let (actual, index) = gate.select_weights(&side_info);

//         assert_eq!(*actual, vec![0.5, 0.6]);
//         assert_eq!(index, 2_usize);
//     }

//     #[test]
//     fn test_update_weights() {
//         let mut gate = Gate::<HalfSpaceContext>::new(2, 3, 10, initialize_balanced_weights);
//         gate.update_weights(0, vec![0.2, 0.1]);
//         let actual = &gate.weights[0];
//         let expected: Vec<f32> = vec![0.2, 0.1];
//         assert_eq!(*actual, expected);
//     }

//     #[test]
//     fn test_initialize_balanced_weights() {
//         let actual = initialize_balanced_weights(2, 2);
//         let expected: Vec<Vec<f32>> = vec![
//             vec![0.5, 0.5],
//             vec![0.5, 0.5],
//             vec![0.5, 0.5],
//             vec![0.5, 0.5],
//         ];
//         assert_eq!(actual, expected);
//     }
// }
