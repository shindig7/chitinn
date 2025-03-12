use ndarray::Array2;

pub trait Layer {
    fn forward(&mut self, inputs: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad: &Array2<f32>) -> Array2<f32>;
}

pub struct DenseLayer {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_cache: Option<Array2<f32>>,
    // Add optimizer state for Adam/SGD
}

pub struct DropoutLayer {
    rate: f32,
    mask: Option<Array2<bool>>,
}