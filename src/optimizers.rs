use ndarray::Array2;

pub enum Optimizer {
    SGD { learning_rate: f32 },
    Adam { lr: f32, beta1: f32, beta2: f32 },
}

impl Optimizer {
    pub fn update(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        // Implement update rule
    }
}