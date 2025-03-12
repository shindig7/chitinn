mod layer;
mod optimizers;

use crate::layer::Layer;
use crate::optimizers::Optimizer;
use ndarray::Array2;

pub struct Sequential {
    input_shape: usize,
    layers: Vec<Box<dyn Layer>>,
    optimizer: Optimizer,
}

impl Sequential {

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, grad: &Array2<f32>) {
        let mut grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    pub fn loss(&self, output: &Array2<f32>, target: &Array2<f32>) -> f32 {
        let diff = output - target;
        0.5 * diff.mapv(|v| v.powi(2)).sum()
    }

    pub fn loss_prim(&self, output: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        output - target
    }

    pub fn train(&mut self, x: &Array2<f32>, y: &Array2<f32>, epochs: usize) -> Vec<f32> {
        let mut losses: Vec<f32> = Vec::new();
        for _ in 0..epochs {
            let output = self.forward(x);
            let loss = self.loss(&output, y);
            losses.push(loss);
            let grad = self.loss_prim(&output, y);
            self.backward(&grad);
        }
        losses
    }
}