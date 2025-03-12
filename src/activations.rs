use ndarray::Array2;

pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::ReLU => x.mapv(|v| if v > 0.0 { v } else {0.0}),
            Self::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Self::Tanh => x.mapv(|v| v.tanh()),
        }
    }
}