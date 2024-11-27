use crate::data::ItemBatch;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{
    lstm::{Lstm, LstmConfig},
    Embedding, EmbeddingConfig, Linear, LinearConfig,
};

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::train::ClassificationOutput;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::ValidStep;

#[derive(Config)]
pub struct ModelConfig {
    vocab_size: usize,
    embedding_dim: usize,
    lstm_dim: usize,
    batch_size: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input: Embedding<B>,
    lstm_layer_a: Lstm<B>,
    lstm_layer_b: Lstm<B>,
    output: Linear<B>,
    vocab_size: usize,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, item: ItemBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.x.dims();
        let device = &self.input.devices()[0];

        let texts = item.x.to_device(device);
        let labels = item.y.to_device(device);

        let output = self.common_fwd(texts);
        let output = output.reshape([batch_size, seq_length*self.vocab_size]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), labels.clone());

        ClassificationOutput {
            loss,
            output: output,
            targets: labels
        }

    }

    pub fn infer(&self, item: Vec<usize>, device: B::Device) -> u32{
        let flat_x_tensor = Tensor::<B, 1, Int>::from_ints(&item[..], &device);
        let batch_size = 1;
        let sequence_length = 64;
        let new_x = flat_x_tensor.reshape([batch_size, sequence_length]);

        let text = new_x.to_device(&device);
        let output = self.common_fwd(text);

        let tmp = output
            .slice([0..1, 0..1])
            .reshape([1, self.vocab_size]);

        let label = softmax(tmp, 1)
            .argmax(1)
            .squeeze::<1>(1)
            .into_scalar();

        label.elem()
    }

    fn common_fwd(&self, data: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let input_emb = self.input.forward(data);
        let (_, out_state) = self.lstm_layer_a.forward(input_emb.clone(), None);
        let (out_ten, _) = self.lstm_layer_b.forward(input_emb, Some(out_state));
        self.output.forward(out_ten)
    }
}

impl<B: AutodiffBackend> TrainStep<ItemBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: ItemBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Perform forward pass
        let item = self.forward(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ItemBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: ItemBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);

        let lstm_layer_a = LstmConfig::new(self.embedding_dim, self.lstm_dim, true).init(device);
        let lstm_layer_b = LstmConfig::new(self.embedding_dim, self.lstm_dim, true).init(device);

        let output = LinearConfig::new(self.lstm_dim, self.vocab_size).init(device);

        Model {
            input,
            lstm_layer_a,
            lstm_layer_b,
            output,
            vocab_size: self.vocab_size,
        }
    }
}
