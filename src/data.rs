use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::{prelude::*, tensor::Tensor};
use serde::{Deserialize, Serialize};

use std::collections::{HashMap};
use std::fs::File;
use std::io::BufReader;
use serde_json;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MyItem {
    pub opt: u8,
    pub asm: String,
}

pub struct MyDataset {
    vocab: HashMap<String, usize>,
    dataset: InMemDataset<MyItem>,
}

impl MyDataset {
    pub fn new(ds_train: &str, split: usize, seed: u64, artifact_dir: &str) -> Result<(Self, Self), std::io::Error> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vocab = HashMap::new();

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_path(ds_train)
            .expect("read from CSV");

        let mut dataset: Vec<MyItem> = rdr
            .deserialize::<MyItem>()
            .map(|x| x.unwrap())
            .collect();

        let mut idx = 0;
        for item in &dataset {
            item.asm.split_ascii_whitespace()
                .for_each(|word|{
                    vocab
                        .entry(word.to_string())
                        .or_insert_with(|| {idx += 1; idx});
                });
        }
        let file = File::create(format!("{artifact_dir}/vocab.json"))?;
        serde_json::to_writer(file, &vocab)?;

        dataset.shuffle(&mut rng);
        let idx_split = dataset.len() * split / 100;
        let validation = dataset.split_off(idx_split);

        Ok((
            Self {
                vocab: vocab.clone(),
                dataset: InMemDataset::new(dataset),
            },
            Self {
                vocab,
                dataset: InMemDataset::new(validation)
            }
        ))

    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn get_vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }
}

impl Dataset<MyItem> for MyDataset {
    fn get(&self, index: usize) -> Option<MyItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone)]
pub struct ItemBatcher<'a, B: Backend> {
    device: B::Device,
    artifact_dir: &'a str,
    seq_len: usize
}

#[derive(Clone, Debug)]
pub struct ItemBatch<B: Backend> {
    pub x: Tensor<B, 2, Int>,
    pub y: Tensor<B, 1, Int>,
}

impl<B: Backend> ItemBatcher<'_, B> {
    pub fn new(device: B::Device, artifact_dir: &'static str, seq_len: usize) -> Self {
        Self { device, artifact_dir, seq_len}
    }
}

impl<B: Backend> Batcher<MyItem, ItemBatch<B>> for ItemBatcher<'_, B> {
    fn batch(&self, items: Vec<MyItem>) -> ItemBatch<B> {
        let batch_size = items.len();
        let sequence_length = self.seq_len;

        let file = File::open(format!("{}/vocab.json", self.artifact_dir)).unwrap();
        let reader = BufReader::new(file);
        let vocab: HashMap<String, usize> = serde_json::from_reader(reader).expect("load vocab");

        let mut flat_input = vec![];
        let mut flat_label = vec![];
        
        for item in items {
            let mut tmp: Vec<usize> = item.asm
            .split_ascii_whitespace()
            .map(|word| vocab.get(word).expect("index for word").to_owned())
            .collect();

            tmp.resize(sequence_length, 0);
            flat_input.extend_from_slice(&tmp[..]);

            flat_label.push(item.opt);
        }

        let flat_x_tensor = Tensor::<B, 1, Int>::from_ints(&flat_input[..], &self.device);
        let flat_y_tensor = Tensor::<B, 1, Int>::from_ints(&flat_label[..], &self.device);


        let new_x = flat_x_tensor.reshape([batch_size, sequence_length]);

        ItemBatch{
            x: new_x,
            y: flat_y_tensor
        }
    }
}
