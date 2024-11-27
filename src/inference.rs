use std::collections::{HashMap};
use std::fs::File;
use std::io::BufReader;
use serde_json;

use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use crate::training::TrainingConfig;
use burn::config::Config;
use burn::record::Recorder;
use burn::module::Module;

pub fn infer<B: AutodiffBackend>(devices: Vec<B::Device>, item: String) -> u32{
    let artifact_dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets");
    let file = File::open(format!("{artifact_dir}/vocab.json")).unwrap();
    let reader = BufReader::new(file);
    let vocab: HashMap<String, usize> = serde_json::from_reader(reader).expect("load vocab");

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("load config");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &devices[0])
        .expect("load model");
    
    let model = config.model_conf.init::<B>(&devices[0]).load_record(record);
    let mut enc_inp = item
        .split_ascii_whitespace()
        .map(|word| vocab.get(word).unwrap_or(&1).to_owned())
        .collect::<Vec<_>>();

    enc_inp.resize(64, 0);
    model.infer(enc_inp, devices[0].clone())

}
