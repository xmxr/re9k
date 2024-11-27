use crate::data::{MyDataset, ItemBatcher};
use crate::model::ModelConfig;
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::DefaultRecorder,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::store::{Aggregate, Direction, Split},
        metric::{AccuracyMetric, CudaMetric, LossMetric},
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 64)]
    pub sequence_length: usize,

    #[config(default = 1)]
    pub num_workers: usize,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    pub optimizer: AdamConfig,

    pub vocab_size: Option<usize>,
    pub model_conf: ModelConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    let path = std::path::Path::new(artifact_dir);
    if path.exists() {
        let mut dir_entries = std::fs::read_dir(artifact_dir)
            .unwrap_or_else(|_| panic!("Error reading directory"));
        if !dir_entries.next().is_none() {
            panic!("Artifact directory not empty!");
        }
    }
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(devices: Vec<B::Device>, ds_train: &str) {
    let artifact_dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets");
    create_artifact_dir(&artifact_dir);

    let seed = 31337;
    B::seed(seed);
    let (dataset_a, dataset_b) = MyDataset::new(&ds_train, 80, seed, &artifact_dir).expect("Could not load diabetes dataset");

    let vocab_size = dataset_a.get_vocab_size();
    let model = ModelConfig::new(
        vocab_size+1, //vocab_size
        256, //embedding_dim
        128, //lstm_units
        64); //batch_size

    let config_optimizer = AdamConfig::new();
    let config = TrainingConfig::new(config_optimizer, model.clone());
    let batcher_train = ItemBatcher::<B>::new(devices[0].clone(), &artifact_dir, config.sequence_length);
    let batcher_test = ItemBatcher::<B::InnerBackend>::new(devices[0].clone(), &artifact_dir, config.sequence_length);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(seed)
        .build(dataset_a);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(seed)
        .build(dataset_b);

    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(DefaultRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(devices.clone())
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model.init(&devices[0]),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("saved config file");


}
