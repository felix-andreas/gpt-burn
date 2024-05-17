use {
    crate::{
        model::{Model, ModelConfig},
        BOLD, RESET,
    },
    burn::{
        module::{AutodiffModule, Module},
        optim::{AdamWConfig, GradientsParams, Optimizer},
        prelude::*,
        record::CompactRecorder,
        tensor::backend::AutodiffBackend,
    },
    rand::prelude::*,
    std::{
        fs,
        time::{Duration, Instant},
    },
};

#[derive(Config)]
pub struct TrainingConfig {
    pub n_steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub batches_per_step: usize,
    pub validation_size: usize,
    pub seed: u64,
    pub model: ModelConfig,
    pub optimizer: AdamWConfig,
}

pub fn train<B: AutodiffBackend>(
    config: &TrainingConfig,
    data_train: Tensor<B, 1, Int>,
    data_val: Tensor<B::InnerBackend, 1, Int>,
    save_checkpoints: bool,
) -> Model<B> {
    let device = data_train.device();
    let mut rng = StdRng::seed_from_u64(config.seed);

    B::seed(config.seed);

    let mut model: Model<B> = config.model.init(&device);
    let mut optimizer = config.optimizer.init::<B, Model<B>>();

    println!(
        "{BOLD}start training{RESET} - parameters: {} \n{}",
        model.num_params(),
        config
    );
    let start = Instant::now();
    for step in 0..config.n_steps {
        // evaluate validation loss
        let loss_val = {
            let (x, y) = get_batch(
                &mut rng,
                data_val.clone(),
                config.validation_size,
                config.model.context_length,
            );
            let model_valid = model.valid();
            let logits = model_valid.forward(x);
            let loss = model_valid.loss(logits, y.clone());
            loss.into_scalar().elem::<f64>()
        };

        let mut loss_sum = 0.0;
        for _ in 0..config.batches_per_step {
            let (x, y) = get_batch(
                &mut rng,
                data_train.clone(),
                config.batch_size,
                config.model.context_length,
            );

            // forward pass
            let logits = model.forward(x);
            let loss = model.loss(logits, y.clone());
            loss_sum += loss.clone().into_scalar().elem::<f32>();

            // backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }

        let elapsed = start.elapsed();

        let format_duration = |duration: Duration| {
            let seconds = duration.as_secs() % 60;
            let minutes = (duration.as_secs() / 60) % 60;
            let hours = (duration.as_secs() / 60) / 60;
            format!("{hours:02}:{minutes:02}:{seconds:02}")
        };
        println!(
            "step {:>3}/{}: train loss: {:.4}, val loss: {:.4} ({}/{})",
            step + 1,
            config.n_steps,
            loss_sum / config.batches_per_step as f32,
            loss_val,
            format_duration(elapsed),
            format_duration(elapsed.mul_f64(config.n_steps as f64 / (step + 1) as f64)),
        );

        if save_checkpoints && (step - 4) % 10 == 0 {
            let model_path = format!(
                ".data/checkpoints/{}_{step}",
                std::time::UNIX_EPOCH.elapsed().unwrap().as_secs(),
            );

            println!("{BOLD}store checkpoint model to: {model_path}{RESET}");
            fs::create_dir_all(&model_path).ok();

            config.save(format!("{model_path}/config.json")).unwrap();
            model
                .clone()
                .save_file(format!("{model_path}/model"), &CompactRecorder::new())
                .unwrap();
        }
    }

    model
}

#[allow(clippy::single_range_in_vec_init)]
fn get_batch<B: Backend>(
    rng: &mut impl Rng,
    data: Tensor<B, 1, Int>,
    batch_size: usize,
    context_length: usize,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let indices = (0..batch_size)
        .map(|_| rng.gen_range(0..data.dims()[0] - context_length))
        .collect::<Vec<_>>();

    let x = Tensor::stack::<2>(
        indices
            .iter()
            .map(|&index| data.clone().slice([index..index + context_length]))
            .collect::<Vec<_>>(),
        0,
    );
    let y = Tensor::stack::<2>(
        indices
            .iter()
            .map(|&index| data.clone().slice([index + 1..index + context_length + 1]))
            .collect::<Vec<_>>(),
        0,
    );
    (x, y)
}
