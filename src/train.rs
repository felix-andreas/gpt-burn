use {
    crate::{
        model::{Model, ModelConfig},
        BOLD, RESET,
    },
    burn::{
        module::Module,
        optim::{AdamWConfig, GradientsParams, Optimizer},
        prelude::*,
        tensor::backend::AutodiffBackend,
    },
    rand::prelude::*,
    std::time::{Duration, Instant},
};

#[derive(Config)]
pub struct TrainingConfig {
    pub n_epochs: usize,
    pub epoch_size: usize,
    pub batch_size: usize,
    pub eval_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub model: ModelConfig,
    pub optimizer: AdamWConfig,
}

pub fn train<B: AutodiffBackend>(
    config: &TrainingConfig,
    data_train: Tensor<B, 1, Int>,
    data_val: Tensor<B, 1, Int>,
) -> Model<B> {
    let mut rng = rand::thread_rng(); // TODO: add seed
    let device = data_train.device();

    B::seed(config.seed);

    let mut model: Model<B> = config.model.init(&device);
    println!(
        "{BOLD}new model{RESET} - parameters: {}, {:?}",
        model.num_params(),
        config.model
    );
    let mut optimizer = config.optimizer.init::<B, Model<B>>();

    for epoch in 0..config.n_epochs {
        // evaluate validation loss
        // TODO: only inference
        let loss_val = {
            let (x, y) = get_batch(
                &mut rng,
                data_val.clone(),
                config.eval_size,
                config.model.context_length,
            );
            let logits = model.forward(x);
            let loss = model.loss(logits, y.clone());
            loss.into_scalar().elem::<f64>()
        };
        let now = Instant::now();
        let mut loss_sum = 0.0;
        for _ in 0..config.epoch_size {
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

        let elapsed = now.elapsed();
        let format_duration = |duration: Duration| {
            let seconds = duration.as_secs() % 60;
            let minutes = (duration.as_secs() / 60) % 60;
            let hours = (duration.as_secs() / 60) / 60;
            format!("{hours:02}:{minutes:02}:{seconds:02}")
        };
        println!(
            "epoch {epoch:03}/{:03}: train loss: {:.4}, val loss: {:.4} ({})",
            config.n_epochs,
            loss_sum / config.epoch_size as f32,
            loss_val,
            format_duration(elapsed)
        );
    }

    model
}

#[allow(clippy::single_range_in_vec_init)]
fn get_batch<B: Backend>(
    rng: &mut ThreadRng,
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
