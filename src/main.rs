#![allow(clippy::too_many_arguments)]

use {
    burn::{
        backend::{Autodiff, Wgpu},
        module::Module,
        optim::AdamWConfig,
        prelude::*,
        record::{CompactRecorder, Recorder},
    },
    clap::{Parser, Subcommand},
    gpt_burn::{
        model::{Model, ModelConfig},
        tokenizer::{CharTokenizer, Tokenizer},
        TrainingConfig, BOLD, RESET,
    },
    std::{
        fs::{self, File},
        io::Read,
    },
};

// region: Main

fn main() {
    type B = Wgpu;
    /* Alternatively use CPU backend */
    // type B = burn::backend::ndarray::NdArray;
    type AutoB = Autodiff<B>;

    match Cli::parse().command {
        Commands::Train {
            dataset_path: data_path,
            model_path,
            batch_size,
            n_epochs,
            n_mega_bytes,
            context_length,
            n_layers,
            n_heads,
            d_model,
            learning_rate,
            save_model,
        } => {
            // cli option defaults
            let data_path = data_path.as_deref().unwrap_or(".data/corpus.txt");
            let n_bytes = n_mega_bytes.unwrap_or(10) << 20;
            // hyperparameters
            let n_epochs = n_epochs.unwrap_or(50);
            let batch_size = batch_size.unwrap_or(64);
            let learning_rate = learning_rate.unwrap_or(3e-4);
            let context_length = context_length.unwrap_or(128);
            let n_layers = n_layers.unwrap_or(6);
            let n_heads = n_heads.unwrap_or(6);
            let d_model = d_model.unwrap_or(384);

            // load data
            println!(
                "{BOLD}load {} MiB data from: {data_path}{RESET}",
                n_bytes >> 20
            );
            let (data_train, data_test, tokenizer) = {
                let device = <Autodiff<B> as Backend>::Device::default();

                let mut file = File::open(data_path).unwrap().take(n_bytes);
                let mut text = String::new();
                file.read_to_string(&mut text).unwrap();

                let tokenizer = CharTokenizer::new();
                let text = text
                    .chars()
                    .filter(|char| tokenizer.ttoi.contains_key(char))
                    .collect::<String>();

                /* Alternatively use `SimpleVowelTokenizer` */
                // let tokenizer = {
                //     let tokens = SimpleVowelTokenizer::tokenize(&text).collect::<Vec<_>>();
                //     SimpleVowelTokenizer::new(&tokens, vocab_size)
                // };

                let mut train = tokenizer.encode(&text);
                let test = train.split_off((0.9 * train.len() as f32) as usize);

                let n_train = train.len();
                let data_train = Tensor::<AutoB, 1, Int>::from_data(
                    Data::new(
                        train.into_iter().map(|e| e as i32).collect(),
                        Shape::new([n_train]),
                    )
                    .convert(),
                    &device,
                );

                let n_test = test.len();
                let data_test = Tensor::<B, 1, Int>::from_data(
                    Data::new(
                        test.into_iter().map(|e| e as i32).collect(),
                        Shape::new([n_test]),
                    )
                    .convert(),
                    &device,
                );
                (data_train, data_test, tokenizer)
            };

            // train
            let config = TrainingConfig {
                n_epochs,
                batch_size,
                epoch_size: 100,
                validation_size: 128,
                seed: 0,
                learning_rate,
                model: ModelConfig {
                    context_length,
                    vocab_size: tokenizer.vocab_size(),
                    d_model,
                    d_hidden: 4 * d_model,
                    n_heads,
                    n_layers,
                    dropout: 0.2,
                },
                optimizer: AdamWConfig::new(),
            };
            let model = gpt_burn::train(&config, data_train, data_test, save_model);

            // store trained model
            if save_model {
                let model_path = model_path.unwrap_or_else(|| {
                    format!(
                        ".data/gpt_{}k_{}context_{}",
                        model.num_params() >> 10,
                        config.model.context_length,
                        std::time::UNIX_EPOCH.elapsed().unwrap().as_secs()
                    )
                });

                println!("{BOLD}store trained model to: {model_path}{RESET}");
                fs::remove_dir_all(&model_path).ok();
                fs::create_dir_all(&model_path).ok();

                /* Uncomment to use `SimpleVowelTokenizer` */
                // tokenizer.save(&format!("{model_path}/tokenizer.bin"));

                config.save(format!("{model_path}/config.json")).unwrap();
                model
                    .clone()
                    .save_file(format!("{model_path}/model"), &CompactRecorder::new())
                    .unwrap();
            }

            // inference
            gpt_burn::run(
                &model,
                &tokenizer,
                EXAMPLE,
                2000,
                config.model.context_length,
            );
        }
        Commands::Run {
            model_path: path,
            prompt,
            n_new_tokens,
        } => {
            let device = <B as Backend>::Device::default();

            let tokenizer = CharTokenizer::new();

            /* Alternatively use `SimpleVowelTokenizer` */
            // let tokenizer = SimpleVowelTokenizer::load(&format!("{path}/tokenizer.bin"));

            let config = TrainingConfig::load(format!("{path}/config.json")).unwrap();
            let record = CompactRecorder::new()
                .load(format!("{path}/model").into(), &device)
                .unwrap();
            let model: Model<B> = config.model.init(&device).load_record(record);

            gpt_burn::run(
                &model,
                &tokenizer,
                &prompt.unwrap_or(EXAMPLE.into()),
                n_new_tokens.unwrap_or(1000),
                config.model.context_length,
            );
        }
    }
}

const EXAMPLE: &str = "Albert Einstein war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft.";

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model
    Train {
        #[arg(short, long)]
        dataset_path: Option<String>,
        #[arg(short, long)]
        model_path: Option<String>,
        #[arg(short, long)]
        n_mega_bytes: Option<u64>,
        #[arg(short, long)]
        n_epochs: Option<usize>,
        #[arg(short, long)]
        batch_size: Option<usize>,
        #[arg(short, long)]
        learning_rate: Option<f64>,
        #[arg(short, long)]
        context_length: Option<usize>,
        #[arg(short, long)]
        n_layers: Option<usize>,
        #[arg(short, long)]
        n_heads: Option<usize>,
        #[arg(short, long)]
        d_model: Option<usize>,
        #[arg(short, long)]
        save_model: bool,
    },
    /// Generate text using pre-trained model
    Run {
        model_path: String,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(short, long)]
        n_new_tokens: Option<usize>,
    },
}
