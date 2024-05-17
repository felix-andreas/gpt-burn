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
        path::PathBuf,
    },
};

fn main() {
    /* Alternatively use CPU backend */
    // type B = burn::backend::ndarray::NdArray;
    type B = Wgpu;
    type AutoB = Autodiff<B>;

    match Cli::parse().command {
        Commands::Train {
            text_corpus,
            output_path,
            no_save,
            n_steps,
            batch_size,
            n_mega_bytes,
            context_length,
            d_model,
            n_layers,
            n_heads,
            learning_rate,
            seed,
            ..
        } => {
            // cli option defaults
            let save = !no_save;

            // load text corpus and tokenizer
            println!(
                "{BOLD}load {} file {text_corpus:?}{RESET} as dataset",
                n_mega_bytes.map_or_else(
                    || "entire".to_string(),
                    |n_mega_bytes| format!("first {n_mega_bytes} MiB of")
                )
            );
            let (data_train, data_test, tokenizer) = {
                let device = <Autodiff<B> as Backend>::Device::default();

                let mut file = File::open(text_corpus)
                    .unwrap()
                    .take(n_mega_bytes.unwrap_or(999) << 20);
                let mut text = String::new();
                file.read_to_string(&mut text).unwrap();

                let tokenizer = CharTokenizer::new();
                let text = text
                    .chars()
                    .filter(|char| tokenizer.ttoi.contains_key(char))
                    .collect::<String>();

                /* Uncomment to use `SimpleVowelTokenizer` */
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
                n_steps,
                batch_size,
                batches_per_step: 100,
                validation_size: 128,
                seed,
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
            let model = gpt_burn::train(&config, data_train, data_test, save);

            // save trained model
            if save {
                let output_path = output_path.unwrap_or_else(|| {
                    format!(
                        ".data/gpt_{}k_{}context_{}",
                        model.num_params() >> 10,
                        config.model.context_length,
                        std::time::UNIX_EPOCH.elapsed().unwrap().as_secs()
                    )
                    .into()
                });

                println!("{BOLD}store trained model to: {output_path:?}{RESET}");
                fs::remove_dir_all(&output_path).ok();
                fs::create_dir_all(&output_path).ok();

                /* Uncomment to use `SimpleVowelTokenizer` */
                // tokenizer.save(&format!("{model_path}/tokenizer.bin"));

                config.save(output_path.join("config.json")).unwrap();
                model
                    .clone()
                    .save_file(output_path.join("model"), &CompactRecorder::new())
                    .unwrap();
            }

            // inference
            println!("{BOLD}generate example text{RESET}");
            gpt_burn::run(
                &model,
                &tokenizer,
                "\n",
                200,
                config.model.context_length,
                seed,
            );
        }
        Commands::Run {
            model_path: path,
            prompt,
            n_new_tokens,
            seed,
            ..
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
                &prompt.unwrap_or("\n".into()),
                n_new_tokens,
                config.model.context_length,
                seed,
            );
        }
    }
}

#[derive(Parser)]
#[clap(disable_help_flag = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    #[clap(long, action = clap::ArgAction::HelpLong)]
    help: Option<bool>,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model
    Train {
        #[arg(short = 'o', long, value_name = "PATH")]
        output_path: Option<PathBuf>,
        #[arg(short = 'c', long, default_value_t = 64)]
        context_length: usize,
        #[arg(short = 'd', long, default_value_t = 64)]
        d_model: usize,
        #[arg(short = 'l', long, default_value_t = 2)]
        n_layers: usize,
        #[arg(short = 'h', long, default_value_t = 2)]
        n_heads: usize,
        #[arg(short = 'n', long, default_value_t = 50)]
        n_steps: usize,
        #[arg(short = 'b', long, default_value_t = 32)]
        batch_size: usize,
        #[arg(short = 'r', long, default_value_t = 0.003)]
        learning_rate: f64,
        #[arg(short = 's', long, default_value_t = 0)]
        seed: u64,
        #[arg(short = 't', long, default_value = ".data/corpus.txt")]
        text_corpus: PathBuf,
        /// Only use first <n> megabytes of dataset for training
        #[arg(short = 'm', long)]
        n_mega_bytes: Option<u64>,
        /// Don't save trained model (useful for debugging)
        #[arg(short = 'x', long)]
        no_save: bool,
        #[arg(long , action = clap::ArgAction::HelpLong)]
        help: Option<bool>,
    },
    /// Generate text using a pre-trained model
    Run {
        model_path: String,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(short, long, default_value_t = 1000)]
        n_new_tokens: usize,
        #[arg(short, long, default_value_t = 0)]
        seed: u64,
        #[arg(long , action = clap::ArgAction::HelpLong)]
        help: Option<bool>,
    },
}
