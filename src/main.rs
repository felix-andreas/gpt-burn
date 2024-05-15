#![allow(clippy::single_range_in_vec_init, clippy::too_many_arguments)]

use {
    burn::{
        backend::{Autodiff, Wgpu},
        module::Module,
        nn::{
            loss::CrossEntropyLossConfig, DropoutConfig, Embedding, EmbeddingConfig, Gelu,
            LayerNorm, LayerNormConfig, Linear, LinearConfig,
        },
        optim::{AdamWConfig, GradientsParams, Optimizer},
        prelude::*,
        record::{CompactRecorder, Recorder},
        tensor::{activation, backend::AutodiffBackend},
    },
    clap::{Parser, Subcommand},
    rand::{distributions::WeightedIndex, prelude::*},
    serde::{Deserialize, Serialize},
    std::{
        collections::HashMap,
        fmt::Debug,
        fs::{self, File},
        io::{BufReader, BufWriter, Read},
        time::{Duration, Instant},
    },
};

// region: Main

fn main() {
    type B = Wgpu;
    // type B = burn::backend::ndarray::NdArray;
    type AutoB = Autodiff<B>;

    match Cli::parse().command {
        Commands::Train {
            dataset_path: data_path,
            model_path,
            n_mega_bytes,
            context_length,
            vocab_size,
            save,
        } => {
            // cli argument defaults
            let data_path = data_path.as_deref().unwrap_or(".data/corpus.txt");
            // hyperparameters
            let n_bytes = n_mega_bytes.unwrap_or(10) << 20;
            let vocab_size = vocab_size.unwrap_or(512);
            let context_length = context_length.unwrap_or(2);

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

                let tokenizer = {
                    let tokens = SimpleVowelTokenizer::tokenize(&text).collect::<Vec<_>>();
                    SimpleVowelTokenizer::new(&tokens, vocab_size)
                };
                // let tokenizer = CharTokenizer::new();
                // let text = text
                //     .chars()
                //     .filter(|char| tokenizer.ttoi.contains_key(char))
                //     .collect::<String>();

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
                let data_test = Tensor::<AutoB, 1, Int>::from_data(
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
                n_epochs: 2,
                epoch_size: 100,
                batch_size: 64,
                eval_size: 128,
                seed: 0,
                learning_rate: 3e-4,
                model: ModelConfig {
                    context_length: 128,
                    vocab_size: tokenizer.vocab_size(),
                    d_model: 32,
                    d_hidden: 4 * 32,
                    n_heads: 2,
                    n_layer: 2,
                    dropout: 0.2,
                },
                optimizer: AdamWConfig::new(),
            };
            let model = train(&config, data_train, data_test);

            // store trained model
            if save {
                let model_path = model_path.unwrap_or_else(|| {
                    format!(
                        ".data/gpt_{}k_{}context_{}",
                        model.num_params() >> 10,
                        context_length,
                        std::time::UNIX_EPOCH.elapsed().unwrap().as_secs()
                    )
                });

                println!("{BOLD}store trained model to: {model_path}{RESET}");
                fs::remove_dir_all(&model_path).ok();
                fs::create_dir_all(&model_path).ok();
                tokenizer.save(&format!("{model_path}/tokenizer.bin"));
                config.save(format!("{model_path}/config.json")).unwrap();
                model
                    .clone()
                    .save_file(format!("{model_path}/model"), &CompactRecorder::new())
                    .unwrap();
            }

            // inference
            run(
                &model,
                &tokenizer,
                "Hallo Welt, wie geht es dir heute?",
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

            let tokenizer = SimpleVowelTokenizer::load(&format!("{path}/tokenizer.bin"));
            // let tokenizer = CharTokenizer::new();

            let config = TrainingConfig::load(format!("{path}/config.json")).unwrap();
            let record = CompactRecorder::new()
                .load(format!("{path}/model").into(), &device)
                .unwrap();
            let model: Model<B> = config.model.init(&device).load_record(record);

            run(
                &model,
                &tokenizer,
                &prompt.unwrap_or("Albert Einstein war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft.".into()),
                n_new_tokens.unwrap_or(1000),
                config.model.context_length,
            );
        }
    }
}

// endregion

// region: CLI

const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

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
        context_length: Option<usize>,
        #[arg(short, long)]
        vocab_size: Option<usize>,
        #[arg(short, long)]
        save: bool,
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

// endregion

// region: Model

#[derive(Debug, Module)]
struct Model<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    linear: Linear<B>,
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    context_length: usize,
    vocab_size: usize,
    n_layer: usize,
    n_heads: usize,
    d_model: usize,
    d_hidden: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl ModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            positional_embedding: EmbeddingConfig::new(self.context_length, self.d_model)
                .init(device),
            linear: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            blocks: (0..self.n_layer)
                .map(|_| BlockConfig::new(self.d_model, self.d_hidden, self.n_heads).init(device))
                .collect(),
            norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, t] = input.dims();

        let x = input.clone();

        let x = {
            let emb_tok = self.token_embedding.forward(x);

            let emb_pos = self
                .positional_embedding
                .forward(Tensor::arange(0..(t as i64), &input.device()).unsqueeze());
            emb_tok + emb_pos
        };

        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x = self.norm.forward(x);
        x = self.linear.forward(x);

        x
    }

    fn loss(&self, logits: Tensor<B, 3>, y: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, t, c] = logits.dims();
        CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.reshape([b * t, c]), y.reshape([b * t]))
    }
}

// Block

#[derive(Debug, Config)]
pub struct BlockConfig {
    d_model: usize,
    d_hidden: usize,
    n_heads: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl BlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            multi_head: MultiHeadAttentionConfig::new(self.d_model, self.n_heads).init(device),
            pwff: PositionWiseFeedForwardConfig::new(self.d_model, self.d_hidden).init(device),
            norm_1: LayerNormConfig::new(self.d_model).init(device),
            norm_2: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

#[derive(Debug, Module)]
struct Block<B: Backend> {
    multi_head: MultiHeadAttention<B>,
    pwff: PositionWiseFeedForward<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
}

impl<B: Backend> Block<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = input.clone();
        let x = self.norm_1.forward(x);
        let x = x.clone() + self.multi_head.forward(x);
        let x = self.norm_2.forward(x);

        x.clone() + self.pwff.forward(x)
    }
}

// Multi-Head Attention

#[derive(Debug, Config)]
struct MultiHeadAttentionConfig {
    d_model: usize,
    n_heads: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl MultiHeadAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let d_k = self.d_model / self.n_heads;
        assert!(d_k * self.n_heads == self.d_model);
        MultiHeadAttention {
            n_heads: self.n_heads,
            d_k,
            query: nn::LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            key: nn::LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            value: nn::LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            out: nn::LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            activation: nn::Gelu::new(),
        }
    }
}

#[derive(Debug, Module)]
struct MultiHeadAttention<B: Backend> {
    n_heads: usize,
    d_k: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
    dropout: nn::Dropout,
    activation: nn::Gelu,
}

impl<B: Backend> MultiHeadAttention<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = input.dims();

        let q = self.query.forward(input.clone());
        let k = self.key.forward(input.clone());
        let v = self.value.forward(input.clone());

        let q = q.reshape([b, t, self.n_heads, self.d_k]).swap_dims(1, 2);
        let k = k.reshape([b, t, self.n_heads, self.d_k]).swap_dims(1, 2);
        let v = v.reshape([b, t, self.n_heads, self.d_k]).swap_dims(1, 2);

        let x = q.matmul(k.transpose()).div_scalar((self.d_k as f32).sqrt());
        let x = {
            let mask = Tensor::<B, 2, Bool>::tril_mask([t, t], 0, &input.device());
            x.mask_fill(mask.unsqueeze(), f32::NEG_INFINITY) // if NaN try 1e-4
        };
        let x = activation::softmax(x, 3);
        let x = self.dropout.forward(x);
        let x = x.matmul(v);

        let x = x.swap_dims(1, 2).reshape([b, t, self.n_heads * self.d_k]);

        self.out.forward(x)
    }
}

// Position-wise Feed-Forward Network

#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    pub d_model: usize,
    pub d_hidden: usize,
    #[config(default = 0.2)]
    pub dropout: f64,
}

impl PositionWiseFeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: LinearConfig::new(self.d_model, self.d_hidden).init(device),
            linear_outer: LinearConfig::new(self.d_hidden, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            gelu: Gelu::new(),
        }
    }
}

#[derive(Debug, Module)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: nn::Linear<B>,
    linear_outer: nn::Linear<B>,
    dropout: nn::Dropout,
    gelu: nn::Gelu,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}

// endregion

// region: Train

#[derive(Config)]
struct TrainingConfig {
    n_epochs: usize,
    epoch_size: usize,
    batch_size: usize,
    eval_size: usize,
    seed: u64,
    learning_rate: f64,
    model: ModelConfig,
    optimizer: AdamWConfig,
}

fn train<B: AutodiffBackend>(
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

// endregion

// region: Inference

fn run<B: Backend>(
    model: &Model<B>,
    tokenizer: &impl Tokenizer,
    prompt: &str,
    n_new_tokens: usize,
    context_length: usize,
) {
    println!("{BOLD}run{RESET}");
    let device = <B as Backend>::Device::default();
    // TODO: add seed
    let mut rng = rand::thread_rng();
    // TODO: inference only!
    let mut ids = tokenizer.encode(prompt);

    for _ in 0..n_new_tokens {
        let x = {
            let ids_sliced = &ids[(ids.len() as isize - context_length as isize).max(0) as usize..];
            Tensor::<B, 2, Int>::from_data(
                Data::new(
                    ids_sliced.iter().map(|&x| x as i32).collect(),
                    Shape::new([1, ids_sliced.len()]),
                )
                .convert(),
                &device,
            )
        };
        let logits = model.forward(x);
        let n = logits.dims()[1];
        let slice = logits.slice([(0..1), (n - 1..n)]).flatten::<1>(0, 2);
        let probs = activation::softmax(slice, 0)
            .into_data()
            .convert::<f32>()
            .value;
        // don't generate <?> special token
        let distribution = WeightedIndex::new(&probs[..probs.len() - 1]).unwrap();
        let prediction = distribution.sample(&mut rng) as usize;
        ids.push(prediction);
        print!("{}", tokenizer.decode(&[prediction]));
    }
    println!()
}

// endregion

// region: Tokenizer

trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

struct CharTokenizer {
    ttoi: HashMap<char, usize>,
    itot: HashMap<usize, char>,
}

#[allow(unused)]
impl CharTokenizer {
    fn new() -> CharTokenizer {
        const CHARS : &str =  "\n abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789äüöÄÖÜß";
        let itot = HashMap::from_iter(CHARS.chars().enumerate());
        let ttoi = HashMap::from_iter(CHARS.chars().enumerate().map(|(token, id)| (id, token)));
        CharTokenizer { ttoi, itot }
    }
}

impl Tokenizer for CharTokenizer {
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|id| self.itot[id]).collect()
    }
    fn encode(&self, text: &str) -> Vec<usize> {
        let n = self.ttoi.len();
        text.chars()
            .map(|char| self.ttoi.get(&char).copied().unwrap_or(n))
            .collect()
    }
    fn vocab_size(&self) -> usize {
        self.ttoi.len() + 1
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SimpleVowelTokenizer {
    ttoi: HashMap<String, usize>,
    itot: HashMap<usize, String>,
}

impl SimpleVowelTokenizer {
    fn new(tokens: &[&str], vocab_size: usize) -> Self {
        println!("{BOLD}build new vocab ...{RESET}");

        const CATCH_ALL_TOKEN: &str = "<?>";
        let mut frequencies = tokens
            .iter()
            .fold(HashMap::new(), |mut map, &token| {
                map.entry(token).and_modify(|freq| *freq += 1).or_insert(1);
                map
            })
            .into_iter()
            .collect::<Vec<_>>();
        frequencies.sort_by_key(|x| x.1);
        frequencies.reverse();
        frequencies.truncate(vocab_size - 1);

        let mut vocab = frequencies.into_iter().map(|x| x.0).collect::<Vec<_>>();
        vocab.sort();
        assert!(!vocab.contains(&CATCH_ALL_TOKEN));
        vocab.push(CATCH_ALL_TOKEN);
        println!("vocab ({}): {:?}", vocab.len(), &vocab);

        let itot = HashMap::<usize, String>::from_iter(
            vocab
                .into_iter()
                .enumerate()
                .map(|(i, x)| (i, x.to_string())),
        );
        let ttoi = HashMap::<String, usize>::from_iter(itot.iter().map(|(&i, t)| (t.clone(), i)));

        // check if vocab is reasonable
        let mut contains = 0;
        tokens.iter().for_each(|&token| {
            contains += ttoi.contains_key(token) as usize;
        });
        println!(
            "share of tokens contained by vocab: {:.3}",
            contains as f32 / tokens.len() as f32
        );

        SimpleVowelTokenizer { ttoi, itot }
    }

    fn save(&self, path: &str) {
        let mut file = BufWriter::new(File::create(path).unwrap());
        bincode::serialize_into(&mut file, &self).unwrap();
    }

    fn load(path: &str) -> SimpleVowelTokenizer {
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);
        bincode::deserialize_from(&mut reader).unwrap()
    }

    fn tokenize(text: &str) -> impl Iterator<Item = &str> {
        let mut token_start = 0;
        let mut prev_char = 'x'; // dummy value
        text.char_indices().filter_map(move |(index, char)| {
            let result = if char.is_whitespace()
                || char.is_ascii_punctuation()
                || prev_char.is_whitespace()
                || prev_char.is_ascii_punctuation()
                || ((['a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü'].contains(&char))
                    && index - token_start > 3)
            {
                let start_index = token_start;
                token_start = index;
                Some(&text[start_index..index])
            } else if index == text.len() - 1 {
                Some(&text[token_start..index + 1])
            } else {
                None
            };
            prev_char = char;
            result
        })
    }
}

impl Tokenizer for SimpleVowelTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let n = self.ttoi.len();
        SimpleVowelTokenizer::tokenize(text)
            .map(|token| self.ttoi.get(token).copied().unwrap_or(n - 1))
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|token| self.itot.get(token).unwrap().clone())
            .collect::<String>()
    }

    fn vocab_size(&self) -> usize {
        self.ttoi.len()
    }
}

// endregion
