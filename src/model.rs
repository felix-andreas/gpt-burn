#![allow(clippy::let_and_return)]
use {
    burn::{
        module::Module,
        nn::{
            loss::CrossEntropyLossConfig, DropoutConfig, Embedding, EmbeddingConfig, Gelu,
            LayerNorm, LayerNormConfig, Linear, LinearConfig,
        },
        prelude::*,
        tensor::activation,
    },
    std::fmt::Debug,
};

// Model

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    dropout: nn::Dropout,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    linear: Linear<B>,
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    pub context_length: usize,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    #[config(default = "0.2")]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            positional_embedding: EmbeddingConfig::new(self.context_length, self.d_model)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            blocks: (0..self.n_layers)
                .map(|_| BlockConfig::new(self.d_model, self.d_hidden, self.n_heads).init(device))
                .collect(),
            norm: LayerNormConfig::new(self.d_model).init(device),
            linear: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = input.clone();

        let x = {
            let emb_tok = self.token_embedding.forward(x.clone());
            let emb_pos = {
                let [_, t] = input.dims();
                self.positional_embedding
                    .forward(Tensor::arange(0..(t as i64), &x.device()).unsqueeze())
            };
            emb_tok + emb_pos
        };
        let x = self.dropout.forward(x);
        let x = self.blocks.iter().fold(x, |x, block| block.forward(x));
        let x = self.norm.forward(x);
        let x = self.linear.forward(x);

        x
    }

    pub fn loss(&self, logits: Tensor<B, 3>, y: Tensor<B, 2, Int>) -> Tensor<B, 1> {
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
            norm_1: LayerNormConfig::new(self.d_model).init(device),
            multi_head: MultiHeadAttentionConfig::new(self.d_model, self.n_heads).init(device),
            norm_2: LayerNormConfig::new(self.d_model).init(device),
            pwff: PositionWiseFeedForwardConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Debug, Module)]
struct Block<B: Backend> {
    norm_1: LayerNorm<B>,
    multi_head: MultiHeadAttention<B>,
    norm_2: LayerNorm<B>,
    pwff: PositionWiseFeedForward<B>,
}

impl<B: Backend> Block<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = input.clone();

        let x = x.clone() + self.multi_head.forward(self.norm_1.forward(x));
        let x = x.clone() + self.pwff.forward(self.norm_2.forward(x));

        x
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
            attn_dropout: nn::DropoutConfig::new(self.dropout).init(),
            resid_dropout: nn::DropoutConfig::new(self.dropout).init(),
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
    attn_dropout: nn::Dropout,
    out: nn::Linear<B>,
    resid_dropout: nn::Dropout,
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
        let x = self.attn_dropout.forward(x);
        let x = x.matmul(v);
        let x = x.swap_dims(1, 2).reshape([b, t, self.n_heads * self.d_k]);
        let x = self.resid_dropout.forward(x);

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
            linear_1: LinearConfig::new(self.d_model, self.d_hidden).init(device),
            gelu: Gelu::new(),
            linear_2: LinearConfig::new(self.d_hidden, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_1: nn::Linear<B>,
    gelu: nn::Gelu,
    linear_2: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_1.forward(input);
        let x = self.gelu.forward(x);
        let x = self.linear_2.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}
