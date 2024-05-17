<div align="center">

# GPT-Burn 🔥

### Implementation of the GPT architecture in Rust 🦀 + [Burn 🔥](https://burn.dev/).

</div>

This project aims to be a clean and concise re-implementation of [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). The model implementation, contained in [`src/model.rs`](src/model.rs), is under 300 lines of code. While this was a fun exercise mostly for (my own) educational purposes, it demonstrates the utility of Rust and Burn in the machine learning domain: The entire project compiles into a single binary, making deployment relatively straightforward.

At the moment, only a character-level tokenizer is supported, so official weights requiring a BPE tokenizer cannot be used yet. However, for fun, you can try out the small toy model I trained ([see inference](#inference)). 

The project also includes a simple CLI for training and inference. 

```
Usage: gpt-burn <COMMAND>

Commands:
  run    Generate text using a pre-trained model
  train  Train a new model
```

## Installation

You can install `gpt-burn` with [Nix](https://nixos.org/):

```sh
nix run github:felix-andreas/gpt-burn
```

Or, install with `cargo`:

```
cargo install --git https://github.com/felix-andreas/gpt-burn
```

Alternatively, clone the repo and build from source:

```sh
nix develop # optional
cargo run --release
```

If you don't use [Nix](https://nixos.org/) and are on a Ubuntu-based distro, you need to install these additional dependencies:

```sh
apt install pkg-config libssl-dev libvulkan1 mesa-vulkan-drivers vulkan-tools
```

## Inference

I trained a toy model with a character-level tokenizer on the [German Wikipedia corpus](https://github.com/GermanT5/wikipedia2corpus) for 20,000 batches (batch size of 128) with the following parameters:

| Parameter      | Value  |
| -------------- | ------ |
| parameters     | 83M    |
| context length | 128    |
| `n_layers`     | 12     |
| `n_heads`      | 12     |
| `d_model`      | 768    |

You can download it [here](https://drive.usercontent.google.com/download?id=1GGLaPnmPQ8Z2B9vJQoI6-K128X9LJKG0&export=download) and extract it afterward. Or, do both in a single command:

```sh
curl -s 'https://drive.usercontent.google.com/download?id=1GGLaPnmPQ8Z2B9vJQoI6-K128X9LJKG0&export=download&confirm=t' | tar xzf -
```

Then, run the model:

```sh
gpt-burn run ./model_83M
```

You should see something along these lines:

```
So wurden bis 1977 679 nachhaltige Wörgler Torbauten vorgeworfen, die Einwohnerzahl Sirkes bestand 2015 bis 1998.
Sie war trotz weniger als 10.000 ausgedehnter Größen wahrscheinlich auf folgende Breitenauflagen mit 932 km.
2016 wurden rund 145 Händen nach Deutschland geladen.
```

Further command line options are:

```
Usage: gpt-burn run [OPTIONS] <MODEL_PATH>

Arguments:
  <MODEL_PATH>

Options:
  -p, --prompt <PROMPT>
  -n, --n-new-tokens <N_NEW_TOKENS>  [default: 1000]
  -s, --seed <SEED>                  [default: 0]
```

## Training

To train your own model, run:

```
gpt-burn train --context-length 128 --n-layers 12 --n-heads 12 --d-model 768 --batch-size 128 --learning-rate 0.0003 --seed 0 --text-corpus ./corpus.txt
```

> [!IMPORTANT]
> Make sure `corpus.txt` is a utf-8 encoded text file!

You can pass most hyperparameters as a command-line option:

```
Usage: gpt-burn train [OPTIONS]

Options:
  -o, --output-path <PATH>
  -c, --context-length <CONTEXT_LENGTH>  [default: 64]
  -d, --d-model <D_MODEL>                [default: 64]
  -l, --n-layers <N_LAYERS>              [default: 2]
  -h, --n-heads <N_HEADS>                [default: 2]
  -n, --n-steps <N_STEPS>                [default: 50]
  -b, --batch-size <BATCH_SIZE>          [default: 32]
  -r, --learning-rate <LEARNING_RATE>    [default: 0.003]
  -s, --seed <SEED>                      [default: 0]
  -t, --text-corpus <TEXT_CORPUS>        [default: .data/corpus.txt]
  -m, --n-mega-bytes <N_MEGA_BYTES>      Only use first <n> megabytes of dataset for training
  -x, --no-save                          Don't save trained model (useful for debugging)
```

## Tokenizer

The model can be used with different tokenizers via the `Tokenizer` trait. Below you see how the following sentence

```
Albert Einstein war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft.
```

is encoded by the different tokenizers.

### Character-level tokenizer

The `CharTokenizer` splits every character into a separate token:

```
Tokens: ["A", "l", "b", "e", "r", "t", " ", "E", "i", "n", "s", "t", "e", "i", "n", " ", "w", "a", "r", " ", "e", "i", "n", " ", "s", "c", "h", "w", "e", "i", "z", "e", "r", "i", "s", "c", "h", "-", "U", "S", "-", "a", "m", "e", "r", "i", "k", "a", "n", "i", "s", "c", "h", "e", "r", " ", "t", "h", "e", "o", "r", "e", "t", "i", "s", "c", "h", "e", "r", " ", "P", "h", "y", "s", "i", "k", "e", "r", " ", "d", "e", "u", "t", "s", "c", "h", "e", "r", " ", "H", "e", "r", "k", "u", "n", "f", "t", "."]
Values: [28, 13, 3, 6, 19, 21, 1, 32, 10, 15, 20, 21, 6, 10, 15, 1, 24, 2, 19, 1, 6, 10, 15, 1, 20, 4, 9, 24, 6, 10, 27, 6, 19, 10, 20, 4, 9, 66, 48, 46, 66, 2, 14, 6, 19, 10, 12, 2, 15, 10, 20, 4, 9, 6, 19, 1, 21, 9, 6, 16, 19, 6, 21, 10, 20, 4, 9, 6, 19, 1, 43, 9, 26, 20, 10, 12, 6, 19, 1, 5, 6, 22, 21, 20, 4, 9, 6, 19, 1, 35, 6, 19, 12, 22, 15, 7, 21, 67]
```

### Simple-vowel tokenizer

The `SimpleVowelTokenizer` splits words before the next vowel if the chunk is longer than three characters, creating a result that resembles syllables:

```
Tokens: ["Albert", " ", "Einst", "ein", " ", "war", " ", "ein", " ", "schw", "eizer", "isch", "-", "US", "-", "amer", "ikan", "isch", "er", " ", "theor", "etisch", "er", " ", "Phys", "iker", " ", "deutsch", "er", " ", "Herk", "unft"]
Values: [2, 0, 3, 9, 0, 19, 0, 9, 0, 16, 10, 15, 1, 6, 1, 7, 13, 15, 11, 0, 17, 12, 11, 0, 5, 14, 0, 8, 11, 0, 4, 18]
```

## References

* [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [OpenAI's GPT-2 Implementation](https://github.com/openai/gpt-2/blob/master/src/model.py)
* [Huggingface's GPT-2 Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
* [Visualization of the GPT Architecture](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer#/media/File:Full_GPT_architecture.svg)
* [Lesson by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [The GPT-3 Architecture, on a Napkin](https://dugas.ch/artificial_curiosity/GPT_architecture.html)
