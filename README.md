<div align="center">

# GPT-Burn 🔥

### Implementation of the GPT architecture in [Burn](https://burn.dev/) 🦀.

</div>

## Usage

You can install `gpt-burn` with Nix:

```
nix run github:felix-andreas/gpt-burn
```

## Training

download from https://github.com/GermanT5/wikipedia2corpus

TODO

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

The `SimpleVowelTokenizer` a

```
Tokens: ["Albert", " ", "Einst", "ein", " ", "war", " ", "ein", " ", "schw", "eizer", "isch", "-", "US", "-", "amer", "ikan", "isch", "er", " ", "theor", "etisch", "er", " ", "Phys", "iker", " ", "deutsch", "er", " ", "Herk", "unft"]
Values: [2, 0, 3, 9, 0, 19, 0, 9, 0, 16, 10, 15, 1, 6, 1, 7, 13, 15, 11, 0, 17, 12, 11, 0, 5, 14, 0, 8, 11, 0, 4, 18]
```

## CLI options

The `gpt-burn` command has multiple subcommmands:

```
Usage: gpt-burn <COMMAND>

Commands:
  train  Train a new model
  run    Generate text using pre-trained model
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

For training, you can pass most hyperparameters as a command-line option:

```
Usage: gpt-burn train [OPTIONS]

Options:
  -o, --output-path <PATH>
  -t, --text-corpus <PATH>
  -m, --mega-bytes <MEGA_BYTES>
  -e, --epochs <EPOCHS>
  -b, --batch-size <BATCH_SIZE>
  -r, --learning-rate <LEARNING_RATE>
  -c, --context-length <CONTEXT_LENGTH>
  -l, --layers <LAYERS>
  -n, --n-heads <N_HEADS>
  -d, --d-model <D_MODEL>
  -s, --save-model
```

For inference, you can pass a model path and the number of new tokens that should be generated:

```
Usage: gpt-burn run [OPTIONS] <MODEL_PATH>

Arguments:
  <MODEL_PATH>

Options:
  -p, --prompt <PROMPT>
  -n, --n-new-tokens <N_NEW_TOKENS>
```

## References

* [Lesson by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [The GPT-3 Architecture, on a Napkin](https://dugas.ch/artificial_curiosity/GPT_architecture.html)
* [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
