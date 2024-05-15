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

## Options

```
Usage: gpt-burn <COMMAND>

Commands:
  train  Train a new model
  run    Generate text using pre-trained model
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

```
Usage: gpt-burn train [OPTIONS]

Options:
  -d, --dataset-path <DATASET_PATH>
  -m, --model-path <MODEL_PATH>
  -n, --n-mega-bytes <N_MEGA_BYTES>
  -n, --n-epochs <N_EPOCHS>
  -l, --learning-rate <LEARNING_RATE>
  -c, --context-length <CONTEXT_LENGTH>
  -n, --n-layers <N_LAYERS>
  -n, --n-heads <N_HEADS>
  -d, --d-model <D_MODEL>
  -s, --save-model
```

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
