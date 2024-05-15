use {
    crate::{model::Model, tokenizer::Tokenizer, BOLD, RESET},
    burn::{prelude::*, tensor::activation},
    rand::{distributions::WeightedIndex, prelude::*},
};

pub fn run<B: Backend>(
    model: &Model<B>,
    tokenizer: &impl Tokenizer,
    prompt: &str,
    n_new_tokens: usize,
    context_length: usize,
    seed: u64,
) {
    let device = <B as Backend>::Device::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ids = tokenizer.encode(prompt);

    print!("{BOLD}{prompt}{RESET}");
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
