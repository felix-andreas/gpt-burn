use gpt_burn::{
    tokenizer::{CharTokenizer, SimpleVowelTokenizer, Tokenizer},
    BOLD, EXAMPLE_TEXT, RESET,
};

fn main() {
    fn print_tokens(tokenizer: &impl Tokenizer) {
        println!(
            "{BOLD}Tokens:{RESET} {:?}",
            tokenizer
                .encode(EXAMPLE_TEXT)
                .into_iter()
                .map(|id| tokenizer.decode(&[id]))
                .collect::<Vec<_>>()
        );
        println!("{BOLD}Values:{RESET} {:?}", tokenizer.encode(EXAMPLE_TEXT));
    }

    println!("{BOLD}Example text:{RESET} {EXAMPLE_TEXT}");

    // CharTokenizer
    println!("{BOLD}CharTokenizer{RESET}",);
    {
        let tokenizer = CharTokenizer::new();
        print_tokens(&tokenizer);
    }

    // SimpleVowelTokenizer
    println!("{BOLD}SimpleVowelTokenizer{RESET}",);
    {
        let tokenizer = {
            let vocab_size = 99;
            let tokens = SimpleVowelTokenizer::tokenize(&EXAMPLE_TEXT).collect::<Vec<_>>();
            SimpleVowelTokenizer::new(&tokens, vocab_size)
        };
        print_tokens(&tokenizer);
    }
}
