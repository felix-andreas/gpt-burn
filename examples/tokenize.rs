use gpt_burn::{
    tokenizer::{CharTokenizer, SimpleVowelTokenizer, Tokenizer},
    BOLD, RESET,
};

fn main() {
    fn print_tokens(tokenizer: &impl Tokenizer, text: &str) {
        println!(
            "{BOLD}Tokens:{RESET} {:?}",
            tokenizer
                .encode(text)
                .into_iter()
                .map(|id| tokenizer.decode(&[id]))
                .collect::<Vec<_>>()
        );
        println!("{BOLD}Values:{RESET} {:?}", tokenizer.encode(text));
    }

    let text = "Albert Einstein war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft.";
    println!("{BOLD}Example text:{RESET} {text}");

    // CharTokenizer
    println!("{BOLD}CharTokenizer{RESET}",);
    {
        let tokenizer = CharTokenizer::new();
        print_tokens(&tokenizer, text);
    }

    // SimpleVowelTokenizer
    println!("{BOLD}SimpleVowelTokenizer{RESET}",);
    {
        let tokenizer = {
            let vocab_size = 99;
            let tokens = SimpleVowelTokenizer::tokenize(text).collect::<Vec<_>>();
            SimpleVowelTokenizer::new(&tokens, vocab_size)
        };
        print_tokens(&tokenizer, text);
    }
}
