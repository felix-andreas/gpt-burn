use {
    crate::{BOLD, RESET},
    serde::{Deserialize, Serialize},
    std::{
        collections::HashMap,
        fmt::Debug,
        fs::File,
        io::{BufReader, BufWriter},
    },
};

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

pub struct CharTokenizer {
    pub ttoi: HashMap<char, usize>,
    pub itot: HashMap<usize, char>,
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl CharTokenizer {
    pub fn new() -> CharTokenizer {
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
pub struct SimpleVowelTokenizer {
    ttoi: HashMap<String, usize>,
    itot: HashMap<usize, String>,
}

impl SimpleVowelTokenizer {
    pub fn new(tokens: &[&str], vocab_size: usize) -> Self {
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

    pub fn save(&self, path: &str) {
        let mut file = BufWriter::new(File::create(path).unwrap());
        bincode::serialize_into(&mut file, &self).unwrap();
    }

    pub fn load(path: &str) -> SimpleVowelTokenizer {
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);
        bincode::deserialize_from(&mut reader).unwrap()
    }

    pub fn tokenize(text: &str) -> impl Iterator<Item = &str> {
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
