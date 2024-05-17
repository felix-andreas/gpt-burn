#![allow(clippy::too_many_arguments)]

pub mod model;
pub mod tokenizer;

mod run;
mod train;

pub use run::run;
pub use train::{train, TrainingConfig};

pub const BOLD: &str = "\x1b[1m";
pub const RESET: &str = "\x1b[0m";
