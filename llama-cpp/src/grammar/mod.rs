//! [GBNF][1] grammar parser
//!
//! > GBNF (GGML BNF) is a format for defining formal grammars to constrain
//! > model outputs in llama.cpp. For example, you can use it to force the model
//! > to generate valid JSON, or speak only in emojis.
//!
//! This module contains utilities to parse [GBNF][1] grammars and to compile
//! them to llama.cpp's binary format. The compiled grammar can then be passed
//! as parameter to the sampler.
//!
//! # Example
//!
//! ```
//! # use llama_cpp::{grammar::{Error, parse_and_compile}, backend::sampling::SamplingParameters};
//! # fn main() -> Result<(), Error> {
//! let compiled = parse_and_compile(r#"
//! root ::= "Hello World"+
//! "#)?;
//!
//! let sampling_parameters = SamplingParameters {
//!     grammar: Some(compiled),
//!     ..Default::default()
//! };
//! # Ok(())
//! # }
//! ```
//!
//! [1]: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

pub mod ast;
pub mod compiler;
mod parser;

use std::path::Path;

use self::compiler::Compiler;
use crate::backend::grammar::Compiled;

/// Grammar parser/compiler error
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("grammar as no 'root' production")]
    NoRoot,

    #[error("production '{symbol}' is undefined")]
    UndefinedSymbol { symbol: String },

    #[error("parse error:\n{0}")]
    Parse(String),

    #[error("invalid char range: {start}-{end}")]
    InvalidCharRange { start: char, end: char },

    #[error("duplicate symbol: {symbol}")]
    DuplicateSymbol { symbol: String },
}

/// Parse source to AST.
pub fn parse<'source>(input: &'source str) -> Result<ast::Grammar<'source>, Error> {
    match parser::parse_grammar_complete(input) {
        Ok((_, ast)) => Ok(ast),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
            Err(Error::Parse(nom::error::convert_error(input, e)))
        }
        _ => unreachable!(),
    }
}

/// Compile AST to binary format.
pub fn compile<'source, 'ast>(ast: &'ast ast::Grammar<'source>) -> Result<Compiled, Error> {
    let mut compiler = Compiler::default();
    compiler.push_ast(ast)?;
    compiler.finish("root".into())
}

/// Parse from text and compile to binary format.
pub fn parse_and_compile(input: &str) -> Result<Compiled, Error> {
    let ast = parse(input)?;
    compile(&ast)
}

/// Compile from source file.
pub fn compile_from_source(path: impl AsRef<Path>) -> Result<Compiled, crate::Error> {
    let source = std::fs::read_to_string(path)?;
    let ast = parse(&source)?;
    let grammar = compile(&ast)?;
    Ok(grammar)
}
