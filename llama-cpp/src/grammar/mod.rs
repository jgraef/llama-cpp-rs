pub mod ast;
pub mod compiler;
mod parser;

use std::path::Path;

pub use ast::Id;

use self::compiler::{
    Compiled,
    Compiler,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("grammar as no 'root' production")]
    NoRoot,

    #[error("production '{0}' is undefined")]
    Undefined(String),

    #[error("parse error:\n{0}")]
    Parse(String),

    #[error("the compiled grammar is invalid. this is most likely a bug. reason: {0}")]
    InvalidCompiled(&'static str),
}

pub fn parse<'source>(input: &'source str) -> Result<ast::Grammar<'source>, Error> {
    match parser::parse_grammar_complete(input) {
        Ok((_, ast)) => Ok(ast),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
            Err(Error::Parse(nom::error::convert_error(input, e)))
        }
        _ => unreachable!(),
    }
}

pub fn compile<'source, 'ast>(ast: &'ast ast::Grammar<'source>) -> Result<Compiled, Error> {
    let mut compiler = Compiler::default();
    compiler.push_ast(ast)?;
    compiler.finish("root".into())
}

pub fn compile_from_source(path: impl AsRef<Path>) -> Result<Compiled, crate::Error> {
    let source = std::fs::read_to_string(path)?;
    let ast = parse(&source)?;
    let grammar = compile(&ast)?;
    Ok(grammar)
}
