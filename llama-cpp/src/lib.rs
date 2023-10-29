#![allow(dead_code)]

pub mod backend;
pub mod grammar;
pub mod loader;
pub mod session;
mod utils;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error")]
    Io(#[from] std::io::Error),

    #[error("backend error")]
    Backend(#[from] crate::backend::Error),

    #[error("grammar error")]
    Grammar(#[from] crate::grammar::Error),
}
