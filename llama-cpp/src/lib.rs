//! Idiomatic Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp).
//!
//! Low-level synchronous bindings can be found in the [`backend`] module.
//!
//! It's recommend to use the high-level asynchronous interfaces found in
//! [`loader`] and [`session`].
//!
//! Todo: support tokio and async-std
//!
//! # Example
//!
//! ```
//! # use std::io::{stdout, Write};
//! # use llama_cpp::{loader::ModelLoader, session::Session, Error};
//! # use futures::{stream::TryStreamExt, pin_mut};
//! # #[tokio::main]
//! # async fn main() -> Result<(), Error> {
//! # let model_path = "../data/TinyLLama-v0.gguf";
//! // load model asynchronously
//! let model = ModelLoader::load(model_path, Default::default())
//!     .wait_for_model()
//!     .await?;
//!
//! // prompt
//! let prompt = "The capital of France is";
//! print!("{}", prompt);
//! stdout().flush()?;
//!
//! // create a session and feed prompt to it
//! let mut session = Session::new(model, Default::default());
//! session.push_text(&prompt, true, false);
//!
//! // create a response stream from it
//! let stream = session.pieces(None, [], false);
//! pin_mut!(stream);
//!
//! // stream LLM output piece by piece
//! while let Some(piece) = stream.try_next().await? {
//!     print!("{piece}");
//!     stdout().flush()?;
//! }
//! # Ok(())
//! # }
//! ```

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
