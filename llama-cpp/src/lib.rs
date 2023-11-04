#![cfg_attr(docsrs, feature(doc_cfg))]

//! Idiomatic Rust bindings for [llama.cpp][1].
//!
//! Low-level synchronous bindings can be found in the [`backend`] module.
//!
//! It's recommend to use the high-level asynchronous interfaces found in
//! [`loader`] and [`session`].
//!
//! # Async Runtime
//!
//! Both [Tokio][2] and [async-std][3] are supported. You choose which one is
//! used by enabling one of the following features:
//!
//!  - `runtime-async-std`
//!  - `runtime-tokio`
//!
//! This will automatically enable the `async` feature, which enables the
//! [`loader`] and [`session`] module.
//!
//! # Example
//!
//! ```
//! # use std::io::{stdout, Write};
//! # use llama_cpp::{loader::ModelLoader, Error, token::Tokenize, session::Session};
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
//! // create an inference session.
//! let session = Session::from_model(model, Default::default());
//!
//! // create a sequence and feed prompt to it.
//! let mut sequence = session.sequence();
//! sequence
//!     .push(Tokenize {
//!         text: prompt,
//!         add_bos: true,
//!         allow_special: false,
//!     })
//!     .await?;
//!
//! // create a response stream from it
//! let stream = inference.stream::<String>(Default::default());
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
//!
//! [1]: https://github.com/ggerganov/llama.cpp
//! [2]: https://tokio.rs/
//! [3]: https://async.rs/

use backend::context::DecodeError;

#[cfg(feature = "async")]
mod async_rt;
pub mod backend;
#[cfg_attr(docsrs, doc(cfg(feature = "grammar")))]
#[cfg(feature = "grammar")]
pub mod grammar;
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
#[cfg(feature = "async")]
pub mod loader;
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
#[cfg(feature = "async")]
pub mod session;
pub mod token;
mod utils;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error")]
    Io(#[from] std::io::Error),

    #[error("backend error")]
    Backend(#[from] crate::backend::Error),

    #[cfg(feature = "grammar")]
    #[error("grammar error")]
    Grammar(#[from] crate::grammar::Error),
}

impl From<DecodeError> for Error {
    fn from(error: DecodeError) -> Self {
        Self::Backend(error.into())
    }
}
