//! Asynchronous model loading
//!
//! See: [`ModelLoader`]

use std::path::Path;

use crate::{
    async_rt::{
        spawn_blocking,
        watch,
        JoinHandle,
    },
    backend::model::{
        Model,
        ModelParameters,
    },
    Error,
};

/// Loads a model asynchronously.
///
/// # Example
///
/// ```
/// # use llama_cpp::{loader::ModelLoader, Error};
/// # #[tokio::main]
/// # async fn main() -> Result<(), Error> {
/// # let model_path = "../data/TinyLLama-v0.gguf";
/// // load model asynchronously
/// let model = ModelLoader::load(model_path, Default::default())
///     .wait_with_progress(|progress| println!("{:.2} % loaded", progress * 100.0))
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct ModelLoader {
    progress: watch::Receiver<f32>,
    join_handle: JoinHandle<Result<Model, Error>>,
}

impl ModelLoader {
    /// Start loading the model file.
    pub fn load(path: impl AsRef<Path>, parameters: ModelParameters) -> Self {
        let (tx, rx) = watch::channel(0.0);
        let path = path.as_ref().to_owned();

        // spawn a thread to asynchronously load the model
        let join_handle = spawn_blocking(move || {
            let _guard = tracing::debug_span!("model loader");
            Model::load(path, &parameters, move |progress| {
                // we don't care about the error, since that just means the receiver has been
                // dropped.
                tx.send(progress).ok();
            })
            .map_err(Error::from)
        });

        Self {
            progress: rx,
            join_handle,
        }
    }

    /// Returns the current loading progress
    pub fn progress(&self) -> f32 {
        *self.progress.borrow()
    }

    /// Waits until the model is ready.
    pub async fn wait_for_model(self) -> Result<Model, Error> {
        self.join_handle.await
    }

    /// Wait until some progress is made and return it.
    ///
    /// This returns `None` if the model finished loading.
    pub async fn wait_for_progress(&mut self) -> Option<f32> {
        match self.progress.changed().await {
            Ok(()) => Some(*self.progress.borrow()),
            // the watch channel returns an error iff the sender has been dropped, i.e. the model
            // loading thread finished.
            Err(_) => None,
        }
    }

    /// Wait until the model is ready, and call a closure with the progress
    /// regularly.
    pub async fn wait_with_progress(mut self, mut f: impl FnMut(f32)) -> Result<Model, Error> {
        loop {
            let Some(progress) = self.wait_for_progress().await
            else {
                break;
            };
            f(progress);
        }

        self.wait_for_model().await
    }
}
