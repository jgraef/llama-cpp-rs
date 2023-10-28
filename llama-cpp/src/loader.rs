use std::path::Path;

use tokio::{
    sync::watch,
    task::JoinHandle,
};

use crate::{
    backend::model::{
        Model,
        ModelParameters,
    },
    error::Error,
};

pub struct ModelLoader {
    progress: watch::Receiver<f32>,
    join_handle: JoinHandle<Result<Model, Error>>,
}

impl ModelLoader {
    pub fn load(path: impl AsRef<Path>, parameters: ModelParameters) -> Self {
        let (tx, rx) = watch::channel(0.0);
        let path = path.as_ref().to_owned();

        // spawn a thread to asynchronously load the model
        let join_handle = tokio::task::spawn_blocking(move || {
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

    pub fn progress(&self) -> f32 {
        *self.progress.borrow()
    }

    pub async fn wait_for_model(self) -> Result<Model, Error> {
        self.join_handle
            .await
            .expect("model loading thread panicked")
    }

    pub async fn wait_for_progress(&mut self) -> Option<f32> {
        match self.progress.changed().await {
            Ok(()) => Some(*self.progress.borrow_and_update()),
            // the watch channel returns an error iff the sender has been dropped, i.e. the model
            // loading thread finished.
            Err(_) => None,
        }
    }

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
