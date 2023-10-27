#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend error")]
    Backend(#[from] crate::backend::Error),
}
