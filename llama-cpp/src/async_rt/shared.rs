pub mod mpsc {
    #[derive(Copy, Clone, Debug, PartialEq, Eq, thiserror::Error)]
    pub enum TryReceiveError {
        #[error("mpsc channel empty")]
        Empty,

        #[error("mpsc channel disconnected")]
        Disconnected,
    }
}
