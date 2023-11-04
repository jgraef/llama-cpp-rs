use std::{
    pin::Pin,
    task::{
        Context,
        Poll,
    },
};

use async_std::task::block_on;
use futures::{
    pin_mut,
    Future,
    Stream,
};

pub fn spawn_blocking<F, R>(f: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    JoinHandle(async_std::task::spawn_blocking(f))
}

pub struct JoinHandle<R>(async_std::task::JoinHandle<R>);

impl<R> Future for JoinHandle<R> {
    type Output = R;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let inner = &mut self.0;
        pin_mut!(inner);
        inner.poll(cx)
    }
}

pub fn run_test<R>(future: impl Future<Output = R>) -> R {
    block_on(future)
}

pub mod oneshot {
    use super::*;

    #[derive(Debug)]
    pub struct Sender<T>(futures::channel::oneshot::Sender<T>);

    impl<T> Sender<T> {
        pub fn send(self, value: T) -> Result<(), T> {
            self.0.send(value)
        }
    }

    #[derive(Debug)]
    pub struct Receiver<T>(futures::channel::oneshot::Receiver<T>);

    impl<T> Receiver<T> {
        pub fn blocking_receive(self) -> Result<T, ReceiveError> {
            block_on(self)
        }
    }

    impl<T> Future for Receiver<T> {
        type Output = Result<T, ReceiveError>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let inner = &mut self.0;
            pin_mut!(inner);
            inner.poll(cx).map(|result| result.map_err(Into::into))
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("oneshot receive error")]
    pub struct ReceiveError(#[from] futures::channel::oneshot::Canceled);

    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let (tx, rx) = futures::channel::oneshot::channel();
        (Sender(tx), Receiver(rx))
    }
}

pub mod mpsc {
    pub use super::super::shared::mpsc::TryReceiveError;
    use super::*;

    impl From<async_std::channel::TryRecvError> for TryReceiveError {
        fn from(e: async_std::channel::TryRecvError) -> Self {
            match e {
                async_std::channel::TryRecvError::Empty => Self::Empty,
                async_std::channel::TryRecvError::Closed => Self::Disconnected,
            }
        }
    }

    pub mod unbounded {
        use super::*;

        #[derive(Debug)]
        pub struct Sender<T>(async_std::channel::Sender<T>);

        impl<T> Sender<T> {
            pub fn send(&self, value: T) -> Result<(), T> {
                // channel is never full
                self.0.try_send(value).map_err(|e| e.into_inner())
            }
        }

        impl<T> Clone for Sender<T> {
            fn clone(&self) -> Self {
                Sender(self.0.clone())
            }
        }

        #[derive(Debug)]
        pub struct Receiver<T>(async_std::channel::Receiver<T>);

        impl<T> Receiver<T> {
            pub fn blocking_receive(&mut self) -> Option<T> {
                // returns error if channel is closed and there are no more messages
                self.0.recv_blocking().ok()
            }

            pub fn close(&mut self) {
                self.0.close();
            }

            pub fn try_receive(&mut self) -> Result<T, TryReceiveError> {
                self.0.try_recv().map_err(Into::into)
            }
        }

        impl<T> Stream for Receiver<T> {
            type Item = T;

            fn poll_next(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                let inner = &mut self.0;
                pin_mut!(inner);
                inner.poll_next(cx)
            }
        }

        pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
            let (tx, rx) = async_std::channel::unbounded();
            (Sender(tx), Receiver(rx))
        }
    }

    pub mod bounded {
        use super::*;

        #[derive(Clone, Debug)]
        pub struct Sender<T>(async_std::channel::Sender<T>);

        impl<T> Sender<T> {
            pub async fn send(&self, value: T) -> Result<(), T> {
                self.0.send(value).await.map_err(|e| e.0)
            }

            pub fn blocking_send(&self, value: T) -> Result<(), T> {
                block_on(self.send(value))
            }
        }

        #[derive(Debug)]
        pub struct Receiver<T>(async_std::channel::Receiver<T>);

        impl<T> Receiver<T> {
            pub fn blocking_receive(&mut self) -> Option<T> {
                self.0.recv_blocking().ok()
            }

            pub fn close(&mut self) {
                self.0.close();
            }

            pub fn try_receive(&mut self) -> Result<T, TryReceiveError> {
                self.0.try_recv().map_err(Into::into)
            }
        }

        impl<T> Stream for Receiver<T> {
            type Item = T;

            fn poll_next(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Option<Self::Item>> {
                let inner = &mut self.0;
                pin_mut!(inner);
                inner.poll_next(cx)
            }
        }

        pub fn channel<T>(size: usize) -> (Sender<T>, Receiver<T>) {
            let (tx, rx) = async_std::channel::bounded(size);
            (Sender(tx), Receiver(rx))
        }
    }
}

pub mod watch {
    use std::ops::Deref;

    #[derive(Debug)]
    pub struct Sender<T>(async_watch::Sender<T>);

    impl<T> Sender<T> {
        pub fn send(&self, value: T) -> Result<(), T> {
            self.0.send(value).map_err(|e| e.value())
        }
    }

    #[derive(Debug)]
    pub struct Receiver<T>(async_watch::Receiver<T>);

    impl<T> Receiver<T> {
        pub fn borrow(&self) -> Ref<'_, T> {
            Ref(self.0.borrow())
        }

        pub async fn changed(&mut self) -> Result<(), ReceiveError> {
            self.0.changed().await.map_err(Into::into)
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("watch receive error")]
    pub struct ReceiveError(#[from] async_watch::error::RecvError);

    #[derive(Debug)]
    pub struct Ref<'a, T>(async_watch::Ref<'a, T>);

    impl<T> Deref for Ref<'_, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.0.deref()
        }
    }

    pub fn channel<T>(initial_value: T) -> (Sender<T>, Receiver<T>) {
        let (tx, rx) = async_watch::channel(initial_value);
        (Sender(tx), Receiver(rx))
    }
}
