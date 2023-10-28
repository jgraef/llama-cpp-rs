//!
//! # Lock-step mode
//!
//! The [`TokenStream`] and [`PieceStream`] can be run in lock-step mode. To
//! enable this pass `lock_step = true` to [`Sampler::tokens`] or
//! [`Sampler::pieces`].
//!
//! In lock-step mode the background thread will wait with sampling the next
//! token until the the stream is polled for the next item. This ensures you
//! can stop the sampling after any token.
//! If you want to continue sampling in the background, before you poll the
//! stream again, you can call [`TokenStream::proceed`] or
//! [`PieceStream::proceed`].
//!
//! If not in lock-step mode the sampling will continue in the background until
//! max tokens or a stop token is reached. This means if you stop the token
//! stream the context may already have additional tokens sampled (and decoded).
//! These tokens will be returned by [`TokenStream::stop`] and
//! [`PieceStream::stop`].

use std::{
    collections::HashSet,
    pin::Pin,
    task::{
        Context,
        Poll,
    },
};

use futures::{
    pin_mut,
    stream::Stream,
};
use tokio::sync::{
    mpsc,
    oneshot,
};

use crate::{
    backend::{
        context::ContextParameters,
        inference::Inference,
        model::{
            Model,
            TokenDecoder,
        },
        sampling::Sampler as SyncSampler,
        Token,
    },
    error::Error,
};

#[derive(Clone, Debug, Default)]
pub struct SessionParameters {
    pub context: ContextParameters,

    // todo: i think we can remove this and just use the max batch size in context.
    pub batch_size: Option<usize>,
}

pub struct Session {
    model: Model,
    tx: mpsc::UnboundedSender<Command>,
}

impl Session {
    pub fn new(model: Model, session_parameters: SessionParameters) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        // spawn thread that runs the sync llama code
        tokio::task::spawn_blocking({
            let model = model.clone();
            move || session_thread(model, session_parameters, rx)
        });

        Self { model, tx }
    }

    fn send_command(&self, command: Command) {
        self.tx.send(command).expect("session thread terminated")
    }

    pub fn push_tokens(&self, tokens: Vec<Token>) {
        self.send_command(Command::Push { tokens });
    }

    pub fn push_text(&self, text: &str, add_bos: bool, allow_special: bool) {
        let tokens = self.model.tokenize(text, add_bos, allow_special);
        self.push_tokens(tokens);
    }

    pub fn sampler<'a>(&'a mut self, sampler: impl Into<SyncSampler>) -> Sampler<'a> {
        self.send_command(Command::SetSampler {
            sampler: sampler.into(),
        });
        Sampler { session: self }
    }
}

fn session_thread(
    model: Model,
    parameters: SessionParameters,
    mut rx: mpsc::UnboundedReceiver<Command>,
) {
    let _span = tracing::debug_span!("session thread started");
    tracing::debug!("model: {}", model.path().display());
    tracing::debug!(?parameters);

    let mut context = model.context(&parameters.context);
    let batch_size = parameters
        .batch_size
        .unwrap_or(parameters.context.n_batch as _);
    let mut inference = Inference::new(&mut context, batch_size);
    let mut sampler = None;

    while let Some(command) = rx.blocking_recv() {
        let result = (|| {
            match command {
                Command::Push { tokens } => {
                    inference.push(&tokens)?;
                }
                Command::SetSampler { sampler: s } => {
                    sampler = Some(s);
                }
                Command::Sample {
                    max_tokens,
                    stop_tokens,
                    tx,
                    lock_step,
                } => {
                    // the session proxy must uphold the invariant that `Command::SetSampler` is
                    // always called at least once before sampling.
                    let sampler = sampler.as_mut().expect("sampler not set");

                    let mut i: usize = 0;

                    while let Some(token) = inference.sample(sampler)? {
                        // if we found a stop token, we stop.
                        if stop_tokens.contains(&token) {
                            tx.blocking_send(SampleStreamItem::Stop {
                                reason: StopReason::StopToken(token),
                            })
                            .ok();
                            break;
                        }

                        // create tx_continue if we run in lock-step mode
                        let (tx_continue, rx_continue) = if lock_step {
                            let (rx, tx) = oneshot::channel();
                            (Some(rx), Some(tx))
                        }
                        else {
                            (None, None)
                        };

                        // send the token to the token stream
                        if tx
                            .blocking_send(SampleStreamItem::Token { token, tx_continue })
                            .is_err()
                        {
                            // the receiver was dropped, so we're done here.
                            break;
                        }

                        // if we're in lock-step mode we wait for the ShouldContinue to arrive.
                        if lock_step {
                            let rx = rx_continue
                                .expect("sampling in lock-step mode, but rx_continue is None");
                            match rx.blocking_recv() {
                                Ok(ShouldContinue::Stop) => {
                                    // the user terminated the stream. we don't need to send the
                                    // stop reason.
                                    break;
                                }
                                // if ShouldContinue::Continue => we continue
                                // if Err, the tx_continue was dropped. that means we don't want to
                                // stop either.
                                _ => {}
                            }
                        }

                        // increment token count
                        i += 1;

                        // if max_tokens is not None and we reached it, we stop.
                        if max_tokens.map(|m| i >= m).unwrap_or_default() {
                            tx.blocking_send(SampleStreamItem::Stop {
                                reason: StopReason::MaxTokens(i),
                            })
                            .ok();
                            break;
                        }
                    }
                }
            }
            Ok::<(), Error>(())
        })();

        if let Err(e) = result {
            tracing::error!("session thread error: {e}");
        }
    }
}

#[derive(Debug)]
enum Command {
    Push {
        tokens: Vec<Token>,
    },
    SetSampler {
        sampler: SyncSampler,
    },
    Sample {
        max_tokens: Option<usize>,
        stop_tokens: HashSet<Token>,
        tx: mpsc::Sender<SampleStreamItem>,
        lock_step: bool,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum StopReason {
    User,
    StopToken(Token),
    MaxTokens(usize),
}

#[derive(Clone, Copy, Debug)]
enum ShouldContinue {
    Continue,
    Stop,
}

enum SampleStreamItem {
    Stop {
        reason: StopReason,
    },
    Token {
        token: Token,
        tx_continue: Option<oneshot::Sender<ShouldContinue>>,
    },
}

pub struct Sampler<'session> {
    session: &'session mut Session,
}

impl<'session> Sampler<'session> {
    pub fn tokens<'sampler>(
        &'sampler mut self,
        max_tokens: Option<usize>,
        stop_tokens: impl IntoIterator<Item = Token>,
        lock_step: bool,
    ) -> TokenStream<'session, 'sampler> {
        let (tx, rx) = mpsc::channel(max_tokens.unwrap_or(TokenStream::DEFAULT_CHANNEL_SIZE));

        self.session.send_command(Command::Sample {
            max_tokens,
            stop_tokens: stop_tokens.into_iter().collect(),
            tx,
            lock_step,
        });

        TokenStream {
            rx: Some(rx),
            _sampler: self,
            stop_reason: None,
            tx_continue: None,
        }
    }

    pub fn pieces<'sampler>(
        &'sampler mut self,
        max_tokens: Option<usize>,
        stop_tokens: impl IntoIterator<Item = Token>,
        lock_step: bool,
    ) -> PieceStream<'session, 'sampler> {
        let token_decoder = self.session.model.token_decoder();

        self.tokens(max_tokens, stop_tokens, lock_step)
            .into_pieces(token_decoder)
    }
}

pub struct TokenStream<'session, 'sampler> {
    rx: Option<mpsc::Receiver<SampleStreamItem>>,

    /// # Note
    ///
    /// We pass the sampler, so that the sampler is exclusively borrowed by this
    /// and thus the user can't start multiple token streams at the same time.
    _sampler: &'sampler mut Sampler<'session>,

    /// the stop reason. we either receive this from the session thread, or set
    /// it ourselves.
    stop_reason: Option<StopReason>,

    /// if in lock-step mode, this is the sender we have to use to continue the
    /// stream. it also allows us to stop the sampling after each token.
    tx_continue: Option<oneshot::Sender<ShouldContinue>>,
}

impl<'session, 'sampler> TokenStream<'session, 'sampler> {
    const DEFAULT_CHANNEL_SIZE: usize = 512;

    fn lock_step_signal(&mut self, should_continue: ShouldContinue) {
        if let Some(tx_continue) = self.tx_continue.take() {
            tx_continue
                .send(should_continue)
                .expect("receiver for tx_continue was dropped by session ead");
        }
    }

    /// Signals the sampler to continue with sampling the next token.
    ///
    /// If not in lock-step mode, this does nothing.
    ///
    /// If the user doesn't call this, it will be called when the stream is
    /// polled. Use this method if you know that you want another token and want
    /// the background thread to sample while you do something else.
    pub fn proceed(&mut self) {
        self.lock_step_signal(ShouldContinue::Continue)
    }

    /// Signals the sampler to stop.
    ///
    /// If not in lock-step mode, this will stop the sampler, but it might
    /// already have sampled additional tokens.
    ///
    /// If in lock-step mode, this will stop the sampler before it samples
    /// another token.
    ///
    /// This also sets the `stop_reason` to `StopReason::User`.
    ///
    /// Returns tokens that were already sampled, but not received by the stream
    /// yet.
    pub fn stop(&mut self) -> Vec<Token> {
        // if `self.rx` is `None`, we already stopped the stream, thus we do nothing.
        let Some(mut rx) = self.rx.take()
        else {
            return vec![];
        };

        // close sampler stream.
        // if not in lock-step mode this would only stop the sampling after the next
        // token(s) have been sampled.
        rx.close();

        // get tokens that are still buffered in the channel
        // note: once we close the stream, there should be no outstanding senders or
        // permits.
        let mut leftover = vec![];
        while let Ok(item) = rx.try_recv() {
            match item {
                SampleStreamItem::Stop { reason: _ } => {
                    // the stream was also terminated by the session thread. we
                    // set this as stop reason.
                }
                SampleStreamItem::Token {
                    token,
                    tx_continue: _,
                } => {
                    // we should only have buffered items left if we're not in lock-step mode, or if
                    // the user called [`TokenStream::proceed`] we don't need to
                    // explicitely signal tx_continue here, since the session thread will know what
                    // to do once the tx_continue is dropped.
                    leftover.push(token);
                }
            }
        }

        // signal sampler to stop (if in lock-step mode)
        // if in lock-step mode, we still need to signal through tx_continue, because
        // the sampling thread is waiting for this.
        self.lock_step_signal(ShouldContinue::Stop);

        // set stop reason
        // we only set this if we don't already have one. `StopReason::User` doesn't
        // convey that much information.
        if self.stop_reason.is_none() {
            self.stop_reason = Some(StopReason::User);
        }

        leftover
    }

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    pub fn into_pieces(self, token_decoder: TokenDecoder) -> PieceStream<'session, 'sampler> {
        PieceStream {
            token_stream: self,
            token_decoder,
        }
    }
}

impl<'session, 'sampler> Stream for TokenStream<'session, 'sampler> {
    type Item = Token;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // if we have a tx_continue stored, we need to send a continue signal on it, so
        // the sampling thread actually continues.
        self.proceed();

        if let Some(rx) = &mut self.rx {
            rx.poll_recv(cx).map(|item_opt| {
                match item_opt {
                    Some(SampleStreamItem::Token { token, tx_continue }) => {
                        self.tx_continue = tx_continue;
                        Some(token)
                    }
                    Some(SampleStreamItem::Stop { reason }) => {
                        self.stop_reason = Some(reason);
                        None
                    }
                    None => None,
                }
            })
        }
        else {
            Poll::Ready(None)
        }
    }
}

pub struct PieceStream<'session, 'sampler> {
    token_stream: TokenStream<'session, 'sampler>,
    token_decoder: TokenDecoder,
}

impl<'session, 'sampler> PieceStream<'session, 'sampler> {
    /// Signals the sampler to continue with sampling the next token.
    ///
    /// If not in lock-step mode, this does nothing.
    ///
    /// If the user doesn't call this, it will be called when the stream is
    /// polled. Use this method if you know that you want another token and want
    /// the background thread to sample while you do something else.
    pub fn proceed(&mut self) {
        self.token_stream.proceed();
    }

    pub fn stop(&mut self) -> Vec<Token> {
        self.token_stream.stop()
    }

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.token_stream.stop_reason()
    }
}

impl<'session, 'sampler> Stream for PieceStream<'session, 'sampler> {
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let token_stream = &mut self.token_stream;
            pin_mut!(token_stream);

            match token_stream.poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // todo: check if we have some stray bytes in the buffer.
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(token)) => {
                    if let Some(piece) = self.token_decoder.decode(token) {
                        // if we have a piece ready, we can return that from the stream.
                        return Poll::Ready(Some(piece));
                    }
                    else {
                        // if we don't have a piece ready, we need to continue
                        // with the next iteration of the loop, polling the
                        // underlying stream again.
                    }
                }
            }
        }
    }
}
