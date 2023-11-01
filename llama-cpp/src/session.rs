//! Inference session
//!
//! This provides asynchronous interfaces to drive a LLM. Internally it spawns
//! a thread which runs the LLM, and communicates with it through a channel.
//!
//! # Example
//!
//! ```
//! # use std::io::{stdout, Write};
//! # use llama_cpp::{loader::ModelLoader, session::Session, Error};
//! # use futures::{stream::TryStreamExt, pin_mut};
//! # #[tokio::main]
//! # async fn main() -> Result<(), Error> {
//! # let model = ModelLoader::load("../data/TinyLLama-v0.gguf", Default::default()).wait_for_model().await?;
//! // first create an inference session.
//! let mut session = Session::new(model, Default::default());
//!
//! // push your prompt into it.
//! session.push_text("Write a poem.", true, false);
//!
//! // get a stream of word pieces.
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
//!
//! # Lock-step mode
//!
//! The [`TokenStream`] and [`PieceStream`] can be run in lock-step mode. To
//! enable this pass `lock_step = true` to [`Session::tokens`] or
//! [`Session::pieces`].
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
//! stream the context may already have additional tokens sampled and decoded.
//! These tokens will be returned by [`TokenStream::stop`] and
//! [`PieceStream::stop`].

use std::{
    collections::HashSet,
    pin::Pin,
    task::{self, Poll},
};

use futures::{
    pin_mut,
    stream::Stream,
};

use crate::{
    async_rt::{
        mpsc,
        oneshot,
        spawn_blocking,
    },
    backend::{
        context::{
            ContextParameters,
            DecodeWarning, Context,
        },
        inference::Inference,
        model::{
            Model,
            TokenDecoder,
        },
        sampling::{
            Sampler,
            SamplingParameters,
        },
        Token,
    },
    Error,
};

/// The [`SessionParameters`] are invalid.
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    #[error("invalid context parameters")]
    Context(#[from] crate::backend::context::CheckError),

    #[error("invalid sampling parameters")]
    Sampling(#[from] crate::backend::sampling::CheckError),

    #[error("invalid batch size: {0}")]
    BatchSize(usize),
}

/// Session parameters
#[derive(Clone, Debug, Default)]
pub struct SessionParameters {
    /// Parameters to create the llama context.
    pub context: ContextParameters,

    /// Parameters for the sampler.
    pub sampling: SamplingParameters,

    /// The batch size to use. If `None` the max batch size from the context
    /// will be used.
    // todo: i think we can remove this and just use the max batch size in context.
    pub batch_size: Option<usize>,
}

impl SessionParameters {
    /// Checks if the session parameters are valid
    pub fn check(&self) -> Result<(), CheckError> {
        self.context.check()?;
        self.sampling.check()?;
        self.batch_size
            .map(|b| {
                if b == 0 {
                    Err(CheckError::BatchSize(b))
                }
                else {
                    Ok::<(), CheckError>(())
                }
            })
            .transpose()?;
        Ok(())
    }
}

/// Inference session
pub struct Session {
    model: Model,
    tx: mpsc::unbounded::Sender<Command>,
}

impl Session {
    /// Creates a new inference session.
    ///
    /// # Panics
    ///
    /// Panics if the session parameters are invalid.
    pub fn new(model: Model, session_parameters: SessionParameters) -> Self {
        session_parameters.check().unwrap();

        let (tx, rx) = mpsc::unbounded::channel();

        // spawn thread that runs the sync llama code
        spawn_blocking({
            let model = model.clone();
            move || session_thread(model, session_parameters, rx)
        });

        Self { model, tx }
    }

    /// Send command to session thread
    fn send_command(&self, command: Command) {
        // the session thread only terminates if the sender to its command queue is
        // dropped, thus this doesn't panic.
        self.tx.send(command).expect("session thread terminated")
    }

    /// Push tokens (your prompt) into the LLM.
    pub fn push_tokens(&self, tokens: Vec<Token>) {
        // todo: pass oneshot to get error back,
        // todo: do we need to check if the tokens are valid?
        self.send_command(Command::Push { tokens });
    }

    /// Tokenize the text (your prompt) and push it into the LLM.
    pub fn push_text(&self, text: &str, add_bos: bool, allow_special: bool) {
        let tokens = self.model.tokenize(text, add_bos, allow_special);
        self.push_tokens(tokens);
    }

    /// Reset the sampler and optionally set new sampling parameters.
    pub fn reset_sampler(&self, parameters: Option<SamplingParameters>) {
        self.send_command(Command::ResetSampler { parameters })
    }

    /// Returns a stream of [`Token`]s.
    pub fn tokens<'session>(
        &'session mut self,
        max_tokens: Option<usize>,
        stop_tokens: impl IntoIterator<Item = Token>,
        lock_step: bool,
    ) -> TokenStream<'session> {
        let (tx, rx) =
            mpsc::bounded::channel(max_tokens.unwrap_or(TokenStream::DEFAULT_CHANNEL_SIZE));

        self.send_command(Command::Sample {
            max_tokens,
            stop_tokens: stop_tokens.into_iter().collect(),
            tx,
            lock_step,
        });

        TokenStream {
            rx: Some(rx),
            _session: self,
            stop_reason: None,
            tx_continue: None,
            warning: None,
        }
    }

    /// Returns a stream of text pieces (words or fragments of words).
    pub fn pieces<'session>(
        &'session mut self,
        max_tokens: Option<usize>,
        stop_tokens: impl IntoIterator<Item = Token>,
        lock_step: bool,
    ) -> PieceStream<'session> {
        let token_decoder = self.model.token_decoder();

        self.tokens(max_tokens, stop_tokens, lock_step)
            .into_pieces(token_decoder)
    }
}

fn session_thread(
    model: Model,
    parameters: SessionParameters,
    mut rx: mpsc::unbounded::Receiver<Command>,
) {
    let _span = tracing::debug_span!("session thread started");
    tracing::debug!("model: {}", model.path().display());
    tracing::trace!(?parameters);

    let mut context = Context::new(model.clone(), &parameters.context);
    let batch_size = parameters
        .batch_size
        .unwrap_or(parameters.context.n_batch as _);
    let mut inference = Inference::new(&mut context, batch_size);
    let mut sampler = Sampler::new(parameters.sampling);

    while let Some(command) = rx.blocking_receive() {
        let result = (|| {
            match command {
                Command::Push { tokens } => {
                    // feed the prompt tokens in case the sampler needs them.
                    for token in &tokens {
                        sampler.feed_prompt_token(*token);
                    }

                    // feed the prompt into inference.
                    // todo: send this error back to the caller. then the session thread doesn't
                    // produce any errors that it doesn't handle. so we can remove the try-catch.
                    inference.push(&tokens)?;
                }
                Command::ResetSampler { parameters } => {
                    let parameters = parameters.unwrap_or_else(|| sampler.parameters().clone());
                    sampler = Sampler::new(parameters);
                }
                Command::Sample {
                    max_tokens,
                    stop_tokens,
                    tx,
                    lock_step,
                } => {
                    let mut i: usize = 0;

                    loop {
                        let sampled = match inference.sample(&mut sampler) {
                            Ok(sampled) => sampled,
                            Err(e) => {
                                // an error occured.
                                tx.blocking_send(SampleStreamItem::Error(e)).ok();
                                break;
                            }
                        };

                        if let Some(warning) = sampled.warning {
                            // send warning to stream
                            tx.blocking_send(SampleStreamItem::Warning(warning)).ok();
                        }

                        // eos, so we stop
                        if sampled.token == model.token_eos() {
                            break;
                        }

                        let token = sampled.token;

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
                            match rx.blocking_receive() {
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
    ResetSampler {
        parameters: Option<SamplingParameters>,
    },
    Sample {
        max_tokens: Option<usize>,
        stop_tokens: HashSet<Token>,
        tx: mpsc::bounded::Sender<SampleStreamItem>,
        lock_step: bool,
    },
}

/// The reason why the LLM output was stopped.
#[derive(Clone, Copy, Debug)]
pub enum StopReason {
    /// The user requested the stop.
    User,

    /// A stop token was encountered.
    StopToken(Token),

    /// The token limit was reached.
    MaxTokens(usize),
}

#[derive(Clone, Copy, Debug)]
enum ShouldContinue {
    Continue,
    Stop,
}

#[derive(Debug)]
enum SampleStreamItem {
    Stop {
        reason: StopReason,
    },
    Token {
        token: Token,
        tx_continue: Option<oneshot::Sender<ShouldContinue>>,
    },
    Warning(DecodeWarning),
    Error(crate::backend::Error),
}

/// Streams individual tokens from a LLM.
pub struct TokenStream<'session> {
    rx: Option<mpsc::bounded::Receiver<SampleStreamItem>>,

    /// # Note
    ///
    /// We pass the session, so that the session is exclusively borrowed by this
    /// and thus the user can't start multiple token streams at the same time.
    _session: &'session mut Session,

    /// the stop reason. we either receive this from the session thread, or set
    /// it ourselves.
    stop_reason: Option<StopReason>,

    /// if in lock-step mode, this is the sender we have to use to continue the
    /// stream. it also allows us to stop the sampling after each token.
    tx_continue: Option<oneshot::Sender<ShouldContinue>>,

    /// The last warning we received
    warning: Option<DecodeWarning>,
}

impl<'session> TokenStream<'session> {
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
    /// yet. If an error occured in the sampling thread, it returns this
    /// instead.
    pub fn stop(&mut self) -> Result<Vec<Token>, Error> {
        // if `self.rx` is `None`, we already stopped the stream, thus we do nothing.
        let Some(mut rx) = self.rx.take()
        else {
            return Ok(vec![]);
        };

        // close sampler stream.
        // if not in lock-step mode this would only stop the sampling after the next
        // token(s) have been sampled.
        rx.close();

        // get tokens that are still buffered in the channel
        // note: once we close the stream, there should be no outstanding senders or
        // permits.
        let mut leftover = vec![];
        while let Ok(item) = rx.try_receive() {
            match item {
                SampleStreamItem::Stop { reason: _ } => {
                    // the stream was also terminated by the session thread. we
                    // set this as stop reason.
                }
                SampleStreamItem::Warning(_) => {
                    // i think we can just ignore any warnings here.
                }
                SampleStreamItem::Error(e) => {
                    // the session thread encountered an error.
                    return Err(e.into());
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

        Ok(leftover)
    }

    /// Returns the stop reason, if the stream has stopped.
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    /// Turns the token stream into a [`PieceStream`]. You can get the
    /// [`TokenDecoder`] from your [`Model`].
    pub fn into_pieces(self, token_decoder: TokenDecoder) -> PieceStream<'session> {
        PieceStream {
            token_stream: self,
            token_decoder,
        }
    }

    /// Returns the last decode warning received from the llama.cpp backend.
    pub fn warning(&self) -> Option<DecodeWarning> {
        self.warning
    }
}

impl<'session> Stream for TokenStream<'session> {
    type Item = Result<Token, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Option<Self::Item>> {
        // if we have a tx_continue stored, we need to send a continue signal on it, so
        // the sampling thread actually continues.
        self.proceed();

        if let Some(rx) = &mut self.rx {
            pin_mut!(rx);
            let mut last_warning = None;

            let poll = loop {
                let next = rx.as_mut().poll_next(cx);

                match next {
                    Poll::Ready(Some(SampleStreamItem::Token { token, tx_continue })) => {
                        self.tx_continue = tx_continue;
                        break Poll::Ready(Some(Ok(token)));
                    }
                    Poll::Ready(Some(SampleStreamItem::Stop { reason })) => {
                        self.stop_reason = Some(reason);
                        break Poll::Ready(None);
                    }
                    Poll::Ready(Some(SampleStreamItem::Warning(warning))) => {
                        last_warning = Some(warning);
                        // poll another item after this
                    }
                    Poll::Ready(Some(SampleStreamItem::Error(e))) => {
                        break Poll::Ready(Some(Err(e.into())))
                    }
                    Poll::Ready(None) => break Poll::Ready(None),
                    Poll::Pending => break Poll::Pending,
                }
            };

            last_warning.map(|w| self.warning = Some(w));
            poll
        }
        else {
            Poll::Ready(None)
        }
    }
}

/// A stream of text pieces.
pub struct PieceStream<'session> {
    token_stream: TokenStream<'session>,
    token_decoder: TokenDecoder,
}

impl<'session> PieceStream<'session> {
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
    /// Returns text that was already sampled, but not received by the stream
    /// yet. If an error occured in the sampling thread, it returns this
    /// instead.
    pub fn stop(&mut self) -> Result<String, Error> {
        let tokens = self.token_stream.stop()?;
        for token in tokens {
            self.token_decoder.push_token(token);
        }
        Ok(self.token_decoder.pop_string().unwrap_or_default())
    }

    /// Returns the stop reason, if the stream has stopped.
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.token_stream.stop_reason()
    }

    /// Turns this into a token stream.
    pub fn into_tokens(self) -> TokenStream<'session> {
        self.token_stream
    }

    /// Returns the last decode warning received from the llama.cpp backend.
    pub fn warning(&self) -> Option<DecodeWarning> {
        self.token_stream.warning()
    }
}

impl<'session> Stream for PieceStream<'session> {
    type Item = Result<String, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let token_stream = &mut self.token_stream;
            pin_mut!(token_stream);

            match token_stream.poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // todo: check if we have some stray bytes in the buffer.
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(token))) => {
                    if let Some(piece) = self.token_decoder.decode(token) {
                        // if we have a piece ready, we can return that from the stream.
                        return Poll::Ready(Some(Ok(piece)));
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
