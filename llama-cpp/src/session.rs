//! Inference Session
//!
//! A [`Session`] manages a single [`Context`]. A [`Session`] can have multiple
//! [`Sequence`]s. Each sequence accepts input (e.g. your prompt) via
//! [`Sequence::push`]. To predict the next tokens in a [`Sequence`] (i.e. get
//! the LLMs output) use [`Sequence::stream`].
//!
//! Both [`Sequence::push`] and [`Sequence::stream`] can work with strings or
//! tokens. Take a look at the [`token`](crate::token) module for the relevant
//! for traits and their implementors. For example [`Sequence::push`] also
//! accepts the [`Tokenize`](crate::token::Tokenize) struct, which gives you
//! more control over how the text is tokenized.
//!
//! [`Sequence`]s can be cloned. Internally this will tell the [`Context`] to
//! copy the sequence's KV cache and other important state. It behaves as if
//! you'd split the sequence at the current point and the output will diverge
//! from this point. This allows you to e.g. sample multiple responses from a
//! single prompt.
//!
//! # Example
//!
//! ```
//! # use std::io::{stdout, Write};
//! # use llama_cpp::{loader::ModelLoader, Error, token::Tokenize, session::Session};
//! # use futures::{stream::TryStreamExt, pin_mut};
//! # #[tokio::main]
//! # async fn main() -> Result<(), Error> {
//! # let model = ModelLoader::load("../data/TinyLLama-v0.gguf", Default::default()).wait_for_model().await?;
//! // first create an inference session.
//! let session = Session::from_model(model, Default::default());
//!
//! // then create a single sequence
//! let mut sequence = session.sequence();
//!
//! // push your prompt into it.
//! sequence.push(Tokenize { text: "A short poem:", add_bos: true, allow_special: false}).await?;
//!
//! // get a stream of word pieces.
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

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{
        atomic::{
            AtomicI32,
            Ordering,
        },
        Arc,
    },
    task::{
        self,
        Poll,
    },
};

use futures::{
    pin_mut,
    stream::Stream,
    Future,
};

use crate::{
    async_rt::{
        mpsc::{
            self,
            TryReceiveError,
        },
        oneshot,
        spawn_blocking,
    },
    backend::{
        context::{
            Context,
            ContextParameters,
            DecodeError,
        },
        model::Model,
        sampling::Sampler,
        SeqId,
        Token,
    },
    token::{
        FromToken,
        IntoTokens,
    },
    utils::IsLast,
    Error,
};

/// Inference session
#[derive(Clone)]
pub struct Session {
    handle: ContextHandle,
}

impl Session {
    /// Creates a new session from a [`Context`].
    pub fn from_context(context: Context) -> Self {
        Self {
            handle: ContextHandle::new(context),
        }
    }

    /// Creates a new session from a [`Model`] and [`ContextParameters`]. This
    /// will create a context for you.
    ///
    /// # Panics
    ///
    /// Panics if the context parameters are invalid.
    pub fn from_model(model: Model, context_parameters: &ContextParameters) -> Self {
        context_parameters.check().unwrap();

        let context = Context::new(model, context_parameters);
        Self::from_context(context)
    }

    /// Start a new sequence.
    pub fn sequence(&self) -> Sequence {
        Sequence::new(self.handle.clone())
    }
}

/// A single text sequence. This can be expanded with your own text using
/// [`Sequence::push`], or by predicted tokens with [`Sequence::stream`].
pub struct Sequence {
    sequence_id: SeqId,
    handle: ContextHandle,
}

impl Sequence {
    fn new(handle: ContextHandle) -> Self {
        let sequence_id = handle.new_sequence_id();
        Self {
            handle,
            sequence_id,
        }
    }

    async fn push_tokens_unchecked(&self, tokens: Vec<Token>) -> Result<(), Error> {
        let (tx_result, rx_result) = oneshot::channel();

        self.handle.send_command(Command::PushTokens {
            sequence_id: self.sequence_id,
            tokens,
            tx_result,
        });

        rx_result
            .await
            .expect("context thread dropped result sender")?;
        Ok(())
    }

    /// Push tokens (e.g. your prompt) into the LLM.
    pub async fn push(&self, input: impl IntoTokens) -> Result<(), Error> {
        let tokens = input.into_tokens(&self.handle.model);

        // check tokens
        for token in &tokens {
            self.handle.model.assert_valid_token(*token);
        }

        self.push_tokens_unchecked(tokens).await
    }

    /// Returns a stream of predicted text.
    pub fn stream<'sequence, Output: FromToken>(
        &'sequence mut self,
        sampler: Sampler,
    ) -> SequenceStream<'sequence, Output> {
        self.handle.send_command(Command::StartSampling {
            sequence_id: self.sequence_id,
            sampler,
        });

        SequenceStream::new(self)
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.handle.send_command_unckecked(Command::DeleteSequence {
            sequence_id: self.sequence_id,
        })
    }
}

impl Clone for Sequence {
    fn clone(&self) -> Self {
        let sequence = Self::new(self.handle.clone());
        self.handle.send_command(Command::CopySequence {
            from_sequence_id: self.sequence_id,
            to_sequence_id: sequence.sequence_id,
        });
        sequence
    }
}

#[derive(Clone)]
struct ContextHandle {
    model: Model,
    tx_command: mpsc::unbounded::Sender<Command>,
    next_sequence_id: Arc<AtomicI32>,
}

impl ContextHandle {
    fn new(context: Context) -> Self {
        let (tx, rx) = mpsc::unbounded::channel();
        let model = context.model().clone();

        // spawn thread that runs the sync llama code
        spawn_blocking(move || context_thread(context, rx));

        Self {
            model,
            tx_command: tx,
            next_sequence_id: Arc::new(AtomicI32::new(0)),
        }
    }

    fn new_sequence_id(&self) -> SeqId {
        self.next_sequence_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a command to context thread
    fn send_command(&self, command: Command) {
        // the context thread only terminates if the sender to its command queue is
        // dropped, thus this doesn't panic.
        self.tx_command
            .send(command)
            .expect("context thread terminated")
    }

    fn send_command_unckecked(&self, command: Command) {
        self.tx_command.send(command).ok();
    }
}

#[derive(Clone, Default)]
struct SequenceState {
    sampling: Option<SequenceSamplingState>,
    previous_tokens: Vec<Token>,
}

#[derive(Clone, Debug)]
struct SequenceSamplingState {
    sampler: Sampler,
}

fn context_thread(mut context: Context, mut rx: mpsc::unbounded::Receiver<Command>) {
    let mut batched = context.batched();
    let mut sequences: HashMap<SeqId, SequenceState> = HashMap::new();
    let mut command_buf = vec![];
    let mut sample_later = vec![];
    let mut copy_later = vec![];

    loop {
        assert!(command_buf.is_empty());
        if !batch_commands(&mut rx, &mut command_buf) {
            break;
        }

        for command in command_buf.drain(..) {
            match command {
                Command::DeleteSequence { sequence_id } => {
                    sequences.remove(&sequence_id);
                    batched.delete_sequence(sequence_id);
                }
                Command::PushTokens {
                    sequence_id,
                    tokens,
                    tx_result,
                } => {
                    let sequence_state = sequences.entry(sequence_id).or_default();

                    // push tokens into batch
                    let mut result = Ok(());
                    for (token, is_last) in IsLast::new(tokens.iter()) {
                        // we already checked that the tokens are valid in
                        // `Sequence::push_tokens`.
                        unsafe {
                            if let Err(e) =
                                batched.push_token_unchecked(*token, sequence_id, is_last)
                            {
                                result = Err(e);
                                break;
                            }
                        };
                    }

                    if result.is_ok() {
                        // store prompt tokens for sampler
                        sequence_state.previous_tokens.extend(tokens);
                    }

                    tx_result.send(result).ok();
                }
                Command::StartSampling {
                    sequence_id,
                    mut sampler,
                } => {
                    let sequence_state = sequences.entry(sequence_id).or_default();
                    sampler.push_previous(&sequence_state.previous_tokens);
                    sequence_state.previous_tokens.clear();
                    sequence_state.sampling = Some(SequenceSamplingState { sampler })
                }
                Command::StopSampling { sequence_id } => {
                    let sequence_state = sequences.get_mut(&sequence_id).expect("no sequence");
                    let sampling_state =
                        sequence_state.sampling.as_mut().expect("no sampling state");
                    sequence_state
                        .previous_tokens
                        .extend(sampling_state.sampler.previous_tokens());
                    sequence_state.sampling = None;
                }
                Command::Sample {
                    sequence_id,
                    tx_result,
                } => {
                    sample_later.push((sequence_id, tx_result));
                }
                Command::CopySequence {
                    from_sequence_id,
                    to_sequence_id,
                } => {
                    copy_later.push((from_sequence_id, to_sequence_id));
                }
            }
        }

        // decode batch if we need to sample or copy kv cache.
        let decode_error = (!sample_later.is_empty() || !copy_later.is_empty())
            .then(|| batched.decode().err())
            .flatten();

        for (from_sequence_id, to_sequence_id) in copy_later.drain(..) {
            // note: currently a sequence can't be split while sampling.
            let from_sequence = sequences.get(&from_sequence_id).expect("no sequence");
            sequences.insert(to_sequence_id, from_sequence.clone());
            batched.copy_sequence(from_sequence_id, to_sequence_id);
        }

        for (sequence_id, tx_result) in sample_later.drain(..) {
            let sequence_state = sequences.get_mut(&sequence_id).expect("no sequence");
            let sampling_state = sequence_state.sampling.as_mut().expect("not sampling");

            if let Some(decode_error) = decode_error {
                tx_result.send(Err(decode_error)).ok();
            }
            else {
                let sample_result = batched.sample(sequence_id, &mut sampling_state.sampler);
                tx_result.send(sample_result).ok();
            }
        }
    }
}

/// Return as many commands as possible. If `block` is `True` and there are no
/// commands in the channel, this will block until a command arrives.
/// Returns `false` if the channel has been closed.
fn batch_commands(rx: &mut mpsc::unbounded::Receiver<Command>, output: &mut Vec<Command>) -> bool {
    loop {
        match rx.try_receive() {
            Ok(command) => output.push(command),
            Err(TryReceiveError::Empty) => {
                if !output.is_empty() {
                    break;
                }

                if let Some(command) = rx.blocking_receive() {
                    output.push(command);
                }
                else {
                    break;
                }
            }
            Err(TryReceiveError::Disconnected) => return false,
        }
    }
    true
}

#[derive(Debug)]
enum Command {
    DeleteSequence {
        sequence_id: SeqId,
    },
    PushTokens {
        sequence_id: SeqId,
        // these are valid tokens.
        tokens: Vec<Token>,
        tx_result: oneshot::Sender<Result<(), DecodeError>>,
    },
    StartSampling {
        sequence_id: SeqId,
        sampler: Sampler,
    },
    StopSampling {
        sequence_id: SeqId,
    },
    Sample {
        sequence_id: SeqId,
        tx_result: oneshot::Sender<Result<Token, DecodeError>>,
    },
    CopySequence {
        from_sequence_id: SeqId,
        to_sequence_id: SeqId,
    },
}

/// Streams individual tokens from a LLM.
struct TokenStream<'sequence> {
    /// # Note
    ///
    /// We pass the session, so that the session is exclusively borrowed by this
    /// and thus the user can't start multiple token streams at the same time.
    sequence: &'sequence mut Sequence,

    rx_result: Option<oneshot::Receiver<Result<Token, DecodeError>>>,
}

impl<'sequence> Stream for TokenStream<'sequence> {
    type Item = Result<Token, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(rx_result) = &mut self.rx_result {
                pin_mut!(rx_result);
                match rx_result.poll(cx) {
                    Poll::Ready(result) => {
                        self.rx_result = None;
                        match result.expect("context thread dropped token sender") {
                            Ok(token) => {
                                if token == self.sequence.handle.model.token_eos() {
                                    return Poll::Ready(None);
                                }
                                else {
                                    return Poll::Ready(Some(Ok(token)));
                                }
                            }
                            Err(e) => return Poll::Ready(Some(Err(e.into()))),
                        }
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
            else {
                let (tx_result, rx_result) = oneshot::channel();
                self.rx_result = Some(rx_result);
                self.sequence.handle.send_command(Command::Sample {
                    sequence_id: self.sequence.sequence_id,
                    tx_result,
                });
            }
        }
    }
}

impl<'sequence> Drop for TokenStream<'sequence> {
    fn drop(&mut self) {
        self.sequence
            .handle
            .send_command_unckecked(Command::StopSampling {
                sequence_id: self.sequence.sequence_id,
            });
    }
}

pub struct SequenceStream<'sequence, Output: FromToken> {
    token_stream: TokenStream<'sequence>,
    state: Output::State,
}

impl<'sequence, Output: FromToken> SequenceStream<'sequence, Output> {
    fn new(sequence: &'sequence mut Sequence) -> Self {
        let state = Output::init(&sequence.handle.model);
        Self {
            token_stream: TokenStream {
                sequence,
                rx_result: None,
            },
            state,
        }
    }
}

impl<'sequence, Output: FromToken> Stream for SequenceStream<'sequence, Output> {
    type Item = Result<Output, Error>;

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
                    if let Some(piece) = Output::from_token(token, &mut self.state) {
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

/*
#[cfg(test)]
mod tests {
    use futures::stream::StreamExt;

    use super::*;
    use crate::{
        backend::grammar::Compiled as Grammar,
        utils::test::model,
    };

    fn inference(grammar: Option<Grammar>) -> Inference {
        model().inference(InferenceParameters {
            context: ContextParameters {
                seed: 1234.into(),
                ..Default::default()
            },
            sampling: SamplingParameters {
                grammar,
                ..Default::default()
            },
            batch_size: None,
        })
    }

    #[cfg(all(feature = "grammar", feature = "runtime-tokio"))]
    #[tokio::test]
    async fn it_reloads_the_grammar() {
        use crate::grammar::parse_and_compile;

        let grammar = parse_and_compile("root ::= \"Hello World\"").unwrap();
        let mut inference = inference(Some(grammar));

        inference
            .push_text("Hello World", true, false)
            .await
            .unwrap();

        let output = inference
            .pieces(Default::default())
            .map(|result| result.unwrap())
            .collect::<String>()
            .await;
        assert_eq!(output, "Hello World");

        // grammar was at end, but should have been reloaded.
        let output = inference
            .pieces(SamplingOptions {
                reload_grammar: false,
                ..Default::default()
            })
            .map(|result| result.unwrap())
            .collect::<String>()
            .await;
        assert_eq!(output, "Hello World");

        // grammar is at end now, so we should get an empty string
        let output = inference
            .pieces(Default::default())
            .map(|result| result.unwrap())
            .collect::<String>()
            .await;
        assert_eq!(output, "");
    }
}
 */
