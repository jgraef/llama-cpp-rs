//! Sampling from a LLM

use std::{
    collections::VecDeque,
    pin::Pin,
};

use super::{
    context::Context,
    grammar::{
        Compiled as CompiledGrammar,
        Loaded as LoadedGrammar,
    },
    Token,
};

/// Sampling parameters
#[derive(Clone, Debug)]
pub struct SamplingParameters {
    pub mode: SamplingMode,
    pub repetition_penalties: Option<RepetitionPenalties>,
    pub grammar: Option<CompiledGrammar>,
}

/// Error returned by [`SamplingParameters::check`]
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    #[error("grammar invalid")]
    Grammar(#[from] super::grammar::CheckError),
}

impl SamplingParameters {
    /// Checks if the parameters are valid.
    pub fn check(&self) -> Result<(), CheckError> {
        // todo
        self.grammar.as_ref().map(|g| g.check()).transpose()?;
        Ok(())
    }

    /// How many previous tokens are kept in the sampling state.
    ///
    /// This currently only depends on repetition penalties.
    pub fn n_prev(&self) -> Option<usize> {
        Some(self.repetition_penalties.as_ref()?.last_n)
    }
}

impl Default for SamplingParameters {
    fn default() -> Self {
        Self {
            mode: Default::default(),
            repetition_penalties: Some(Default::default()),
            grammar: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SamplingMode {
    MiroStat {
        temperature: f32,

        tau: f32,

        eta: f32,

        /// # TODO
        ///
        /// This is a const 100 in `llama.cpp/common/sampling.cpp`
        m: i32,
    },
    MiroStatV2 {
        temperature: f32,

        tau: f32,

        eta: f32,
    },
    Greedy,
    Temperature {
        temperature: f32,

        top_k: i32,

        top_p: f32,

        n_probs: usize,

        tfs_z: f32,

        typical_p: f32,
    },
}

impl Default for SamplingMode {
    fn default() -> Self {
        Self::Temperature {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            n_probs: 0,
            tfs_z: 1.0,
            typical_p: 1.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RepetitionPenalties {
    pub last_n: usize,
    pub repeat: Option<f32>,
    pub frequency: Option<f32>,
    pub present: Option<f32>,
    pub penalize_nl: bool,
}

impl Default for RepetitionPenalties {
    fn default() -> Self {
        Self {
            last_n: 64,
            repeat: Some(1.10),
            frequency: None,
            present: None,
            penalize_nl: true,
        }
    }
}

#[derive(Clone, Debug)]
struct Buffer {
    tokens: VecDeque<Token>,
    buf_size: usize,
}

impl Buffer {
    pub fn new(buf_size: usize) -> Self {
        Self {
            tokens: VecDeque::with_capacity(buf_size),
            buf_size,
        }
    }

    pub fn push(&mut self, token: Token) {
        self.tokens.push_back(token);
        if self.tokens.len() > self.buf_size {
            self.tokens.pop_front();
        }
    }

    pub fn push_all(&mut self, tokens: &[Token]) {
        let tokens = &tokens[tokens.len().saturating_sub(self.buf_size)..];
        for token in tokens {
            self.push(*token);
        }
    }

    pub fn get_last_n(&mut self, mut last_n: usize) -> &[Token] {
        let n = self.tokens.len();

        // take at most as many tokens as we have.
        if last_n > n {
            last_n = n;
        }

        if last_n == 0 {
            return &[];
        }

        let start = n - last_n;
        let end = n;

        let (a, _) = self.tokens.as_slices();

        if start < a.len() && end <= a.len() {
            // all elements are in the first half of the buffer
            let (a, _) = self.tokens.as_slices();
            &a[start..end]
        }
        else if start >= a.len() && end >= a.len() {
            // all elements are in the second half of the buffer
            let (a, b) = self.tokens.as_slices();
            &b[start - a.len()..end - a.len()]
        }
        else {
            // we need to make the buffer contiguous
            let c = self.tokens.make_contiguous();
            &c[start..end]
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    parameters: SamplingParameters,

    /// mu for mirostat sampling
    mu: f32,

    /// grammar for grammar sampling.
    grammar: Option<LoadedGrammar>,

    /// previous tokens for repetition penalties.
    token_buf: Option<Buffer>,
}

impl Sampler {
    /// Creates a new sampler
    ///
    /// # Panics
    ///
    /// Panics if the sampling parameters are invalid.
    pub fn new(parameters: SamplingParameters) -> Self {
        parameters.check().unwrap();

        let grammar = parameters.grammar.as_ref().map(|grammar| grammar.load());

        let token_buf = parameters.n_prev().map(|n| Buffer::new(n));

        Self {
            parameters,
            mu: 0.0,
            grammar,
            token_buf,
        }
    }

    /// Returns the [`SamplingParameters`]
    pub fn parameters(&self) -> &SamplingParameters {
        &self.parameters
    }

    /// Returns the previous tokens, if they're being kept by the sampler state.
    pub fn previous_tokens<'a>(&'a self) -> impl Iterator<Item = Token> + 'a {
        self.token_buf
            .as_ref()
            .map(|b| b.tokens.iter())
            .into_iter()
            .flatten()
            .copied()
    }

    /// Sample a token from the candidates.
    pub fn sample(&mut self, mut candidates: Candidates, context: &mut Context) -> Token {
        let token_data = &mut candidates.data as *mut _;

        // apply repetition penalties
        // note: we don't care if the previous token buffer has invalid tokens in it.
        // the llama.cpp function called here doesn't care. it just counts occurences.
        if let Some(repetition_penalties) = &self.parameters.repetition_penalties {
            let buffer = self.token_buf.as_mut().expect("no token buffer");
            let last_tokens = buffer.get_last_n(repetition_penalties.last_n);

            // get newline index and logit to reset later
            let nl = context.model().token_nl();
            let nl = (!repetition_penalties.penalize_nl)
                .then(|| {
                    candidates
                        .buf
                        .iter()
                        .enumerate()
                        .find_map(|(i, c)| (c.id == nl).then_some((i, c.logit)))
                })
                .flatten();

            unsafe {
                llama_cpp_sys::llama_sample_repetition_penalties(
                    context.handle,
                    token_data,
                    last_tokens.as_ptr(),
                    last_tokens.len(),
                    repetition_penalties.repeat.unwrap_or(1.0),
                    repetition_penalties.frequency.unwrap_or_default(),
                    repetition_penalties.present.unwrap_or_default(),
                );
            }

            // reset newline logit
            if let Some((i, logit)) = nl {
                candidates.buf[i].logit = logit;
            }
        }

        // note: grammar sampling doesn't work if this is done in another order! if
        // something breaks, check the llama.cpp sampling code.
        if let Some(grammar) = &self.grammar {
            unsafe {
                llama_cpp_sys::llama_sample_grammar(context.handle, token_data, grammar.handle);
            }
        }

        // sample next token
        let token = unsafe {
            match self.parameters.mode {
                SamplingMode::MiroStat {
                    tau,
                    eta,
                    m,
                    temperature,
                } => {
                    llama_cpp_sys::llama_sample_temp(context.handle, token_data, temperature);

                    llama_cpp_sys::llama_sample_token_mirostat(
                        context.handle,
                        token_data,
                        tau,
                        eta,
                        m,
                        &mut self.mu as _,
                    )
                }

                SamplingMode::MiroStatV2 {
                    tau,
                    eta,
                    temperature,
                } => {
                    llama_cpp_sys::llama_sample_temp(context.handle, token_data, temperature);

                    llama_cpp_sys::llama_sample_token_mirostat_v2(
                        context.handle,
                        token_data,
                        tau,
                        eta,
                        &mut self.mu as _,
                    )
                }

                SamplingMode::Greedy => {
                    llama_cpp_sys::llama_sample_token_greedy(context.handle, token_data)
                }

                SamplingMode::Temperature {
                    temperature,
                    top_k,
                    top_p,
                    n_probs,
                    tfs_z,
                    typical_p,
                } => {
                    let min_keep = std::cmp::min(1, n_probs);

                    llama_cpp_sys::llama_sample_temp(context.handle, token_data, temperature);

                    llama_cpp_sys::llama_sample_top_k(context.handle, token_data, top_k, min_keep);

                    llama_cpp_sys::llama_sample_tail_free(
                        context.handle,
                        token_data,
                        tfs_z,
                        min_keep,
                    );

                    llama_cpp_sys::llama_sample_typical(
                        context.handle,
                        token_data,
                        typical_p,
                        min_keep,
                    );

                    llama_cpp_sys::llama_sample_top_p(context.handle, token_data, top_p, min_keep);

                    llama_cpp_sys::llama_sample_token(context.handle, token_data)
                }
            }
        };

        // feed token into previous token accumulator
        self.push_previous(&[token]);

        // feed token into grammar
        //
        // It's very important that we only feed tokens to the grammar that can be
        // accepted by it.
        if let Some(grammar) = &mut self.grammar {
            // see safety section in [`Grammar::accept_token`].
            unsafe {
                grammar.accept_token(context, token);
            }
        }

        token
    }

    /// Push tokens into the previous token buffer.
    pub fn push_previous(&mut self, tokens: &[Token]) {
        if let Some(token_buf) = &mut self.token_buf {
            token_buf.push_all(tokens);
        }
    }

    /// Reloads the grammar state machine. Tells the sampler to start the
    /// grammar from its root node again.
    pub fn reload_grammar(&mut self) {
        self.grammar = self
            .parameters
            .grammar
            .as_ref()
            .map(|grammar| grammar.load());
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

#[derive(Clone, Debug)]
pub struct Candidates {
    buf: Pin<Vec<llama_cpp_sys::llama_token_data>>,
    data: llama_cpp_sys::llama_token_data_array,
}

impl Candidates {
    pub fn from_logits(logits: &[f32]) -> Self {
        let mut buf = Vec::with_capacity(logits.len());

        for (i, logit) in logits.iter().enumerate() {
            let token = i as Token;

            buf.push(llama_cpp_sys::llama_token_data {
                id: token,
                logit: *logit,
                p: 0.0,
            });
        }

        // pin it before we get the pointer to it.
        let mut buf = Pin::new(buf);

        let data = llama_cpp_sys::llama_token_data_array {
            data: buf.as_mut_ptr(),
            size: buf.len(),
            sorted: false,
        };

        Self { buf, data }
    }
}
