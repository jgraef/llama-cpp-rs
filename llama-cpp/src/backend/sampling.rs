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
            repetition_penalties: None,
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
        while self.tokens.len() > self.buf_size {
            self.tokens.pop_front();
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
    pub fn sample(&mut self, candidates: &mut Candidates, context: &mut Context) -> Token {
        let candidates = &mut candidates.data as *mut _;

        // apply repetition penalties
        if let Some(_repetition_penalties) = &self.parameters.repetition_penalties {
            todo!("implement repetition penalties");
        }

        // note: grammar sampling doesn't work if this is done in another order! if
        // something breaks, check the llama.cpp sampling code.
        if let Some(grammar) = &self.grammar {
            unsafe {
                llama_cpp_sys::llama_sample_grammar(context.handle, candidates, grammar.handle);
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
                    llama_cpp_sys::llama_sample_temp(context.handle, candidates, temperature);

                    llama_cpp_sys::llama_sample_token_mirostat(
                        context.handle,
                        candidates,
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
                    llama_cpp_sys::llama_sample_temp(context.handle, candidates, temperature);

                    llama_cpp_sys::llama_sample_token_mirostat_v2(
                        context.handle,
                        candidates,
                        tau,
                        eta,
                        &mut self.mu as _,
                    )
                }

                SamplingMode::Greedy => {
                    llama_cpp_sys::llama_sample_token_greedy(context.handle, candidates)
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

                    llama_cpp_sys::llama_sample_temp(context.handle, candidates, temperature);

                    llama_cpp_sys::llama_sample_top_k(context.handle, candidates, top_k, min_keep);

                    llama_cpp_sys::llama_sample_tail_free(
                        context.handle,
                        candidates,
                        tfs_z,
                        min_keep,
                    );

                    llama_cpp_sys::llama_sample_typical(
                        context.handle,
                        candidates,
                        typical_p,
                        min_keep,
                    );

                    llama_cpp_sys::llama_sample_top_p(context.handle, candidates, top_p, min_keep);

                    llama_cpp_sys::llama_sample_token(context.handle, candidates)
                }
            }
        };

        // feed token into previous token accumulator
        self.feed_previous(token);

        // feed token into grammar
        //
        // It's very important that we only feed tokens to the grammar that were just
        // sampled using the grammar.
        if let Some(grammar) = &mut self.grammar {
            // see safety section in [`Grammar::accept_token`].
            unsafe {
                grammar.accept_token(context, token);
            }
        }

        token
    }

    fn feed_previous(&mut self, token: Token) {
        if let Some(token_buf) = &mut self.token_buf {
            token_buf.push(token);
        }
    }

    pub fn feed_prompt_token(&mut self, token: Token) {
        // todo: is this unsafe, since we can feed invalid tokens?
        self.feed_previous(token);
    }
}

// is this a good way to do this?
pub struct Candidates {
    _buf: Pin<Vec<llama_cpp_sys::llama_token_data>>,

    // this references `_buf`, so we need to make sure it can only be accessed while `_buf` still
    // exists.
    data: llama_cpp_sys::llama_token_data_array,
}

impl Candidates {
    pub fn new(logits: &[f32]) -> Self {
        let mut buf = Vec::with_capacity(logits.len());
        for (i, logit) in logits.iter().enumerate() {
            buf.push(llama_cpp_sys::llama_token_data {
                id: i as _,
                logit: *logit,
                p: 0.0,
            });
        }

        let data = llama_cpp_sys::llama_token_data_array {
            data: buf.as_mut_ptr(),
            size: buf.len(),
            sorted: false,
        };

        Self {
            _buf: Pin::new(buf),
            data,
        }
    }
}
