use std::pin::Pin;

use super::{
    context::Context,
    grammar::Grammar,
    Token,
};

#[derive(Clone, Debug)]
pub struct SamplingParameters {
    pub mode: SamplingMode,
    pub repetition_penalties: Option<RepetitionPenalties>,
    pub soft_max: bool,
    pub top_k: Option<TopK>,
    pub top_p: Option<TopP>,
    pub temperature: f32,
    pub grammar: Option<Grammar>,
}

impl Default for SamplingParameters {
    fn default() -> Self {
        Self {
            mode: SamplingMode::Propability,
            repetition_penalties: None,
            soft_max: false,
            top_k: Some(TopK { k: 40, min_keep: 1 }),
            top_p: Some(TopP {
                p: 0.9,
                min_keep: 1,
            }),
            temperature: 0.4,
            grammar: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SamplingMode {
    MiroStat { tau: f32, eta: f32, m: i32 },
    MiroStatV2 { tau: f32, eta: f32 },
    Greedy,
    Propability,
}

impl Default for SamplingMode {
    fn default() -> Self {
        Self::Propability
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RepetitionPenalties {
    // todo
}

#[derive(Copy, Clone, Debug)]
pub struct TopK {
    pub k: i32,
    pub min_keep: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct TopP {
    pub p: f32,
    pub min_keep: usize,
}

#[derive(Clone, Debug)]
pub struct Sampler {
    parameters: SamplingParameters,

    /// mu for mirostat sampling
    mu: f32,
}

impl Sampler {
    pub fn new(parameters: SamplingParameters) -> Self {
        Self {
            parameters,
            mu: 0.0,
        }
    }

    // check if we need Context as mut-borrow.
    pub fn sample(&mut self, candidates: &mut Candidates, context: &mut Context) -> Token {
        let candidates = &mut candidates.data as *mut _;

        // apply sampling parameters
        if let Some(_repetition_penalties) = &self.parameters.repetition_penalties {
            todo!("implement repetition penalties");
        }
        if self.parameters.soft_max {
            unsafe {
                llama_cpp_sys::llama_sample_softmax(context.handle, candidates);
            }
        }
        if let Some(top_k) = &self.parameters.top_k {
            unsafe {
                llama_cpp_sys::llama_sample_top_k(
                    context.handle,
                    candidates,
                    top_k.k,
                    top_k.min_keep,
                );
            }
        }
        if let Some(top_p) = &self.parameters.top_p {
            unsafe {
                llama_cpp_sys::llama_sample_top_p(
                    context.handle,
                    candidates,
                    top_p.p,
                    top_p.min_keep,
                );
            }
        }
        unsafe {
            llama_cpp_sys::llama_sample_temp(
                context.handle,
                candidates,
                self.parameters.temperature,
            );
        }
        if let Some(grammar) = &self.parameters.grammar {
            unsafe {
                llama_cpp_sys::llama_sample_grammar(context.handle, candidates, grammar.handle);
            }
        }

        // sample next token
        let token = match self.parameters.mode {
            SamplingMode::MiroStat { tau, eta, m } => unsafe {
                llama_cpp_sys::llama_sample_token_mirostat(
                    context.handle,
                    candidates,
                    tau,
                    eta,
                    m,
                    &mut self.mu as _,
                )
            },
            SamplingMode::MiroStatV2 { tau, eta } => unsafe {
                llama_cpp_sys::llama_sample_token_mirostat_v2(
                    context.handle,
                    candidates,
                    tau,
                    eta,
                    &mut self.mu as _,
                )
            },
            SamplingMode::Greedy => unsafe {
                llama_cpp_sys::llama_sample_token_greedy(context.handle, candidates)
            },
            SamplingMode::Propability => unsafe {
                llama_cpp_sys::llama_sample_token(context.handle, candidates)
            },
        };

        // feed token into grammar
        if let Some(grammar) = &mut self.parameters.grammar {
            grammar.accept_token(context, token);
        }

        token
    }
}

impl From<SamplingParameters> for Sampler {
    fn from(parameters: SamplingParameters) -> Self {
        Self::new(parameters)
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
