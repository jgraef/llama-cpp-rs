//! LLM context

use std::slice;

use itertools::Itertools;

use super::{
    batch::Batch,
    default_n_threads,
    model::Model,
    sampling::{Sampler, Candidates},
    Error,
    Pos,
    Token,
    DEFAULT_SEED,
};
use crate::utils::IsLast;

/// Parameters for a llama context.
#[derive(Clone, Debug)]
pub struct ContextParameters {
    /// RNG seed, `None` for random.
    pub seed: Option<u32>,

    /// Text context. If set to `None`, it will take the parameter from the
    /// model.
    pub n_ctx: Option<u32>,

    /// Prompt processing maximum batch size.
    pub n_batch_max: u32,

    /// Number of threads to use for generation.
    pub n_threads: Option<u32>,

    /// Number of threads to use for batch processing.
    pub n_threads_batch: Option<u32>,

    /// RoPE base frequency. If set to `None`, it will take the parameter from
    /// the model.
    ///
    /// See <https://github.com/ggerganov/llama.cpp/pull/2054>
    pub rope_freq_base: Option<f32>,

    // RoPE frequency scaling factor. If set to `None`, it will take the parameter from the model.
    /// See <https://github.com/ggerganov/llama.cpp/pull/2054>
    pub rope_fre_scale: Option<f32>,

    /// If true, use experimental mul_mat_q kernels
    pub mul_mat_q: bool,

    /// Use fp16 for KV cache, fp32 otherwise.
    pub kv_cache_type: KvCacheType,

    /// The `llama_eval()` call computes all logits, not just the last one.
    pub logits_all: bool,

    /// Store embeddings
    pub embedding: bool,
}

/// Error returned by [`ContextParameters::check`]
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    // todo
}

impl ContextParameters {
    /// Checks of the context parameters are valid.
    pub fn check(&self) -> Result<(), CheckError> {
        // todo
        Ok(())
    }

    fn to_ffi(&self) -> llama_cpp_sys::llama_context_params {
        llama_cpp_sys::llama_context_params {
            seed: self.seed.unwrap_or(u32::MAX),
            n_ctx: self.n_ctx.unwrap_or(0),
            n_batch: self.n_batch_max,
            n_threads: self.n_threads.unwrap_or(default_n_threads()),
            n_threads_batch: self.n_threads_batch.unwrap_or(default_n_threads()),
            rope_freq_base: self.rope_freq_base.unwrap_or_default(),
            rope_freq_scale: self.rope_fre_scale.unwrap_or_default(),
            mul_mat_q: self.mul_mat_q,
            f16_kv: matches!(self.kv_cache_type, KvCacheType::F16),
            logits_all: self.logits_all,
            embedding: self.embedding,
        }
    }
}

impl Default for ContextParameters {
    fn default() -> Self {
        Self {
            seed: Some(DEFAULT_SEED),
            n_ctx: Some(512),
            n_batch_max: 512,
            n_threads: None,
            n_threads_batch: None,
            rope_freq_base: None,
            rope_fre_scale: None,
            mul_mat_q: true,
            kv_cache_type: Default::default(),
            logits_all: false,
            embedding: false,
        }
    }
}

/// A llama context.
///
/// This is the low-level interface to drive a LLM.
pub struct Context {
    pub(super) handle: *mut llama_cpp_sys::llama_context,

    /// This is important, since it keeps the model from being dropped, while we
    /// still have a context to it.
    model: Model,

    /// The number of tokens in the last batch_decode
    n_tokens: i32,

    /// Maximum batch size
    n_batch_max: u32,
}

unsafe impl Send for Context {}

impl Context {
    /// Creates a new context.
    ///
    /// # Panics
    ///
    /// Panics if the context parameters are invalid.
    pub fn new(model: Model, parameters: &ContextParameters) -> Self {
        parameters.check().unwrap();

        let n_batch_max = parameters.n_batch_max;

        let parameters = parameters.to_ffi();
        tracing::trace!("creating context: {:#?}", parameters);

        let handle =
            unsafe { llama_cpp_sys::llama_new_context_with_model(model.inner.handle, parameters) };

        Self {
            handle,
            model,
            n_tokens: 0,
            n_batch_max,
        }
    }

    /// Creates a decoder for batched decoding.
    ///
    /// # Panics
    ///
    /// Panics if the `batch_size` is 0.
    pub fn decoder(&mut self, batch_size: usize) -> Decoder {
        Decoder::new(self, batch_size)
    }

    /// Return the model this context uses.
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn n_ctx(&mut self) -> u32 {
        unsafe { llama_cpp_sys::llama_n_ctx(self.handle) as u32 }
    }

    /// Returns the internal state as bytes.
    ///
    /// See also: [`load_state`](Context::load_state)
    pub fn save_state(&self) -> Vec<u8> {
        let state_size = unsafe { llama_cpp_sys::llama_get_state_size(self.handle) };

        let mut buf = Vec::with_capacity(state_size);
        buf.resize(state_size, 0);

        // this doesn't seem to return an error ever.
        let bytes_written =
            unsafe { llama_cpp_sys::llama_copy_state_data(self.handle, buf.as_mut_ptr()) };
        buf.truncate(bytes_written);

        buf
    }

    /// Load state from data.
    ///
    /// # Safety
    ///
    /// llama.cpp just asserts that everything is alright, so the program will
    /// crash if the data is incorrect :(
    pub unsafe fn load_state(&mut self, data: &[u8]) -> Result<(), Error> {
        let state_size = unsafe { llama_cpp_sys::llama_get_state_size(self.handle) };

        if data.len() < state_size {
            return Err(Error::InvalidStateDataLength {
                expected: state_size,
                got: data.len(),
            });
        }

        // note: the ffi binding wants *mut for the data, but I'm pretty sure it doesn't
        // modify it.
        llama_cpp_sys::llama_set_state_data(self.handle, data.as_ptr() as *mut _);

        Ok(())
    }

    pub fn set_n_threads(&mut self, n_threads: u32, n_threads_batch: u32) {
        unsafe {
            llama_cpp_sys::llama_set_n_threads(self.handle, n_threads, n_threads_batch);
        }
    }

    /// Decode a batch
    ///
    /// # Panics
    ///
    /// Panics if the number of tokens in the batch are 0 or if they exceed the
    /// max batch size, or if `batch.n_embd` is invalid.
    pub fn decode_batch(&mut self, batch: &Batch) -> Result<Option<DecodeWarning>, Error> {
        tracing::trace!("calling llama_decode");

        assert!(batch.data.n_tokens > 0);
        // or should we rather check that the batch size is smaller than n_batch?
        assert!(batch.data.n_tokens <= self.n_batch_max as _);
        assert!(batch.n_embd() == 0 || batch.n_embd() == self.model.n_embd() as _);

        // remember this for later. some arrays in the context are resized according to
        // this (and n_vocab)
        self.n_tokens = batch.data.n_tokens;

        let ret = unsafe { llama_cpp_sys::llama_decode(self.handle, batch.data) };
        match ret {
            0 => Ok(None),
            1 => Ok(Some(DecodeWarning::NoKvSlot)),
            _ if ret < 0 => Err(Error::DecodeError),
            _ if ret > 0 => Ok(Some(DecodeWarning::Unknown(ret))),
            _ => unreachable!(),
        }
    }

    /// Returns the logits for the i-th token that was decoded with the last
    /// batch.
    ///
    /// # Panics
    ///
    /// Panics if the token index is out of bounds.
    pub fn get_logits_ith<'a>(&'a self, i: i32) -> &'a [f32] {
        assert!(i >= 0 && i < self.n_tokens, "invalid logits index");

        let n_vocab = self.model.n_vocab();
        unsafe {
            let data = llama_cpp_sys::llama_get_logits_ith(self.handle, i);
            assert!(!data.is_null());
            slice::from_raw_parts(data, n_vocab as usize)
        }
    }

    fn get_embeddings<'a>(&'a self) -> Option<&'a [f32]> {
        // according to llama.cpp this array has the size of `n_embd` from the model.
        // but i think it's only initialized if the context parameter `embedding` is
        // true.
        let n_embd = self.model.n_embd();

        unsafe {
            let data = llama_cpp_sys::llama_get_embeddings(self.handle);
            if data.is_null() {
                None
            }
            else {
                Some(slice::from_raw_parts(data, n_embd as usize))
            }
        }
    }

    /// Returns performance information.
    pub fn timings(&self) -> Timings {
        unsafe { llama_cpp_sys::llama_get_timings(self.handle) }
    }

    pub fn reset_timings(&mut self) {
        unsafe {
            llama_cpp_sys::llama_reset_timings(self.handle);
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        tracing::trace!("calling llama_free (context)");
        unsafe { llama_cpp_sys::llama_free(self.handle) }
    }
}

#[derive(Copy, Clone, Debug, thiserror::Error)]
pub enum DecodeWarning {
    #[error("no KV slot found")]
    NoKvSlot,

    #[error("unknown warning: {0}")]
    Unknown(i32),
}

#[derive(Clone, Copy, Debug)]
pub enum KvCacheType {
    F16,
    F32,
}

impl Default for KvCacheType {
    fn default() -> Self {
        Self::F16
    }
}

pub type Timings = llama_cpp_sys::llama_timings;

/// Helper to do batched decoding and sampling.
pub struct Decoder<'a> {
    context: &'a mut Context,
    batch: Batch,
    pos: Pos,
    logits_pos: Option<i32>,
}

impl<'a> Decoder<'a> {
    /// Creates a decoder for batched decoding.
    ///
    /// # Panics
    ///
    /// Panics if the `batch_size` is 0.
    pub fn new(context: &'a mut Context, batch_size: usize) -> Self {
        assert!(batch_size > 0);

        // for now we'll have `n_seq_max=1`.
        let batch = Batch::new_tokens(batch_size, 1);

        Self {
            context,
            batch,
            pos: 0,
            logits_pos: None,
        }
    }

    /// Decode tokens.
    ///
    /// If `logits_for_last_token` is `true` this will produce logits for the
    /// last token. You want this if you want to sample tokens next. For
    /// embedding you can set this to `false`.
    pub fn decode(&mut self, tokens: &[Token], logits_for_last_token: bool) -> Result<(), Error> {
        for token in tokens {
            self.context.model.assert_valid_token(*token);
        }

        unsafe { self.decode_unchecked(tokens, logits_for_last_token) }
    }

    /// Decode tokens.
    ///
    /// # Safety
    ///
    /// This doesn't check if the provided tokens are valid. If you pass invalid
    /// tokens, the llama.cpp backend might crash.
    pub unsafe fn decode_unchecked(
        &mut self,
        tokens: &[Token],
        logits_for_last_token: bool,
    ) -> Result<(), Error> {
        for chunk in &IsLast::new(tokens.into_iter()).chunks(self.batch.size()) {
            self.batch.clear();

            for (token, is_last) in chunk {
                // add token to batch
                self.batch.add_token(*token, self.pos, &[0], is_last);

                // generate logits for the last position
                if is_last && logits_for_last_token {
                    self.logits_pos = Some(self.pos);
                }

                self.pos += 1;
            }

            // decode
            if let Some(warning) = self.context.decode_batch(&self.batch)? {
                // todo
                tracing::warn!("{}", warning);
                //warnings.push(warning);
            }
        }

        Ok(())
    }

    /// Samples the next token and decodes it.
    ///
    /// # Panics
    ///
    /// Panics if no tokens have been pushed for decoding.
    pub fn sample_and_decode(&mut self, sampler: &mut Sampler) -> Result<SampleResult, Error> {
        // todo: this should return an error, right?
        let logits_pos = self
            .logits_pos
            .expect("sample was called before a prompt was feed into inference");
        let logits = self.context.get_logits_ith(logits_pos);

        // sample next token
        let candidates = Candidates::from_logits(logits);
        let next_token = sampler.sample(candidates, &mut self.context);

        sampler.push_previous(next_token);

        // clear batch to feed in sampled token
        self.batch.clear();

        // add sampled token to batch
        // note: since we sampled this token, we can assume its valid.
        unsafe { self.batch.add_token(next_token, self.pos, &[0], true) };

        // generate logits for the only token that is in the batch
        self.logits_pos = Some(0);

        self.pos += 1;

        // decode
        let warning = self.context.decode_batch(&mut self.batch)?;

        Ok(SampleResult {
            token: next_token,
            warning,
        })
    }

    pub fn get_embeddings(&self) -> Option<&[f32]> {
        self.context.get_embeddings()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SampleResult {
    pub token: Token,
    pub warning: Option<DecodeWarning>,
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::context::ContextParameters,
        utils::test::{
            context,
            model,
        },
    };

    #[test]
    #[should_panic]
    fn it_rejects_invalid_tokens() {
        let mut context = context();
        let mut decoder = context.decoder(512);
        decoder.decode(&[i32::MAX], false).unwrap();
    }

    #[test]
    fn get_embeddings_from_new_context() {
        let context = model().context(&ContextParameters {
            embedding: true,
            ..Default::default()
        });
        let _embeddings = context
            .get_embeddings()
            .expect("context returned no embeddings");
    }

    #[test]
    fn get_embeddings_from_decoded_tokens() {
        let model = model();
        let tokens = model.tokenize("Hello World", true, false);

        let mut context = model.context(&ContextParameters {
            embedding: true,
            ..Default::default()
        });

        let mut decoder = context.decoder(512);
        decoder.decode(&tokens, false).unwrap();

        let embeddings = context
            .get_embeddings()
            .expect("context returned no embeddings");
        assert_ne!(embeddings[0], 0.0);
    }
}
