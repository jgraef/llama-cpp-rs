//! LLM context

use std::slice;

use itertools::Itertools;

use super::{
    batch::Batch,
    default_n_threads,
    model::Model,
    sampling::{
        Candidates,
        Sampler,
    },
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
            n_ctx: self.n_ctx.unwrap_or_default(),
            n_batch: self.n_batch_max,
            n_threads: self.n_threads.unwrap_or(default_n_threads()),
            n_threads_batch: self.n_threads_batch.unwrap_or(default_n_threads()),
            rope_freq_base: self.rope_freq_base.unwrap_or_default(),
            rope_freq_scale: self.rope_fre_scale.unwrap_or_default(),
            mul_mat_q: self.mul_mat_q,
            f16_kv: matches!(self.kv_cache_type, KvCacheType::F16),
            logits_all: false, // we don't need this, since we use the batch API.
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
    n_tokens: u32,

    /// Context size
    n_ctx: u32,

    /// Total number of tokens decoded
    n_past: u32,

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
        assert!(
            !handle.is_null(),
            "llama_new_context_with_model returned NULL"
        );

        let n_ctx = unsafe { llama_cpp_sys::llama_n_ctx(handle) } as _;

        Self {
            handle,
            model,
            n_tokens: 0,
            n_ctx,
            n_past: 0,
            n_batch_max,
        }
    }

    /// Creates a decoder for batched decoding.
    ///
    /// # Panics
    ///
    /// Panics if the `batch_size` is 0.
    pub fn decoder(&mut self, batch_size: impl Into<Option<u32>>) -> Decoder {
        let batch_size = batch_size.into().unwrap_or(self.n_batch_max);
        Decoder::new(self, batch_size)
    }

    /// Return the model this context uses.
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn n_ctx(&mut self) -> u32 {
        self.n_ctx
    }

    pub fn n_batch_max(&self) -> u32 {
        self.n_batch_max
    }

    pub fn n_past(&self) -> u32 {
        self.n_past
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
    /// This returns an error if the call to `llama_decode` fails, or if the kv
    /// cache is full.
    ///
    /// # Panics
    ///
    /// Panics if the number of tokens in the batch exceeds the
    /// max batch size, or if `batch.n_embd` is invalid.
    pub fn decode_batch(&mut self, batch: &Batch) -> Result<(), Error> {
        tracing::trace!("calling llama_decode");

        // if the batch is empty, we do nothing.
        if batch.data.n_tokens == 0 {
            return Ok(());
        }

        assert!(batch.data.n_tokens > 0);
        // or should we rather check that the batch size is smaller than n_batch?
        assert!(batch.data.n_tokens <= self.n_batch_max as _);
        assert!(batch.n_embd() == 0 || batch.n_embd() == self.model.n_embd() as _);

        // remember this for later. some arrays in the context are resized according to
        // this (and n_vocab)
        self.n_tokens = batch.data.n_tokens as _;

        // we need to know when we run out of context.
        self.n_past += batch.data.n_tokens as u32;

        let ret = unsafe { llama_cpp_sys::llama_decode(self.handle, batch.data) };

        match ret {
            0 => Ok(()),
            1 => Err(Error::KvCacheFull),
            _ if ret < 0 => Err(Error::DecodeError),
            _ if ret > 0 => panic!("llama_decode returned unknown warning: {ret}"),
            _ => unreachable!(),
        }
    }

    pub fn swap(&mut self, n_keep: u32) {
        /* taken from llama.cpp/examples/main/main.cpp

            const int n_left    = n_past - params.n_keep - 1;
            const int n_discard = n_left/2;

            llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
            llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

            n_past -= n_discard;
        */

        println!("n_keep = {n_keep}, n_past={}", self.n_past);
        let n_left = (self.n_past as i32) - (n_keep as i32) - 1;
        let n_discard = n_left / 2;
        println!("n_left = {n_left}, n_discard = {n_discard}");
        assert!(n_discard > 0);

        // todo
        let seq_id = 0;
        let n_keep = n_keep as i32;

        tracing::debug!(n_keep, n_left, n_discard, "context swap");

        unsafe {
            llama_cpp_sys::llama_kv_cache_seq_rm(
                self.handle,
                seq_id,
                n_keep + 1,
                n_keep + n_discard + 1,
            );
            llama_cpp_sys::llama_kv_cache_seq_shift(
                self.handle,
                seq_id,
                n_keep + n_discard + 1,
                self.n_past as i32,
                -n_discard,
            );
        }

        self.n_past = self.n_past
            .checked_sub(n_discard as u32)
            .expect("bug: discarded more than we have");
    }

    pub fn swap_if_needed(&mut self, n_tokens: u32, n_keep: u32) {
        if self.n_past + n_tokens > self.n_ctx {
            self.swap(n_keep);
        }
    }

    /// Returns the logits for the i-th token that was decoded with the last
    /// batch.
    ///
    /// # Panics
    ///
    /// Panics if the token index is out of bounds.
    pub fn get_logits_ith<'a>(&'a self, i: u32) -> &'a [f32] {
        assert!(i < self.n_tokens, "invalid logits index");

        let n_vocab = self.model.n_vocab();
        unsafe {
            // this array is n_tokens * n_vocab elements large.
            let data = llama_cpp_sys::llama_get_logits_ith(self.handle, i as _);
            assert!(!data.is_null());
            slice::from_raw_parts(data, n_vocab as usize)
        }
    }

    pub fn get_embeddings<'a>(&'a self) -> Option<&'a [f32]> {
        // according to llama.cpp this array has the size of `n_embd` from the model.
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
// todo: rename
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
    pub fn new(context: &'a mut Context, batch_size: u32) -> Self {
        assert!(
            batch_size > 0 && batch_size <= context.n_batch_max,
            "invalid batch size"
        );

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
    ///
    /// # Panics
    ///
    /// Panics if one if the passed tokens is invalid.
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
        for chunk in IsLast::new(tokens.into_iter())
            .chunks(self.batch.size() as _)
            .into_iter()
        {
            self.batch.clear();

            for (i, (token, is_last)) in chunk.into_iter().enumerate() {
                // add token to batch
                self.batch.add_token(*token, self.pos, &[0], is_last);

                // generate logits for the last position
                if is_last && logits_for_last_token {
                    self.logits_pos = Some(i as _);
                }

                self.pos += 1;
            }

            // todo: make this optional
            self.context.swap_if_needed(self.batch.data.n_tokens as u32, 4);

            // decode
            self.context.decode_batch(&self.batch)?;
        }

        Ok(())
    }

    /// Samples the next token and decodes it.
    ///
    /// # Panics
    ///
    /// Panics if no tokens have been pushed for decoding.
    pub fn sample_and_decode(&mut self, sampler: &mut Sampler) -> Result<Token, Error> {
        // todo: this should return an error, right?
        let logits_pos = self
            .logits_pos
            .expect("batch hasn't decoded to any logits yet");
        let logits = self.context.get_logits_ith(logits_pos as _);

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
        self.context.decode_batch(&mut self.batch)?;

        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use lipsum::lipsum;

    use super::*;
    use crate::{
        backend::{
            context::ContextParameters,
            sampling::Sampler,
        },
        utils::test::{
            context,
            model,
        },
    };

    fn sample_until_eos<'a>(decoder: &mut Decoder<'a>, sampler: &mut Sampler) -> Vec<Token> {
        let mut tokens = vec![];
        loop {
            let token = decoder.sample_and_decode(sampler).unwrap();
            if token == decoder.context.model().token_eos() {
                break;
            }
            tokens.push(token);
        }
        tokens
    }

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
            seed: Some(1234),
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

    #[test]
    fn decode_twice() {
        let mut context = context();
        let tokens = context.model().tokenize("Hello World", true, false);
        let mut decoder = context.decoder(tokens.len() as u32);
        decoder.decode(&tokens, false).unwrap();
        decoder.decode(&tokens, false).unwrap();
    }

    #[test]
    fn it_decodes_long_input() {
        let mut context = context();
        println!("n_ctx = {}", context.n_ctx());

        let tokens = context.model().tokenize(&lipsum(512), true, false);
        let mut decoder = context.decoder(64);
        decoder.decode(&tokens, true).unwrap();

        let mut sampler = Sampler::new(Default::default());
        let tokens = sample_until_eos(&mut decoder, &mut sampler);
        println!("{:?}", tokens);
        //assert_eq!(token, 23);
    }

    #[test]
    fn batch_size_doesnt_influence_output() {
        let tokens = model().tokenize("Hello World", true, false);

        let gen = |batch_size: u32| {
            let mut context = context();
            let mut decoder = context.decoder(Some(batch_size));
            decoder.decode(&tokens, true).unwrap();
            let mut sampler = Sampler::new(Default::default());
            sample_until_eos(&mut decoder, &mut sampler)
        };

        let output1 = gen(512);
        let output2 = gen(32);
        assert_eq!(output1, output2);
    }
}
