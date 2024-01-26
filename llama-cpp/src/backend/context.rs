//! LLM context

use std::{
    collections::HashMap,
    slice,
};

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
    SeqId,
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

    /// The number of tokens in the last batch_decode. We need to know this
    /// because some buffers' size depend on it.
    n_tokens: u32,

    /// Context size
    n_ctx: u32,

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
            n_batch_max,
        }
    }

    pub fn manager(&mut self) -> ContextManager {
        ContextManager::new(self, std::cmp::min(self.n_ctx, self.n_batch_max))
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
    pub unsafe fn decode_batch_unchecked(&mut self, batch: &Batch) -> Result<(), DecodeError> {
        // if the batch is empty, we do nothing.
        if batch.data.n_tokens == 0 {
            return Ok(());
        }

        tracing::trace!("calling llama_decode");

        assert!(batch.data.n_tokens > 0);
        // or should we rather check that the batch size is smaller than n_batch?
        assert!(batch.data.n_tokens <= self.n_batch_max as _);
        assert!(batch.n_embd() == 0 || batch.n_embd() == self.model.n_embd() as _);

        // remember this for later. some arrays in the context are resized according to
        // this (and n_vocab)
        self.n_tokens = batch.data.n_tokens as _;

        let ret = unsafe { llama_cpp_sys::llama_decode(self.handle, batch.data) };

        match ret {
            0 => Ok(()),
            1 => Err(DecodeError::KvCacheFull),
            _ if ret < 0 => Err(DecodeError::Failed(ret)),
            _ if ret > 0 => Err(DecodeError::UnknownWarning(ret)),
            _ => unreachable!(),
        }
    }

    /*fn swap(&mut self, n_keep: u32) -> u32 {
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

        self.n_past = self
            .n_past
            .checked_sub(n_discard as u32)
            .expect("bug: discarded more than we have");

        n_discard as u32
    }

    pub fn swap_if_needed(&mut self, n_tokens: u32, n_keep: u32) -> u32 {
        if self.n_past + n_tokens > self.n_ctx {
            self.swap(n_keep)
        }
        else {
            0
        }
    }*/

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

#[derive(Copy, Clone, Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("decode failed: {0}")]
    Failed(i32),

    #[error("kv cache is full")]
    KvCacheFull,

    #[error("unknown warning: {0}")]
    UnknownWarning(i32),
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

#[derive(Clone, Debug, Default)]
struct SequenceState {
    position: Pos,
    logits_indices: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

/// Helper to do batched decoding on a [`Context`].
pub struct ContextManager<'a> {
    context: &'a mut Context,
    batch: Batch,
    sequences: HashMap<SeqId, SequenceState>,
}

impl<'a> ContextManager<'a> {
    pub fn new(context: &'a mut Context, batch_size: u32) -> Self {
        // `n_seq_max` specifies how many sequences a token can belong to, not how many
        // sequences we can have.
        let batch = Batch::new_tokens(batch_size, 1);

        Self {
            context,
            batch,
            sequences: HashMap::new(),
        }
    }

    pub unsafe fn push_token_unchecked(
        &mut self,
        token: Token,
        seq_id: SeqId,
        calculate_logits: bool,
    ) -> Result<(), DecodeError> {
        let sequence = self.sequences.entry(seq_id).or_default();

        // remember logits index
        if calculate_logits {
            sequence.logits_indices.push(self.batch.n_tokens());
        }

        // add token to batch
        self.batch
            .add_token(token, sequence.position, &[seq_id], calculate_logits);

        // increment sequence's position
        sequence.position += 1;

        // decode if batch is full
        if self.batch.is_full() {
            self.decode()?;
        }

        Ok(())
    }

    pub fn push_token(
        &mut self,
        token: Token,
        seq_id: SeqId,
        calculate_logits: bool,
    ) -> Result<(), DecodeError> {
        self.context.model.assert_valid_token(token);
        unsafe { self.push_token_unchecked(token, seq_id, calculate_logits) }
    }

    pub fn push_tokens(
        &mut self,
        tokens: &[Token],
        seq_id: SeqId,
        calculate_logits_for_last_token: bool,
    ) -> Result<(), DecodeError> {
        for (token, is_last) in IsLast::new(tokens.iter()) {
            self.push_token(*token, seq_id, calculate_logits_for_last_token && is_last)?;
        }

        Ok(())
    }

    pub fn decode(&mut self) -> Result<(), DecodeError> {
        if self.batch.is_empty() {
            return Ok(());
        }

        // since we constructed the batch it should be safe to decode it.
        unsafe { self.context.decode_batch_unchecked(&self.batch)? };

        // clear batch
        self.batch.clear();

        // copy logits
        for sequence in self.sequences.values_mut() {
            for logits_index in &sequence.logits_indices {
                let logits = self.context.get_logits_ith(*logits_index);
                sequence.logits.push(logits.to_owned());
            }
            sequence.logits_indices.clear();
        }

        Ok(())
    }

    pub fn take_logits(&mut self, seq_id: SeqId) -> Vec<Vec<f32>> {
        let Some(sequence) = self.sequences.get_mut(&seq_id)
        else {
            return vec![];
        };
        std::mem::replace(&mut sequence.logits, vec![])
    }

    pub fn delete_sequence(&mut self, seq_id: SeqId) {
        tracing::debug!(seq_id, "delete sequence");

        self.sequences.remove(&seq_id);

        // remove sequence from kv cache
        unsafe {
            llama_cpp_sys::llama_kv_cache_seq_rm(self.context.handle, seq_id, -1, -1);
        }
    }

    pub fn sample(&mut self, seq_id: SeqId, sampler: &mut Sampler) -> Result<Token, DecodeError> {
        self.decode()?;

        // get last logits and discard the rest.
        let logits = self
            .take_logits(seq_id)
            .pop()
            .expect("no logits calculated");
        let candidates = Candidates::from_logits(&logits);

        let token = sampler.sample(candidates, &mut self.context);

        self.push_token(token, seq_id, true)?;

        Ok(token)
    }

    pub fn copy_sequence(&mut self, from_sequence_id: SeqId, to_sequence_id: SeqId) {
        tracing::debug!(from_sequence_id, to_sequence_id, "copy sequence");

        let Some(from_sequence) = self.sequences.get(&from_sequence_id)
        else {
            // if we copy from a sequence we have no data for, we are basically copying the
            // empty sequence, so we just return :3
            return;
        };

        // copy sequence state
        let to_sequence = from_sequence.clone();
        self.sequences.insert(to_sequence_id, to_sequence);

        // copy kv-cache
        unsafe {
            llama_cpp_sys::llama_kv_cache_seq_cp(
                self.context.handle,
                from_sequence_id,
                to_sequence_id,
                -1,
                -1,
            );
        }
    }
}

impl<'a> Drop for ContextManager<'a> {
    fn drop(&mut self) {
        // clear kv cache
        unsafe {
            llama_cpp_sys::llama_kv_cache_tokens_rm(self.context.handle, -1, -1);
        }
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

    fn sample_until_eos<'a>(
        context_manager: &mut ContextManager<'a>,
        sampler: &mut Sampler,
    ) -> Vec<Token> {
        let mut tokens = vec![];
        loop {
            let token = context_manager.sample(0, sampler).unwrap();
            if token == context_manager.context.model().token_eos() {
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
        let mut context_manager = context.manager();
        context_manager.push_token(i32::MAX, 0, false).unwrap();
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

        let mut context_manager = context.manager();
        context_manager.push_tokens(&tokens, 0, false).unwrap();
        context_manager.decode().unwrap();
        drop(context_manager);

        let embeddings = context
            .get_embeddings()
            .expect("context returned no embeddings");
        assert_ne!(embeddings[0], 0.0);
    }

    #[test]
    #[ignore = "context swapping doesn't work yet"]
    fn it_decodes_long_input() {
        let mut context = context();
        println!("n_ctx = {}", context.n_ctx());

        let tokens = context.model().tokenize(&lipsum(512), true, false);
        let mut context_manager = ContextManager::new(&mut context, 64);
        context_manager.push_tokens(&tokens, 0, true).unwrap();

        let mut sampler = Sampler::new(Default::default());
        let tokens = sample_until_eos(&mut context_manager, &mut sampler);
        println!("{:?}", tokens);
        //assert_eq!(token, 23);
    }

    #[test]
    fn batch_size_doesnt_influence_output() {
        let tokens = model().tokenize("Hello World", true, false);

        let gen = |batch_size: u32| {
            let mut context = context();
            let mut context_manager = ContextManager::new(&mut context, batch_size);
            context_manager.push_tokens(&tokens, 0, true).unwrap();
            let mut sampler = Sampler::new(Default::default());
            sample_until_eos(&mut context_manager, &mut sampler)
        };

        let output1 = gen(512);
        let output2 = gen(32);
        assert_eq!(output1, output2);
    }
}
