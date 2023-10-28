use std::{
    marker::PhantomData,
    slice,
};

use super::{
    batch::Batch,
    model::Model,
    Error,
    DEFAULT_N_THREADS,
    DEFAULT_SEED,
};

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

#[derive(Clone, Debug)]
pub struct ContextParameters {
    /// RNG seed, `None` for random.
    pub seed: Option<u32>,

    /// Text context. If set to `None`, it will take the parameter from the
    /// model.
    pub n_ctx: Option<u32>,

    /// Prompt processing maximum batch size.
    pub n_batch: u32,

    /// Number of threads to use for generation.
    pub n_threads: Option<u32>,

    /// Number of threads to use for batch processing.
    pub n_threads_batch: Option<u32>,

    /// RoPE base frequency. If set to `None`, it will take the parameter from
    /// the model.
    ///
    /// See https://github.com/ggerganov/llama.cpp/pull/2054
    pub rope_freq_base: Option<f32>,

    // RoPE frequency scaling factor. If set to `None`, it will take the parameter from the model.
    /// See https://github.com/ggerganov/llama.cpp/pull/2054
    pub rope_fre_scale: Option<f32>,

    /// If true, use experimental mul_mat_q kernels
    pub mul_mat_q: bool,

    /// Use fp16 for KV cache, fp32 otherwise.
    pub kv_cache_type: KvCacheType,

    /// The `llama_eval()` call computes all logits, not just the last one.
    pub logits_all: bool,

    /// Embedding mode only
    pub embedding_only: bool,
}

impl ContextParameters {
    fn to_ffi(&self) -> llama_cpp_sys::llama_context_params {
        llama_cpp_sys::llama_context_params {
            seed: self.seed.unwrap_or(u32::MAX),
            n_ctx: self.n_ctx.unwrap_or(0),
            n_batch: self.n_batch,
            n_threads: self.n_threads.unwrap_or(*DEFAULT_N_THREADS),
            n_threads_batch: self.n_threads_batch.unwrap_or(*DEFAULT_N_THREADS),
            rope_freq_base: self.rope_freq_base.unwrap_or_default(),
            rope_freq_scale: self.rope_fre_scale.unwrap_or_default(),
            mul_mat_q: self.mul_mat_q,
            f16_kv: matches!(self.kv_cache_type, KvCacheType::F16),
            logits_all: self.logits_all,
            embedding: self.embedding_only,
        }
    }
}

impl Default for ContextParameters {
    fn default() -> Self {
        Self {
            seed: Some(DEFAULT_SEED),
            n_ctx: Some(512),
            n_batch: 512,
            n_threads: None,
            n_threads_batch: None,
            rope_freq_base: None,
            rope_fre_scale: None,
            mul_mat_q: true,
            kv_cache_type: Default::default(),
            logits_all: false,
            embedding_only: false,
        }
    }
}

pub struct Context {
    pub(super) handle: *mut llama_cpp_sys::llama_context,

    /// This is important, since it keeps the model from being dropped, while we
    /// still have a context to it.
    model: Model,
}

unsafe impl Send for Context {}

impl Context {
    pub(super) fn new(model: Model, parameters: &ContextParameters) -> Self {
        let parameters = parameters.to_ffi();
        tracing::trace!("creating context: {:#?}", parameters);

        let handle =
            unsafe { llama_cpp_sys::llama_new_context_with_model(model.inner.handle, parameters) };

        Self { handle, model }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn n_ctx(&mut self) -> u32 {
        unsafe { llama_cpp_sys::llama_n_ctx(self.handle) as u32 }
    }

    pub fn get_state(&mut self) -> Vec<u8> {
        let state_size = unsafe { llama_cpp_sys::llama_get_state_size(self.handle) };

        let mut buf = Vec::with_capacity(state_size);
        buf.resize(state_size, 0);

        // this doesn't seem to return an error ever.
        let bytes_written =
            unsafe { llama_cpp_sys::llama_copy_state_data(self.handle, buf.as_mut_ptr()) };
        buf.truncate(bytes_written);

        buf
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), Error> {
        let state_size = unsafe { llama_cpp_sys::llama_get_state_size(self.handle) };
        if data.len() < state_size {
            return Err(Error::InvalidStateDataLength {
                expected: state_size,
                got: data.len(),
            });
        }

        unsafe {
            // note: the ffi binding wants *mut, but I'm pretty sure it doesn't modify it.
            llama_cpp_sys::llama_set_state_data(self.handle, data.as_ptr() as *mut _);
        }

        Ok(())
    }

    pub fn set_n_threads(&mut self, n_threads: u32, n_threads_batch: u32) {
        unsafe {
            llama_cpp_sys::llama_set_n_threads(self.handle, n_threads, n_threads_batch);
        }
    }

    pub(super) fn decode(&mut self, batch: &Batch) -> Result<Option<DecodeWarning>, Error> {
        tracing::trace!("calling llama_decode");

        assert!(batch.data.n_tokens > 0);

        let ret = unsafe { llama_cpp_sys::llama_decode(self.handle, batch.data) };
        match ret {
            0 => Ok(None),
            1 => Ok(Some(DecodeWarning::NoKvSlot)),
            _ if ret < 0 => Err(Error::DecodeError),
            _ if ret > 0 => Ok(Some(DecodeWarning::Unknown(ret))),
            _ => unreachable!(),
        }
    }

    pub(super) fn get_logits_ith<'a>(&'a self, i: i32) -> &'a [f32] {
        // todo: make sure this is always safe!

        let n_vocab = self.model.n_vocab();
        unsafe {
            let data = llama_cpp_sys::llama_get_logits_ith(self.handle, i);
            slice::from_raw_parts(data, n_vocab as usize)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        tracing::trace!("calling llama_free (context)");
        unsafe { llama_cpp_sys::llama_free(self.handle) }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeWarning {
    #[error("no KV slot found")]
    NoKvSlot,

    #[error("unknown warning: {0}")]
    Unknown(i32),
}

pub struct Logits<'a> {
    data: *const f32,
    n_vocab: usize,
    _lifetime: PhantomData<&'a ()>,
}
