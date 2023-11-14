//! Large Language Model

use std::{
    ffi::{
        c_void,
        CStr,
        CString,
    },
    path::Path,
    ptr,
    sync::Arc,
};

use super::{
    context::{
        Context,
        ContextParameters,
    },
    ffi_path,
    llama_init,
    Error,
    Token,
    TokenType,
    VocabType,
    MAX_DEVICES,
};

#[derive(Clone, Debug)]
pub struct ModelParameters {
    pub n_gpu_layers: Option<u32>,
    pub main_gpu: u32,
    pub tensor_split: Option<[f32; MAX_DEVICES]>,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl ModelParameters {
    /// # Safety
    ///
    /// Caller must ensure that the returned pointer only lives as long as the
    /// `ModelParameters` struct lives.
    unsafe fn to_ffi(
        &self,
        progress_callback: llama_cpp_sys::llama_progress_callback,
        progress_callback_user_data: *mut c_void,
    ) -> llama_cpp_sys::llama_model_params {
        llama_cpp_sys::llama_model_params {
            n_gpu_layers: self.n_gpu_layers.map(|n| n as i32).unwrap_or(-1),
            main_gpu: self.main_gpu as i32,
            tensor_split: self
                .tensor_split
                .as_ref()
                .map(|t| t.as_ptr())
                .unwrap_or(ptr::null()),
            progress_callback,
            progress_callback_user_data,
            vocab_only: self.vocab_only,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            n_gpu_layers: None,
            main_gpu: 0,
            tensor_split: None,
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

pub(super) struct ModelInner {
    pub(super) handle: *mut llama_cpp_sys::llama_model,
    n_vocab: i32,
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe {
            tracing::trace!("calling llama_free_model");
            llama_cpp_sys::llama_free_model(self.handle);
        }
    }
}

// todo: check these?
// i'm somewhat sure that the model is only read by llama.cpp
unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

/// A large language model
///
/// This internally uses an `Arc`, so it's cheap to clone.
#[derive(Clone)]
pub struct Model {
    pub(super) inner: Arc<ModelInner>,
}

impl Model {
    /// Loads the model from a file. This method is blocking. If you want to
    /// load the model asynchronously, use
    /// [`ModelLoader`](crate::loader::ModelLoader)
    pub fn load<P: FnMut(f32)>(
        path: impl AsRef<Path>,
        parameters: &ModelParameters,
        mut progress: P,
    ) -> Result<Self, Error> {
        llama_init();

        let path = path.as_ref();
        tracing::debug!("loading model: {}", path.display());

        // we pass a pointer to the Box rather than the boxes pointer itself, so that we
        // definitely clean up its memory when it's dropped.
        let progress_callback_user_data = &mut progress as *mut _ as *mut c_void;
        let c_path = ffi_path(path)?;

        unsafe extern "C" fn progress_callback<P: FnMut(f32)>(
            progress: f32,
            user_data: *mut c_void,
        ) {
            let f = &mut *(user_data as *mut P);
            f(progress);
        }

        let handle = unsafe {
            let parameters =
                parameters.to_ffi(Some(progress_callback::<P>), progress_callback_user_data);
            llama_cpp_sys::llama_load_model_from_file(c_path.as_ptr(), parameters)
        };

        if handle.is_null() {
            tracing::debug!("llama_load_model_from file returned NULL");
            Err(Error::ModelLoadFailed {
                path: path.to_owned(),
            })
        }
        else {
            tracing::debug!("model loaded");

            let n_vocab = unsafe { llama_cpp_sys::llama_n_vocab(handle) };

            Ok(Model {
                inner: Arc::new(ModelInner { handle, n_vocab }),
            })
        }
    }

    /// Creates a new context.
    ///
    /// # Panics
    ///
    /// Panics if the context parameters are invalid.
    pub fn context(&self, context_parameters: &ContextParameters) -> Context {
        Context::new(self.clone(), context_parameters)
    }

    pub fn vocab_type(&self) -> VocabType {
        VocabType::from_ffi(unsafe { llama_cpp_sys::llama_vocab_type(self.inner.handle) })
    }

    pub fn n_vocab(&self) -> u32 {
        self.inner.n_vocab as _
    }

    pub fn n_ctx_train(&self) -> u32 {
        unsafe { llama_cpp_sys::llama_n_ctx_train(self.inner.handle) as u32 }
    }

    pub fn n_embd(&self) -> u32 {
        unsafe { llama_cpp_sys::llama_n_embd(self.inner.handle) as u32 }
    }

    /// Get the model's RoPE frequency scaling factor.
    pub fn rope_freq_scale_train(&self) -> f32 {
        unsafe { llama_cpp_sys::llama_rope_freq_scale_train(self.inner.handle) }
    }

    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_sys::llama_model_size(self.inner.handle) }
    }

    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_sys::llama_model_n_params(self.inner.handle) }
    }

    pub fn token_bos(&self) -> Token {
        unsafe { llama_cpp_sys::llama_token_bos(self.inner.handle) }
    }

    pub fn token_eos(&self) -> Token {
        unsafe { llama_cpp_sys::llama_token_eos(self.inner.handle) }
    }

    pub fn token_nl(&self) -> Token {
        unsafe { llama_cpp_sys::llama_token_nl(self.inner.handle) }
    }

    pub fn tokenize(&self, text: &str, add_bos: bool, allow_special: bool) -> Vec<Token> {
        tracing::trace!(text, "tokenize");

        // this is exactly how llama.cpp/common.cpp does it
        let token_count = text.len() + add_bos.then_some(1).unwrap_or_default();

        let text_len = text.len() as i32;
        let text = text.as_ptr() as _;

        tracing::trace!(token_count, text_len);

        let mut token_buf = Vec::with_capacity(token_count);
        token_buf.resize(token_count, 0);

        tracing::trace!("calling llama_tokenize");
        let ret = unsafe {
            llama_cpp_sys::llama_tokenize(
                self.inner.handle,
                text,
                text_len,
                token_buf.as_mut_ptr(),
                token_count as _,
                add_bos,
                allow_special,
            )
        };
        tracing::trace!(ret);

        // todo: are there cases where you could have more tokens that input characters?
        if ret < 0 {
            panic!("fixme: expected not more tokens than input characters.");
        }

        let token_count = ret as _;
        token_buf.truncate(token_count);

        token_buf
    }

    /// Writes the token bytes into an output buffer.
    ///
    /// # Panics
    ///
    /// Panics if the token is not in the vocabulary.
    pub fn token_to_piece(&self, token: Token, output: &mut Vec<u8>) {
        self.assert_valid_token(token);

        const BUF_SIZE: usize = 32;
        let mut buf = [0u8; BUF_SIZE];

        let n = unsafe {
            llama_cpp_sys::llama_token_to_piece(
                self.inner.handle,
                token,
                buf.as_mut_ptr() as _,
                BUF_SIZE as _,
            ) as usize
        };

        output.extend_from_slice(&buf[..n]);
    }

    /// Returns the text for a token. If you want to turn a token stream into
    /// text, you should use [`Model::token_decoder`]. This turns special tokens
    /// into a special representation. E.g. it turns a newline into
    /// '<0x0A>'.
    pub fn get_token_text<'a>(&'a self, token: Token) -> Result<&'a str, Error> {
        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_sys::llama_token_get_text(
                self.inner.handle,
                token,
            ))
        };

        Ok(c_str.to_str()?)
    }

    pub fn get_token_score(&self, token: Token) -> f32 {
        self.assert_valid_token(token);

        unsafe { llama_cpp_sys::llama_token_get_score(self.inner.handle, token) }
    }

    pub fn get_token_type(&self, token: Token) -> TokenType {
        self.assert_valid_token(token);

        let ty = unsafe { llama_cpp_sys::llama_token_get_type(self.inner.handle, token) };

        TokenType::from_ffi(ty)
    }

    /// Creates a new token decoder.
    pub fn token_decoder(&self) -> TokenDecoder {
        TokenDecoder::new(self.clone())
    }

    pub fn get_tensor<'a>(&'a self, name: &str) -> Option<&'a Tensor> {
        let name = CString::new(name).expect("tensor name contains a null byte");

        unsafe {
            let tensor = llama_cpp_sys::llama_get_model_tensor(self.inner.handle, name.as_ptr());
            if tensor.is_null() {
                None
            }
            else {
                Some(&*tensor)
            }
        }
    }

    pub(crate) fn assert_valid_token(&self, token: Token) {
        if !self.is_valid_token(token) {
            panic!("invalid token: {token}");
        }
    }

    /// Returns if the token is valid for this model.
    pub fn is_valid_token(&self, token: Token) -> bool {
        token >= 0 && token < self.inner.n_vocab
    }
}

/// Buffer to decode tokens.
///
/// [`Model`]s produce [`Token`]s and these don't necessarily translate directly
/// to valid UTF-8. It might be necessary to first buffer a few tokens (decoded
/// into bytes).
pub struct TokenDecoder {
    model: Model,
    buf: Vec<u8>,
    pub strip_leading_space: bool,
}

impl TokenDecoder {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            buf: Vec::with_capacity(32),
            strip_leading_space: false,
        }
    }

    pub fn push_token(&mut self, token: Token) {
        self.model.token_to_piece(token, &mut self.buf);
    }

    /// Returns the internal buffer as `String`. This returns `None` if the
    /// UTF-8 parsing fails, which means that more tokens are needed.
    pub fn pop_string(&mut self) -> Option<String> {
        let data = std::mem::replace(&mut self.buf, vec![]);

        let data = if self.strip_leading_space {
            self.strip_leading_space = false;
            if let Some(stripped) = data.strip_prefix(b" ") {
                stripped.to_owned()
            }
            else {
                data
            }
        }
        else {
            data
        };

        match String::from_utf8(data) {
            Ok(s) => Some(s),
            Err(e) => {
                // put the bytes back into the buffer
                // note, that they are now already stripped, which is fine.
                self.buf = e.into_bytes();
                None
            }
        }
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn decode(&mut self, token: Token) -> Option<String> {
        self.push_token(token);
        let s = self.pop_string()?;
        self.clear();
        Some(s)
    }
}

/// A GGML tensor.
pub type Tensor = llama_cpp_sys::ggml_tensor;
