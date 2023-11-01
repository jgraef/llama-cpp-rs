//! Low-level bindings for llama.cpp
//!
//! This module provides low-level wrappers around the FFI interface. Everything
//! here is synchronous, meaning it can't be awaited and will block the current
//! thread. Since LLM inference is a slow process, it's recommended to use the
//! high-level interface, which runs the synchronous code in a separate thread.
//!
//! # Example
//!
//! ```
//! # use std::io::{stdout, Write};
//! # use llama_cpp::{backend::{model::Model, context::{Context, Decoder}, sampling::Sampler}, Error};
//! # fn main() -> Result<(), Error> {
//! # let model_path = "../data/TinyLLama-v0.gguf";
//! // load model (with default model parameters)
//! let model = Model::load(
//!     model_path,
//!     &Default::default(),
//!     |progress| println!("progress: {} %", progress * 100.0)
//! )?;
//!
//! // create context (with default context parameters)
//! let mut context = model.context(&Default::default());
//!
//! // create batched decoder (with batch size 512)
//! let mut decoder = context.decoder(512);
//!
//! // prompt
//! let prompt = "The capital of France is";
//! print!("{prompt}");
//! stdout().flush()?;
//!
//! // tokenize and decode prompt
//! let tokens = model.tokenize(prompt, true, false);
//! decoder.decode(&tokens, true);
//!
//! // create sampler
//! let mut sampler = Sampler::new(Default::default());
//!
//! let mut token_decoder = model.token_decoder();
//! loop {
//!     // sample and decode the sampled token
//!     let token = decoder.sample_and_decode(&mut sampler)?.token;
//!
//!     // if it's an eos token, we are done.
//!     if token == model.token_eos() {
//!         break;
//!     }
//!
//!     // decode token
//!     token_decoder.push_token(token);
//!     if let Some(text) = token_decoder.pop_string() {
//!         print!("{text}");
//!         stdout().flush()?;
//!     }
//! }
//!
//! # Ok(())
//! # }
//! ```
//!
//! # Safety
//!
//! We tried to make sure all the public interfaces in this module are safe to
//! use and don't cause crashes or undefined behaviour, but we might have missed
//! a few edge-cases. If you encounter a crash, please [open an issue][1].
//!
//! Some functions in here are unsafe, because they assume specific invariants,
//! e.g. that tokens are valid.
//!
//! [1]: https://github.com/jgraef/llama-cpp-rs/issues

pub mod batch;
pub mod context;
pub mod grammar;
pub mod model;
pub mod quantization;
pub mod sampling;

use std::{
    cell::OnceCell,
    ffi::{
        c_char,
        c_void,
        CStr,
        CString,
    },
    path::{
        Path,
        PathBuf,
    },
    ptr,
    sync::{
        Mutex,
        Once,
    },
};

/// Max number of devices that can be used to split a tensor computations in
/// between.
///
/// This is defined by the llama.cpp library.
pub const MAX_DEVICES: usize = llama_cpp_sys::llama_define_max_devices;

/// Default seed for RNG.
///
/// This is defined by the llama.cpp library.
pub const DEFAULT_SEED: u32 = llama_cpp_sys::llama_define_default_seed;

const DEFAULT_N_THREADS: OnceCell<u32> = OnceCell::new();

/// Default number of threads. This returns the number of physical cores, as reported by the [num_cpus](https://docs.rs/num_cpus/latest/num_cpus/) crate.
pub fn default_n_threads() -> u32 {
    *DEFAULT_N_THREADS.get_or_init(|| num_cpus::get_physical() as _)
}

unsafe extern "C" fn llama_log_callback(
    level: llama_cpp_sys::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    let Ok(text) = CStr::from_ptr(text).to_str()
    else {
        return;
    };
    let text = text.trim_end_matches('\n');
    match level {
        llama_cpp_sys::ggml_log_level::GGML_LOG_LEVEL_ERROR => {
            tracing::error!(target: "llama.cpp", "{}", text)
        }
        llama_cpp_sys::ggml_log_level::GGML_LOG_LEVEL_WARN => {
            tracing::warn!(target: "llama.cpp", "{}", text)
        }
        llama_cpp_sys::ggml_log_level::GGML_LOG_LEVEL_INFO => {
            tracing::info!(target: "llama.cpp", "{}", text)
        }
    }
}

static LLAMA_INIT: Once = Once::new();

pub(self) fn llama_init() {
    LLAMA_INIT.call_once(|| {
        unsafe {
            // initialize llama.cpp backend
            llama_cpp_sys::llama_backend_init(false);

            // set log callback to use tracing
            llama_cpp_sys::llama_log_set(Some(llama_log_callback), ptr::null_mut());
        }
    });
}

static SYSTEM_INFO_MUTEX: Mutex<()> = Mutex::new(());

/// Returns llama.cpp's system info (`llama_print_system_info()`).
///
/// This contains the various features enabled in the llama.cpp library and is
/// useful for diagnostics.
pub fn system_info() -> String {
    // the system info function writes to a static buffer, so we need to lock.
    let _guard = SYSTEM_INFO_MUTEX
        .lock()
        .expect("system info mutex poisened");

    let info = unsafe { CStr::from_ptr(llama_cpp_sys::llama_print_system_info()) };

    info.to_str()
        .expect("system info contains invalid utf-8")
        .to_owned()
}

pub(self) fn ffi_path(path: impl AsRef<Path>) -> Result<CString, Error> {
    Ok(CString::new(path.as_ref().as_os_str().as_encoded_bytes())?)
}

/// Backend error
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tried to pass a string containing null bytes to FFI interface.
    #[error("string contains null byte")]
    NulError(#[from] std::ffi::NulError),

    /// Could not load the model.
    #[error("failed to load model: {path}")]
    ModelLoadFailed { path: PathBuf },

    /// Could not quantize the model.
    #[error("quantization failed")]
    QuantizationFailed,

    /// The context expects the state data to be a specific size.
    #[error("expected state data with {expected} bytes, but got {got} bytes")]
    InvalidStateDataLength { expected: usize, got: usize },

    /// Decoder error
    #[error("model decode failed")]
    DecodeError,

    /// Incorrect UTF-8 encoding of `&str`
    #[error("utf-8 error")]
    StrUtf8Error(#[from] std::str::Utf8Error),

    /// Incorrect UTF-8 encoding of `String`
    #[error("utf-8 error")]
    StringUtf8Error(#[from] std::string::FromUtf8Error),
}

/// The number type used by a model.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub enum FType {
    AllF32,
    MostlyF16,
    MostlyQ4_0,
    MostlyQ4_1,
    MostlyQ4_1SomeF16,
    MostlyQ8_0,
    MostlyQ5_0,
    MostlyQ5_1,
    MostlyQ2_K,
    MostlyQ3_K_S,
    MostlyQ3_K_M,
    MostlyQ3_K_L,
    MostlyQ4_K_S,
    MostlyQ4_K_M,
    MostlyQ5_K_S,
    MostlyQ5_K_M,
    MostlyQ6_K,
    Guessed,
}

impl FType {
    fn to_ffi(&self) -> llama_cpp_sys::llama_ftype {
        match self {
            FType::AllF32 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_ALL_F32,
            FType::MostlyF16 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_F16,
            FType::MostlyQ4_0 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_0,
            FType::MostlyQ4_1 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1,
            FType::MostlyQ4_1SomeF16 => {
                llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16
            }
            FType::MostlyQ8_0 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q8_0,
            FType::MostlyQ5_0 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_0,
            FType::MostlyQ5_1 => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_1,
            FType::MostlyQ2_K => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q2_K,
            FType::MostlyQ3_K_S => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_S,
            FType::MostlyQ3_K_M => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_M,
            FType::MostlyQ3_K_L => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_L,
            FType::MostlyQ4_K_S => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_S,
            FType::MostlyQ4_K_M => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M,
            FType::MostlyQ5_K_S => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_K_S,
            FType::MostlyQ5_K_M => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_K_M,
            FType::MostlyQ6_K => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q6_K,
            FType::Guessed => llama_cpp_sys::llama_ftype::LLAMA_FTYPE_GUESSED,
        }
    }

    #[allow(dead_code)]
    fn from_ffi(x: llama_cpp_sys::llama_ftype) -> Self {
        match x {
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_ALL_F32 => FType::AllF32,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_F16 => FType::MostlyF16,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_0 => FType::MostlyQ4_0,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1 => FType::MostlyQ4_1,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 => {
                FType::MostlyQ4_1SomeF16
            }
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q8_0 => FType::MostlyQ8_0,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_0 => FType::MostlyQ5_0,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_1 => FType::MostlyQ5_1,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q2_K => FType::MostlyQ2_K,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_S => FType::MostlyQ3_K_S,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_M => FType::MostlyQ3_K_M,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q3_K_L => FType::MostlyQ3_K_L,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_S => FType::MostlyQ4_K_S,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_K_M => FType::MostlyQ4_K_M,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_K_S => FType::MostlyQ5_K_S,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_K_M => FType::MostlyQ5_K_M,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_MOSTLY_Q6_K => FType::MostlyQ6_K,
            llama_cpp_sys::llama_ftype::LLAMA_FTYPE_GUESSED => FType::Guessed,
        }
    }
}

/// The vocabulary type used by a model.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VocabType {
    Spm,
    Bpe,
}

impl VocabType {
    fn from_ffi(x: llama_cpp_sys::llama_vocab_type) -> Self {
        match x {
            llama_cpp_sys::llama_vocab_type::LLAMA_VOCAB_TYPE_SPM => Self::Spm,
            llama_cpp_sys::llama_vocab_type::LLAMA_VOCAB_TYPE_BPE => Self::Bpe,
        }
    }
}

/// A token.
///
/// This is defined by llama.cpp to be just an `i32`.
///
/// You can use [`Model::tokenize`](model::Model::tokenize) to turn a string
/// into tokens and [`TokenDecoder`](model::TokenDecoder) turn a token into its
/// corresponding text.
pub type Token = llama_cpp_sys::llama_token;

/// Type of a token.
///
/// You can get the type of a token by calling
/// [`Model::get_token_type`](model::Model::get_token_type).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_ffi(x: llama_cpp_sys::llama_token_type) -> Self {
        match x {
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_UNDEFINED => Self::Undefined,
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_NORMAL => Self::Normal,
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_UNKNOWN => Self::Unknown,
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_CONTROL => Self::Control,
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_USER_DEFINED => Self::UserDefined,
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_UNUSED => Self::Unused,
            llama_cpp_sys::llama_token_type::LLAMA_TOKEN_TYPE_BYTE => Self::Byte,
        }
    }
}

pub type Pos = llama_cpp_sys::llama_pos;
pub type SeqId = llama_cpp_sys::llama_seq_id;
