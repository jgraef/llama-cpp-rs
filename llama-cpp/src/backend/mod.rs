pub mod batch;
pub mod context;
pub mod grammar;
pub mod inference;
pub mod model;
pub mod quantization;
pub mod sampling;

use std::{
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

use lazy_static::lazy_static;

pub const MAX_DEVICES: usize = llama_cpp_sys::llama_define_max_devices;
pub const DEFAULT_SEED: u32 = llama_cpp_sys::llama_define_default_seed;

lazy_static! {
    pub static ref DEFAULT_N_THREADS: u32 = num_cpus::get_physical() as u32;
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("string contains null byte")]
    NulError(#[from] std::ffi::NulError),

    #[error("failed to load model: {path}")]
    ModelLoadFailed { path: PathBuf },

    #[error("quantization failed")]
    QuantizationFailed,

    #[error("expected state data with {expected} bytes, but got {got} bytes")]
    InvalidStateDataLength { expected: usize, got: usize },

    #[error("model decode failed")]
    DecodeError,

    #[error("utf-8 error")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("failed to load grammar")]
    GrammarLoadFailed,
}

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

pub enum VocabType {
    Spm,
    Bpe,
}

impl VocabType {
    pub fn from_ffi(x: llama_cpp_sys::llama_vocab_type) -> Self {
        match x {
            llama_cpp_sys::llama_vocab_type::LLAMA_VOCAB_TYPE_SPM => Self::Spm,
            llama_cpp_sys::llama_vocab_type::LLAMA_VOCAB_TYPE_BPE => Self::Bpe,
        }
    }
}

pub type Token = llama_cpp_sys::llama_token;

pub type Pos = llama_cpp_sys::llama_pos;

pub type SeqId = llama_cpp_sys::llama_seq_id;
