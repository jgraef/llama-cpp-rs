use std::path::Path;

use super::{
    ffi_path,
    Error,
    FType,
    DEFAULT_N_THREADS,
};

/// Model quantization parameters
#[derive(Clone, Debug)]
pub struct QuantizationParameters {
    /// Number of threads to use for quantizing, if <=0 will use
    /// std::thread::hardware_concurrency()
    pub n_threads: Option<u32>,

    /// quantize to this `llama_ftype`
    pub ftype: FType,

    /// Allow quantizing non-f32/f16 tensors.
    pub allow_requantize: bool,

    /// Quantize `output.weight`.
    pub quantize_output_tensor: bool,

    /// Only copy tensors. `ftype`, `allow_requantize` and
    /// quantize_output_tensor are ignored.
    pub only_copy: bool,
}

impl QuantizationParameters {
    pub fn to_ffi(&self) -> llama_cpp_sys::llama_model_quantize_params {
        llama_cpp_sys::llama_model_quantize_params {
            nthread: self.n_threads.unwrap_or(*DEFAULT_N_THREADS) as i32,
            ftype: self.ftype.to_ffi(),
            allow_requantize: self.allow_requantize,
            quantize_output_tensor: self.quantize_output_tensor,
            only_copy: self.only_copy,
        }
    }
}

impl Default for QuantizationParameters {
    fn default() -> Self {
        Self {
            n_threads: None,
            ftype: FType::MostlyQ5_1,
            allow_requantize: false,
            quantize_output_tensor: true,
            only_copy: false,
        }
    }
}

pub async fn quantize(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    parameters: &QuantizationParameters,
) -> Result<(), Error> {
    let input = ffi_path(input)?;
    let output = ffi_path(output)?;
    let params = parameters.to_ffi();

    let result = tokio::task::spawn_blocking(move || {
        unsafe {
            llama_cpp_sys::llama_model_quantize(
                input.as_ptr(),
                output.as_ptr(),
                &params as *const _,
            )
        }
    })
    .await
    .ok()
    .unwrap_or_default();

    if result == 0 {
        return Ok(());
    }
    else {
        return Err(Error::QuantizationFailed);
    }
}
