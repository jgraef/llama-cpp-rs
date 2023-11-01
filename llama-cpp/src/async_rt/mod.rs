#![allow(dead_code)]

/// Thin wrappers around async stuff, so we can switch between runtimes.
mod shared;

cfg_if::cfg_if! {
    if #[cfg(feature = "runtime-tokio")] {
        mod tokio_rt;
        pub use tokio_rt::*;
    } else if #[cfg(feature = "runtime-async-std")] {
        mod async_std_rt;
        pub use async_std_rt::*;
    } else {
        std::compile_error!("you need to select an async runtime by enabling a feature.");
    }
}
