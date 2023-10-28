use std::{
    env,
    path::{
        Path,
        PathBuf,
    },
    process::Command,
};

use bindgen::EnumVariation;

const OFFLINE: bool = true;

fn init_submodule(llama_cpp_path: &Path) {
    if !llama_cpp_path.join("CMakeLists.txt").exists() {
        if OFFLINE {
            panic!(
                "llama.cpp git submodule not intialized. Run `git submodule update --init` in '{}'`.",
                llama_cpp_path.display()
            )
        }
        else {
            Command::new("git")
                .args(&["submodule", "update", "--init"])
                .current_dir(llama_cpp_path)
                .status()
                .expect("Git is needed to retrieve the llama.cpp source files");
        }
    }
}

fn compile_llama_cpp(llama_cpp: &Path) -> PathBuf {
    let mut config = cmake::Config::new(llama_cpp);
    config
        .profile("Release")
        .define("CMAKE_CONFIGURATION_TYPES", "Release")
        .define("LLAMA_STATIC", "ON")
        .define("LLAMA_STANDALONE", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        // i think we need this, since we're statically linking.
        // this will also enable avx, avx2, fma. we might not want this
        .define("LLAMA_NATIVE", "OFF");
    //.define("LLAMA_AVX", "ON")
    //.define("LLAMA_AVX2", "ON");
    config.build().join("lib")
}

fn main() {
    let llama_cpp = Path::new("llama.cpp").canonicalize().unwrap();

    // fetch git submodule
    init_submodule(&llama_cpp);

    // compile llama.cpp
    let compiled = compile_llama_cpp(&llama_cpp);

    // statically link to llama.cpp
    println!("cargo:rustc-link-search=native={}", compiled.display());
    println!("cargo:rustc-link-lib=static=llama");

    // we need stdc++
    // todo: this might not work on other platforms.
    println!("cargo:rustc-flags=-l dylib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_args(&["-x", "c++"])
        .clang_arg("-std=c++17")
        .clang_arg(&format!("-I{}", llama_cpp.display()))
        .allowlist_item("llama_.*")
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
