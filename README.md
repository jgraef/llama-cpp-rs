
# Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)

**This crate is still under development**

The crate `llama-cpp` contains idiomatic Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp).
It offers a low-level synchronous API and a (somewhat) high-level asynchronous API.

A simple command line interface that also serves as an example is included in `llama-cpp-cli`

`llama-cpp-sys` contains the low-level FFI bindings to llama.cpp. It has the llama.cpp source code in a git submodule.
The build script takes care of building, and linking to llama.cpp. It links statically to it and generates bindings using bindgen.
Make sure to pull the submodule:

```bash
git submodule update --init -- llama-cpp-sys/llama.cpp/
```

## Example

Examples are located in `examples/`. They are standalone crates.

```rust
// load model asynchronously
let model = ModelLoader::load(&model_path, Default::default())
    .wait_for_model()
    .await?;

// prompt
let prompt = "The capital of Paris is";
print!("{}", prompt);
stdout().flush()?;

// create a session and feed prompt to it
let mut session = Session::new(model, Default::default());
session.push_text(&prompt, true, false);

// create a sampler and a response stream from it
let mut sampler = session.sampler(SamplingParameters::default());
let stream = sampler.pieces(None, [], false);
pin_mut!(stream);

// stream LLM output piece by piece
while let Some(piece) = stream.next().await {
    print!("{piece}");
    stdout().flush()?;
}
```
