
# Rust bindings for [llama.cpp][1]

**This crate is still in early development**

The crate `llama-cpp` contains idiomatic Rust bindings for [llama.cpp][1].
It offers a low-level synchronous API and a high-level asynchronous API.

A simple command line interface that also serves as an example is included in `llama-cpp-cli`.

`llama-cpp-sys` contains the low-level FFI bindings to [llama.cpp][1]. It has the [llama.cpp][1] source code in a git submodule.
The build script takes care of building, and linking to [llama.cpp][1]. It links statically to it and generates bindings using bindgen.
Make sure to pull the submodule:

```bash
git submodule update --init -- llama-cpp-sys/llama.cpp/
```

## Async Runtime

Both [Tokio][2] and [async-std][3] are supported. You choose which one is used by enabling one of the following features

 - `runtime-async-std`
 - `runtime-tokio`

## Features

 - [x] Text Generation
 - [_] Embedding
 - [_] GPU support
 - [_] LORA
 - [x] grammar sampling
 - [_] repetition penalties
 - [_] beam search
 - [_] classifier free guidance
 - [x] tokio runtime
 - [x] async-std runtime

## Example

Examples are located in `examples/`. They are standalone crates.

```rust
// load model asynchronously
let model = ModelLoader::load(&model_path, Default::default())
    .wait_for_model()
    .await?;

// prompt
let prompt = "The capital of France is";
print!("{}", prompt);
stdout().flush()?;

// create a session and feed prompt to it
let mut session = Session::new(model, Default::default());
session.push_text(&prompt, true, false);

// create a response stream from it
let stream = session.pieces(None, [], false);
pin_mut!(stream);

// stream LLM output piece by piece
while let Some(piece) = stream.try_next().await? {
    print!("{piece}");
    stdout().flush()?;
}
```

[1]: https://github.com/ggerganov/llama.cpp
[2]: https://tokio.rs/
[3]: https://async.rs/