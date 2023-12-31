//! This is a somewhat useless example. It generates a grammar of all emojis
//! using the [emoji][1] crate and samples from it. It doesn't really work 😿
//!
//! [1] https://crates.io/crates/emoji

use std::io::{
    stdout,
    Write,
};

use color_eyre::eyre::Error;
use futures::{
    pin_mut,
    TryStreamExt,
};
use llama_cpp::{
    backend::{
        grammar::Compiled,
        sampling::{
            Sampler,
            SamplingParameters,
        },
    },
    grammar::compiler::Buffer,
    loader::ModelLoader,
    session::Session,
    token::Tokenize,
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    // initialize error handling and logging.
    dotenvy::dotenv().ok();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    // create emoji grammar
    let mut rule1 = Buffer::default();
    for (i, emoji) in emoji::lookup_by_glyph::iter_emoji().enumerate() {
        if i > 0 {
            rule1.alt();
        }
        for c in emoji.glyph.chars() {
            rule1.char(c);
        }
    }
    rule1.end();
    let mut rule2 = Buffer::default();
    rule2.rule_ref(0);
    rule2.rule_ref(1);
    rule2.end();
    let grammar = Compiled {
        root: 1,
        rules: vec![rule1.elements, rule2.elements],
    };

    let sampling_parameters = SamplingParameters {
        grammar: Some(grammar),
        ..Default::default()
    };

    // load model asynchronously
    let model_path = "../../data/TinyLLama-v0.gguf";
    let model = ModelLoader::load(&model_path, Default::default())
        .wait_for_model()
        .await?;

    // create an inference session
    let session = Session::from_model(model, &Default::default());
    let mut sequence = session.sequence();

    // feed prompt to it.
    sequence
        .push(Tokenize {
            text: "Once upon a time",
            add_bos: true,
            allow_special: false,
        })
        .await?;

    // create a response stream from it
    let stream = sequence.stream::<String>(Sampler::new(sampling_parameters));
    pin_mut!(stream);

    // stream LLM output piece by piece
    while let Some(piece) = stream.try_next().await? {
        print!("{piece}");
        stdout().flush()?;
    }

    println!("");
    Ok(())
}
