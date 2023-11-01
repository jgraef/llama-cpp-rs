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
    backend::{sampling::SamplingParameters, grammar::Compiled},
    grammar::compiler::Buffer,
    loader::ModelLoader,
    session::{
        Session,
        SessionParameters,
    },
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

    // load model asynchronously
    let model_path = "../../data/TinyLLama-v0.gguf";
    let model = ModelLoader::load(&model_path, Default::default())
        .wait_for_model()
        .await?;

    // create a session
    let mut session = Session::new(
        model,
        SessionParameters {
            sampling: SamplingParameters {
                grammar: Some(grammar),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    // feed prompt to it.
    session.push_text("Once upon a time", true, false);

    // create a response stream from it
    let stream = session.pieces(None, [], false);
    pin_mut!(stream);

    // stream LLM output piece by piece
    while let Some(piece) = stream.try_next().await? {
        print!("{piece}");
        stdout().flush()?;
    }

    println!("");
    Ok(())
}
