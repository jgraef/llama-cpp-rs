use std::{
    io::{
        stdout,
        Write,
    },
    path::Path,
};

use color_eyre::eyre::Error;
use futures::{
    pin_mut,
    StreamExt,
};
use llama_cpp::{
    backend::sampling::SamplingParameters,
    loader::ModelLoader,
    session::Session,
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    // initialize error handling and logging.
    dotenvy::dotenv().ok();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    // check if model exists
    let model_path = Path::new("orca-mini-3b-gguf2-q4_0.gguf");
    if !model_path.exists() {
        eprintln!("Model not found");
        eprintln!("You can download a small model here: https://huggingface.co/TheBloke/orca_mini_3B-GGML");
        panic!("no model");
    }

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
    let mut sampler = session.sampler(SamplingParameters::default())?;
    let stream = sampler.pieces(None, [], false);
    pin_mut!(stream);

    // stream LLM output piece by piece
    while let Some(piece) = stream.next().await {
        print!("{piece}");
        stdout().flush()?;
    }

    println!("");
    Ok(())
}
