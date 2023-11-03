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
    TryStreamExt,
};
use llama_cpp::loader::ModelLoader;

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
    let prompt = "The capital of France is";
    print!("{}", prompt);
    stdout().flush()?;

    // create an inference session
    let mut inference = model.inference(Default::default());

    // feed prompt to it.
    inference.push_text(&prompt, true, false).await?;

    // create a response stream from it
    let stream = inference.pieces(Default::default());
    pin_mut!(stream);

    // stream LLM output piece by piece
    while let Some(piece) = stream.try_next().await? {
        print!("{piece}");
        stdout().flush()?;
    }

    println!("");
    Ok(())
}
