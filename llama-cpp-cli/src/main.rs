use std::{
    io::{
        stdout,
        Write,
    },
    path::PathBuf,
};

use color_eyre::eyre::Error;
use futures::{
    pin_mut,
    TryStreamExt,
};
use inquire::InquireError;
use llama_cpp::{
    backend::{
        context::ContextParameters,
        sampling::SamplingParameters,
        system_info,
    },
    inference::{
        Inference,
        InferenceParameters,
    },
    loader::ModelLoader,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct ModelOptions {
    #[structopt(short, long)]
    model: PathBuf,

    #[structopt(short, long)]
    seed: Option<u32>,

    #[structopt(short, long)]
    grammar: Option<PathBuf>,
}

impl ModelOptions {
    async fn inference(&self) -> Result<Inference, Error> {
        let grammar = self
            .grammar
            .as_ref()
            .map(|path| llama_cpp::grammar::compile_from_source(path))
            .transpose()?;

        let inference_parameters = InferenceParameters {
            context: ContextParameters {
                seed: self.seed,
                n_ctx: Some(512),
                ..Default::default()
            },
            sampling: SamplingParameters {
                grammar,
                ..Default::default()
            },
            batch_size: Some(64),
        };

        let model = ModelLoader::load(&self.model, Default::default())
            .wait_for_model()
            .await?;

        Ok(model.inference(inference_parameters))
    }
}

#[derive(Debug, StructOpt)]
enum Args {
    Generate {
        #[structopt(flatten)]
        model_options: ModelOptions,

        prompt: String,
    },
    Chat {
        #[structopt(flatten)]
        model_options: ModelOptions,
    },
    SystemInfo,
    PrintVocab {
        #[structopt(short, long)]
        model_path: PathBuf,
    },
}

impl Args {
    pub async fn run(self) -> Result<(), Error> {
        match self {
            Self::Generate {
                model_options,
                prompt,
            } => {
                generate(model_options, &prompt).await?;
            }
            Self::Chat { model_options } => {
                chat(model_options).await?;
            }
            Self::SystemInfo => {
                let info = system_info();
                println!("{}", info);
            }
            Self::PrintVocab { model_path } => {
                let model = ModelLoader::load(&model_path, Default::default())
                    .wait_for_model()
                    .await?;

                let n_vocab = model.n_vocab();

                for i in 0..n_vocab {
                    let s = model.get_token_text(i as _)?;

                    println!("{i} = {s}");
                }
            }
        }

        Ok(())
    }
}

async fn generate(model_options: ModelOptions, prompt: &str) -> Result<(), Error> {
    let mut inference = model_options.inference().await?;

    print!("{}", prompt);
    stdout().flush()?;

    inference.push_text(&prompt, true, false).await?;

    let stream = inference.pieces(Default::default());
    pin_mut!(stream);

    while let Some(piece) = stream.try_next().await? {
        print!("{piece}");
        stdout().flush()?;
    }

    println!("");

    Ok(())
}

async fn chat(model_options: ModelOptions) -> Result<(), Error> {
    let mut inference = model_options.inference().await?;
    let mut is_first = true;

    loop {
        let text = match prompt("").await {
            Ok(text) => text,
            Err(InquireError::OperationCanceled) | Err(InquireError::OperationInterrupted) => break,
            Err(e) => return Err(e.into()),
        };

        inference.push_text(&text, is_first, true).await?;

        let stream = inference.pieces(Default::default());
        pin_mut!(stream);

        while let Some(piece) = stream.try_next().await? {
            print!("{piece}");
            stdout().flush()?;
        }
        println!("");

        is_first = false;
    }

    Ok(())
}

async fn prompt(prompt: impl ToString) -> Result<String, InquireError> {
    let prompt = prompt.to_string();
    Ok(
        tokio::task::spawn_blocking(move || inquire::Text::new(&prompt).prompt())
            .await
            .expect("join error")?,
    )
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let args = Args::from_args();
    args.run().await?;

    Ok(())
}
