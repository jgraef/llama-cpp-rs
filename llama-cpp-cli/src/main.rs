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
use llama_cpp::{
    backend::{
        context::ContextParameters,
        sampling::SamplingParameters,
        system_info,
    },
    inference::InferenceParameters,
    loader::ModelLoader,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
enum Args {
    Generate {
        #[structopt(short, long)]
        model: PathBuf,

        #[structopt(short, long)]
        seed: Option<u32>,

        #[structopt(short, long)]
        grammar: Option<PathBuf>,

        prompt: String,
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
                model,
                seed,
                grammar,
                prompt,
            } => {
                let grammar = grammar
                    .map(|path| llama_cpp::grammar::compile_from_source(path))
                    .transpose()?;

                let inference_parameters = InferenceParameters {
                    context: ContextParameters {
                        seed,
                        n_ctx: Some(512),
                        ..Default::default()
                    },
                    sampling: SamplingParameters {
                        grammar,
                        ..Default::default()
                    },
                    batch_size: Some(64),
                };

                let model = ModelLoader::load(&model, Default::default())
                    .wait_for_model()
                    .await?;

                print!("{}", prompt);
                stdout().flush()?;

                let mut inference = model.inference(inference_parameters);

                inference.push_text(&prompt, true, false).await?;

                let stream = inference.pieces(None, [], false);
                pin_mut!(stream);

                while let Some(piece) = stream.try_next().await? {
                    print!("{piece}");
                    stdout().flush()?;
                }

                println!("");
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
                let mut buf = Vec::with_capacity(32);

                for i in 0..n_vocab {
                    model.get_token_text(i as _, &mut buf);

                    if let Ok(s) = std::str::from_utf8(&buf) {
                        println!("{i} = '{s}'");
                    }
                    else if buf.len() == 1 {
                        println!("{i} = 0x{:02x}", buf[0]);
                    }
                    else {
                        println!("{i} = {:?}", buf);
                    }

                    buf.clear();
                }
            }
        }

        Ok(())
    }
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
