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
    StreamExt,
};
use llama_cpp::{
    backend::{
        context::ContextParameters,
        sampling::{
            SamplingMode,
            SamplingParameters,
            TopK,
            TopP,
        },
        system_info,
    },
    loader::ModelLoader,
    session::{
        Session,
        SessionParameters,
    },
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
                let session_parameters = SessionParameters {
                    context: ContextParameters {
                        seed,
                        n_ctx: Some(512),
                        n_batch: 512,
                        ..Default::default()
                    },
                    batch_size: Some(64),
                };

                let grammar = grammar
                    .map(|path| llama_cpp::grammar::compile_from_source(path))
                    .transpose()?;

                let sampling_parameters = SamplingParameters {
                    mode: SamplingMode::Propability,
                    repetition_penalties: None,
                    soft_max: false,
                    top_k: Some(TopK { k: 40, min_keep: 1 }),
                    top_p: Some(TopP {
                        p: 0.9,
                        min_keep: 1,
                    }),
                    temperature: 0.4,
                    grammar,
                };

                let model = ModelLoader::load(&model, Default::default())
                    .wait_for_model()
                    .await?;

                print!("{}", prompt);
                stdout().flush()?;

                let mut session = Session::new(model, session_parameters);

                session.push_text(&prompt, true, false);

                let mut sampler = session.sampler(sampling_parameters)?;

                let stream = sampler.pieces(None, [], false);
                pin_mut!(stream);

                while let Some(piece) = stream.next().await {
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
                    model.token_to_piece(i as _, &mut buf);

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
