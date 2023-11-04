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
    TryStreamExt,
};
use inquire::InquireError;
use llama_cpp::{
    backend::{
        context::ContextParameters,
        sampling::{
            Sampler,
            SamplingParameters,
        },
        system_info,
    },
    loader::ModelLoader,
    session::Session,
    token::Tokenize,
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
    async fn session(&self) -> Result<Session, Error> {
        let context_parameters = ContextParameters {
            seed: self.seed,
            n_ctx: Some(512),
            ..Default::default()
        };

        let model = ModelLoader::load(&self.model, Default::default())
            .wait_for_model()
            .await?;

        let context = model.context(&context_parameters);

        let session = Session::from_context(context);

        Ok(session)
    }

    fn sampler(&self) -> Result<Sampler, Error> {
        let grammar = self
            .grammar
            .as_ref()
            .map(|path| llama_cpp::grammar::compile_from_source(path))
            .transpose()?;

        let sampling_parameters = SamplingParameters {
            grammar,
            ..Default::default()
        };
        Ok(Sampler::new(sampling_parameters))
    }
}

#[derive(Debug, StructOpt)]
enum Args {
    Generate {
        #[structopt(flatten)]
        model_options: ModelOptions,

        #[structopt(long)]
        parallel: Option<usize>,

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
                parallel,
                prompt,
            } => {
                generate(&prompt, model_options, parallel).await?;
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

async fn generate(
    prompt: &str,
    model_options: ModelOptions,
    parallel: Option<usize>,
) -> Result<(), Error> {
    let session = model_options.session().await?;
    let sampler = model_options.sampler()?;

    let parallel = parallel.unwrap_or(1);
    if parallel == 0 {
        return Ok(());
    }

    if parallel == 1 {
        print!("{prompt}");
        stdout().flush()?;
    }
    else {
        println!("{prompt} ... generating");
    }

    // create sequence and feed prompt to it.
    let sequence = session.sequence();
    sequence
        .push(Tokenize {
            text: &prompt,
            add_bos: true,
            allow_special: false,
        })
        .await?;

    // split sequence
    let mut sequences = (1..parallel).map(|_| sequence.clone()).collect::<Vec<_>>();
    sequences.push(sequence);

    // sample in parallel
    let mut join_handles = sequences
        .into_iter()
        .map(|mut sequence| {
            let sampler = sampler.clone();

            tokio::spawn(async move {
                let mut output = String::new();

                let stream = sequence.stream::<String>(sampler);
                pin_mut!(stream);

                while let Some(piece) = stream.try_next().await? {
                    if parallel == 1 {
                        print!("{piece}");
                        stdout().flush()?;
                    }
                    else {
                        output.push_str(&piece);
                    }
                }

                Ok::<_, Error>(output)
            })
        })
        .collect::<Vec<_>>();

    if parallel == 1 {
        join_handles.pop().unwrap().await??;
        println!("");
    }
    else {
        let outputs = futures::stream::iter(join_handles)
            .then(|r| async { r.await.unwrap() })
            .try_collect::<Vec<_>>()
            .await?;

        for (i, output) in outputs.iter().enumerate() {
            println!("Output #{}:", i + 1);
            println!("{output}");
            println!("");
        }
    }

    Ok(())
}

async fn chat(model_options: ModelOptions) -> Result<(), Error> {
    let session = model_options.session().await?;
    let sampler = model_options.sampler()?;
    let mut sequence = session.sequence();
    let mut is_first = true;

    loop {
        let text = match prompt("").await {
            Ok(text) => text,
            Err(InquireError::OperationCanceled) | Err(InquireError::OperationInterrupted) => break,
            Err(e) => return Err(e.into()),
        };

        sequence
            .push(Tokenize {
                text: &text,
                add_bos: is_first,
                allow_special: false,
            })
            .await?;

        let stream = sequence.stream::<String>(sampler.clone());
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
