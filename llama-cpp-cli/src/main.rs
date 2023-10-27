use std::path::PathBuf;

use color_eyre::eyre::Error;
use llama_cpp::backend::{
    context::ContextParameters,
    inference::Inference,
    model::Model,
    sampling::Sampler,
    system_info,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(subcommand)]
    command: Command,
}

impl Args {
    pub async fn run(self) -> Result<(), Error> {
        match self.command {
            Command::Complete { model_path, prompt } => {
                let model = Model::load(&model_path, Default::default())
                    .wait_for_model()
                    .await?;

                let prompt = model.tokenize(&prompt, true, false);
                tracing::debug!("prompt tokens: {:?}", prompt);

                let n_len = 32;

                let mut context_params = ContextParameters::default();
                context_params.n_ctx = Some(n_len);
                context_params.n_batch = n_len;
                let mut context = model.context(&context_params);

                let mut inference = Inference::new(&mut context, prompt.len());
                inference.push(&prompt)?;

                let sampler_params = Default::default();
                let mut sampler = Sampler::new(&sampler_params);

                while let Some(token) = inference.sample(&mut sampler)? {
                    let mut response = String::new();
                    model.token_to_piece(token, &mut response)?;
                    tracing::info!(token_id = token, token = response);
                }
            }
            Command::SystemInfo => {
                let info = system_info();
                println!("{}", info);
            }
        }

        Ok(())
    }
}

#[derive(Debug, StructOpt)]
enum Command {
    Complete {
        #[structopt(short, long)]
        model_path: PathBuf,

        prompt: String,
    },
    SystemInfo,
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
