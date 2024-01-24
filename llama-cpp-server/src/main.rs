mod api;

use std::path::PathBuf;

use axum::{
    routing,
    Router,
};
use color_eyre::eyre::Error;
use llama_cpp::{
    loader::ModelLoader,
    session::Session,
};
use structopt::StructOpt;
use tokio::net::TcpListener;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::api::ApiDoc;

#[derive(Debug, StructOpt)]
struct Args {
    model: PathBuf,

    #[structopt(short, long, default_value = "0.0.0.0:7860")]
    address: String,
}

impl Args {
    pub async fn run(self) -> Result<(), Error> {
        let model = ModelLoader::load(&self.model, Default::default())
            .wait_for_model()
            .await?;
        let context = model.context(&Default::default());
        let session = Session::from_context(context);

        let app = Router::new()
            .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
            .route("/generate", routing::post(api::generate))
            .with_state(session);

        tracing::info!("listening on: http://{}", self.address);
        let listener = TcpListener::bind(self.address).await?;
        axum::serve(listener, app.into_make_service()).await?;

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
