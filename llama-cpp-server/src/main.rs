mod api;

use std::path::PathBuf;

use axum::{
    response::Html, routing, Router
};
use color_eyre::eyre::Error;
use llama_cpp::loader::ModelLoader;
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

        let app = Router::new()
            .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
            .route("/", routing::get(index))
            .route("/status", routing::get(api::status))
            .route("/generate", routing::post(api::generate))
            .with_state(model);

        tracing::info!("listening on: http://{}", self.address);
        let listener = TcpListener::bind(self.address).await?;
        axum::serve(listener, app.into_make_service()).await?;

        Ok(())
    }
}

async fn index() -> Html<&'static str> {
    Html(r#"
<html>
    <head>
        <title>llama-cpp-server</title>
    </head>
    <body>
        API server built with <a href="https://github.com/jgraef/llama-cpp-rs">llama-cpp-rs</a>.<br>

        <a href="/swagger-ui">Swagger UI</a>
    </body>
</html>
    "#)
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
