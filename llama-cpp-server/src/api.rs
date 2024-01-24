use async_stream::try_stream;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::Event,
        IntoResponse,
        Response,
        Sse,
    },
    Json,
};
use futures::{
    Stream,
    StreamExt,
    TryStreamExt,
};
use llama_cpp::{
    backend::sampling::Sampler,
    session::{
        Sequence,
        Session,
    },
    token::BeginOfSequence,
};
use serde::{
    Deserialize,
    Serialize,
};
use utoipa::{
    OpenApi,
    ToSchema,
};

#[derive(OpenApi)]
#[openapi(
    paths(
        generate
    ),
    components(
        //schemas(GenerateRequest, GenerateResponse)
    ),
    tags(
        //(name = "todo", description = "Todo items management API")
    )
)]
pub struct ApiDoc;

#[derive(Debug, Serialize, thiserror::Error)]
pub enum ApiError {
    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<llama_cpp::Error> for ApiError {
    fn from(value: llama_cpp::Error) -> Self {
        Self::Internal(value.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self)).into_response()
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct GenerateRequest {
    input: String,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct GenerateResponse {
    output: String,
}

/// Generate a text completion from a prompt.
#[utoipa::path(
    post,
    path = "/generate",
    request_body = GenerateRequest,
    responses(
        (status = 200, description = "Generated text completion successfully", body = GenerateResponse)
    )
)]
pub async fn generate(
    State(session): State<Session>,
    request: Json<GenerateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let sampler = Sampler::new(Default::default());

    let mut sequence = session.sequence();
    sequence.push(BeginOfSequence).await?;
    sequence.push(request.input.as_str()).await?;

    if request.stream {
        fn stream(
            mut sequence: Sequence,
            sampler: Sampler,
        ) -> impl Stream<Item = Result<Event, ApiError>> {
            try_stream! {
                let mut stream = sequence.stream::<String>(sampler);
                while let Some(text) = stream.try_next().await? {
                    yield Event::default().json_data(GenerateResponse { output: text }).unwrap();
                }
            }
        }

        Ok(Sse::new(stream(sequence, sampler)).into_response())
    }
    else {
        let mut output = String::new();
        let mut stream = sequence.stream::<String>(sampler);
        while let Some(token) = stream.try_next().await? {
            output.push_str(&token);
        }

        Ok(Json(GenerateResponse { output }).into_response())
    }
}
