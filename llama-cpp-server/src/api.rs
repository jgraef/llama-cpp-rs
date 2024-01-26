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
    TryStreamExt,
};
use llama_cpp::{
    backend::{
        context::ContextParameters,
        model::Model,
        sampling::{
            Sampler,
            SamplingMode,
            SamplingParameters,
        },
    },
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
        schemas(GenerateRequest, GenerateResponse)
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

impl From<llama_cpp::grammar::Error> for ApiError {
    fn from(value: llama_cpp::grammar::Error) -> Self {
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
    temperature: Option<f32>,
    #[serde(default)]
    greedy: bool,
    grammar: Option<String>,
    seed: Option<u32>,
    n_ctx: Option<u32>,
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
    State(model): State<Model>,
    request: Json<GenerateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let mut context_parameters = ContextParameters::default();
    context_parameters.seed = request.seed;
    context_parameters.n_ctx = request.n_ctx;
    let context = model.context(&context_parameters);
    let session = Session::from_context(context);

    let mut sampling_parameters = SamplingParameters::default();
    if let Some(grammar) = &request.grammar {
        let grammar = llama_cpp::grammar::parse_and_compile(grammar)?;
        sampling_parameters.grammar = Some(grammar);
    }
    if let Some(t) = request.temperature {
        match &mut sampling_parameters.mode {
            SamplingMode::Temperature { temperature, .. } => *temperature = t,
            _ => {}
        }
    }
    if request.greedy {
        sampling_parameters.mode = SamplingMode::Greedy;
    }
    let sampler = Sampler::new(sampling_parameters);

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


#[derive(Debug, Serialize)]
pub struct StatusResponse {
    system_info: String,
}

/// Returns server status information.
#[utoipa::path(
    get,
    path = "/status",
    responses(
        (status = 200, body = StatusResponse)
    )
)]
pub async fn status(State(_model): State<Model>) -> Json<StatusResponse> {
    Json(StatusResponse {
        system_info: llama_cpp::backend::system_info()
    })
}