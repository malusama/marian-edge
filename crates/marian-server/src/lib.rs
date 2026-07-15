use std::{sync::Arc, time::Instant};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use marian_core::{TranslateError, TranslationInput, Translator};
use serde::{Deserialize, Serialize};
use tower_http::{
    catch_panic::CatchPanicLayer,
    cors::{AllowOrigin, Any, CorsLayer},
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::TraceLayer,
};

const MAX_TEXT_BYTES: usize = 64 * 1024;
const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone)]
pub struct AppState {
    translator: Translator,
    started: Instant,
}

impl AppState {
    pub fn new(translator: Translator) -> Self {
        Self {
            translator,
            started: Instant::now(),
        }
    }

    pub fn translator(&self) -> &Translator {
        &self.translator
    }
}

pub fn router(state: AppState, allow_origin: Option<HeaderValue>) -> Router {
    let request_id = header::HeaderName::from_static(REQUEST_ID_HEADER);
    let app = Router::new()
        .route("/translate", post(translate))
        .route("/imme", post(immersive_translate))
        .route("/detect", post(detect))
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .with_state(Arc::new(state))
        .layer(DefaultBodyLimit::max(MAX_TEXT_BYTES))
        .layer(CatchPanicLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::new(request_id.clone()))
        .layer(SetRequestIdLayer::new(request_id, MakeRequestUuid));

    let Some(origin) = allow_origin else {
        return app;
    };
    let cors = if origin == HeaderValue::from_static("*") {
        CorsLayer::new()
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers([header::CONTENT_TYPE])
            .allow_origin(Any)
    } else {
        CorsLayer::new()
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers([header::CONTENT_TYPE])
            .allow_origin(AllowOrigin::exact(origin))
    };
    app.layer(cors)
}

#[derive(Debug, Deserialize)]
pub struct TranslateRequest {
    pub text: String,
    #[serde(default, alias = "source_lang")]
    pub from: Option<String>,
    #[serde(alias = "target_lang")]
    pub to: String,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranslateResponse {
    pub text: String,
    pub from: String,
    pub to: String,
}

async fn translate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TranslateRequest>,
) -> Result<Json<TranslateResponse>, ApiError> {
    validate_text(&request.text)?;
    let source = normalize_source(request.from.as_deref(), &request.text)?;
    let target = normalize_lang(&request.to)?;
    let mut input = TranslationInput::new(request.text, &source, &target);
    input.max_output_tokens = request.max_output_tokens.unwrap_or(512).clamp(1, 2_048);
    let result = state.translator.translate(input).await?;
    Ok(Json(TranslateResponse {
        text: result.text,
        from: source,
        to: target,
    }))
}

#[derive(Debug, Deserialize)]
pub struct ImmersiveRequest {
    #[serde(default)]
    pub source_lang: Option<String>,
    pub target_lang: String,
    pub text_list: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImmersiveItem {
    pub detected_source_lang: String,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImmersiveResponse {
    pub translations: Vec<ImmersiveItem>,
}

async fn immersive_translate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImmersiveRequest>,
) -> Result<Json<ImmersiveResponse>, ApiError> {
    if request.text_list.len() > 256 {
        return Err(ApiError::bad_request(
            "text_list may contain at most 256 items",
        ));
    }
    let target = normalize_lang(&request.target_lang)?;
    let default_source = request.source_lang.as_deref();

    // Validate the complete request before starting any background work. A
    // malformed later item must not leave earlier translations running after
    // the HTTP request has already failed.
    let inputs = request
        .text_list
        .into_iter()
        .map(|text| {
            validate_text(&text)?;
            let source = normalize_source(default_source, &text)?;
            Ok((text, source))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;

    let sources = inputs
        .iter()
        .map(|(_, source)| source.clone())
        .collect::<Vec<_>>();
    let logical_inputs = inputs
        .into_iter()
        .map(|(text, source)| TranslationInput::new(text, source, target.clone()))
        .collect();
    let translations = state
        .translator
        .translate_many(logical_inputs)
        .await?
        .into_iter()
        .zip(sources)
        .map(|(output, source)| ImmersiveItem {
            detected_source_lang: source,
            text: output.text,
        })
        .collect();
    Ok(Json(ImmersiveResponse { translations }))
}

#[derive(Debug, Deserialize)]
pub struct DetectRequest {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetectResponse {
    pub language: String,
}

async fn detect(Json(request): Json<DetectRequest>) -> Result<Json<DetectResponse>, ApiError> {
    validate_text(&request.text)?;
    let language = detect_language(&request.text);
    Ok(Json(DetectResponse {
        language: language.into(),
    }))
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[derive(Serialize)]
struct InfoResponse {
    status: &'static str,
    ready: bool,
    version: &'static str,
    revision: &'static str,
    uptime_seconds: u64,
    backend: marian_core::BackendInfo,
}

async fn info(State(state): State<Arc<AppState>>) -> Json<InfoResponse> {
    let ready = state.translator.is_ready();
    Json(InfoResponse {
        status: if ready { "ok" } else { "draining" },
        ready,
        version: env!("CARGO_PKG_VERSION"),
        revision: env!("MARIAN_EDGE_BUILD_GIT_SHA"),
        uptime_seconds: state.started.elapsed().as_secs(),
        backend: state.translator.backend_info().clone(),
    })
}

async fn livez() -> StatusCode {
    StatusCode::OK
}

async fn readyz(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.translator.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.translator.stats().prometheus(),
    )
}

fn validate_text(text: &str) -> Result<(), ApiError> {
    if text.trim().is_empty() {
        return Err(ApiError::bad_request("text must not be empty"));
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(ApiError::payload_too_large("text exceeds 64 KiB"));
    }
    Ok(())
}

fn normalize_source(source: Option<&str>, text: &str) -> Result<String, ApiError> {
    match source.map(str::trim) {
        None | Some("") => Ok(detect_language(text).into()),
        Some(language) if language.eq_ignore_ascii_case("auto") => Ok(detect_language(text).into()),
        Some(language) => normalize_lang(language),
    }
}

fn normalize_lang(language: &str) -> Result<String, ApiError> {
    let language = language.trim().to_ascii_lowercase();
    let primary = language.split(['-', '_']).next().unwrap_or_default();
    if !(2..=3).contains(&primary.len()) || !primary.bytes().all(|byte| byte.is_ascii_lowercase()) {
        return Err(ApiError::bad_request("invalid language code"));
    }
    Ok(primary.into())
}

fn detect_language(text: &str) -> &'static str {
    let mut cjk = 0usize;
    let mut letters = 0usize;
    for character in text.chars() {
        if matches!(character, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}') {
            cjk += 1;
        }
        if character.is_alphabetic() {
            letters += 1;
        }
    }
    if cjk * 3 >= letters.max(1) {
        "zh"
    } else {
        "en"
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    retry_after: bool,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            retry_after: false,
        }
    }

    fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: message.into(),
            retry_after: false,
        }
    }
}

impl From<TranslateError> for ApiError {
    fn from(error: TranslateError) -> Self {
        let (status, retry_after) = match error {
            TranslateError::QueueFull | TranslateError::ShuttingDown => {
                (StatusCode::SERVICE_UNAVAILABLE, true)
            }
            TranslateError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, false),
            TranslateError::Backend(marian_core::BackendError::InvalidInput(_))
            | TranslateError::Backend(marian_core::BackendError::UnsupportedDirection(_)) => {
                (StatusCode::UNPROCESSABLE_ENTITY, false)
            }
            TranslateError::Backend(_) | TranslateError::WorkerStopped => {
                (StatusCode::INTERNAL_SERVER_ERROR, false)
            }
        };
        Self {
            status,
            message: error.to_string(),
            retry_after,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct ErrorBody {
            error: String,
        }
        let mut response = (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response();
        if self.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use marian_core::{EchoBackend, SchedulerConfig};
    use tower::ServiceExt;

    use super::*;

    fn test_app() -> (Router, Translator) {
        let translator = Translator::start(
            SchedulerConfig {
                batch_window: Duration::from_micros(100),
                ..SchedulerConfig::default()
            },
            || Ok(EchoBackend),
        )
        .unwrap();
        (router(AppState::new(translator.clone()), None), translator)
    }

    #[tokio::test]
    async fn translate_contract_normalizes_regional_language_codes() {
        let (app, translator) = test_app();
        let response = app
            .oneshot(
                Request::post("/translate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"text":"hello","from":"en-US","to":"zh-CN"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key(REQUEST_ID_HEADER));
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: TranslateResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(body.text, "hello");
        assert_eq!(body.from, "en");
        assert_eq!(body.to, "zh");
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn rejects_empty_input() {
        let (app, translator) = test_app();
        let response = app
            .oneshot(
                Request::post("/translate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text":"  ","to":"zh"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn readiness_changes_during_shutdown() {
        let (app, translator) = test_app();
        let ready = app
            .clone()
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);
        translator.shutdown().await;
        let draining = app
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(draining.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn immersive_sorts_across_length_buckets_and_restores_input_order() {
        let (app, translator) = test_app();
        let response = app
            .oneshot(
                Request::post("/imme")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"source_lang":"en_US","target_lang":"zh-Hans","text_list":["a","ninechars","xyz","this sentence is deliberately much longer"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: ImmersiveResponse = serde_json::from_slice(&body).unwrap();
        let texts: Vec<_> = body
            .translations
            .iter()
            .map(|translation| translation.text.as_str())
            .collect();
        assert_eq!(
            texts,
            [
                "a",
                "ninechars",
                "xyz",
                "this sentence is deliberately much longer"
            ]
        );
        let stats = translator.stats().snapshot();
        assert_eq!(stats.accepted, 4);
        assert_eq!(stats.completed, 4);
        // Exact batch boundaries are deliberately scheduler/timing dependent.
        // Dynamic batching itself has deterministic coverage in marian-core.
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn immersive_adapter_preserves_reserved_placeholder_bytes_and_order() {
        let (app, translator) = test_app();
        let placeholders = ["{0} Hello {1} world.", "<b0></b0> Hello <b1></b1> world."];
        let body = serde_json::to_vec(&serde_json::json!({
            "source_lang": "en",
            "target_lang": "zh",
            "text_list": placeholders
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::post("/imme")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: ImmersiveResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(body.translations[0].text, placeholders[0]);
        assert_eq!(body.translations[1].text, placeholders[1]);
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn immersive_validates_every_item_before_submitting_work() {
        let (app, translator) = test_app();
        let response = app
            .oneshot(
                Request::post("/imme")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"source_lang":"en","target_lang":"zh","text_list":["valid","  "]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(translator.stats().snapshot().accepted, 0);
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn documented_body_item_and_endpoint_shape_limits_are_enforced() {
        let (app, translator) = test_app();
        let oversized = serde_json::to_vec(&serde_json::json!({
            "text": "a".repeat(MAX_TEXT_BYTES),
            "from": "en",
            "to": "zh"
        }))
        .unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::post("/translate")
                    .header("content-type", "application/json")
                    .body(Body::from(oversized))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let too_many_items = serde_json::to_vec(&serde_json::json!({
            "source_lang": "en",
            "target_lang": "zh",
            "text_list": vec!["a"; 257]
        }))
        .unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::post("/imme")
                    .header("content-type", "application/json")
                    .body(Body::from(too_many_items))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let response = app
            .oneshot(
                Request::post("/imme")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"from":"en","to":"zh","text_list":["hello"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn detect_and_health_contracts_are_stable() {
        let (app, translator) = test_app();
        let detect_response = app
            .clone()
            .oneshot(
                Request::post("/detect")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let detect_body: serde_json::Value = serde_json::from_slice(
            &detect_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
        assert_eq!(detect_body, serde_json::json!({"language": "en"}));

        let info_response = app
            .clone()
            .oneshot(Request::get("/info").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let info_body: serde_json::Value = serde_json::from_slice(
            &info_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
        assert_eq!(info_body["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(info_body["revision"], env!("MARIAN_EDGE_BUILD_GIT_SHA"));

        let health_response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let health_body: serde_json::Value = serde_json::from_slice(
            &health_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
        assert_eq!(health_body, serde_json::json!({"status": "ok"}));
        translator.shutdown().await;
    }

    #[tokio::test]
    async fn cors_is_opt_in() {
        let (app, translator) = test_app();
        let response = app
            .oneshot(
                Request::get("/health")
                    .header("origin", "https://example.test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            !response
                .headers()
                .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        );
        translator.shutdown().await;

        let translator = Translator::start(SchedulerConfig::default(), || Ok(EchoBackend)).unwrap();
        let app = router(
            AppState::new(translator.clone()),
            Some(HeaderValue::from_static("*")),
        );
        let response = app
            .oneshot(
                Request::get("/health")
                    .header("origin", "https://example.test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("*"))
        );
        translator.shutdown().await;
    }
}
