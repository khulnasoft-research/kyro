use crate::api::tokenizer::LuminaTokenizer;
use crate::scheduler::continuous_batching::{Request, Scheduler};
use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::{Notify, RwLock};
use tracing::Span;

pub struct AppState {
    pub scheduler: Arc<RwLock<Scheduler>>,
    pub notify: Arc<Notify>,
    pub registry: prometheus::Registry,
    pub tokenizer: Option<LuminaTokenizer>,
}

impl AppState {
    pub fn new(
        scheduler: Arc<RwLock<Scheduler>>,
        notify: Arc<Notify>,
        registry: prometheus::Registry,
        tokenizer: Option<LuminaTokenizer>,
    ) -> Self {
        Self {
            scheduler,
            notify,
            registry,
            tokenizer,
        }
    }
}

// ── Request types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop: Option<StopCondition>,
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    pub best_of: Option<usize>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    MultiModal(Vec<ContentItem>),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ContentItem {
    #[serde(rename = "type")]
    pub item_type: String,
    pub text: Option<String>,
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

// ── Response types ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ── Error types ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

fn error_response(
    status: axum::http::StatusCode,
    message: &str,
    error_type: &str,
    code: &str,
) -> axum::response::Response {
    let body = Json(ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            error_type: error_type.to_string(),
            code: code.to_string(),
        },
    });
    (status, body).into_response()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn extract_prompt_text(messages: &[Message]) -> String {
    let mut text = String::new();
    for msg in messages {
        match &msg.content {
            MessageContent::Text(t) => {
                text.push_str(&format!("{}: {}\n", msg.role, t));
            }
            MessageContent::MultiModal(items) => {
                text.push_str(&format!("{}: ", msg.role));
                for item in items {
                    if let Some(ref t) = item.text {
                        text.push_str(t);
                    }
                }
                text.push('\n');
            }
        }
    }
    text
}

// ── Handlers ───────────────────────────────────────────────────

pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = Vec::new();
    match encoder.encode(&state.registry.gather(), &mut buffer) {
        Ok(()) => axum::response::Response::builder()
            .header(
                axum::http::header::CONTENT_TYPE,
                prometheus::TextEncoder::new().format_type(),
            )
            .body(axum::body::Body::from(buffer))
            .unwrap(),
        Err(e) => axum::response::Response::builder()
            .status(500)
            .body(axum::body::Body::from(format!(
                "Failed to encode metrics: {}",
                e
            )))
            .unwrap(),
    }
}

#[tracing::instrument(skip(state), fields(request_id))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let request_id = rand::random::<u64>();
    tracing::Span::current().record("request_id", request_id);

    if payload.messages.is_empty() {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            "empty_messages",
        );
    }

    // Encode prompt with tokenizer or use fallback
    let prompt_text = extract_prompt_text(&payload.messages);
    let prompt_tokens = match &state.tokenizer {
        Some(tk) => match tk.encode(&prompt_text) {
            Ok(tokens) => tokens,
            Err(e) => {
                return error_response(
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    &format!("Failed to tokenize prompt: {}", e),
                    "invalid_request_error",
                    "tokenization_failed",
                );
            }
        },
        None => {
            // Fallback: use fixed prompt tokens when no tokenizer is loaded
            vec![1, 2, 3]
        }
    };

    let max_tokens = payload.max_tokens.unwrap_or(50);
    let temperature = payload.temperature.unwrap_or(1.0);
    let top_p = payload.top_p.unwrap_or(1.0);
    let top_k = payload.top_k;
    let frequency_penalty = payload.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = payload.presence_penalty.unwrap_or(0.0);
    let best_of = payload.best_of.unwrap_or(1).max(1);
    let seed = payload.seed;

    // Convert logit_bias from token string -> f32 to token id -> f32
    let logit_bias = payload.logit_bias.map(|bias_map| {
        bias_map
            .into_iter()
            .filter_map(|(token_str, bias)| token_str.parse::<u32>().ok().map(|id| (id, bias)))
            .collect::<std::collections::HashMap<u32, f32>>()
    });

    // Convert stop strings to token IDs
    let stop_strings: Vec<String> = match payload.stop {
        Some(StopCondition::Single(s)) => vec![s],
        Some(StopCondition::Multiple(v)) => v,
        None => Vec::new(),
    };
    let stop_token_ids = stop_strings
        .iter()
        .filter_map(|s| state.tokenizer.as_ref().map(|tk| tk.encode(s).unwrap_or_default()))
        .filter(|tokens| !tokens.is_empty())
        .collect();

    let created = unix_now();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();

    let request = Request {
        id: request_id,
        prompt_tokens,
        generated_tokens: Vec::new(),
        max_tokens,
        is_prefill: true,
        cached_prefix_len: 0,
        prefill_cursor: 0,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        logit_bias,
        stop_token_ids,
        best_of,
        seed,
        token_sender: Some(tx),
        deadline: Some(tokio::time::Instant::now() + tokio::time::Duration::from_secs(300)),
        grammar_processor: None,
    };

    // Add request to scheduler
    {
        let mut sched = state.scheduler.write().await;
        sched.add_request(request);
    }
    state.notify.notify_one();

    if payload.stream.unwrap_or(false) {
        let stream = async_stream::stream! {
            let mut all_tokens: Vec<u32> = Vec::new();

            while let Some(token) = rx.recv().await {
                all_tokens.push(token);

                // Decode the accumulated tokens to get readable text
                let content = match &state.tokenizer {
                    Some(tk) => tk.decode(&all_tokens).unwrap_or_else(|_| format!("token_{}", token)),
                    None => format!("token_{}", token),
                };

                let chunk = ChatCompletionStreamResponse {
                    id: format!("chatcmpl-{}", request_id),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: payload.model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            content: Some(content),
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap_or_else(|e| {
                    format!("{{\"error\": \"serialization failed: {}\"}}", e)
                });
                yield Ok::<Event, Infallible>(Event::default().data(data));
            }

            // Send the [DONE] signal
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream).into_response()
    } else {
        let mut all_tokens: Vec<u32> = Vec::new();
        while let Some(token) = rx.recv().await {
            all_tokens.push(token);
        }

        let content = match &state.tokenizer {
            Some(tk) => tk.decode(&all_tokens).unwrap_or_else(|_| {
                all_tokens
                    .iter()
                    .map(|t| format!("token_{}", t))
                    .collect::<Vec<_>>()
                    .join("")
            }),
            None => all_tokens
                .iter()
                .map(|t| format!("token_{}", t))
                .collect::<Vec<_>>()
                .join(""),
        };

        let prompt_tokens_count = match &state.tokenizer {
            Some(tk) => tk.encode(&prompt_text).map(|t| t.len()).unwrap_or(0),
            None => 0,
        };

        Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", request_id),
            object: "chat.completion".to_string(),
            created,
            model: payload.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(content),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_tokens_count,
                completion_tokens: all_tokens.len(),
                total_tokens: prompt_tokens_count + all_tokens.len(),
            },
        })
        .into_response()
    }
}

pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(|| async { "OK" }))
        .route("/metrics", get(metrics_handler))
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
                .make_span_with(|_request: &axum::extract::Request<_>| {
                    tracing::info_span!("http_request")
                })
                .on_response(
                    |response: &axum::response::Response,
                     _latency: std::time::Duration,
                     _span: &Span| {
                        tracing::info!(status = response.status().as_u16(), "HTTP response")
                    },
                ),
        )
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_prompt_text_single_message() {
        let msgs = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hello".to_string()),
        }];
        let text = extract_prompt_text(&msgs);
        assert_eq!(text, "user: Hello\n");
    }

    #[test]
    fn test_extract_prompt_text_multiple_messages() {
        let msgs = vec![
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hi".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Text("Hello!".to_string()),
            },
        ];
        let text = extract_prompt_text(&msgs);
        assert_eq!(text, "user: Hi\nassistant: Hello!\n");
    }

    #[test]
    fn test_extract_prompt_text_multimodal() {
        let msgs = vec![Message {
            role: "user".to_string(),
            content: MessageContent::MultiModal(vec![ContentItem {
                item_type: "text".to_string(),
                text: Some("Look at this".to_string()),
                image_url: None,
            }]),
        }];
        let text = extract_prompt_text(&msgs);
        assert_eq!(text, "user: Look at this\n");
    }

    #[test]
    fn test_error_response_format() {
        let status = axum::http::StatusCode::BAD_REQUEST;
        let resp = error_response(status, "test error", "invalid_request_error", "test_code");
        assert_eq!(resp.status(), 400);
    }

    #[test]
    fn test_error_response_422() {
        let status = axum::http::StatusCode::UNPROCESSABLE_ENTITY;
        let resp = error_response(status, "bad input", "invalid_request_error", "bad_input");
        assert_eq!(resp.status(), 422);
    }

    #[test]
    fn test_usage_defaults() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_unix_now_returns_recent_timestamp() {
        let now = unix_now();
        // Should be a 2025-2026 timestamp (roughly 1.7-1.8 billion)
        assert!(now > 1_700_000_000, "timestamp too old: {}", now);
        assert!(now < 2_000_000_000, "timestamp too far in future: {}", now);
    }
}
