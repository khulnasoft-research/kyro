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
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

pub struct AppState {
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub notify: Arc<Notify>,
}

impl AppState {
    pub fn new(scheduler: Arc<Mutex<Scheduler>>, notify: Arc<Notify>) -> Self {
        Self { scheduler, notify }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
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

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
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

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let request_id = rand::random::<u64>();

    let request = Request {
        id: request_id,
        prompt_tokens: vec![1, 2, 3], // TODO: Use real tokenizer
        generated_tokens: Vec::new(),
        max_tokens: 50,
        is_prefill: true,
        cached_prefix_len: 0,
        prefill_cursor: 0,
        temperature: payload.temperature.unwrap_or(1.0),
        top_p: payload.top_p.unwrap_or(1.0),
        token_sender: Some(tx),
        grammar_processor: None,
    };

    // Add request to scheduler
    {
        let mut sched = state.scheduler.lock().await;
        sched.add_request(request);
    }
    state.notify.notify_one();

    if payload.stream.unwrap_or(false) {
        let stream = async_stream::stream! {
            while let Some(token) = rx.recv().await {
                let chunk = ChatCompletionStreamResponse {
                    id: format!("chatcmpl-{}", request_id),
                    object: "chat.completion.chunk".to_string(),
                    created: 1677652288,
                    model: "kyro-llama3".to_string(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            content: Some(format!("token_{}", token)),
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }
        };

        Sse::new(stream).into_response()
    } else {
        let mut full_content = String::new();
        while let Some(token) = rx.recv().await {
            full_content.push_str(&format!("token_{}", token));
        }

        Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", request_id),
            object: "chat.completion".to_string(),
            created: 1677652288,
            model: payload.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(full_content),
                },
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response()
    }
}

pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(|| async { "OK" }))
        .with_state(state)
}
