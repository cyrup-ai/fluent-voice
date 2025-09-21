#![allow(unused_imports)]
use crate::endpoints::{ElevenLabsEndpoint, RequestBody};
use crate::error::Error::HttpError;
// WebSocket functionality is currently disabled
// use crate::error::WebSocketError;
// use crate::endpoints::genai::tts::ws::*;
use futures_util::{SinkExt, Stream, StreamExt, pin_mut};
use reqwest::{Method, header::CONTENT_TYPE};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
// WebSocket functionality is currently disabled
// use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
// use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const XI_API_KEY_HEADER: &str = "xi-api-key";
const APPLICATION_JSON: &str = "application/json";

/// HTTP/3 QUIC-enabled ElevenLabs client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Enable 0-RTT early data for reduced latency
    pub enable_early_data: bool,
    /// Maximum QUIC connection idle timeout
    pub max_idle_timeout: Duration,
    /// Per-stream receive window size (bytes)
    pub stream_receive_window: u64,
    /// Connection-wide receive window size (bytes)  
    pub conn_receive_window: u64,
    /// Send window size (bytes)
    pub send_window: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            enable_early_data: true,
            max_idle_timeout: Duration::from_secs(30),
            stream_receive_window: 1024 * 1024,    // 1MB
            conn_receive_window: 10 * 1024 * 1024, // 10MB
            send_window: 1024 * 1024,              // 1MB
        }
    }
}

#[derive(Clone)]
pub struct ElevenLabsClient {
    inner: reqwest::Client,
    api_key: String,
}

impl ElevenLabsClient {
    pub fn from_env() -> Result<Self> {
        Self::from_env_with_config(ClientConfig::default())
    }

    pub fn from_env_with_config(config: ClientConfig) -> Result<Self> {
        // Try all possible ElevenLabs API key environment variables
        let api_key = std::env::var("ELEVENLABS_API_KEY")
            .or_else(|_| std::env::var("ELEVEN_API_KEY"))
            .or_else(|_| std::env::var("ELEVEN_LABS_API_KEY"))
            .map_err(|_| "No ElevenLabs API key found. Set ELEVENLABS_API_KEY, ELEVEN_API_KEY, or ELEVEN_LABS_API_KEY environment variable")?;

        // Debug: Show first/last few characters of the API key
        println!(
            "🔑 Using API key: {}...{} (length: {})",
            &api_key[..std::cmp::min(8, api_key.len())],
            &api_key[std::cmp::max(api_key.len().saturating_sub(4), 8)..],
            api_key.len()
        );

        Self::new_with_config(api_key, config)
    }

    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::new_with_config(api_key, ClientConfig::default())
    }

    pub fn new_with_config(api_key: impl Into<String>, config: ClientConfig) -> Result<Self> {
        // Install default crypto provider for rustls (ignore if already installed)
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

        // Create root cert store with webpki certificates
        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        // Configure TLS with manual root certificates and ALPN for HTTP/3
        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        // Configure ALPN protocols for HTTP/3 QUIC
        let mut tls_config = tls_config;
        tls_config.alpn_protocols = vec![b"h3".to_vec()];

        let client = reqwest::Client::builder()
            .use_preconfigured_tls(tls_config)
            .http3_prior_knowledge()
            .tls_early_data(config.enable_early_data)
            .http3_max_idle_timeout(config.max_idle_timeout)
            .http3_stream_receive_window(config.stream_receive_window)
            .http3_conn_receive_window(config.conn_receive_window)
            .http3_send_window(config.send_window)
            .build()
            .map_err(|e| format!("Failed to create HTTP/3 client: {}", e))?;

        Ok(Self {
            inner: client,
            api_key: api_key.into(),
        })
    }

    pub async fn hit<T: ElevenLabsEndpoint>(&self, endpoint: T) -> Result<T::ResponseBody> {
        let mut builder = self
            .inner
            .request(T::METHOD, endpoint.url()?)
            .header(XI_API_KEY_HEADER, &self.api_key)
            .version(http::Version::HTTP_3);

        if matches!(T::METHOD, Method::POST | Method::PATCH) {
            let request_body = endpoint.request_body().await?;
            builder = match request_body {
                RequestBody::Json(json) => {
                    builder.header(CONTENT_TYPE, APPLICATION_JSON).json(&json)
                }
                RequestBody::Multipart(form) => builder.multipart(form),
                RequestBody::Empty => return Err("request must have a body".into()),
            };
        }

        let resp = builder.send().await?;

        if !resp.status().is_success() {
            return Err(Box::new(HttpError(resp.json().await?)));
        }

        endpoint.response_body(resp).await
    }

    /// Hit an endpoint and return both headers and response body
    #[allow(dead_code)] // False positive: method is used in fluent_voice_impl.rs
    pub async fn hit_with_headers<T: ElevenLabsEndpoint>(
        &self,
        endpoint: T,
    ) -> Result<(http::HeaderMap, T::ResponseBody)> {
        let mut builder = self
            .inner
            .request(T::METHOD, endpoint.url()?)
            .header(XI_API_KEY_HEADER, &self.api_key)
            .version(http::Version::HTTP_3);

        if matches!(T::METHOD, Method::POST | Method::PATCH) {
            let request_body = endpoint.request_body().await?;
            builder = match request_body {
                RequestBody::Json(json) => {
                    builder.header(CONTENT_TYPE, APPLICATION_JSON).json(&json)
                }
                RequestBody::Multipart(form) => builder.multipart(form),
                RequestBody::Empty => return Err("request must have a body".into()),
            };
        }

        let resp = builder.send().await?;

        if !resp.status().is_success() {
            return Err(Box::new(HttpError(resp.json().await?)));
        }

        // Extract headers before consuming response
        let headers = resp.headers().clone();
        let body = endpoint.response_body(resp).await?;

        Ok((headers, body))
    }

    // WebSocket functionality is currently disabled
    #[allow(dead_code)]
    const FLUSH_JSON: &'static str = r#"{"text":" ","flush":true}"#;
    #[allow(dead_code)]
    const EOS_JSON: &'static str = r#"{"text":""}"#;

    // WebSocket functionality is currently disabled
    /*
    pub async fn hit_ws<S>(
        &self,
        mut endpoint: WebSocketTTS<S>,
    ) -> Result<impl Stream<Item = Result<WebSocketTTSResponse>>>
    where
        S: Stream<Item = String> + Send + 'static,
    {
        let (ws_stream, _) = connect_async(endpoint.url()).await?;
        let (mut writer, mut reader) = ws_stream.split();
        let (tx_to_caller, rx_for_caller) =
            futures_channel::mpsc::unbounded::<Result<WebSocketTTSResponse>>();

        // Perhaps remove api key setter from bos_message
        // as it is already set in the client ?
        if endpoint.body.bos_message.authorization.is_none() {
            endpoint.body.bos_message.xi_api_key = Some(self.api_key.clone());
        }

        let _reader_t: JoinHandle<Result<()>> = tokio::spawn(async move {
            while let Some(msg_result) = reader.next().await {
                let msg = msg_result?;
                match msg {
                    Message::Text(text) => {
                        let response: WebSocketTTSResponse = serde_json::from_str(&text)?;
                        tx_to_caller.unbounded_send(Ok(response))?;
                    }
                    Message::Close(msg) => {
                        if let Some(close_frame) = msg {
                            if close_frame.code == CloseCode::Normal {
                                continue;
                            } else {
                                tx_to_caller.unbounded_send(Err(Box::new(
                                    WebSocketError::NonNormalCloseCode(
                                        close_frame.reason.to_string(),
                                    ),
                                )))?;
                            }
                        } else {
                            tx_to_caller.unbounded_send(Err(Box::new(
                                WebSocketError::ClosedWithoutCloseFrame,
                            )))?;
                        }
                    }
                    _ => tx_to_caller
                        .unbounded_send(Err(Box::new(WebSocketError::UnexpectedMessageType)))?,
                }
            }
            Ok(())
        });

        let _thread: JoinHandle<Result<()>> = tokio::spawn(async move {
            let bos_message = endpoint.body.bos_message;
            writer.send(bos_message.to_message()?).await?;

            let text_stream = endpoint.body.text_stream;
            pin_mut!(text_stream);

            while let Some(chunk) = text_stream.next().await {
                writer.send(chunk.to_message()?).await?;
            }

            if endpoint.body.flush {
                writer.send(Message::from(Self::FLUSH_JSON)).await?;
            }


            writer.send(Message::from(Self::EOS_JSON)).await?;

            Ok(())
        });
        Ok(rx_for_caller)
    }
    */
}

impl From<(reqwest::Client, String)> for ElevenLabsClient {
    fn from((client, api_key): (reqwest::Client, String)) -> Self {
        Self {
            inner: client,
            api_key,
        }
    }
}
