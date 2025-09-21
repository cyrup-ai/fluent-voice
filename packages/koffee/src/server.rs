//! TCP server functionality for Koffee
//!
//! This module implements a sophisticated TCP server for real-time wake word detection
//! with support for multiple concurrent clients, audio streaming, and detection results.

use crate::{KoffeeCandle, Result};

use log::{error, info, warn};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time::timeout;

/// Maximum number of concurrent connections
const MAX_CONNECTIONS: usize = 100;

/// Connection timeout in seconds
const CONNECTION_TIMEOUT: u64 = 300; // 5 minutes

/// Maximum audio chunk size in bytes
const MAX_AUDIO_CHUNK_SIZE: usize = 8192;

/// Rate limiting: max requests per minute per client
const MAX_REQUESTS_PER_MINUTE: u32 = 1000;

/// Run a TCP server that accepts audio input and returns wake-word detections
pub async fn run_tcp_server(detector: Arc<Mutex<KoffeeCandle>>, port: u16) -> Result<()> {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .map_err(|e| format!("Failed to bind TCP server to port {}: {}", port, e))?;

    info!("🚀 TCP server listening on port {}", port);
    info!("Server configuration:");
    info!("  - Max connections: {}", MAX_CONNECTIONS);
    info!("  - Connection timeout: {}s", CONNECTION_TIMEOUT);
    info!("  - Max audio chunk size: {} bytes", MAX_AUDIO_CHUNK_SIZE);
    info!(
        "  - Rate limit: {} requests/minute per client",
        MAX_REQUESTS_PER_MINUTE
    );

    // Initialize connection manager
    let connection_manager = Arc::new(ConnectionManager::new(detector));

    // Start cleanup task for expired connections
    let cleanup_manager = connection_manager.clone();
    tokio::spawn(async move {
        cleanup_manager.cleanup_expired_connections().await;
    });

    // Accept connections loop
    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let manager = connection_manager.clone();

                // Check connection limits
                if manager.get_connection_count().await >= MAX_CONNECTIONS {
                    warn!(
                        "Connection limit reached, rejecting connection from {}",
                        addr
                    );
                    drop(stream);
                    continue;
                }

                info!("New connection from {}", addr);

                // Spawn task to handle client connection
                tokio::spawn(async move {
                    if let Err(e) = handle_client_connection(stream, addr, manager).await {
                        error!("Client connection error ({}): {}", addr, e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Connection manager for handling multiple clients
struct ConnectionManager {
    detector: Arc<Mutex<KoffeeCandle>>,
    connections: Arc<RwLock<HashMap<SocketAddr, ClientSession>>>,
}

impl ConnectionManager {
    fn new(detector: Arc<Mutex<KoffeeCandle>>) -> Self {
        Self {
            detector,
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn create_session(&self, addr: SocketAddr) -> Result<ClientSession> {
        let session = ClientSession::new(addr);

        let mut connections = self.connections.write().await;
        connections.insert(addr, session.clone());

        info!("Created session for client {}", addr);
        Ok(session)
    }

    async fn remove_session(&self, addr: SocketAddr) {
        let mut connections = self.connections.write().await;
        connections.remove(&addr);
        info!("Removed session for client {}", addr);
    }

    async fn get_connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.len()
    }

    async fn cleanup_expired_connections(&self) {
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await; // Check every minute

            let mut connections = self.connections.write().await;
            let mut expired_addrs = Vec::new();

            for (addr, session) in connections.iter() {
                if session.is_expired() {
                    expired_addrs.push(*addr);
                }
            }

            for addr in expired_addrs {
                connections.remove(&addr);
                info!("Cleaned up expired connection: {}", addr);
            }
        }
    }

    async fn process_audio(
        &self,
        addr: SocketAddr,
        audio_data: &[u8],
    ) -> Result<Option<DetectionResult>> {
        // Rate limiting check
        {
            let mut connections = self.connections.write().await;
            if let Some(session) = connections.get_mut(&addr) {
                if !session.check_rate_limit() {
                    return Err("Rate limit exceeded".to_string());
                }
            }
        }

        // Convert audio data to f32 samples
        let audio_samples = convert_audio_bytes_to_samples(audio_data)?;

        // Process with wake word detector
        let mut detector = self
            .detector
            .lock()
            .map_err(|e| format!("Failed to lock detector: {}", e))?;

        // Simulate wake word detection (replace with actual detection logic)
        if let Some(detection) = detector.process_samples(&audio_samples) {
            Ok(Some(DetectionResult {
                wake_word: detection.name.clone(),
                confidence: detection.score,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            }))
        } else {
            Ok(None)
        }
    }
}

/// Client session with rate limiting and connection tracking
#[derive(Clone)]
struct ClientSession {
    addr: SocketAddr,
    created_at: Instant,
    last_activity: Arc<Mutex<Instant>>,
    request_count: Arc<Mutex<u32>>,
    last_minute_reset: Arc<Mutex<Instant>>,
}

impl ClientSession {
    fn new(addr: SocketAddr) -> Self {
        let now = Instant::now();
        Self {
            addr,
            created_at: now,
            last_activity: Arc::new(Mutex::new(now)),
            request_count: Arc::new(Mutex::new(0)),
            last_minute_reset: Arc::new(Mutex::new(now)),
        }
    }

    fn is_expired(&self) -> bool {
        if let Ok(last_activity) = self.last_activity.lock() {
            last_activity.elapsed() > Duration::from_secs(CONNECTION_TIMEOUT)
        } else {
            true // If lock is poisoned, consider expired for safety
        }
    }

    fn check_rate_limit(&self) -> bool {
        let now = Instant::now();

        // Update last activity
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = now;
        }

        // Check if we need to reset the minute counter
        if let Ok(mut last_reset) = self.last_minute_reset.lock() {
            if now.duration_since(*last_reset) >= Duration::from_secs(60) {
                if let Ok(mut count) = self.request_count.lock() {
                    *count = 0;
                }
                *last_reset = now;
            }
        }

        // Check rate limit
        if let Ok(mut count) = self.request_count.lock() {
            if *count >= MAX_REQUESTS_PER_MINUTE {
                return false;
            }
            *count += 1;
            true
        } else {
            false // If lock is poisoned, deny request for safety
        }
    }

    /// Get connection duration for monitoring
    fn connection_duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Log connection statistics
    fn log_connection_stats(&self) {
        let duration = self.connection_duration();
        let request_count = self.request_count.lock().map(|c| *c).unwrap_or(0);
        info!(
            "Connection stats for {}: duration={}s, requests={}",
            self.addr,
            duration.as_secs(),
            request_count
        );
    }
}

/// Wake word detection result
#[derive(Debug, Clone)]
struct DetectionResult {
    wake_word: String,
    confidence: f32,
    timestamp: u64,
}

/// Audio streaming protocol messages
#[derive(Debug)]
enum ProtocolMessage {
    AudioChunk { data: Vec<u8> },
    Ping,
    Disconnect,
}

/// Handle individual client connection with full protocol support
async fn handle_client_connection(
    mut stream: TcpStream,
    addr: SocketAddr,
    manager: Arc<ConnectionManager>,
) -> Result<()> {
    // Create client session
    let session = manager.create_session(addr).await?;

    // Set connection timeout
    let mut buffer = vec![0u8; MAX_AUDIO_CHUNK_SIZE];

    info!("Client {} connected, starting audio processing", addr);

    loop {
        // Read message with timeout
        let bytes_read = match timeout(Duration::from_secs(30), stream.read(&mut buffer)).await {
            Ok(Ok(0)) => {
                session.log_connection_stats();
                info!("Client {} disconnected", addr);
                break;
            }
            Ok(Ok(n)) => n,
            Ok(Err(e)) => {
                session.log_connection_stats();
                error!("Read error from client {}: {}", addr, e);
                break;
            }
            Err(_) => {
                warn!("Client {} timed out", addr);
                break;
            }
        };

        // Parse protocol message
        let message = parse_protocol_message(&buffer[..bytes_read])?;

        match message {
            ProtocolMessage::AudioChunk { data } => {
                // Process audio data
                match manager.process_audio(addr, &data).await {
                    Ok(Some(detection)) => {
                        // Send detection result back to client
                        let response = serialize_detection_result(&detection)?;
                        if let Err(e) = stream.write_all(&response).await {
                            error!("Failed to send response to client {}: {}", addr, e);
                            break;
                        }
                    }
                    Ok(None) => {
                        // No detection, send keep-alive
                        let keep_alive = serialize_keep_alive()?;
                        if let Err(e) = stream.write_all(&keep_alive).await {
                            error!("Failed to send keep-alive to client {}: {}", addr, e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Audio processing error for client {}: {}", addr, e);
                        let error_response = serialize_error(&e.to_string())?;
                        let _ = stream.write_all(&error_response).await;
                        break;
                    }
                }
            }
            ProtocolMessage::Ping => {
                // Respond to ping
                let pong = serialize_pong()?;
                if let Err(e) = stream.write_all(&pong).await {
                    error!("Failed to send pong to client {}: {}", addr, e);
                    break;
                }
            }
            ProtocolMessage::Disconnect => {
                info!("Client {} requested disconnect", addr);
                break;
            }
        }
    }

    // Cleanup
    manager.remove_session(addr).await;
    Ok(())
}

/// Parse incoming protocol message from binary data
fn parse_protocol_message(data: &[u8]) -> Result<ProtocolMessage> {
    if data.is_empty() {
        return Err("Empty message".to_string());
    }

    match data[0] {
        0x01 => {
            // Audio chunk message
            if data.len() < 5 {
                return Err("Invalid audio chunk message".to_string());
            }
            let chunk_size = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
            if data.len() < 5 + chunk_size {
                return Err("Incomplete audio chunk".to_string());
            }
            Ok(ProtocolMessage::AudioChunk {
                data: data[5..5 + chunk_size].to_vec(),
            })
        }
        0x02 => Ok(ProtocolMessage::Ping),
        0x03 => Ok(ProtocolMessage::Disconnect),
        _ => Err(format!("Unknown message type: {}", data[0])),
    }
}

/// Serialize detection result to binary format
fn serialize_detection_result(detection: &DetectionResult) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    response.push(0x81); // Detection result message type

    // Wake word name length and data
    let name_bytes = detection.wake_word.as_bytes();
    response.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    response.extend_from_slice(name_bytes);

    // Confidence score
    response.extend_from_slice(&detection.confidence.to_le_bytes());

    // Timestamp
    response.extend_from_slice(&detection.timestamp.to_le_bytes());

    Ok(response)
}

/// Serialize keep-alive message
fn serialize_keep_alive() -> Result<Vec<u8>> {
    Ok(vec![0x82]) // Keep-alive message type
}

/// Serialize pong response
fn serialize_pong() -> Result<Vec<u8>> {
    Ok(vec![0x83]) // Pong message type
}

/// Serialize error message
fn serialize_error(error_msg: &str) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    response.push(0x84); // Error message type

    let error_bytes = error_msg.as_bytes();
    response.extend_from_slice(&(error_bytes.len() as u32).to_le_bytes());
    response.extend_from_slice(error_bytes);

    Ok(response)
}

/// Convert audio bytes to f32 samples for processing
fn convert_audio_bytes_to_samples(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err("Invalid audio data length".to_string());
    }

    let mut samples = Vec::with_capacity(data.len() / 2);

    for chunk in data.chunks_exact(2) {
        let sample_bytes = [chunk[0], chunk[1]];
        let sample_i16 = i16::from_le_bytes(sample_bytes);
        let sample_f32 = sample_i16 as f32 / 32768.0;
        samples.push(sample_f32);
    }

    Ok(samples)
}
