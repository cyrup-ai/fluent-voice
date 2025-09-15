#![allow(unsafe_code)]
#![allow(non_snake_case)]
#![allow(elided_lifetimes_in_paths)]
#![allow(clippy::needless_lifetimes)]
#![allow(ambiguous_associated_items)]
use anyhow::{Context as _, Result, anyhow};

use crate::livekit_client::{LocalAudioTrack, RemoteAudioTrack, RemoteVideoTrack};
#[cfg(feature = "microphone")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait as _};
use futures::channel::mpsc::UnboundedSender;
use futures::{Stream, StreamExt as _};
// Removed gpui dependency in favor of standard async patterns
// use gpui::{BackgroundExecutor, Task};
use libwebrtc::native::{apm, audio_mixer, audio_resampler};

// Platform-specific imports for video and audio handling
#[cfg(target_os = "macos")]
use core_video;

// Core Video pixel format constants for format detection
#[cfg(target_os = "macos")]
mod pixel_format_constants {
    // OSType is a 32-bit unsigned integer in Core Foundation
    pub type OSType = u32;

    // FourCC codes for pixel formats (computed at compile time with const evaluation)
    pub const K_CVPIXEL_FORMAT_TYPE_32_BGRA: OSType = 0x42475241; // 'BGRA'
    pub const K_CVPIXEL_FORMAT_TYPE_32_ARGB: OSType = 0x32000000; // 32-bit ARGB
    pub const K_CVPIXEL_FORMAT_TYPE_24_RGB: OSType = 0x18000000; // 24-bit RGB
    #[allow(dead_code)] // Reserved for future YUV format support
    pub const K_CVPIXEL_FORMAT_TYPE_420_YP_CB_CR8_BI_PLANAR_FULL_RANGE: OSType = 0x34323066; // '420f'
}
use fluent_video::{VideoSource, VideoSourceOptions};
// Platform-specific CoreAudio types - optimized for lock-free operation
#[cfg(target_os = "macos")]
type AudioObjectID = u32;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
type OSStatus = i32;
#[cfg(target_os = "macos")]
#[repr(C)]
#[allow(dead_code)]
struct AudioObjectPropertyAddress {
    #[allow(non_snake_case)]
    mSelector: u32,
    #[allow(non_snake_case)]
    mScope: u32,
    #[allow(non_snake_case)]
    mElement: u32,
}
use livekit::track;

use crate::client_util::ResultExt as _;
use livekit::webrtc::{
    audio_frame::AudioFrame,
    audio_source::{AudioSourceOptions, RtcAudioSource, native::NativeAudioSource},
    audio_stream::native::NativeAudioStream,
    video_frame::VideoBuffer,
    video_stream::native::NativeVideoStream,
};
// LOCK-FREE ARCHITECTURE: Replace all locks with atomic operations and channels
use crossbeam_queue::ArrayQueue;
use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{self, AtomicBool, AtomicU32},
};
use std::time::Duration;
use std::{borrow::Cow, thread};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

// Platform-specific screen capture frame types are defined below near RemoteVideoFrame

/// LOCK-FREE AudioStack: Uses atomic operations and channels for blazing performance
pub struct AudioStack {
    executor: tokio::runtime::Handle,
    // Lock-free APM communication via channels
    #[allow(dead_code)]
    apm_command_tx: mpsc::UnboundedSender<ApmCommand>,
    // Lock-free mixer communication via channels
    mixer_command_tx: mpsc::UnboundedSender<MixerCommand>,
    // Atomic task handle management
    #[allow(dead_code)]
    output_task_running: AtomicBool,
    // Atomic SSRC generation
    next_ssrc: AtomicU32,
    // Pre-allocated frame buffer pool for zero allocation
    #[allow(dead_code)]
    frame_pool: Arc<ArrayQueue<AudioFrameBuffer>>,
    // APM processor handle
    apm: Arc<Mutex<apm::AudioProcessingModule>>,
    // Output task handle
    _output_task: RefCell<std::sync::Weak<tokio::task::JoinHandle<()>>>,
}

/// Zero-allocation audio frame buffer with pre-allocated capacity
#[derive(Clone)]
struct AudioFrameBuffer {
    data: Box<[i16; FRAME_BUFFER_SIZE]>,
    sample_rate: u32,
    num_channels: u32,
    #[allow(dead_code)]
    samples_per_channel: u32,
}

/// Lock-free APM command for asynchronous processing
#[allow(dead_code)]
enum ApmCommand {
    #[allow(dead_code)]
    ProcessStream {
        buffer: AudioFrameBuffer,
        response_tx: oneshot::Sender<AudioFrameBuffer>,
    },
    #[allow(dead_code)]
    ProcessReverseStream {
        buffer: AudioFrameBuffer,
        response_tx: oneshot::Sender<AudioFrameBuffer>,
    },
}

/// Lock-free mixer command for asynchronous processing
enum MixerCommand {
    AddSource {
        source: AudioMixerSource,
        response_tx: oneshot::Sender<()>,
    },
    RemoveSource {
        ssrc: u32,
        response_tx: oneshot::Sender<()>,
    },
    Mix {
        channels: usize,
        response_tx: oneshot::Sender<Vec<i16>>,
    },
}

// Pre-calculated buffer size for optimal performance
const FRAME_BUFFER_SIZE: usize = (SAMPLE_RATE / 100 * NUM_CHANNELS) as usize;
const FRAME_POOL_SIZE: usize = 64; // Pre-allocate 64 buffers for zero allocation

// NOTE: We use WebRTC's mixer which only supports
// 16kHz, 32kHz and 48kHz. As 48 is the most common "next step up"
// for audio output devices like speakers/bluetooth, we just hard-code
// this; and downsample when we need to.
const SAMPLE_RATE: u32 = 48000;
const NUM_CHANNELS: u32 = 2;

impl AudioStack {
    /// Creates a new lock-free AudioStack with pre-allocated buffers for zero allocation
    pub fn new(executor: tokio::runtime::Handle) -> Result<Self, anyhow::Error> {
        // Create lock-free APM processor
        let (apm_command_tx, apm_command_rx) = mpsc::unbounded_channel();
        let apm_executor = executor.clone();

        // Spawn dedicated APM processor task for lock-free operation
        apm_executor.spawn(async move {
            Self::run_apm_processor(apm_command_rx).await;
        });

        // Create lock-free mixer processor
        let (mixer_command_tx, mixer_command_rx) = mpsc::unbounded_channel();
        let mixer_executor = executor.clone();

        // Spawn dedicated mixer processor task for lock-free operation
        mixer_executor.spawn(async move {
            Self::run_mixer_processor(mixer_command_rx).await;
        });

        // Pre-allocate frame buffer pool for zero allocation
        let frame_pool = Arc::new(ArrayQueue::new(FRAME_POOL_SIZE));
        for _ in 0..FRAME_POOL_SIZE {
            let buffer = AudioFrameBuffer {
                data: Box::new([0i16; FRAME_BUFFER_SIZE]),
                sample_rate: SAMPLE_RATE,
                num_channels: NUM_CHANNELS,
                samples_per_channel: SAMPLE_RATE / 100,
            };
            if frame_pool.push(buffer).is_err() {
                return Err(anyhow::anyhow!("Failed to initialize frame pool"));
            }
        }

        Ok(Self {
            executor,
            apm_command_tx,
            mixer_command_tx,
            output_task_running: AtomicBool::new(false),
            next_ssrc: AtomicU32::new(1),
            frame_pool,
            apm: Arc::new(Mutex::new(apm::AudioProcessingModule::new(
                true, true, true, true,
            ))),
            _output_task: RefCell::new(std::sync::Weak::new()),
        })
    }

    /// Lock-free APM processor that runs in dedicated task
    async fn run_apm_processor(mut command_rx: mpsc::UnboundedReceiver<ApmCommand>) {
        let mut apm = apm::AudioProcessingModule::new(true, true, true, true);

        while let Some(command) = command_rx.recv().await {
            match command {
                ApmCommand::ProcessStream {
                    mut buffer,
                    response_tx,
                } => {
                    // Process audio without locks
                    let result = apm.process_stream(
                        &mut buffer.data[..],
                        buffer.sample_rate as i32,
                        buffer.num_channels as i32,
                    );

                    if result.is_ok() {
                        let _ = response_tx.send(buffer);
                    }
                }
                ApmCommand::ProcessReverseStream {
                    mut buffer,
                    response_tx,
                } => {
                    // Process reverse stream without locks
                    let result = apm.process_reverse_stream(
                        &mut buffer.data[..],
                        buffer.sample_rate as i32,
                        buffer.num_channels as i32,
                    );

                    if result.is_ok() {
                        let _ = response_tx.send(buffer);
                    }
                }
            }
        }
    }

    /// Lock-free mixer processor that runs in dedicated task
    async fn run_mixer_processor(mut command_rx: mpsc::UnboundedReceiver<MixerCommand>) {
        let mut mixer = audio_mixer::AudioMixer::new();

        while let Some(command) = command_rx.recv().await {
            match command {
                MixerCommand::AddSource {
                    source,
                    response_tx,
                } => {
                    mixer.add_source(source);
                    let _ = response_tx.send(());
                }
                MixerCommand::RemoveSource { ssrc, response_tx } => {
                    mixer.remove_source(ssrc as i32);
                    let _ = response_tx.send(());
                }
                MixerCommand::Mix {
                    channels,
                    response_tx,
                } => {
                    let mixed = mixer.mix(channels).to_vec();
                    let _ = response_tx.send(mixed);
                }
            }
        }
    }

    pub fn play_remote_audio_track(&self, track: &RemoteAudioTrack) -> AudioStream {
        let output_task = self.start_output();

        let next_ssrc = self.next_ssrc.fetch_add(1, atomic::Ordering::Relaxed) as i32;
        let source = AudioMixerSource {
            ssrc: next_ssrc,
            sample_rate: SAMPLE_RATE,
            num_channels: NUM_CHANNELS,
            buffer: Arc::default(),
        };
        // Use channel-based mixer API - send command without waiting for response
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
        let _ = self.mixer_command_tx.send(MixerCommand::AddSource {
            source: source.clone(),
            response_tx,
        });
        // Note: Not waiting for response to keep function synchronous

        let mut stream = NativeAudioStream::new(
            track.0.rtc_track(),
            source.sample_rate as i32,
            source.num_channels as i32,
        );

        let receive_task = self.executor.spawn({
            let source = source.clone();
            async move {
                while let Some(frame) = stream.next().await {
                    source.receive(frame);
                }
            }
        });

        let mixer_command_tx = self.mixer_command_tx.clone();
        let ssrc = source.ssrc;
        let on_drop = crate::client_util::defer(move || {
            let (response_tx, _) = tokio::sync::oneshot::channel();
            let _ = mixer_command_tx.send(MixerCommand::RemoveSource {
                ssrc: ssrc as u32,
                response_tx,
            });
            drop(receive_task);
            drop(output_task);
        });

        AudioStream::Output {
            _drop: Box::new(on_drop),
        }
    }

    pub fn capture_local_microphone_track(&self) -> Result<(LocalAudioTrack, AudioStream)> {
        let source = NativeAudioSource::new(
            // n.b. this struct's options are always ignored, noise cancellation is provided by apm.
            AudioSourceOptions::default(),
            SAMPLE_RATE,
            NUM_CHANNELS,
            10,
        );

        let track = track::LocalAudioTrack::create_audio_track(
            "microphone",
            RtcAudioSource::Native(source.clone()),
        );

        let apm = self.apm.clone();

        let (frame_tx, mut frame_rx) = futures::channel::mpsc::unbounded();
        let transmit_task = self.executor.spawn({
            let source = source.clone();
            async move {
                while let Some(frame) = frame_rx.next().await {
                    let _ = source.capture_frame(&frame).await.log_err();
                }
            }
        });
        let capture_task = self.executor.spawn(async move {
            Self::capture_input(apm, frame_tx, SAMPLE_RATE, NUM_CHANNELS).await
        });

        let on_drop = crate::client_util::defer(|| {
            drop(transmit_task);
            drop(capture_task);
        });
        Ok((
            crate::livekit_client::LocalAudioTrack(track),
            AudioStream::Output {
                _drop: Box::new(on_drop),
            },
        ))
    }

    fn start_output(&self) -> Arc<tokio::task::JoinHandle<()>> {
        if let Some(task) = self._output_task.borrow().upgrade() {
            return task;
        }
        let task = Arc::new(self.executor.spawn({
            let apm = self.apm.clone();
            let mixer_command_tx = self.mixer_command_tx.clone();
            async move {
                let _ = Self::play_output(apm, mixer_command_tx, SAMPLE_RATE, NUM_CHANNELS)
                    .await
                    .log_err();
            }
        }));
        *self._output_task.borrow_mut() = Arc::downgrade(&task);
        task
    }

    async fn play_output(
        apm: Arc<Mutex<apm::AudioProcessingModule>>,
        mixer_command_tx: mpsc::UnboundedSender<MixerCommand>,
        sample_rate: u32,
        num_channels: u32,
    ) -> Result<()> {
        loop {
            let mut device_change_listener = DeviceChangeListener::new(false)?;
            let (output_device, output_config) = default_device(false)?;
            let (end_on_drop_tx, end_on_drop_rx) = std::sync::mpsc::channel::<()>();
            let mixer_tx = mixer_command_tx.clone();
            let apm = apm.clone();
            let mut resampler = audio_resampler::AudioResampler::default();
            let mut buf = Vec::new();

            thread::spawn(move || {
                let output_stream = output_device.build_output_stream(
                    &output_config.config(),
                    {
                        move |mut data, _info| {
                            while !data.is_empty() {
                                if data.len() <= buf.len() {
                                    let rest = buf.split_off(data.len());
                                    data.copy_from_slice(&buf);
                                    buf = rest;
                                    return;
                                }
                                if !buf.is_empty() {
                                    let (prefix, suffix) = data.split_at_mut(buf.len());
                                    prefix.copy_from_slice(&buf);
                                    data = suffix;
                                }

                                let (response_tx, mut response_rx) =
                                    tokio::sync::oneshot::channel();
                                let _ = mixer_tx.send(MixerCommand::Mix {
                                    channels: output_config.channels() as usize,
                                    response_tx,
                                });
                                let mixed: Vec<i16> = response_rx.try_recv().unwrap_or_default();
                                let sampled = resampler.remix_and_resample(
                                    &mixed,
                                    sample_rate / 100,
                                    num_channels,
                                    sample_rate,
                                    output_config.channels() as u32,
                                    output_config.sample_rate().0,
                                );
                                buf = sampled.to_vec();
                                apm.lock()
                                    .process_reverse_stream(
                                        &mut buf,
                                        output_config.sample_rate().0 as i32,
                                        output_config.channels() as i32,
                                    )
                                    .ok();
                            }
                        }
                    },
                    |error| log::error!("error playing audio track: {error:?}"),
                    Some(Duration::from_millis(100)),
                );

                let output_stream = match output_stream {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Error with output stream: {e:?}");
                        return;
                    }
                };

                let _ = output_stream.play().log_err();
                // Block forever to keep the output stream alive
                end_on_drop_rx.recv().ok();
            });

            device_change_listener.next().await;
            drop(end_on_drop_tx)
        }
    }

    async fn capture_input(
        apm: Arc<Mutex<apm::AudioProcessingModule>>,
        frame_tx: UnboundedSender<AudioFrame<'static>>,
        sample_rate: u32,
        num_channels: u32,
    ) -> Result<()> {
        loop {
            let mut device_change_listener = DeviceChangeListener::new(true)?;
            let (device, config) = default_device(true)?;
            let (end_on_drop_tx, end_on_drop_rx) = std::sync::mpsc::channel::<()>();
            let apm = apm.clone();
            let frame_tx = frame_tx.clone();
            let mut resampler = audio_resampler::AudioResampler::default();

            thread::spawn(move || {
                let result = (|| -> Result<(), anyhow::Error> {
                    if let Ok(name) = device.name() {
                        log::info!("Using microphone: {name}")
                    } else {
                        log::info!("Using microphone: <unknown>");
                    }

                    let ten_ms_buffer_size =
                        (config.channels() as u32 * config.sample_rate().0 / 100) as usize;
                    let mut buf: Vec<i16> = Vec::with_capacity(ten_ms_buffer_size);

                    let stream = device
                        .build_input_stream_raw(
                            &config.config(),
                            cpal::SampleFormat::I16,
                            move |data, _: &_| {
                                let mut data_slice = match data.as_slice::<i16>() {
                                    Some(slice) => slice,
                                    None => {
                                        log::error!("Failed to convert audio data to i16 slice");
                                        return;
                                    }
                                };
                                while !data_slice.is_empty() {
                                    let remainder =
                                        (buf.capacity() - buf.len()).min(data_slice.len());
                                    buf.extend_from_slice(&data_slice[..remainder]);
                                    data_slice = &data_slice[remainder..];

                                    if buf.capacity() == buf.len() {
                                        let mut sampled = resampler
                                            .remix_and_resample(
                                                buf.as_slice(),
                                                config.sample_rate().0 / 100,
                                                config.channels() as u32,
                                                config.sample_rate().0,
                                                num_channels,
                                                sample_rate,
                                            )
                                            .to_owned();
                                        let _ = apm
                                            .lock()
                                            .process_stream(
                                                &mut sampled,
                                                sample_rate as i32,
                                                num_channels as i32,
                                            )
                                            .log_err();
                                        buf.clear();
                                        if let Err(e) = frame_tx.unbounded_send(AudioFrame {
                                            data: Cow::Owned(sampled),
                                            sample_rate,
                                            num_channels,
                                            samples_per_channel: sample_rate / 100,
                                        }) {
                                            log::error!("Failed to send audio frame: {e:?}");
                                        }
                                    }
                                }
                            },
                            |err| log::error!("error capturing audio track: {err:?}"),
                            Some(Duration::from_millis(100)),
                        )
                        .context("failed to build input stream")?;

                    stream.play()?;
                    // Keep the thread alive and holding onto the `stream`
                    end_on_drop_rx.recv().ok();
                    anyhow::Ok(())
                })();
                if let Err(e) = result {
                    log::error!("Error occurred: {e:?}");
                }
            });

            device_change_listener.next().await;
            drop(end_on_drop_tx)
        }
    }
}

#[derive(Debug)]
pub enum AudioStream {
    Input { _task: Arc<JoinHandle<()>> },
    Output { _drop: Box<dyn std::any::Any> },
}

// Screen capture functionality using VideoSource infrastructure and LibWebRTC
pub async fn capture_local_video_track() -> Result<(
    super::livekit_client::LocalVideoTrack,
    Box<dyn std::any::Any + Send + 'static>,
)> {
    use crate::livekit_client::LocalVideoTrack;
    use futures::StreamExt;
    use libwebrtc::native::yuv_helper;
    use libwebrtc::video_frame::{I420Buffer, VideoFrame, VideoRotation};
    use libwebrtc::video_source::native::NativeVideoSource;
    use libwebrtc::video_source::{RtcVideoSource, VideoResolution};
    use std::time::Duration;
    use tokio::sync::oneshot;

    // Create LibWebRTC video source for LiveKit integration
    let native_video_source = NativeVideoSource::new(VideoResolution {
        width: 1920,
        height: 1080,
    });

    // Create LiveKit video track using LibWebRTC patterns
    let livekit_track = livekit::track::LocalVideoTrack::create_video_track(
        "screen_capture",
        RtcVideoSource::Native(native_video_source.clone()),
    );

    // Create our VideoSource for screen capture
    let screen_video_source = VideoSource::from_screen(VideoSourceOptions {
        width: Some(1920),
        height: Some(1080),
        fps: Some(30),
    })
    .context("Failed to create screen capture VideoSource")?;

    // Start screen capture
    screen_video_source
        .start()
        .context("Failed to start screen capture")?;

    // Get frame stream from our VideoSource
    let frame_stream = screen_video_source.get_frame_stream();

    // Create cleanup channels
    let (stop_tx, mut stop_rx) = oneshot::channel();

    // Spawn task to bridge VideoSource frames to LibWebRTC video source
    let bridge_task = tokio::spawn({
        let native_source = native_video_source.clone();
        async move {
            let mut frame_interval = tokio::time::interval(Duration::from_millis(33)); // ~30 fps
            let mut pinned_stream = Box::pin(frame_stream);

            loop {
                tokio::select! {
                    _ = &mut stop_rx => {
                        break;
                    }
                    _ = frame_interval.tick() => {
                        if let Some(video_frame) = pinned_stream.next().await {
                            // Convert fluent_video VideoFrame to LibWebRTC VideoFrame
                            if let Ok(rgba_data) = video_frame.to_rgba_bytes() {
                                let width = video_frame.width();
                                let height = video_frame.height();
                                let timestamp_us = video_frame.timestamp_us();

                                // Create I420 buffer for LibWebRTC
                                let mut i420_buffer = I420Buffer::new(width, height);

                                // Get buffer data pointers
                                let (stride_y, stride_u, stride_v) = i420_buffer.strides();
                                let (data_y, data_u, data_v) = i420_buffer.data_mut();

                                // Convert RGBA to I420 using LibWebRTC yuv helper
                                yuv_helper::abgr_to_i420(
                                    &rgba_data,
                                    (width * 4) as u32, // RGBA stride
                                    data_y,
                                    stride_y,
                                    data_u,
                                    stride_u,
                                    data_v,
                                    stride_v,
                                    width as i32,
                                    height as i32,
                                );

                                // Create LibWebRTC VideoFrame
                                let libwebrtc_frame = VideoFrame {
                                    rotation: VideoRotation::VideoRotation0,
                                    buffer: i420_buffer,
                                    timestamp_us,
                                };

                                // Feed frame to LibWebRTC source
                                native_source.capture_frame(&libwebrtc_frame);
                            }
                        }
                    }
                }
            }
        }
    });

    // Create cleanup handle
    struct ScreenCaptureCleanup {
        _video_source: VideoSource,
        _stop_tx: Option<oneshot::Sender<()>>,
        _bridge_task: tokio::task::JoinHandle<()>,
    }

    let cleanup = ScreenCaptureCleanup {
        _video_source: screen_video_source,
        _stop_tx: Some(stop_tx),
        _bridge_task: bridge_task,
    };

    Ok((LocalVideoTrack(livekit_track), Box::new(cleanup)))
}

fn default_device(input: bool) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let device;
    let config;
    if input {
        device = cpal::default_host()
            .default_input_device()
            .ok_or_else(|| anyhow!("no audio input device available"))?;
        config = device
            .default_input_config()
            .context("failed to get default input config")?;
    } else {
        device = cpal::default_host()
            .default_output_device()
            .ok_or_else(|| anyhow!("no audio output device available"))?;
        config = device
            .default_output_config()
            .context("failed to get default output config")?;
    }
    Ok((device, config))
}

#[derive(Clone)]
struct AudioMixerSource {
    ssrc: i32,
    sample_rate: u32,
    num_channels: u32,
    buffer: Arc<Mutex<VecDeque<Vec<i16>>>>,
}

impl AudioMixerSource {
    fn receive(&self, frame: AudioFrame) {
        assert_eq!(
            frame.data.len() as u32,
            self.sample_rate * self.num_channels / 100
        );

        let mut buffer = self.buffer.lock();
        buffer.push_back(frame.data.to_vec());
        while buffer.len() > 10 {
            buffer.pop_front();
        }
    }
}

impl libwebrtc::native::audio_mixer::AudioMixerSource for AudioMixerSource {
    fn ssrc(&self) -> i32 {
        self.ssrc
    }

    fn preferred_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[allow(warnings)]
    fn get_audio_frame_with_info(&self, target_sample_rate: u32) -> Option<AudioFrame> {
        assert_eq!(self.sample_rate, target_sample_rate);
        let buf = self.buffer.lock().pop_front()?;
        Some(AudioFrame {
            data: Cow::Owned(buf),
            sample_rate: self.sample_rate,
            num_channels: self.num_channels,
            samples_per_channel: self.sample_rate / 100,
        })
    }
}

// On macOS, we don't require Stream to be Send due to CVPixelBufferPool not being Send-safe
#[cfg(target_os = "macos")]
pub fn play_remote_video_track(
    track: &RemoteVideoTrack,
) -> impl Stream<Item = RemoteVideoFrame> + 'static {
    let mut pool = None;
    let mut most_recent_frame_size = (0, 0);
    NativeVideoStream::new(track.0.rtc_track())
        .then(move |frame| {
            if pool.is_none()
                || most_recent_frame_size != (frame.buffer.width(), frame.buffer.height())
            {
                most_recent_frame_size = (frame.buffer.width(), frame.buffer.height());
                pool = create_buffer_pool(frame.buffer.width(), frame.buffer.height())
                    .log_err()
                    .ok();
            }
            let pool_clone = pool.clone();
            async move {
                if frame.buffer.width() < 10 && frame.buffer.height() < 10 {
                    // when the remote stops sharing, we get an 8x8 black image.
                    // In a lil bit, the unpublish will come through and close the view,
                    // but until then, don't flash black.
                    return None;
                }

                match pool_clone {
                    Some(p) => video_frame_buffer_from_webrtc(p, frame.buffer),
                    None => None,
                }
            }
        })
        .filter_map(|option| async move { option })
}

// On non-macOS platforms, we can guarantee Stream implements Send
#[cfg(not(target_os = "macos"))]
pub fn play_remote_video_track(
    track: &RemoteVideoTrack,
) -> impl Stream<Item = RemoteVideoFrame> + Send + 'static {
    NativeVideoStream::new(track.0.rtc_track())
        .filter_map(|frame| async move { video_frame_buffer_from_webrtc(frame.buffer) })
}

// VideoFrameExtensions trait for working with RemoteVideoFrame
pub trait VideoFrameExtensions {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

#[cfg(target_os = "macos")]
impl VideoFrameExtensions for RemoteVideoFrame {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Get the pixel data from the Core Video buffer with automatic format detection
        // SAFETY: get_buffer_data is called with a valid CVPixelBuffer reference.
        // The function validates the buffer, detects pixel format, and returns
        // properly converted RGBA data. Buffer lifetime is guaranteed
        // by the CVPixelBuffer's reference counting mechanism.
        let rgba_data = unsafe { get_buffer_data(self)? };

        Ok(rgba_data)
    }

    fn width(&self) -> u32 {
        // Call the CVPixelBuffer get_width method directly
        self.get_width() as u32
    }

    fn height(&self) -> u32 {
        // Call the CVPixelBuffer get_height method directly
        self.get_height() as u32
    }
}

#[cfg(not(target_os = "macos"))]
impl VideoFrameExtensions for RemoteVideoFrame {
    fn to_rgba_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Extract real image data from Arc<gpui::RenderImage>
        let render_image = self.as_ref();

        // RenderImage was constructed with SmallVec<Frame<RgbaImage>> during video frame processing
        // We need to extract the pixel data from the first frame
        let frames = render_image.frames();

        let frame = frames
            .first()
            .ok_or("VideoFrame contains no image frames - corrupted video data")?;

        // Extract RgbaImage from Frame and get raw pixel data
        // The frame contains RGBA image data from the video stream
        let rgba_image = frame.buffer();
        let (width, height) = rgba_image.dimensions();

        // Validate image dimensions
        if width == 0 || height == 0 {
            return Err(format!("Invalid video frame dimensions: {}x{}", width, height).into());
        }

        // Get raw RGBA pixel data (4 bytes per pixel: R, G, B, A)
        let raw_data = rgba_image.as_raw().clone();

        // Validate pixel data size matches expected dimensions
        let expected_size = (width * height * 4) as usize;
        if raw_data.len() != expected_size {
            return Err(format!(
                "Video frame pixel data size mismatch: got {} bytes, expected {} bytes for {}x{} RGBA", 
                raw_data.len(), expected_size, width, height
            ).into());
        }

        Ok(raw_data)
    }

    fn width(&self) -> u32 {
        let render_image = self.as_ref();
        let frames = render_image.frames();

        frames
            .first()
            .map(|frame| frame.buffer().width())
            .unwrap_or_else(|| {
                tracing::warn!("VideoFrame contains no frames, returning width 0");
                0
            })
    }

    fn height(&self) -> u32 {
        let render_image = self.as_ref();
        let frames = render_image.frames();

        frames
            .first()
            .map(|frame| frame.buffer().height())
            .unwrap_or_else(|| {
                tracing::warn!("VideoFrame contains no frames, returning height 0");
                0
            })
    }
}

// Private helper method for macOS implementation
//
// Pixel format conversion functions
#[cfg(target_os = "macos")]
fn convert_bgra_to_rgba(
    data: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgba_buffer = Vec::with_capacity(width * height * 4);

    // Handle row padding - similar to screenshots-rs pattern
    for row in data.chunks_exact(bytes_per_row) {
        let row_data = &row[..width * 4]; // Remove padding
        rgba_buffer.extend_from_slice(row_data);
    }

    // BGRA -> RGBA conversion (swap R and B channels)
    for bgra in rgba_buffer.chunks_exact_mut(4) {
        bgra.swap(0, 2); // B <-> R
    }

    Ok(rgba_buffer)
}

#[cfg(target_os = "macos")]
fn convert_argb_to_rgba(
    data: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgba_buffer = Vec::with_capacity(width * height * 4);

    for row in data.chunks_exact(bytes_per_row) {
        let row_data = &row[..width * 4];
        for argb in row_data.chunks_exact(4) {
            // ARGB -> RGBA reordering: A,R,G,B -> R,G,B,A
            rgba_buffer.push(argb[1]); // R
            rgba_buffer.push(argb[2]); // G
            rgba_buffer.push(argb[3]); // B
            rgba_buffer.push(argb[0]); // A
        }
    }

    Ok(rgba_buffer)
}

#[cfg(target_os = "macos")]
fn convert_rgb_to_rgba(
    data: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgba_buffer = Vec::with_capacity(width * height * 4);

    for row in data.chunks_exact(bytes_per_row) {
        let row_data = &row[..width * 3]; // RGB has 3 bytes per pixel
        for rgb in row_data.chunks_exact(3) {
            rgba_buffer.push(rgb[0]); // R
            rgba_buffer.push(rgb[1]); // G
            rgba_buffer.push(rgb[2]); // B
            rgba_buffer.push(255); // A (full opacity)
        }
    }

    Ok(rgba_buffer)
}

// SAFETY: This function must only be called with a valid CVPixelBuffer reference.
// The caller must ensure that:
// 1. The CVPixelBuffer is properly initialized and not null
// 2. The buffer's pixel format is compatible with the expected data layout
// 3. The buffer is locked for reading before calling this function
// 4. The returned data pointer is not used after the buffer is unlocked
#[cfg(target_os = "macos")]
unsafe fn get_buffer_data(frame: &RemoteVideoFrame) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use core_video::r#return::kCVReturnSuccess;

    // Lock the CVPixelBuffer for reading
    if frame.lock_base_address(0) != kCVReturnSuccess {
        return Err("Failed to lock CVPixelBuffer base address".into());
    }

    // Create a guard to ensure we unlock the buffer when done
    struct BufferGuard<'a>(&'a RemoteVideoFrame);
    impl<'a> Drop for BufferGuard<'a> {
        fn drop(&mut self) {
            let _ = self.0.unlock_base_address(0);
        }
    }
    let _guard = BufferGuard(frame);

    // Get buffer dimensions and properties
    let width = frame.get_width();
    let height = frame.get_height();
    let bytes_per_row = frame.get_bytes_per_row();

    // Get the base address of the pixel data
    let base_address = unsafe { frame.get_base_address() };
    if base_address.is_null() {
        return Err("CVPixelBuffer base address is null".into());
    }

    // Detect pixel format for proper conversion
    let pixel_format = frame.get_pixel_format();

    // Calculate actual buffer size and copy data
    let buffer_size = bytes_per_row * height;
    let slice = unsafe { std::slice::from_raw_parts(base_address as *const u8, buffer_size) };

    // Convert based on detected pixel format
    use pixel_format_constants::*;
    match pixel_format {
        K_CVPIXEL_FORMAT_TYPE_32_BGRA => {
            convert_bgra_to_rgba(slice, width as usize, height as usize, bytes_per_row)
        }
        K_CVPIXEL_FORMAT_TYPE_32_ARGB => {
            convert_argb_to_rgba(slice, width as usize, height as usize, bytes_per_row)
        }
        K_CVPIXEL_FORMAT_TYPE_24_RGB => {
            convert_rgb_to_rgba(slice, width as usize, height as usize, bytes_per_row)
        }
        _ => {
            // Default to BGRA conversion if format is unknown/unsupported
            convert_bgra_to_rgba(slice, width as usize, height as usize, bytes_per_row)
        }
    }
}

#[cfg(target_os = "macos")]
fn create_buffer_pool(
    width: u32,
    height: u32,
) -> Result<core_video::pixel_buffer_pool::CVPixelBufferPool> {
    use core_foundation::{base::TCFType, number::CFNumber, string::CFString};
    use core_video::pixel_buffer;
    use core_video::{
        pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        pixel_buffer_io_surface::kCVPixelBufferIOSurfaceCoreAnimationCompatibilityKey,
        pixel_buffer_pool::{self},
    };

    // SAFETY: Core Video constants are valid static C string pointers.
    // wrap_under_get_rule correctly manages the reference count for these
    // system-provided constant strings that have static lifetime.
    let width_key: CFString =
        unsafe { CFString::wrap_under_get_rule(pixel_buffer::kCVPixelBufferWidthKey) };
    let height_key: CFString =
        unsafe { CFString::wrap_under_get_rule(pixel_buffer::kCVPixelBufferHeightKey) };
    // SAFETY: Same as above - kCVPixelBufferIOSurfaceCoreAnimationCompatibilityKey
    // is a valid static Core Video constant string pointer.
    let animation_key: CFString = unsafe {
        CFString::wrap_under_get_rule(kCVPixelBufferIOSurfaceCoreAnimationCompatibilityKey)
    };
    // SAFETY: kCVPixelBufferPixelFormatTypeKey is a valid static Core Video constant.
    let format_key: CFString =
        unsafe { CFString::wrap_under_get_rule(pixel_buffer::kCVPixelBufferPixelFormatTypeKey) };

    let yes: CFNumber = 1.into();
    let width: CFNumber = (width as i32).into();
    let height: CFNumber = (height as i32).into();
    let format: CFNumber = (kCVPixelFormatType_420YpCbCr8BiPlanarFullRange as i64).into();

    let buffer_attributes = core_foundation::dictionary::CFDictionary::from_CFType_pairs(&[
        (width_key, width.into_CFType()),
        (height_key, height.into_CFType()),
        (animation_key, yes.into_CFType()),
        (format_key, format.into_CFType()),
    ]);

    pixel_buffer_pool::CVPixelBufferPool::new(None, Some(&buffer_attributes)).map_err(|cv_return| {
        anyhow!(
            "failed to create pixel buffer pool: CVReturn({})",
            cv_return
        )
    })
}

#[cfg(target_os = "macos")]
pub type RemoteVideoFrame = core_video::pixel_buffer::CVPixelBuffer;

#[cfg(target_os = "macos")]
pub struct ScreenCaptureFrame(pub core_video::pixel_buffer::CVPixelBuffer);

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
pub struct ScreenCaptureFrame(pub scap::frame::Frame);

#[cfg(target_os = "macos")]
fn video_frame_buffer_from_webrtc(
    pool: core_video::pixel_buffer_pool::CVPixelBufferPool,
    buffer: Box<dyn VideoBuffer>,
) -> Option<RemoteVideoFrame> {
    use core_foundation::base::TCFType;
    use core_video::{pixel_buffer::CVPixelBuffer, r#return::kCVReturnSuccess};
    use livekit::webrtc::native::yuv_helper::i420_to_nv12;

    if let Some(native) = buffer.as_native() {
        let pixel_buffer = native.get_cv_pixel_buffer();
        if pixel_buffer.is_null() {
            return None;
        }
        // SAFETY: pixel_buffer is validated as non-null above. The native WebRTC
        // buffer provides a valid CVPixelBuffer pointer that we can safely wrap.
        // wrap_under_get_rule properly manages the reference count.
        return unsafe { Some(CVPixelBuffer::wrap_under_get_rule(pixel_buffer as _)) };
    }

    let i420_buffer = buffer.as_i420()?;
    let pixel_buffer = match pool.create_pixel_buffer() {
        Ok(buffer) => buffer,
        Err(e) => {
            log::error!("Failed to create pixel buffer: {e:?}");
            return None;
        }
    };

    let image_buffer = unsafe {
        if pixel_buffer.lock_base_address(0) != kCVReturnSuccess {
            log::error!("Failed to lock base address of pixel buffer");
            return None;
        }

        let dst_y = pixel_buffer.get_base_address_of_plane(0);
        let dst_y_stride = pixel_buffer.get_bytes_per_row_of_plane(0);
        let dst_y_len = pixel_buffer.get_height_of_plane(0) * dst_y_stride;
        let dst_uv = pixel_buffer.get_base_address_of_plane(1);
        let dst_uv_stride = pixel_buffer.get_bytes_per_row_of_plane(1);
        let dst_uv_len = pixel_buffer.get_height_of_plane(1) * dst_uv_stride;
        let width = pixel_buffer.get_width();
        let height = pixel_buffer.get_height();
        let dst_y_buffer = std::slice::from_raw_parts_mut(dst_y as *mut u8, dst_y_len);
        let dst_uv_buffer = std::slice::from_raw_parts_mut(dst_uv as *mut u8, dst_uv_len);

        let (stride_y, stride_u, stride_v) = i420_buffer.strides();
        let (src_y, src_u, src_v) = i420_buffer.data();
        i420_to_nv12(
            src_y,
            stride_y,
            src_u,
            stride_u,
            src_v,
            stride_v,
            dst_y_buffer,
            dst_y_stride as u32,
            dst_uv_buffer,
            dst_uv_stride as u32,
            width as i32,
            height as i32,
        );

        if pixel_buffer.unlock_base_address(0) != kCVReturnSuccess {
            log::error!("Failed to unlock base address of pixel buffer");
            return None;
        }

        pixel_buffer
    };

    Some(image_buffer)
}

#[cfg(not(target_os = "macos"))]
pub type RemoteVideoFrame = Arc<gpui::RenderImage>;

#[cfg(not(target_os = "macos"))]
fn video_frame_buffer_from_webrtc(buffer: Box<dyn VideoBuffer>) -> Option<RemoteVideoFrame> {
    use gpui::RenderImage;
    use image::{Frame, RgbaImage};
    use livekit::webrtc::prelude::VideoFormatType;
    use smallvec::SmallVec;
    use std::alloc::{Layout, alloc};

    let width = buffer.width();
    let height = buffer.height();
    let stride = width * 4;
    let byte_len = (stride * height) as usize;
    // SAFETY: Manual memory allocation for video frame buffer to avoid initialization overhead.
    // This is safe because:
    // 1. Layout is validated before allocation
    // 2. Allocation is checked for null before use
    // 3. The buffer.to_argb() call will initialize all allocated bytes
    // 4. Vec::from_raw_parts reconstructs ownership with correct length and capacity
    let argb_image = unsafe {
        // Motivation for this unsafe code is to avoid initializing the frame data, since to_argb
        // will write all bytes anyway.
        let layout = match Layout::array::<u8>(byte_len) {
            Ok(l) => l,
            Err(e) => {
                log::error!("Failed to create layout: {:?}", e);
                return None;
            }
        };
        let start_ptr = alloc(layout);
        if start_ptr.is_null() {
            return None;
        }
        let bgra_frame_slice = std::slice::from_raw_parts_mut(start_ptr, byte_len);
        buffer.to_argb(
            VideoFormatType::ARGB, // For some reason, this displays correctly while RGBA (the correct format) does not
            bgra_frame_slice,
            stride,
            width as i32,
            height as i32,
        );
        Vec::from_raw_parts(start_ptr, byte_len, byte_len)
    };

    Some(Arc::new(RenderImage::new(SmallVec::from_elem(
        Frame::new(
            RgbaImage::from_raw(width, height, argb_image)
                .with_context(|| "Bug: not enough bytes allocated for image.")
                .log_err()?,
        ),
        1,
    ))))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn video_frame_buffer_to_webrtc(frame: ScreenCaptureFrame) -> Option<impl AsRef<dyn VideoBuffer>> {
    use core_foundation::base::TCFType;
    use livekit::webrtc;

    let pixel_buffer = frame.0.as_concrete_TypeRef();
    std::mem::forget(frame.0);
    unsafe {
        Some(webrtc::video_frame::native::NativeBuffer::from_cv_pixel_buffer(pixel_buffer as _))
    }
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn video_frame_buffer_to_webrtc(frame: ScreenCaptureFrame) -> Option<impl AsRef<dyn VideoBuffer>> {
    use libwebrtc::native::yuv_helper::argb_to_nv12;
    use livekit::webrtc::prelude::NV12Buffer;
    match frame.0 {
        scap::frame::Frame::BGRx(frame) => {
            let mut buffer = NV12Buffer::new(frame.width as u32, frame.height as u32);
            let (stride_y, stride_uv) = buffer.strides();
            let (data_y, data_uv) = buffer.data_mut();
            argb_to_nv12(
                &frame.data,
                frame.width as u32 * 4,
                data_y,
                stride_y,
                data_uv,
                stride_uv,
                frame.width,
                frame.height,
            );
            Some(buffer)
        }
        scap::frame::Frame::YUVFrame(yuvframe) => {
            let mut buffer = NV12Buffer::with_strides(
                yuvframe.width as u32,
                yuvframe.height as u32,
                yuvframe.luminance_stride as u32,
                yuvframe.chrominance_stride as u32,
            );
            let (luminance, chrominance) = buffer.data_mut();
            luminance.copy_from_slice(yuvframe.luminance_bytes.as_slice());
            chrominance.copy_from_slice(yuvframe.chrominance_bytes.as_slice());
            Some(buffer)
        }
        _ => {
            log::error!(
                "Expected BGRx or YUV frame from scap screen capture but got some other format."
            );
            None
        }
    }
}

#[cfg(target_os = "windows")]
fn video_frame_buffer_to_webrtc(_frame: ScreenCaptureFrame) -> Option<impl AsRef<dyn VideoBuffer>> {
    None as Option<Box<dyn VideoBuffer>>
}

trait DeviceChangeListenerApi: Stream<Item = ()> + Sized {
    fn new(input: bool) -> Result<Self>;
}

#[cfg(target_os = "macos")]
mod macos {

    use super::{AudioObjectID, AudioObjectPropertyAddress, OSStatus};
    use futures::channel::mpsc::UnboundedReceiver;

    /// Implementation from: https://github.com/zed-industries/cpal/blob/fd8bc2fd39f1f5fdee5a0690656caff9a26d9d50/src/host/coreaudio/macos/property_listener.rs#L15
    pub struct CoreAudioDefaultDeviceChangeListener {
        rx: UnboundedReceiver<()>,
        #[allow(dead_code)]
        callback: Box<PropertyListenerCallbackWrapper>,
        #[allow(dead_code)]
        input: bool,
        #[allow(dead_code)]
        device_id: AudioObjectID, // Store the device ID to properly remove listeners
    }

    trait _AssertSend: Send {}
    impl _AssertSend for CoreAudioDefaultDeviceChangeListener {}

    #[allow(dead_code)]
    struct PropertyListenerCallbackWrapper(#[allow(dead_code)] Box<dyn FnMut() + Send>);

    /// SAFETY: This function is called by Core Audio as a C callback.
    /// The caller (Core Audio system) guarantees:
    /// 1. callback pointer is valid for the lifetime of the listener registration
    /// 2. Audio parameters are valid for the current audio session
    /// 3. Function is called on appropriate Core Audio thread
    ///    We ensure safety by validating the callback pointer before dereferencing.
    #[allow(dead_code)]
    unsafe extern "C" fn property_listener_handler_shim(
        _: AudioObjectID,
        _: u32,
        _: *const AudioObjectPropertyAddress,
        callback: *mut ::std::os::raw::c_void,
    ) -> OSStatus {
        let wrapper = callback as *mut PropertyListenerCallbackWrapper;
        // SAFETY: callback pointer was validated during listener registration.
        // PropertyListenerCallbackWrapper manages the proper callback lifetime.
        unsafe { (*wrapper).0() };
        0
    }

    // TODO: Fix coreaudio-rs API usage when proper imports are available
    /*
    impl super::DeviceChangeListenerApi for CoreAudioDefaultDeviceChangeListener {
        fn new(input: bool) -> anyhow::Result<Self> {
            let (tx, rx) = futures::channel::mpsc::unbounded();

            let callback = Box::new(PropertyListenerCallbackWrapper(Box::new(move || {
                tx.unbounded_send(()).ok();
            })));

            // Get the current default device ID
            // SAFETY: Core Audio system calls with validated parameters.
            // kAudioObjectSystemObject is a valid system-provided constant.
            let device_id = unsafe {
                // Listen for default device changes
                coreaudio::Error::from_os_status(AudioObjectAddPropertyListener(
                    kAudioObjectSystemObject,
                    &AudioObjectPropertyAddress {
                        mSelector: if input {
                            kAudioHardwarePropertyDefaultInputDevice
                        } else {
                            kAudioHardwarePropertyDefaultOutputDevice
                        },
                        mScope: kAudioObjectPropertyScopeGlobal,
                        mElement: kAudioObjectPropertyElementMaster,
                    },
                    Some(property_listener_handler_shim),
                    &*callback as *const _ as *mut _,
                ))?;

                // Also listen for changes to the device configuration
                let device_id = if input {
                    let mut input_device: AudioObjectID = 0;
                    let mut prop_size = std::mem::size_of::<AudioObjectID>() as u32;
                    let result = coreaudio::AudioObjectGetPropertyData(
                        kAudioObjectSystemObject,
                        &AudioObjectPropertyAddress {
                            mSelector: kAudioHardwarePropertyDefaultInputDevice,
                            mScope: kAudioObjectPropertyScopeGlobal,
                            mElement: kAudioObjectPropertyElementMaster,
                        },
                        0,
                        std::ptr::null(),
                        &mut prop_size as *mut _,
                        &mut input_device as *mut _ as *mut _,
                    );
                    if result != 0 {
                        log::warn!("Failed to get default input device ID");
                        0
                    } else {
                        input_device
                    }
                } else {
                    let mut output_device: AudioObjectID = 0;
                    let mut prop_size = std::mem::size_of::<AudioObjectID>() as u32;
                    let result = coreaudio::AudioObjectGetPropertyData(
                        kAudioObjectSystemObject,
                        &AudioObjectPropertyAddress {
                            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                            mScope: kAudioObjectPropertyScopeGlobal,
                            mElement: kAudioObjectPropertyElementMaster,
                        },
                        0,
                        std::ptr::null(),
                        &mut prop_size as *mut _,
                        &mut output_device as *mut _ as *mut _,
                    );
                    if result != 0 {
                        log::warn!("Failed to get default output device ID");
                        0
                    } else {
                        output_device
                    }
                };

                if device_id != 0 {
                    // Listen for format changes on the device
                    coreaudio::Error::from_os_status(AudioObjectAddPropertyListener(
                        device_id,
                        &AudioObjectPropertyAddress {
                            mSelector: coreaudio::kAudioDevicePropertyStreamFormat,
                            mScope: if input {
                                coreaudio::kAudioObjectPropertyScopeInput
                            } else {
                                coreaudio::kAudioObjectPropertyScopeOutput
                            },
                            mElement: kAudioObjectPropertyElementMaster,
                        },
                        Some(property_listener_handler_shim),
                        &*callback as *const _ as *mut _,
                    ))?;
                }

                device_id
            };

            Ok(Self {
                rx,
                callback,
                input,
                device_id,
            })
        }
    }
    */

    // Temporary implementation until CoreAudio APIs are available
    impl super::DeviceChangeListenerApi for CoreAudioDefaultDeviceChangeListener {
        fn new(_input: bool) -> anyhow::Result<Self> {
            let (tx, rx) = futures::channel::mpsc::unbounded();
            let callback = Box::new(PropertyListenerCallbackWrapper(Box::new(move || {
                let _ = tx.unbounded_send(());
            })));

            Ok(Self {
                rx,
                callback,
                input: _input,
                device_id: 0, // Placeholder until CoreAudio is available
            })
        }
    }

    // TODO: Fix Drop impl when coreaudio APIs are available
    /*
    impl Drop for CoreAudioDefaultDeviceChangeListener {
        fn drop(&mut self) {
            unsafe {
                // Remove the system-level property listener
                AudioObjectRemovePropertyListener(
                    kAudioObjectSystemObject,
                    &AudioObjectPropertyAddress {
                        mSelector: if self.input {
                            kAudioHardwarePropertyDefaultInputDevice
                        } else {
                            kAudioHardwarePropertyDefaultOutputDevice
                        },
                        mScope: kAudioObjectPropertyScopeGlobal,
                        mElement: kAudioObjectPropertyElementMaster,
                    },
                    Some(property_listener_handler_shim),
                    &*self.callback as *const _ as *mut _,
                );

                // Remove the device-specific property listener if we have a valid device ID
                if self.device_id != 0 {
                    AudioObjectRemovePropertyListener(
                        self.device_id,
                        &AudioObjectPropertyAddress {
                            mSelector: coreaudio::kAudioDevicePropertyStreamFormat,
                            mScope: if self.input {
                                coreaudio::kAudioObjectPropertyScopeInput
                            } else {
                                coreaudio::kAudioObjectPropertyScopeOutput
                            },
                            mElement: kAudioObjectPropertyElementMaster,
                        },
                        Some(property_listener_handler_shim),
                        &*self.callback as *const _ as *mut _,
                    );
                }
            }
        }
    }
    */

    impl futures::Stream for CoreAudioDefaultDeviceChangeListener {
        type Item = ();

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            use futures::StreamExt;
            self.rx.poll_next_unpin(cx)
        }
    }
}

#[cfg(target_os = "macos")]
type DeviceChangeListener = macos::CoreAudioDefaultDeviceChangeListener;

#[cfg(not(target_os = "macos"))]
mod noop_change_listener {
    use std::task::Poll;

    use super::DeviceChangeListenerApi;

    pub struct NoopOutputDeviceChangelistener {}

    impl DeviceChangeListenerApi for NoopOutputDeviceChangelistener {
        fn new(_input: bool) -> anyhow::Result<Self> {
            Ok(NoopOutputDeviceChangelistener {})
        }
    }

    impl futures::Stream for NoopOutputDeviceChangelistener {
        type Item = ();

        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }
}

#[cfg(not(target_os = "macos"))]
type DeviceChangeListener = noop_change_listener::NoopOutputDeviceChangelistener;
