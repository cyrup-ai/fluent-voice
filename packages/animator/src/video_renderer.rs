use egui_wgpu::RenderState;
use futures::StreamExt;
use livekit::webrtc::{
    native::yuv_helper as yuv, prelude::*, video_stream::native::NativeVideoStream,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct VideoRenderer {
    internal: Arc<Mutex<RendererInternal>>,
    _rtc_track: RtcVideoTrack,
    _render_task: Option<tokio::task::JoinHandle<()>>,
}

struct RendererInternal {
    render_state: RenderState,
    width: u32,
    height: u32,
    rgba_data: Vec<u8>,
    texture: Option<egui_wgpu::wgpu::Texture>,
    texture_view: Option<egui_wgpu::wgpu::TextureView>,
    egui_tex: Option<egui::TextureId>,
    frame_count: u64,
    last_frame_time: Instant,
    dropped_frames: u64,
}

impl VideoRenderer {
    pub fn new(
        rt_handle: &tokio::runtime::Handle,
        render_state: RenderState,
        rtc_track: RtcVideoTrack,
    ) -> Self {
        let internal = Arc::new(Mutex::new(RendererInternal {
            render_state,
            width: 0,
            height: 0,
            rgba_data: Vec::with_capacity(1920 * 1080 * 4), // Pre-allocate for common resolution
            texture: None,
            texture_view: None,
            egui_tex: None,
            frame_count: 0,
            last_frame_time: Instant::now(),
            dropped_frames: 0,
        }));

        let mut sink = NativeVideoStream::new(rtc_track.clone());
        let internal_clone = internal.clone();

        // Use async task instead of blocking thread
        let render_task = rt_handle.spawn(async move {
            let mut frame_interval = tokio::time::interval(Duration::from_millis(16)); // ~60fps max
            frame_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            while let Some(frame) = sink.next().await {
                // Skip frame if we're falling behind
                if frame_interval.tick().await.elapsed() > Duration::from_millis(8) {
                    internal_clone.lock().dropped_frames += 1;
                    continue;
                }

                let result = {
                    let mut internal = internal_clone.lock();
                    internal.process_frame(frame)
                };

                if let Err(e) = result {
                    log::error!("Failed to process video frame: {:?}", e);
                }
            }
        });

        Self {
            _rtc_track: rtc_track,
            internal,
            _render_task: Some(render_task),
        }
    }

    pub fn resolution(&self) -> (u32, u32) {
        let internal = self.internal.lock();
        (internal.width, internal.height)
    }

    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.internal.lock().egui_tex
    }

    pub fn stats(&self) -> (u64, u64, f32) {
        let internal = self.internal.lock();
        let fps = if internal.frame_count > 0 {
            internal.frame_count as f32 / internal.last_frame_time.elapsed().as_secs_f32()
        } else {
            0.0
        };
        (internal.frame_count, internal.dropped_frames, fps)
    }
}

impl RendererInternal {
    fn process_frame(
        &mut self,
        frame: livekit::webrtc::video_frame::VideoFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let buf = frame.buffer.to_i420();
        let (w, h) = (buf.width(), buf.height());

        self.ensure_texture_size(w, h)?;

        let stride_rgba = w * 4;
        let (sy, su, sv) = buf.strides();
        let (dy, du, dv) = buf.data();

        // Ensure rgba_data has correct size
        let required_size = (w * h * 4) as usize;
        if self.rgba_data.len() < required_size {
            self.rgba_data.resize(required_size, 0);
        }

        yuv::i420_to_abgr(
            dy,
            sy,
            du,
            su,
            dv,
            sv,
            &mut self.rgba_data[..required_size],
            stride_rgba,
            w as i32,
            h as i32,
        );

        if let Some(texture) = &self.texture {
            self.render_state.queue.write_texture(
                egui_wgpu::wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: egui_wgpu::wgpu::Origin3d::default(),
                    aspect: egui_wgpu::wgpu::TextureAspect::default(),
                },
                &self.rgba_data[..required_size],
                egui_wgpu::wgpu::TexelCopyBufferLayout {
                    bytes_per_row: Some(stride_rgba),
                    rows_per_image: None,
                },
                egui_wgpu::wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );

            self.frame_count += 1;
        }

        Ok(())
    }

    fn ensure_texture_size(&mut self, w: u32, h: u32) -> Result<(), Box<dyn std::error::Error>> {
        if self.width == w && self.height == h {
            return Ok(());
        }

        self.width = w;
        self.height = h;

        // Create texture with optimal format
        let texture_desc = egui_wgpu::wgpu::TextureDescriptor {
            label: Some("lk-video-tex"),
            usage: egui_wgpu::wgpu::TextureUsages::TEXTURE_BINDING
                | egui_wgpu::wgpu::TextureUsages::COPY_DST,
            dimension: egui_wgpu::wgpu::TextureDimension::D2,
            size: egui_wgpu::wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            mip_level_count: 1,
            format: egui_wgpu::wgpu::TextureFormat::Rgba8Unorm, // More efficient than sRGB
            view_formats: &[egui_wgpu::wgpu::TextureFormat::Rgba8Unorm],
        };

        self.texture = Some(self.render_state.device.create_texture(&texture_desc));

        let view_desc = egui_wgpu::wgpu::TextureViewDescriptor {
            label: Some("lk-video-tex-view"),
            format: Some(egui_wgpu::wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(egui_wgpu::wgpu::TextureViewDimension::D2),
            aspect: egui_wgpu::wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        };

        self.texture_view = self.texture.as_ref().map(|t| t.create_view(&view_desc));

        if let Some(texture_view) = &self.texture_view {
            if let Some(egui_id) = self.egui_tex {
                self.render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &self.render_state.device,
                        texture_view,
                        egui_wgpu::wgpu::FilterMode::Linear,
                        egui_id,
                    );
            } else {
                self.egui_tex = Some(self.render_state.renderer.write().register_native_texture(
                    &self.render_state.device,
                    texture_view,
                    egui_wgpu::wgpu::FilterMode::Linear,
                ));
            }
        }

        Ok(())
    }
}

impl Drop for VideoRenderer {
    fn drop(&mut self) {
        if let Some(task) = self._render_task.take() {
            task.abort();
        }
    }
}
