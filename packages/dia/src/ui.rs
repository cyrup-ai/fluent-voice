//! Dioxus 0.6 broadcast meter and preset selector components.
//! Provides real-time BS.1770 loudness visualization.

#[cfg(feature = "ui")]
pub mod ui_components {
    use dioxus::prelude::*;
    use gloo_timers::callback::Interval;
    use wasm_bindgen::{JsCast, JsValue};
    use web_sys::{self, CanvasRenderingContext2d, HtmlCanvasElement};

    use crate::audio::enhanced_normalizer::{EnhancedNormalizer, LoudnessPreset};

    // -----------------------------------------------------------------------------
    // Component Props
    // -----------------------------------------------------------------------------

    #[derive(Props, PartialEq, Clone)]
    pub struct BroadcastMeterProps {
        pub normalizer: Signal<Option<EnhancedNormalizer>>,
    }

    #[derive(Props, PartialEq, Clone)]
    pub struct PresetSelectorProps {
        pub normalizer: Signal<Option<EnhancedNormalizer>>,
    }

    // -----------------------------------------------------------------------------
    // Helper Components
    // -----------------------------------------------------------------------------

    /// Creates a meter reading item with label and value
    fn meter_item(
        label: &'static str,
        value: impl std::fmt::Display,
        class: &'static str,
    ) -> Element {
        rsx! {
            div {
                span { class: "label", "{label}" }
                span { class: "{class}", "{value}" }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Broadcast Meter Component
    // -----------------------------------------------------------------------------

    #[component]
    pub fn BroadcastMeter(props: BroadcastMeterProps) -> Element {
        // Local state for meter values
        let mut lufs = use_signal(|| -60.0_f64);
        let mut true_peak = use_signal(|| 0.0_f64);
        let mut gain_db = use_signal(|| 0.0_f64);
        let mut history = use_signal(Vec::<f64>::new);
        let mut preset = use_signal(|| LoudnessPreset::Voice);

        // 100ms polling effect to update meter values
        {
            let normalizer = props.normalizer.clone();
            let mut lufs = lufs.clone();
            let mut true_peak = true_peak.clone();
            let mut gain_db = gain_db.clone();
            let mut history = history.clone();
            let mut preset = preset.clone();

            use_effect(move || {
                let interval = Interval::new(100, move || {
                    if let Some(norm) = normalizer.read().as_ref() {
                        // Loudness measurements
                        let cur_lufs = norm
                            .get_integrated_lufs()
                            .unwrap_or_else(|| norm.get_momentary_lufs());
                        lufs.set(cur_lufs);

                        // True-peak measurements
                        let buf_f32: Vec<f32> =
                            norm.get_lufs_history().iter().map(|&x| x as f32).collect();
                        true_peak.set(norm.get_true_peak(&buf_f32) as f64);

                        // Gain calculation
                        gain_db.set(20.0 * f64::from(norm.get_current_gain().log10()));

                        // Update history and preset
                        history.set(norm.get_lufs_history());
                        preset.set(norm.get_current_preset());
                    }
                });

                // Cleanup function
                (move || {
                    drop(interval);
                })()
            });
        }

        // Canvas drawing effect to display history graph
        {
            let history_signal = history.clone();

            use_effect(move || {
                // Draw history on canvas
                if let Some(window) = web_sys::window().as_ref() {
                    if let Some(document) = window.document() {
                        if let Some(element) = document.get_element_by_id("lufs-history") {
                            if let Some(canvas) = element.dyn_ref::<HtmlCanvasElement>() {
                                if let Ok(ctx_obj) = canvas.get_context("2d") {
                                    if let Some(ctx_obj_unwrapped) = ctx_obj {
                                        if let Ok(ctx) =
                                            ctx_obj_unwrapped.dyn_into::<CanvasRenderingContext2d>()
                                        {
                                            // Get canvas dimensions
                                            let w = canvas.width() as f64;
                                            let h = canvas.height() as f64;

                                            // Clear canvas and draw grid
                                            ctx.clear_rect(0.0, 0.0, w, h);
                                            // Use the standard stroke style API despite deprecation warnings
                                            ctx.set_stroke_style(&JsValue::from_str("#333"));
                                            ctx.set_line_width(1.0);

                                            // Draw horizontal grid lines
                                            for db in [-60.0, -40.0, -20.0, 0.0] {
                                                let y = h - ((db + 60.0) / 60.0) * h;
                                                ctx.begin_path();
                                                ctx.move_to(0.0, y);
                                                ctx.line_to(w, y);
                                                ctx.stroke();
                                            }

                                            // Draw history line
                                            let hist = history_signal.read();
                                            if !hist.is_empty() {
                                                // Use the standard stroke style API despite deprecation warnings
                                                ctx.set_stroke_style(&JsValue::from_str("#00aff0"));
                                                ctx.set_line_width(2.0);
                                                ctx.begin_path();

                                                let step = w / hist.len() as f64;
                                                for (i, &v) in hist.iter().enumerate() {
                                                    let x = i as f64 * step;
                                                    let y = h
                                                        - ((v + 60.0).clamp(0.0, 60.0) / 60.0 * h);
                                                    if i == 0 {
                                                        ctx.move_to(x, y);
                                                    } else {
                                                        ctx.line_to(x, y);
                                                    }
                                                }
                                                ctx.stroke();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // No cleanup needed
                (|| {})()
            });
        }

        // Pre-compute meter items
        let lufs_item = meter_item(
            "LUFS:",
            format!("{:.1}", *lufs.read()),
            lufs_class(*lufs.read()),
        );
        let tp_item = meter_item(
            "True Peak:",
            format!("{:.1}", *true_peak.read()),
            tp_class(*true_peak.read()),
        );
        let gain_item = meter_item("Gain:", format!("{:.1} dB", *gain_db.read()), "value");
        let preset_item = meter_item("Preset:", preset_name(*preset.read()), "value");
        let target_item = meter_item("Target:", preset_target(*preset.read()), "value");

        // Render component UI
        rsx! {
            div {
                class: "meter-wrapper",
                h3 { "Broadcast Audio Meters" }
                canvas {
                    id: "lufs-history",
                    width: "480",
                    height: "120"
                }
                div {
                    class: "meter-readings",
                }
                {lufs_item}
                {tp_item}
                {gain_item}
                div {
                    class: "preset-info",
                }
                {preset_item}
                {target_item}
                PresetSelector { normalizer: props.normalizer }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Preset Selector Component
    // -----------------------------------------------------------------------------

    #[component]
    pub fn PresetSelector(props: PresetSelectorProps) -> Element {
        // Local state for preset selection
        let mut selected = use_signal(|| LoudnessPreset::Voice);

        // Initialize from normalizer if available
        {
            let normalizer = props.normalizer.clone();
            let mut selected = selected.clone();

            use_effect(move || {
                // Set the value using Dioxus 0.6 signal pattern
                let current_preset = normalizer
                    .read()
                    .as_ref()
                    .map(|norm| norm.get_current_preset())
                    .unwrap_or(LoudnessPreset::Voice);

                // Clone before setting to avoid borrowing issues
                let value = current_preset.clone();
                selected.set(value);
                (|| {})()
            });
        }

        // Click handler factory for preset buttons
        let make_click = |p: LoudnessPreset| {
            let mut normalizer = props.normalizer.clone();
            let mut selected = selected.clone();

            move |_| {
                // Clone the preset value to use in closures
                let preset_value = p;

                // In Dioxus 0.6, all signals implement Copy, so we can clone them safely
                // Update the normalizer first
                normalizer.write().as_mut().map(|norm| {
                    norm.set_preset(preset_value);
                });

                // Update the selected signal
                selected.set(preset_value);
            }
        };

        // Render preset selector buttons
        rsx! {
            div {
                class: "preset-buttons",
                h4 { "Presets" }
                div {
                    button {
                        class: if *selected.read() == LoudnessPreset::YouTube { "selected" } else { "" },
                        onclick: make_click(LoudnessPreset::YouTube),
                        "YouTube"
                    }
                    button {
                        class: if *selected.read() == LoudnessPreset::Broadcast { "selected" } else { "" },
                        onclick: make_click(LoudnessPreset::Broadcast),
                        "Broadcast"
                    }
                    button {
                        class: if *selected.read() == LoudnessPreset::Streaming { "selected" } else { "" },
                        onclick: make_click(LoudnessPreset::Streaming),
                        "Streaming"
                    }
                }
                div {
                    button {
                        class: if *selected.read() == LoudnessPreset::Podcast { "selected" } else { "" },
                        onclick: make_click(LoudnessPreset::Podcast),
                        "Podcast"
                    }
                    button {
                        class: if *selected.read() == LoudnessPreset::Voice { "selected" } else { "" },
                        onclick: make_click(LoudnessPreset::Voice),
                        "Voice"
                    }
                    button {
                        class: if *selected.read() == LoudnessPreset::Telephony { "selected" } else { "" },
                        onclick: make_click(LoudnessPreset::Telephony),
                        "Telephony"
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Style Helpers
    // -----------------------------------------------------------------------------

    /// Returns CSS class based on LUFS value
    fn lufs_class(v: f64) -> &'static str {
        if v > -9.0 {
            "value-high"
        } else if v > -16.0 {
            "value-warning"
        } else {
            "value-ok"
        }
    }

    /// Returns CSS class based on true-peak value
    fn tp_class(v: f64) -> &'static str {
        if v > 0.0 {
            "value-high"
        } else if v > -1.0 {
            "value-warning"
        } else {
            "value-ok"
        }
    }

    /// Returns the display name for a loudness preset
    fn preset_name(p: LoudnessPreset) -> &'static str {
        match p {
            LoudnessPreset::YouTube => "YouTube",
            LoudnessPreset::Broadcast => "Broadcast (EBU R128)",
            LoudnessPreset::Streaming => "Streaming",
            LoudnessPreset::Podcast => "Podcast",
            LoudnessPreset::Voice => "Voice",
            LoudnessPreset::Telephony => "Telephony",
            LoudnessPreset::Custom => "Custom",
        }
    }

    /// Returns the target LUFS value for a loudness preset
    fn preset_target(p: LoudnessPreset) -> &'static str {
        match p {
            LoudnessPreset::YouTube => "-14 LUFS",
            LoudnessPreset::Broadcast => "-23 LUFS",
            LoudnessPreset::Streaming => "-16 LUFS",
            LoudnessPreset::Podcast => "-16 LUFS",
            LoudnessPreset::Voice => "-14 LUFS",
            LoudnessPreset::Telephony => "-18 LUFS",
            LoudnessPreset::Custom => "–",
        }
    }
}

// Provide stubs when UI is not enabled
#[cfg(not(feature = "ui"))]
pub mod ui_components {
    // Empty module when UI is not enabled
}
