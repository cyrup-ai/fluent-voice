/* eslint-disable @typescript-eslint/no-this-alias */

// <reference types="@types/audioworklet" />

import { BUFFER_ADAPTATION, CONFIG_PRESETS, MESSAGE_TYPES } from "./constants";
import { applyFade, nowMs, validateConfig } from "./lib/utils";

import { AudioBufferPool } from "./lib/audio-buffer-pool";
import { FrameSequencer } from "./lib/frame-sequencer";
import { NetworkQualityEstimator } from "./lib/network-quality-estimator";
import { PerformanceMonitor } from "./lib/performance-monitor";

/* ------------------------------------------------------------------ *
 *  Typed aliases                                                      *
 * ------------------------------------------------------------------ */
type Cfg = typeof CONFIG_PRESETS.moshi;

/* READY payload */
interface ReadyMsg {
    type: typeof MESSAGE_TYPES.READY;
    sampleRate: number;
    timestamp: number;
    config: Cfg;
}

/* Periodic STATS payload */
interface StatsMsg {
    type: typeof MESSAGE_TYPES.STATS;
    totalAudioPlayed: number;
    actualAudioPlayed: number;
    delay: number;
    minDelay: number;
    maxDelay: number;
    bufferUnderruns: number;
    bufferOverruns: number;
    consecutiveUnderruns: number;
    consecutiveOverruns: number;
    stableFrameCount: number;
    adaptationCount: number;
    performance: PerformanceMonitor["metrics"];
    network: ReturnType<NetworkQualityEstimator["getMetrics"]>;
    sequencer: ReturnType<FrameSequencer["getStats"]>;
    memoryUsage: {
        frames: number;
        estimatedBytes: number;
        poolStats: AudioBufferPool["stats"];
    };
    bufferHealth: {
        current: number;
        max: number;
        partial: number;
        initial: number;
    };
    timestamp: number;
}

/* ------------------------------------------------------------------ *
 *  UniversalAudioProcessor                                            *
 * ------------------------------------------------------------------ */
class UniversalAudioProcessor extends AudioWorkletProcessor {
    /* ——— configuration & helpers —————————————— */
    private cfg: Cfg = CONFIG_PRESETS.moshi;
    private readonly pool = new AudioBufferPool();
    private readonly perf = new PerformanceMonitor(256);
    private readonly net: NetworkQualityEstimator;
    private readonly seq = new FrameSequencer();

    /* ——— JS-side frame ring —————————————————— */
    private frames: Float32Array[] = new Array(1024);
    private head = 0;
    private tail = 0;
    private off0 = 0;
    private _bufferedSamples = 0;
    private _framesInFlight = 0;

    /* ——— playback state ———————————————————— */
    private started = false;
    private remainPartial = 0;
    private firstChunk = true;

    /* ——— stats counters ——————————————————— */
    private lastStats = 0;
    private recvSinceLast = 0;
    private totalPlay = 0;
    private actualPlay = 0;

    private m = {
        bufUnd: 0,
        bufOvr: 0,
        conUnd: 0,
        conOvr: 0,
        stable: 0,
        adapt: 0,
        maxD: 0,
        minD: 2_000,
        avgD: 0,
    };

    // Mutable buffer settings to work around read-only config
    private mutablePartialBufferSamples: number =
        CONFIG_PRESETS.moshi.partialBufferSamples;
    private mutableMaxBufferSamples: number =
        CONFIG_PRESETS.moshi.maxBufferSamples;

    /* ---------------------------------------------------------------- *
     *  constructor                                                     *
     * ---------------------------------------------------------------- */
    constructor(options: AudioWorkletNodeOptions) {
        super();

        /* load config */
        this.cfg = validateConfig(
            (options.processorOptions as any)?.config ?? {},
            this.cfg,
        );
        this.mutablePartialBufferSamples = this.cfg.partialBufferSamples;
        this.mutableMaxBufferSamples = this.cfg.maxBufferSamples;

        /* metrics helpers */
        this.net = new NetworkQualityEstimator((globalThis as any).sampleRate);

        /* message handler */
        (this as any).port.onmessage = (ev: MessageEvent) =>
            this.onMsg(ev.data);

        /* announce readiness */
        const ready: ReadyMsg = {
            type: MESSAGE_TYPES.READY,
            sampleRate: (globalThis as any).sampleRate,
            timestamp: nowMs(),
            config: this.cfg,
        };
        (this as any).port.postMessage(ready);
    }

    /* ---------------------------------------------------------------- *
     *  message handler                                                 *
     * ---------------------------------------------------------------- */
    private onMsg(msg: any) {
        switch (msg?.type) {
            case MESSAGE_TYPES.RESET:
                this.reset();
                break;

            case MESSAGE_TYPES.CONFIG:
                this.cfg = validateConfig(msg.config, this.cfg);
                this.mutablePartialBufferSamples =
                    this.cfg.partialBufferSamples;
                this.mutableMaxBufferSamples = this.cfg.maxBufferSamples;
                this.reset();
                break;

            case MESSAGE_TYPES.AUDIO_FRAME:
                this.ingest(msg.frame, msg.sequence);
                break;
        }
    }

    /* ---------------------------------------------------------------- *
     *  reset processor state                                           *
     * ---------------------------------------------------------------- */
    private reset() {
        this.head = this.tail = this.off0 = 0;
        this._bufferedSamples = 0;
        this._framesInFlight = 0;
        this.started = false;
        this.remainPartial = 0;
        this.firstChunk = true;
        this.m = {
            ...this.m,
            bufUnd: 0,
            bufOvr: 0,
            conUnd: 0,
            conOvr: 0,
            stable: 0,
            adapt: 0,
            maxD: 0,
            minD: 2_000,
            avgD: 0,
        };
        this.net.reset();
        this.perf.reset();
        this.mutablePartialBufferSamples = this.cfg.partialBufferSamples;
        this.mutableMaxBufferSamples = this.cfg.maxBufferSamples;
    }

    /* ---------------------------------------------------------------- *
     *  JS-ring helpers (push/pop)                                      *
     * ---------------------------------------------------------------- */
    private push(f: Float32Array) {
        if ((this.tail + 1) % this.frames.length === this.head) {
            const old = this.frames;
            this.frames = new Array(old.length * 2);
            let j = 0;
            for (let i = this.head; i !== this.tail; i = (i + 1) % old.length)
                this.frames[j++] = old[i];
            this.head = 0;
            this.tail = j;
        }
        this.frames[this.tail] = f;
        this.tail = (this.tail + 1) % this.frames.length;
        this._bufferedSamples += f.length;
        this._framesInFlight++;
    }
    private pop(): Float32Array | null {
        if (this.head === this.tail) return null;
        const f = this.frames[this.head];
        this.head = (this.head + 1) % this.frames.length;
        this._framesInFlight--;
        return f;
    }

    /* ---------------------------------------------------------------- *
     *  ingest network frame                                            *
     * ---------------------------------------------------------------- */
    private ingest(frame: Float32Array, seq?: number) {
        const t = nowMs();
        this.net.recordFrameArrival(t, frame.length * 4, seq);

        /* sequence re-ordering */
        if (seq !== undefined && this.cfg.outOfOrderFrameHandling) {
            const res = this.seq.add(frame, seq);
            if (res.status === "play") {
                this.push(res.frame);
                for (const extra of res.extra) this.push(extra);
            } else if (res.status === "buffer") {
                /* nothing to do – will flush later */
                return;
            } else return; // discarded
        } else this.push(frame);

        this.recvSinceLast++;

        /* maybe start playback */
        if (
            !this.started &&
            this._bufferedSamples >= this.cfg.initialBufferSamples
        )
            this.startPlayback();

        /* overrun detection */
        if (this._bufferedSamples > this.maxThreshold()) {
            this.m.bufOvr++;
            this.m.conOvr++;
            this.m.conUnd = 0;
            this.m.stable = 0;
            this.dropExcess();
            if (
                this.m.conOvr >= BUFFER_ADAPTATION.OVERRUN_THRESHOLD &&
                this.cfg.adaptiveBuffering
            )
                this.enlargeBuffers();
        } else {
            this.m.conOvr = 0;
            this.m.stable++;
            if (
                this.m.stable >= BUFFER_ADAPTATION.STABLE_THRESHOLD &&
                this.cfg.adaptiveBuffering &&
                this.cfg.prioritizeLatency
            )
                this.shrinkBuffers();
        }
    }

    /* ---------------------------------------------------------------- *
     *  playback helpers                                                *
     * ---------------------------------------------------------------- */
    private maxThreshold() {
        return (
            this.cfg.initialBufferSamples +
            this.mutablePartialBufferSamples +
            this.mutableMaxBufferSamples
        );
    }
    private startPlayback() {
        this.started = true;
        this.remainPartial = this.mutablePartialBufferSamples;
        this.firstChunk = true;
    }
    private dropExcess() {
        const target =
            this.cfg.initialBufferSamples + this.mutablePartialBufferSamples;
        let excess = this._bufferedSamples - target;
        if (excess <= 0) return;

        let dropped = 0;
        while (excess > 0 && this.head !== this.tail) {
            const frame = this.frames[this.head];
            const availableInFrame = frame.length - this.off0;
            const toDrop = Math.min(excess, availableInFrame);

            this.off0 += toDrop;
            excess -= toDrop;
            dropped += toDrop;

            if (this.off0 >= frame.length) {
                this.pool.release(this.pop()!);
                this.off0 = 0;
            }
        }
        this._bufferedSamples -= dropped;

        /* expand max buffer to survive future burst */
        this.mutableMaxBufferSamples = Math.min(
            this.mutableMaxBufferSamples + this.cfg.maxBufferSamplesIncrement,
            this.cfg.maxMaxBufferWithIncrements,
        );
    }
    private enlargeBuffers() {
        const rec = this.net.getRecommendedBufferSettings();
        const rate = BUFFER_ADAPTATION.ADAPTATION_RATE;
        this.mutablePartialBufferSamples = Math.min(
            Math.max(
                this.mutablePartialBufferSamples,
                Math.floor(rec.partialBufferSamples * (1 + rate)),
            ),
            this.cfg.maxPartialWithIncrements,
        );
        this.mutableMaxBufferSamples = Math.min(
            Math.max(
                this.mutableMaxBufferSamples,
                Math.floor(rec.maxBufferSamples * (1 + rate)),
            ),
            this.cfg.maxMaxBufferWithIncrements,
        );
        this.m.adapt++;
        this.m.conUnd = this.m.conOvr = 0;
    }
    private shrinkBuffers() {
        const rate = BUFFER_ADAPTATION.ADAPTATION_RATE / 2;
        this.mutablePartialBufferSamples = Math.max(
            CONFIG_PRESETS.moshi.partialBufferSamples,
            Math.floor(this.mutablePartialBufferSamples * (1 - rate)),
        );
        this.mutableMaxBufferSamples = Math.max(
            CONFIG_PRESETS.moshi.maxBufferSamples,
            Math.floor(this.mutableMaxBufferSamples * (1 - rate)),
        );
        this.m.stable = BUFFER_ADAPTATION.STABLE_THRESHOLD / 2;
    }

    /* ---------------------------------------------------------------- *
     *  output helper                                                   *
     * ---------------------------------------------------------------- */
    private copyOut(dst: Float32Array): number {
        let wr = 0;
        while (wr < dst.length && this.head !== this.tail) {
            const f = this.frames[this.head];
            const avail = f.length - this.off0;
            const need = dst.length - wr;
            const cp = Math.min(avail, need);
            dst.set(f.subarray(this.off0, this.off0 + cp), wr);
            wr += cp;
            this.off0 += cp;
            if (this.off0 >= f.length) {
                this.pool.release(this.pop()!);
                this.off0 = 0;
            }
        }
        return wr;
    }

    /* ---------------------------------------------------------------- *
     *  main process()                                                  *
     * ---------------------------------------------------------------- */
    process(_in: Float32Array[][], outArr: Float32Array[][]): boolean {
        const out = outArr[0][0];
        this.perf.begin();
        out.fill(0);

        if (!this.started) {
            this.remainPartial -= out.length;
            this.perf.end(this.cfg.frameProcessingBudgetMs);
            return true;
        }

        const written = this.copyOut(out);
        this._bufferedSamples -= written;

        if (this.firstChunk) {
            const fade = Math.min(this.cfg.fadeInSamples, written);
            applyFade(out, 0, fade, true, this.cfg.fadeCurve);
            this.firstChunk = false;
        }

        if (written < out.length) {
            /* underrun */
            this.m.bufUnd++;
            this.m.conUnd++;
            this.m.conOvr = 0;
            this.m.stable = 0;
            const fade = Math.min(this.cfg.fadeOutSamples, written);
            applyFade(out, written - fade, fade, false, this.cfg.fadeCurve);
            this.started = false;

            this.mutablePartialBufferSamples = Math.min(
                this.mutablePartialBufferSamples +
                    this.cfg.partialBufferIncrement,
                this.cfg.maxPartialWithIncrements,
            );

            if (
                this.m.conUnd >= BUFFER_ADAPTATION.UNDERRUN_THRESHOLD &&
                this.cfg.adaptiveBuffering
            )
                this.enlargeBuffers();
        } else this.m.conUnd = 0;

        /* delay metrics */
        const d = this._bufferedSamples / (globalThis as any).sampleRate;
        this.m.maxD = Math.max(this.m.maxD, d);
        this.m.minD = Math.min(this.m.minD, d);
        this.m.avgD =
            this.m.avgD * this.cfg.smoothingFactor +
            d * (1 - this.cfg.smoothingFactor);

        this.totalPlay += out.length / (globalThis as any).sampleRate;
        this.actualPlay += written / (globalThis as any).sampleRate;

        this.perf.end(this.cfg.frameProcessingBudgetMs);

        /* stats throttle */
        const now = nowMs();
        if (now - this.lastStats > 250) {
            const stats: StatsMsg = {
                type: MESSAGE_TYPES.STATS,
                totalAudioPlayed: this.totalPlay,
                actualAudioPlayed: this.actualPlay,
                delay: this.m.avgD,
                minDelay: this.m.minD,
                maxDelay: this.m.maxD,
                bufferUnderruns: this.m.bufUnd,
                bufferOverruns: this.m.bufOvr,
                consecutiveUnderruns: this.m.conUnd,
                consecutiveOverruns: this.m.conOvr,
                stableFrameCount: this.m.stable,
                adaptationCount: this.m.adapt,
                performance: this.perf.metrics,
                network: this.net.getMetrics(),
                sequencer: this.seq.getStats(),
                memoryUsage: {
                    frames: this._framesInFlight,
                    estimatedBytes: 4 * this._bufferedSamples,
                    poolStats: this.pool.stats,
                },
                bufferHealth: {
                    current: this._bufferedSamples,
                    max: this.maxThreshold(),
                    partial: this.mutablePartialBufferSamples,
                    initial: this.cfg.initialBufferSamples,
                },
                timestamp: now,
            };
            (this as any).port.postMessage(stats);
            this.lastStats = now;
            this.recvSinceLast = 0;
        }

        return true;
    }
}

(globalThis as any).registerProcessor(
    "universal-audio-processor",
    UniversalAudioProcessor,
);
