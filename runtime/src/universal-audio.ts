===============================
========= src/universal-audio.ts ===========
===============================
/* eslint-disable @typescript-eslint/no-this-alias */

import { CONSTANTS, MESSAGE_TYPES } from "./constants.js";
import {
  applyFade,
  nowMs,
  validateConfig
} from "./lib/utils.js";

import { SharedByteRing } from "./lib/shared-byte-ring.js";
import { AudioBufferPool } from "./lib/audio-buffer-pool.js";
import { PerformanceMonitor } from "./lib/performance-monitor.js";
import { FrameSequencer } from "./lib/frame-sequencer.js";
import { NetworkQualityEstimator } from "./lib/network-quality-estimator.js";

/* ------------------------------------------------------------------ *
 *  Typed aliases                                                      *
 * ------------------------------------------------------------------ */
type Cfg = typeof CONSTANTS.CONFIG_PRESETS.moshi;

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
  performance: ReturnType<PerformanceMonitor["metrics"]>;
  network: ReturnType<NetworkQualityEstimator["getMetrics"]>;
  sequencer: ReturnType<FrameSequencer["getStats"]>;
  memoryUsage: {
    frames: number;
    estimatedBytes: number;
    poolStats: ReturnType<AudioBufferPool["stats"]>;
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
  private cfg: Cfg = CONSTANTS.CONFIG_PRESETS.moshi;
  private readonly pool = new AudioBufferPool();
  private readonly perf = new PerformanceMonitor(256);
  private readonly net: NetworkQualityEstimator;
  private readonly seq = new FrameSequencer();
  private readonly ring: SharedByteRing | null;

  /* ——— JS-side frame ring —————————————————— */
  private frames: Float32Array[] = new Array(1024);
  private head = 0;
  private tail = 0;
  private off0 = 0;

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
    bufUnd: 0, bufOvr: 0,
    conUnd: 0, conOvr: 0,
    stable: 0, adapt: 0,
    maxD: 0, minD: 2_000,
    avgD: 0
  };

  /* ---------------------------------------------------------------- *
   *  constructor                                                     *
   * ---------------------------------------------------------------- */
  constructor(options: AudioWorkletNodeOptions) {
    super();

    /* load config */
    this.cfg = validateConfig(
      (options.processorOptions as any)?.config ?? {},
      this.cfg
    );

    /* shared ring (optional) */
    this.ring = (options.processorOptions as any)?.ringBytes
      ? new SharedByteRing((options.processorOptions as any).ringBytes)
      : null;

    /* metrics helpers */
    this.net = new NetworkQualityEstimator(sampleRate);

    /* message handler */
    this.port.onmessage = ev => this.onMsg(ev.data);

    /* announce readiness */
    const ready: ReadyMsg = {
      type: MESSAGE_TYPES.READY,
      sampleRate: sampleRate,
      timestamp: nowMs(),
      config: this.cfg
    };
    this.port.postMessage(ready);
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
    this.started = false;
    this.remainPartial = 0;
    this.firstChunk = true;
    this.m = { ...this.m, bufUnd:0,bufOvr:0,conUnd:0,conOvr:0,
               stable:0,adapt:0,maxD:0,minD:2_000,avgD:0 };
    this.net.reset();
    this.perf.reset();
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
  }
  private pop(): Float32Array | null {
    if (this.head === this.tail) return null;
    const f = this.frames[this.head];
    this.head = (this.head + 1) % this.frames.length;
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
    if (!this.started && this.buffered() >= this.cfg.initialBufferSamples)
      this.startPlayback();

    /* overrun detection */
    if (this.buffered() > this.maxThreshold()) {
      this.m.bufOvr++; this.m.conOvr++; this.m.conUnd = 0; this.m.stable = 0;
      this.dropExcess();
      if (this.m.conOvr >= CONSTANTS.BUFFER_ADAPTATION.OVERRUN_THRESHOLD &&
          this.cfg.adaptiveBuffering) this.enlargeBuffers();
    } else {
      this.m.conOvr = 0; this.m.stable++;
      if (this.m.stable >= CONSTANTS.BUFFER_ADAPTATION.STABLE_THRESHOLD &&
          this.cfg.adaptiveBuffering && this.cfg.prioritizeLatency)
        this.shrinkBuffers();
    }
  }

  /* ---------------------------------------------------------------- *
   *  playback helpers                                                *
   * ---------------------------------------------------------------- */
  private buffered(): number {
    if (this.head === this.tail) return 0;
    let t = -this.off0;
    for (let i = this.head; i !== this.tail; i = (i + 1) % this.frames.length)
      t += this.frames[i].length;
    return Math.max(0, t);
  }
  private maxThreshold() {
    return this.cfg.initialBufferSamples +
           this.cfg.partialBufferSamples +
           this.cfg.maxBufferSamples;
  }
  private startPlayback() {
    this.started = true;
    this.remainPartial = this.cfg.partialBufferSamples;
    this.firstChunk = true;
  }
  private dropExcess() {
    const target = this.cfg.initialBufferSamples + this.cfg.partialBufferSamples;
    while (this.buffered() > target && this.head !== this.tail) {
      const fLen = this.frames[this.head].length - this.off0;
      const need = this.buffered() - target;
      const rm = Math.min(fLen, need);
      this.off0 += rm;
      if (this.off0 >= this.frames[this.head].length) {
        this.pool.release(this.pop()!);
        this.off0 = 0;
      }
    }
    /* expand max buffer to survive future burst */
    this.cfg.maxBufferSamples = Math.min(
      this.cfg.maxBufferSamples + this.cfg.maxBufferSamplesIncrement,
      this.cfg.maxMaxBufferWithIncrements
    );
  }
  private enlargeBuffers() {
    const rec = this.net.getRecommendedBufferSettings();
    const rate = CONSTANTS.BUFFER_ADAPTATION.ADAPTATION_RATE;
    this.cfg.partialBufferSamples = Math.min(
      Math.max(
        this.cfg.partialBufferSamples,
        Math.floor(rec.partialBufferSamples * (1 + rate))
      ),
      this.cfg.maxPartialWithIncrements
    );
    this.cfg.maxBufferSamples = Math.min(
      Math.max(
        this.cfg.maxBufferSamples,
        Math.floor(rec.maxBufferSamples * (1 + rate))
      ),
      this.cfg.maxMaxBufferWithIncrements
    );
    this.m.adapt++;
    this.m.conUnd = this.m.conOvr = 0;
  }
  private shrinkBuffers() {
    const rate = CONSTANTS.BUFFER_ADAPTATION.ADAPTATION_RATE / 2;
    this.cfg.partialBufferSamples = Math.max(
      CONSTANTS.CONFIG_PRESETS.moshi.partialBufferSamples,
      Math.floor(this.cfg.partialBufferSamples * (1 - rate))
    );
    this.cfg.maxBufferSamples = Math.max(
      CONSTANTS.CONFIG_PRESETS.moshi.maxBufferSamples,
      Math.floor(this.cfg.maxBufferSamples * (1 - rate))
    );
    this.m.stable = CONSTANTS.BUFFER_ADAPTATION.STABLE_THRESHOLD / 2;
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
      wr += cp; this.off0 += cp;
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

    if (this.firstChunk) {
      const fade = Math.min(this.cfg.fadeInSamples, written);
      applyFade(out, 0, fade, true, this.cfg.fadeCurve);
      this.firstChunk = false;
    }

    if (written < out.length) {
      /* underrun */
      this.m.bufUnd++; this.m.conUnd++; this.m.conOvr = 0; this.m.stable = 0;
      const fade = Math.min(this.cfg.fadeOutSamples, written);
      applyFade(out, written - fade, fade, false, this.cfg.fadeCurve);
      this.started = false;

      this.cfg.partialBufferSamples = Math.min(
        this.cfg.partialBufferSamples + this.cfg.partialBufferIncrement,
        this.cfg.maxPartialWithIncrements
      );

      if (this.m.conUnd >= CONSTANTS.BUFFER_ADAPTATION.UNDERRUN_THRESHOLD &&
          this.cfg.adaptiveBuffering) this.enlargeBuffers();
    } else this.m.conUnd = 0;

    /* delay metrics */
    const d = this.buffered() / sampleRate;
    this.m.maxD = Math.max(this.m.maxD, d);
    this.m.minD = Math.min(this.m.minD, d);
    this.m.avgD = this.m.avgD * this.cfg.smoothingFactor + d *
                  (1 - this.cfg.smoothingFactor);

    this.totalPlay += out.length / sampleRate;
    this.actualPlay += written / sampleRate;

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
          frames:
            this.tail >= this.head
              ? this.tail - this.head
              : this.frames.length - this.head + this.tail,
          estimatedBytes: 4 * this.buffered(),
          poolStats: this.pool.stats
        },
        bufferHealth: {
          current: this.buffered(),
          max: this.maxThreshold(),
          partial: this.cfg.partialBufferSamples,
          initial: this.cfg.initialBufferSamples
        },
        timestamp: now
      };
      this.port.postMessage(stats);
      this.lastStats = now; this.recvSinceLast = 0;
    }

    return true;
  }
}

registerProcessor("universal-audio-processor", UniversalAudioProcessor);
