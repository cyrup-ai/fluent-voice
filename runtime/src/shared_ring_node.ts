/**
 * shared-ring-node.ts
 * ───────────────────
 * Main-thread helper that:
 *
 *  1. Ensures `universal-audio.ts` is loaded as an AudioWorkletModule.
 *  2. Creates an `AudioWorkletNode` wired to that processor.
 *  3. Allocates a `SharedArrayBuffer` ring and passes it via
 *     `processorOptions`.
 *  4. Returns an **`enqueue()`** function that the caller can feed with
 *     `Float32Array` PCM frames (mono, 32-bit float, sample-rate-matching
 *     the AudioContext) – frames are copied zero-copy into the shared ring.
 *
 * Usage
 * ─────
 *   const ctx = new AudioContext({ sampleRate: 48_000 });
 *   const { node, enqueue } = await createSharedRingNode(ctx, {
 *     ringSeconds: 0.5          // 500 ms of capacity
 *   });
 *
 *   incomingSocket.onPCM = pcm => {
 *     if (!enqueue(pcm))
 *       console.warn("Ring full – frame dropped");
 *   };
 */

import { createRingBuffer, SharedByteRing } from "./lib/shared-byte-ring.js";

/** Caches module loading promises to avoid redundant `addModule` calls. */
const moduleLoadPromises = new WeakMap<
    BaseAudioContext,
    Map<string, Promise<void>>
>();

export interface RingNodeOptions {
    /** Ring capacity in seconds (default 0.5 s). */
    ringSeconds?: number;
    /** Explicit sample-rate (defaults to ctx.sampleRate). */
    sampleRate?: number;
    /** Worklet script URL (default "universal-audio.js"). */
    url?: string;
    /** Mono/stereo etc. (default 1). */
    channelCount?: number;
}

/** Producer-side interface returned to caller. */
export interface RingWriter {
    /** Push PCM frame; returns `false` if ring currently full. */
    enqueue(frame: Float32Array): boolean;
    /** Shared buffer for diagnostics / transfer. */
    sab: SharedArrayBuffer;
}

/** The result of creating a shared ring node. */
export interface SharedRingNode {
    /** The `AudioWorkletNode` that consumes from the ring. */
    node: AudioWorkletNode;
    /** Function to push audio frames into the ring. */
    enqueue: RingWriter["enqueue"];
    /** The underlying `SharedArrayBuffer` for the ring. */
    sab: SharedArrayBuffer;
}

/**
 * Create node + writer.
 * The node is auto-connected to `ctx.destination`; caller may disconnect.
 */
export async function createSharedRingNode(
    ctx: BaseAudioContext,
    opts: RingNodeOptions = {},
): Promise<SharedRingNode> {
    /* defaults */
    const {
        ringSeconds = 0.5,
        sampleRate = ctx.sampleRate,
        url = "universal-audio.js",
        channelCount = 1,
    } = opts;

    /* 1. Load module once, robustly */
    if (!moduleLoadPromises.has(ctx)) {
        moduleLoadPromises.set(ctx, new Map());
    }
    const urlPromises = moduleLoadPromises.get(ctx)!;

    if (!urlPromises.has(url)) {
        urlPromises.set(url, ctx.audioWorklet.addModule(url));
    }
    await urlPromises.get(url)!;

    /* 2. SAB allocation */
    const capSamples = Math.ceil(sampleRate * ringSeconds);
    const sab = createRingBuffer(capSamples);
    const ring = new SharedByteRing(sab);

    /* 3. Worklet node */
    const node = new AudioWorkletNode(ctx, "universal-audio-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [channelCount],
        processorOptions: { ringBytes: sab, sampleRate },
    });

    /* auto connect – consumer may change routing */
    node.connect(ctx.destination);

    /* 4. writer */
    return {
        node,
        enqueue: (pcm) => ring.push(pcm),
        sab,
    };
}
