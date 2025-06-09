declare const currentTime: number; // seconds in AudioWorkletGlobalScope
declare const sampleRate: number; // Hz

import { CONSTANTS } from "../constants.js";

/* ----- time helpers --------------------------------------------------- */
export const asMs = (s: number, sr: number) => (s * 1000) / sr;
export const asSamples = (ms: number, sr: number) =>
    Math.round((ms * sr) / 1000);
export const nowMs = () => currentTime * 1000;

/* ----- fade LUT ------------------------------------------------------- */
const FADE_LUT = new Map<string, Float32Array>();
function fadeTable(len: number, inOut: "in" | "out", curve: string) {
    const k = `${len}|${inOut}|${curve}`;
    if (FADE_LUT.has(k)) return FADE_LUT.get(k)!;
    const t = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        const x = i / len;
        switch (curve) {
            case "exponential":
                t[i] = inOut === "in" ? x * x : (1 - x) * (1 - x);
                break;
            case "logarithmic":
                t[i] = inOut === "in" ? Math.sqrt(x) : Math.sqrt(1 - x);
                break;
            case "sinusoidal":
                t[i] =
                    inOut === "in"
                        ? Math.sin((x * Math.PI) / 2)
                        : Math.sin(((1 - x) * Math.PI) / 2);
                break;
            default:
                t[i] = inOut === "in" ? x : 1 - x;
        }
    }
    FADE_LUT.set(k, t);
    return t;
}
export function applyFade(
    buf: Float32Array,
    off: number,
    len: number,
    fadeIn: boolean,
    curve: string,
) {
    if (len <= 0) return;
    const tbl = fadeTable(len, fadeIn ? "in" : "out", curve);
    for (let i = 0; i < len; i++) buf[off + i] *= tbl[i];
}

/* ----- config validation --------------------------------------------- */
export function validateConfig(cfg: any, def: any) {
    const out = { ...def };
    for (const [k, v] of Object.entries(def))
        if (k in cfg && typeof cfg[k] === typeof v) (out as any)[k] = cfg[k];
    const perf = ["balanced", "quality", "latency", "reliability"];
    if (!perf.includes(out.performanceMode)) out.performanceMode = "balanced";
    return out;
}

/* ----- jitter / bandwidth helpers ------------------------------------ */
export function calculateJitter(arr: number[]) {
    if (arr.length < 3) return 0;
    const deltas = arr.slice(1).map((t, i) => t - arr[i]);
    const avg = deltas.reduce((s, d) => s + d, 0) / deltas.length;
    const var_ = deltas.reduce((s, d) => s + (d - avg) ** 2, 0) / deltas.length;
    return Math.sqrt(var_) / 1000; /* seconds */
}

/* ----- network-driven buffer recommendation -------------------------- */
export function estimateBufferSize(
    jitter: number,
    loss: number,
    conn: string,
    sr: number,
) {
    const base =
        CONSTANTS.CONNECTION_TYPE_BUFFER_SIZES[
            conn as keyof typeof CONSTANTS.CONNECTION_TYPE_BUFFER_SIZES
        ] ?? CONSTANTS.CONNECTION_TYPE_BUFFER_SIZES.unknown;
    const totalSec = base + jitter * 3 + loss * 0.5;
    const init = asSamples(totalSec * 1000, sr);
    return {
        initialBufferSamples: init,
        partialBufferSamples: init >> 3,
        maxBufferSamples: init >> 3,
        partialBufferIncrement: init >> 4,
        maxBufferSamplesIncrement: init >> 4,
        maxPartialWithIncrements: init >> 1,
        maxMaxBufferWithIncrements: init >> 1,
    };
}
