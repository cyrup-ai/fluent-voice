/**
 * PerformanceMonitor
 * ──────────────────
 * Captures per-block processing latency inside the AudioWorklet thread
 * and provides aggregate statistics.
 *
 * Design goals:
 * • Zero allocations in the hot path (`begin` / `end`)
 * •  O(1) constant memory footprint (ring histogram)
 * •  Quick percentile queries for debugging (p50 / p95)
 *
 * Typical usage:
 *
 *   const perf = new PerformanceMonitor(256);   // keep last 256 samples
 *
 *   process() {
 *     perf.begin();
 *     … heavy DSP …
 *     const dt = perf.end(5);   // 5-ms soft budget
 *     if (dt > 5) console.warn("overrun", dt);
 *   }
 */

export class PerformanceMonitor {
    private readonly hist: Float32Array;
    private idx = 0;
    private readonly size: number;

    /* running totals */
    private t0 = 0;
    private sum = 0;
    private count = 0;
    private max = 0;
    private overruns = 0;

    constructor(windowSize = 128) {
        this.size = windowSize;
        this.hist = new Float32Array(windowSize);
    }

    /** Mark start of timed region. */
    begin(): void {
        this.t0 = nowMs();
    }

    /**
     * Mark end of region; returns elapsed milliseconds.
     * @param budget soft budget for overrun counter
     */
    end(budget: number): number {
        const dt = nowMs() - this.t0;

        /* ring histogram */
        this.hist[this.idx] = dt;
        this.idx = (this.idx + 1) % this.size;

        /* aggregates */
        this.sum += dt;
        this.count++;
        if (dt > this.max) this.max = dt;
        if (dt > budget) this.overruns++;

        return dt;
    }

    /** Reset all statistics & histogram. */
    reset(): void {
        this.hist.fill(0);
        this.idx = 0;
        this.sum = this.count = this.max = this.overruns = 0;
    }

    /** Average over *all* samples since last reset. */
    mean(): number {
        return this.count ? this.sum / this.count : 0;
    }

    /** Max single-sample latency. */
    peak(): number {
        return this.max;
    }

    /** Overrun ratio (samples exceeding given budget / total). */
    overrunRatio(): number {
        return this.count ? this.overruns / this.count : 0;
    }

    /**
     * Percentile over the rolling histogram (p in 0…1).
     * Uses a copy sort to avoid altering histogram order; O(N log N) but N≤256.
     */
    percentile(p: number): number {
        if (p <= 0) return 0;
        if (p >= 1) return this.max;
        const tmp = Array.from(this.hist);
        tmp.sort((a, b) => a - b);
        const idx = Math.floor(p * (tmp.length - 1));
        return tmp[idx];
    }

    /** Export snapshot suitable for postMessage / JSON. */
    get metrics() {
        return {
            mean: this.mean(),
            max: this.max,
            overruns: this.overruns,
            samples: this.count,
            p50: this.percentile(0.5),
            p95: this.percentile(0.95),
        };
    }
}
