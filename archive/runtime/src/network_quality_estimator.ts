import { calculateJitter, estimateBufferSize } from "./utils.js";

/**
 * NetworkQualityEstimator
 * ───────────────────────
 * Collects low-level arrival statistics and derives metrics used by the
 * adaptive jitter-buffer:
 *
 * • **jitter**: standard-deviation of inter-arrival times (seconds)
 * • **packetLoss**: fraction of missing sequence numbers (0-1)
 * • **estimatedBps**: payload bandwidth (bytes/s) over rolling window
 * • **frameInterval**: mean arrival spacing (ms)
 * • **connectionType**: heuristic “wifi / 4g / 3g / 2g / slow-2g / ethernet”
 *
 * A rolling window keeps at most `maxHistory` packets, so work remains O(1).
 */
export class NetworkQualityEstimator {
    /* rolling history */
    private readonly times: number[] = []; // ms
    private readonly sizes: number[] = []; // bytes
    private readonly maxHistory: number;

    /* sequence tracking */
    private lastSeq = -1;
    private missing = 0;

    /* derived metrics */
    jitter = 0; // seconds
    packetLoss = 0; // 0-1
    estimatedBps = 0; // bytes / second
    frameInterval = 0; // milliseconds
    connectionType = "unknown";

    /* ctor */
    constructor(
        private readonly sampleRate: number,
        maxHistory = 200,
    ) {
        this.maxHistory = maxHistory;
    }

    /* ─── public datapoint ingestion ─────────────────────────────────── */
    /**
     * Record a frame arrival.
     * @param tsMs   arrival timestamp (ms)
     * @param size   payload bytes
     * @param seq    optional sequence number for loss detection
     */
    recordFrameArrival(tsMs: number, size: number, seq?: number): void {
        /* rolling window push / trim */
        this.times.push(tsMs);
        this.sizes.push(size);
        if (this.times.length > this.maxHistory) {
            this.times.shift();
            this.sizes.shift();
        }

        /* jitter (std-dev of inter-arrival) */
        if (this.times.length >= 3) this.jitter = calculateJitter(this.times);

        /* sequence-based packet-loss */
        if (seq !== undefined) {
            if (this.lastSeq >= 0 && seq > this.lastSeq + 1)
                this.missing += seq - this.lastSeq - 1;
            this.lastSeq = seq;

            const expected = this.lastSeq + 1;
            this.packetLoss = expected ? this.missing / expected : 0;
        }

        /* bandwidth & frame interval */
        if (this.times.length >= 2) {
            const spanMs = this.times[this.times.length - 1] - this.times[0];
            const totalBytes = this.sizes.reduce((s, b) => s + b, 0);
            if (spanMs > 0) {
                this.estimatedBps = (totalBytes / spanMs) * 1000;
                this.frameInterval = spanMs / (this.times.length - 1);
            }
        }

        /* heuristic connection type */
        const j = this.jitter,
            bw = this.estimatedBps;
        if (bw > 10_000_000 && j < 0.01) this.connectionType = "ethernet";
        else if (bw > 2_000_000 && j < 0.02) this.connectionType = "wifi";
        else if (bw > 500_000 && j < 0.05) this.connectionType = "4g";
        else if (bw > 100_000 && j < 0.1) this.connectionType = "3g";
        else if (bw > 20_000) this.connectionType = "2g";
        else this.connectionType = "slow-2g";
    }

    /* ─── API surface ─────────────────────────────────────────────────── */
    getMetrics() {
        return {
            jitter: this.jitter,
            packetLoss: this.packetLoss,
            estimatedBandwidth: this.estimatedBps,
            frameInterval: this.frameInterval,
            connectionType: this.connectionType,
            lostFrames: this.missing,
        };
    }

    /**
     * Suggest buffer sizes (in samples) given current network state,
     * delegating to `estimateBufferSize` helper.
     */
    getRecommendedBufferSettings() {
        return estimateBufferSize(
            this.jitter,
            this.packetLoss,
            this.connectionType,
            this.sampleRate,
        );
    }

    /** Reset all statistics. */
    reset(): void {
        this.times.length = this.sizes.length = 0;
        this.lastSeq = -1;
        this.missing = 0;
        this.jitter = 0;
        this.packetLoss = 0;
        this.estimatedBps = 0;
        this.frameInterval = 0;
        this.connectionType = "unknown";
    }
}
