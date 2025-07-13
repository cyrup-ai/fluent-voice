/**
 * FrameSequencer
 * ──────────────
 * Re-orders incoming audio frames based on monotonically increasing
 * 32-bit sequence numbers.  It supports:
 *
 *   • Early-arriving frames (buffered until their turn)
 *   • Late frames (arrive after we already advanced) — can be dropped
 *   • Huge gaps (connection reset) — force re-sync
 *
 * Memory is bounded: at most `maxBuffered` frames are retained.
 *
 * Typical workflow
 * ────────────────
 *   const seq = new FrameSequencer();
 *   const r = seq.add(frame, sequenceNumber);
 *
 *   if (r.status === "play")      play(r.frame);
 *   else if (r.status === "buffer")  // ignore – will auto flush later
 *   else if (r.status === "discard") // optionally log r.reason
 *
 * The caller should repeatedly poll `drain()` after a “play” event
 * to flush any newly contiguous frames.
 */

export interface PlayResult {
    status: "play";
    frame: Float32Array;
    /** additional frames that became contiguous */
    extra: Float32Array[];
}

export interface BufferResult {
    status: "buffer";
    buffered: number; // count of frames currently buffered
}

export interface DiscardResult {
    status: "discard";
    reason: "too_old" | "duplicate";
}

type Result = PlayResult | BufferResult | DiscardResult;

export interface SequencerStats {
    reorderedFrames: number;
    droppedLateFrames: number;
    droppedOldFrames: number;
    duplicates: number;
    maxReorderDistance: number;
    bufferedFrames: number;
    nextExpected: number;
}

export class FrameSequencer {
    private readonly buf = new Map<number, Float32Array>(); // seq → frame
    private nextSeq = 0;
    private minBufferedSeq: number | null = null; // For fast oldest lookup

    constructor(
        private readonly maxBuffered = 50,
        private readonly maxGap = 1_000 /* sequences considered a reset */,
    ) {}

    add(frame: Float32Array, seq: number): Result {
        /* first ever frame -> establish baseline */
        if (this.nextSeq === 0) {
            this.nextSeq = seq + 1;
            return { status: "play", frame, extra: [] };
        }

        /* duplicate? */
        if (seq < this.nextSeq && this.buf.has(seq))
            return { status: "discard", reason: "duplicate" };

        /* expected → play immediately */
        if (seq === this.nextSeq) {
            const extra: Float32Array[] = [];
            this.nextSeq++;

            /* flush contiguous buffered frames */
            while (this.buf.has(this.nextSeq)) {
                extra.push(this.buf.get(this.nextSeq)!);
                this.buf.delete(this.nextSeq);
                this.nextSeq++;
            }
            // Update minBufferedSeq
            if (this.buf.size === 0) {
                this.minBufferedSeq = null;
            } else {
                this.minBufferedSeq = Math.min(...this.buf.keys());
            }
            return { status: "play", frame, extra };
        }

        /* massive gap → treat as stream reset */
        if (seq > this.nextSeq + this.maxGap) {
            this.buf.clear();
            this.minBufferedSeq = null;
            this.nextSeq = seq + 1;
            return { status: "play", frame, extra: [] };
        }

        /* future frame -> buffer */
        if (seq > this.nextSeq) {
            this.buf.set(seq, frame);
            if (this.minBufferedSeq === null || seq < this.minBufferedSeq) {
                this.minBufferedSeq = seq;
            }

            /* enforce memory bound */
            if (this.buf.size > this.maxBuffered) {
                // Drop the oldest buffered seq
                // Use minBufferedSeq for O(1) lookup if possible
                let oldest: number;
                if (
                    this.minBufferedSeq !== null &&
                    this.buf.has(this.minBufferedSeq)
                ) {
                    oldest = this.minBufferedSeq;
                } else {
                    oldest = Math.min(...this.buf.keys());
                }
                this.buf.delete(oldest);

                // Update minBufferedSeq
                if (this.buf.size === 0) {
                    this.minBufferedSeq = null;
                } else {
                    // Only update if we just deleted the min
                    if (oldest === this.minBufferedSeq) {
                        this.minBufferedSeq = Math.min(...this.buf.keys());
                    }
                }
            }

            return { status: "buffer", buffered: this.buf.size };
        }

        /* late frame (< nextSeq) */
        const distance = this.nextSeq - seq;
        if (distance > this.maxBuffered)
            return { status: "discard", reason: "too_old" };

        /* mildly late frame — better to play than drop */
        return { status: "play", frame, extra: [] };
    }

    /** Flush any contiguous frames already buffered. */
    drain(): Float32Array[] {
        const list: Float32Array[] = [];
        while (this.buf.has(this.nextSeq)) {
            list.push(this.buf.get(this.nextSeq)!);
            this.buf.delete(this.nextSeq);
            this.nextSeq++;
        }
        // Update minBufferedSeq
        if (this.buf.size === 0) {
            this.minBufferedSeq = null;
        } else {
            this.minBufferedSeq = Math.min(...this.buf.keys());
        }
        return list;
    }

    /** Diagnostics snapshot. */
    getStats(): SequencerStats {
        return {
            reorderedFrames: 0, // kept for compatibility – could track
            droppedLateFrames: 0, // ""
            droppedOldFrames: 0, // ""
            duplicates: 0, // ""
            maxReorderDistance: 0, // ""
            bufferedFrames: this.buf.size,
            nextExpected: this.nextSeq,
        };
    }

    /** Hard reset. */
    reset(): void {
        this.buf.clear();
        this.nextSeq = 0;
        this.minBufferedSeq = null;
    }
}
