/**
 * SharedByteRing
 * ──────────────
 * A lock-free single-producer / single-consumer ring buffer for 32-bit
 * PCM samples, implemented on top of `SharedArrayBuffer`.
 *
 * The layout (little-endian 32-bit words):
 *
 *   Int32[0]  writeIdx   → next position producer will write
 *   Int32[1]  readIdx    → next position consumer will read
 *   Int32[2]  capacity   → total sample slots
 *   Float32[] data       → PCM payload, length == capacity
 *
 * Synchronisation:
 * ────────────────
 * • Producer **only** writes `writeIdx` (W) and never touches R.
 * • Consumer **only** writes `readIdx`  (R) and never touches W.
 * • Both may *read* the other index.
 * • One sample slot is deliberately left unused so the “empty” state
 *   (W == R) is unambiguous; the ring is *full* when advancing W would
 *   make it equal to R.
 *
 * Blocking helpers (`waitForSpace`, `waitForData`) use `Atomics.wait`
 * (fallback to busy-yield if unavailable) so the producer can sleep
 * until the consumer drains, and vice-versa.
 */

const HEADER_BYTES = 12; // 3 × Int32 = 12
const I32_WRITE = 0,
    I32_READ = 1,
    I32_CAP = 2;

/* ------------------------------------------------------------------ */
export class SharedByteRing {
    private readonly i32: Int32Array;
    private readonly f32: Float32Array;
    readonly capacity: number;

    constructor(sab: SharedArrayBuffer) {
        this.i32 = new Int32Array(sab, 0, 3);
        this.f32 = new Float32Array(sab, HEADER_BYTES);
        this.capacity = Atomics.load(this.i32, I32_CAP);
        if (this.capacity <= 1)
            throw new Error("capacity must be >1 (one slot kept empty)");
    }

    /* ——— state helpers ——————————————————————————————— */
    /** Samples currently queued and readable by consumer. */
    available(): number {
        const w = Atomics.load(this.i32, I32_WRITE);
        const r = Atomics.load(this.i32, I32_READ);
        return w >= r ? w - r : this.capacity - (r - w);
    }

    /** Free space (in samples) for producer to write. */
    space(): number {
        // subtract 1 to keep at least one slot free
        return this.capacity - 1 - this.available();
    }

    /* ——— consumer API ———————————————————————————————— */
    /**
     * Pop up to `dst.length` samples into `dst`.
     * @return number of samples copied (0 means ring empty).
     */
    pop(dst: Float32Array): number {
        let r = Atomics.load(this.i32, I32_READ);
        let copied = 0;

        while (copied < dst.length) {
            const w = Atomics.load(this.i32, I32_WRITE);
            if (r === w) break; // empty

            const chunk = r < w ? w - r : this.capacity - r;
            const want = Math.min(chunk, dst.length - copied);

            dst.set(this.f32.subarray(r, r + want), copied);
            r = (r + want) % this.capacity;
            copied += want;
        }

        if (copied) {
            Atomics.store(this.i32, I32_READ, r);
            Atomics.notify(this.i32, I32_WRITE, 1); // wake producer
        }
        return copied;
    }

    /** Block (or spin-yield) until at least `needed` samples are available. */
    waitForData(needed = 1): void {
        while (this.available() < needed) {
            if (Atomics.wait)
                Atomics.wait(
                    this.i32,
                    I32_WRITE,
                    Atomics.load(this.i32, I32_WRITE),
                    50,
                );
            else Atomics.yield?.();
        }
    }

    /* ——— producer API ———————————————————————————————— */
    /**
     * Push PCM from `src` into ring. Returns `true` on success.
     * If not enough contiguous space exists, **no** data is written and
     * the function returns `false`.
     */
    push(src: Float32Array): boolean {
        const need = src.length;
        if (need > this.space()) return false;

        let w = Atomics.load(this.i32, I32_WRITE);

        const first = Math.min(need, this.capacity - w);
        this.f32.set(src.subarray(0, first), w);
        w = (w + first) % this.capacity;

        const remain = need - first;
        if (remain) {
            this.f32.set(src.subarray(first), w);
            w = (w + remain) % this.capacity;
        }

        Atomics.store(this.i32, I32_WRITE, w);
        Atomics.notify(this.i32, I32_READ, 1); // wake consumer
        return true;
    }

    /** Block (or spin) until at least `need` samples of free space. */
    waitForSpace(need: number): void {
        while (this.space() < need) {
            if (Atomics.wait)
                Atomics.wait(
                    this.i32,
                    I32_READ,
                    Atomics.load(this.i32, I32_READ),
                    50,
                );
            else Atomics.yield?.();
        }
    }
}

/* ------------------------------------------------------------------ *
 *  Helper: create new SharedArrayBuffer with header initialised      *
 * ------------------------------------------------------------------ */
export function createRingBuffer(capacitySamples: number): SharedArrayBuffer {
    if (capacitySamples <= 1)
        throw new Error("capacitySamples must be >1 (one slot kept empty)");
    const sab = new SharedArrayBuffer(
        HEADER_BYTES + capacitySamples * Float32Array.BYTES_PER_ELEMENT,
    );
    const i32 = new Int32Array(sab, 0, 3);
    i32[I32_WRITE] = 0;
    i32[I32_READ] = 0;
    i32[I32_CAP] = capacitySamples;
    return sab;
}
