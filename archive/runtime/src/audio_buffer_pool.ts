/**
 * AudioBufferPool
 * ───────────────
 * Simple freelist for Float32Array blocks to minimise garbage-collector
 * churn inside the AudioWorklet thread.  Buffers are keyed by their
 * `.length` (sample count).  Typical usage:
 *
 *   const pool = new AudioBufferPool(    // pre-allocate 20 × 960-sample blocks
 *     { preallocate: [ { size: 960, count: 20 } ] }
 *   );
 *
 *   // obtain a zero-filled buffer
 *   const buf = pool.get(960);
 *
 *   // … fill with PCM …
 *
 *   // return to freelist
 *   pool.release(buf);
 *
 * Diagnostic counters are exposed via `stats`.
 */

export interface PoolOptions {
    /**
     * Pre-allocate buffers so the first burst avoids `new Float32Array`.
     * Example: `[ { size: 960, count: 20 }, { size: 2048, count: 4 } ]`
     */
    preallocate?: { size: number; count: number }[];
    /** Maximum freelist length per size before trim() starts culling. */
    maxPerSize?: number;
}

export class AudioBufferPool {
    // Use number keys for faster access and less string conversion
    private readonly pool = new Map<number, Float32Array[]>();
    private readonly maxPer: number;

    /** Rolling statistics since construction. */
    readonly stats = {
        created: 0,
        reused: 0,
        released: 0,
        trimmed: 0,
    };

    constructor(opts: PoolOptions = {}) {
        this.maxPer = opts.maxPerSize ?? 32;

        /* optional pre-allocation */
        if (opts.preallocate) {
            for (const { size, count } of opts.preallocate) {
                if (!this.pool.has(size)) this.pool.set(size, []);
                const arr = this.pool.get(size)!;
                for (let i = 0; i < count; i++) {
                    arr.push(new Float32Array(size));
                    this.stats.created++;
                }
            }
        }
    }

    /**
     * Obtain a zero-filled buffer of the requested size.
     * If no freelist entry is available, a new one is created.
     */
    get(size: number): Float32Array {
        const list = this.pool.get(size);
        if (list && list.length) {
            this.stats.reused++;
            return list.pop()!; // `!` – list is non-empty
        }

        this.stats.created++;
        return new Float32Array(size); // JS engine zeros new TypedArray
    }

    /**
     * Return a buffer to the freelist.  The buffer is zeroed before storage
     * to avoid leaking prior audio data across users of the pool.
     */
    release(buf: Float32Array): void {
        const size = buf.length;
        let list = this.pool.get(size);
        if (!list) {
            list = [];
            this.pool.set(size, list);
        }

        // Use a fast zeroing method for large buffers
        if (buf.length > 128) {
            // Unroll for large buffers
            let i = 0,
                len = buf.length;
            const block = 32;
            for (; i + block <= len; i += block) {
                buf.fill(0, i, i + block);
            }
            if (i < len) buf.fill(0, i, len);
        } else {
            buf.fill(0);
        }

        list.push(buf);
        this.stats.released++;

        /* opportunistic trim if freelist grew too large */
        if (list.length > this.maxPer) this.trim(size);
    }

    /**
     * Trim freelist for a specific size (or all sizes if key omitted) down
     * to `maxPerSize` buffers, releasing the excess for GC.
     */
    trim(key?: number): void {
        const trimOne = (k: number) => {
            const list = this.pool.get(k);
            if (list && list.length > this.maxPer) {
                const excess = list.length - this.maxPer;
                list.length = this.maxPer; // drop references
                this.stats.trimmed += excess;
            }
        };

        if (typeof key === "number") trimOne(key);
        else for (const k of this.pool.keys()) trimOne(k);
    }

    /**
     * Snapshot of current freelist occupancy useful for debugging.
     */
    snapshot() {
        const entries: { size: number; count: number }[] = [];
        for (const [k, v] of this.pool.entries())
            entries.push({ size: k, count: v.length });
        return entries;
    }
}
