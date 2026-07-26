# ringbuf-hpp

A header-only, lock-free, single-producer single-consumer ring buffer for C++17 and later. It was written for the usual audio problem: getting samples or events between a real-time callback thread and a UI thread without ever taking a lock, allocating, or blocking on the audio side.

Single header, no build step, no dependencies beyond the standard library.

## Installing

Copy `include/ringbuf.hpp` into your project and include it.

```cpp
#include "ringbuf.hpp"
```

That is the whole installation. There is a CMake target if you want one (`add_subdirectory(ringbuf-hpp)`, then link `ringbuf::ringbuf`), but it only sets an include directory.

## Basic use

Capacity is a template parameter and is rounded up to the next power of two, which lets the implementation mask instead of divide when wrapping.

```cpp
#include "ringbuf.hpp"
#include <thread>
#include <cstdio>

rb::ringbuf<float, 4096> fifo;

void audio_thread() {
    float block[128];
    for (int i = 0; i < 128; ++i) block[i] = process_one_sample();

    // Never blocks. Returns how many elements were actually written.
    const size_t written = fifo.push_bulk(block, 128);
    if (written < 128) {
        // Consumer is behind. Drop, and count it — do not retry here.
        ++dropped_blocks;
    }
}

void ui_thread() {
    float scratch[512];
    while (running) {
        const size_t got = fifo.pop_bulk(scratch, 512);
        update_meter(scratch, got);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}
```

Single-element access is also available:

```cpp
if (!fifo.push(sample)) { /* full */ }

float out;
while (fifo.pop(out)) { consume(out); }
```

Both `push` and `pop` are wait-free and never allocate. `size()` and `empty()` are approximate by nature — they read the other thread's index, which may advance the moment after you look — so treat them as hints. The return value of `push`/`pop` is the only thing you should branch on.

The buffer holds `Capacity - 1` elements. One slot is left empty to keep the full and empty states distinguishable without a separate flag.

## Element type requirements

`T` must be trivially copyable and nothrow default-constructible. Storage is a plain array, constructed once when the buffer is constructed; pushing assigns into an existing slot rather than placement-newing. This is deliberate — it keeps the producer path free of anything that could throw or allocate. If you need to move non-trivial objects across the boundary, pass a raw pointer or an index into a preallocated pool instead.

## Memory ordering

The implementation uses two `std::atomic<size_t>` indices, `write_` and `read_`, each padded to its own cache line (64 bytes by default, override with `-DRB_CACHELINE=…`) to avoid false sharing between the two threads.

The ordering discipline is the standard acquire/release handshake:

- The producer reads its own `write_` with `memory_order_relaxed` — it is the only writer, so no synchronization is needed to observe its own value.
- The producer reads `read_` with `memory_order_acquire` to determine free space. This pairs with the consumer's release store and guarantees the producer sees that the consumer has finished reading those slots.
- The producer writes the data, then stores the new `write_` with `memory_order_release`. That store publishes the element writes.
- The consumer mirrors this exactly: relaxed load of `read_`, acquire load of `write_`, read the data, release store of `read_`.

No sequential consistency is required anywhere, and there is no `seq_cst` fence in the hot path. The correctness argument rests entirely on the C++11 memory model, not on any particular hardware's ordering, so it holds on weakly ordered architectures.

Indices are monotonically increasing and masked at access time. On a 64-bit `size_t` this means overflow is not a practical concern; on a 32-bit target the wraparound is still handled correctly because capacity is a power of two and the arithmetic is unsigned.

## What this is not

It is not multi-producer or multi-consumer. Two producer threads calling `push` concurrently is undefined behaviour, full stop — there is no debug assertion that can reliably catch it, so this is on you. It is not a bounded queue with blocking semantics; there is no `wait_and_pop`. It does not grow.

## Tested on

| Platform | Compiler | Notes |
| --- | --- | --- |
| Linux x86-64 | GCC 11, 13; Clang 15, 17 | primary development target |
| Linux aarch64 | GCC 12 | weak ordering, exercised under stress test |
| macOS arm64 | Apple Clang 15 | CoreAudio callback, real workload |
| Windows x86-64 | MSVC 19.38 | `/std:c++17` and `/std:c++20` |

The test suite includes a stress test that runs a producer and consumer for 30 seconds with randomized block sizes and verifies the consumed sequence is exactly the produced sequence with no gaps or duplicates. It is also run under ThreadSanitizer on the Linux configurations; TSan is clean.

## Licence

MIT.
