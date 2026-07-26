# rotenc-mod

A Linux kernel module that exposes a quadrature rotary encoder wired to two GPIO pins as a character device. Each `read()` returns the net position change since the previous read, as a signed integer.

This exists because the obvious userspace approach — polling `/sys/class/gpio` or waiting on `gpiod` line events — drops steps when you spin the knob fast and the scheduler isn't feeling generous. `rotenc-mod` decodes in an interrupt handler, so the count stays honest.

It is deliberately small. If you need button presses, multiple encoders, or proper input-subsystem integration, use `rotary-encoder.c` in the mainline tree instead. This module is for when you want a file descriptor and nothing else.

## Requirements

- A kernel with GPIO interrupt support (essentially all of them on ARM SBCs and x86 boards with a GPIO controller)
- Kernel headers for the running kernel
- An encoder with A and B outputs, common ground, and detents or not — both work

Tested on 5.15, 6.1, and 6.6 on a Raspberry Pi 4 and a BeagleBone Black.

## Building

Install headers for whatever kernel you're actually running:

```sh
# Debian/Ubuntu/Raspberry Pi OS
sudo apt install build-essential linux-headers-$(uname -r)

# Fedora
sudo dnf install kernel-devel-$(uname -r)
```

Then:

```sh
git clone https://github.com/example/rotenc-mod
cd rotenc-mod
make
```

The `Makefile` builds out-of-tree against `/lib/modules/$(shell uname -r)/build`. If your headers live somewhere else, override it:

```sh
make KDIR=/path/to/kernel/headers
```

You should end up with `rotenc.ko` in the source directory.

## Loading

The module takes two required parameters, the GPIO numbers for the A and B channels:

```sh
sudo insmod rotenc.ko gpio_a=17 gpio_b=27
```

These are global GPIO numbers as the kernel sees them, not physical header pins. On a Pi the BCM numbering matches; on other boards check `/sys/kernel/debug/gpio` to map base offsets.

Optional parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `gpio_a` | — | GPIO number for channel A (required) |
| `gpio_b` | — | GPIO number for channel B (required) |
| `debounce_us` | `500` | Minimum microseconds between accepted transitions |
| `invert` | `0` | Set to `1` to flip the sign of reported deltas |
| `pullup` | `1` | Enable the internal pull-up on both lines |

On success the module creates `/dev/rotenc0`. It's owned by root with mode `0600` by default; add a udev rule if you want it readable by a normal user:

```
# /etc/udev/rules.d/60-rotenc.rules
KERNEL=="rotenc0", MODE="0660", GROUP="gpio"
```

Unload with `sudo rmmod rotenc`.

Failures show up in `dmesg`. The most common one is `rotenc: failed to request gpio 17 (-16)`, which means something else already owns the pin — usually a device tree overlay or a leftover sysfs export.

## Read semantics

`/dev/rotenc0` is a blocking character device with a deliberately narrow contract:

- A `read()` of at least `sizeof(int)` bytes returns exactly one `int` in native byte order: the accumulated position delta since the last successful read.
- Reading resets the accumulator to zero. Deltas are not cumulative across reads.
- If the accumulator is zero, `read()` blocks until the knob moves. Open with `O_NONBLOCK` to get `-EAGAIN` instead.
- Clockwise rotation produces positive values, counter-clockwise negative — unless `invert=1`, or unless you wired A and B backwards, which amounts to the same thing.
- Buffers shorter than `sizeof(int)` return `-EINVAL`. Longer buffers are fine; only four bytes are written.
- `poll()`/`select()`/`epoll` work. The device reports `POLLIN | POLLRDNORM` whenever the accumulator is non-zero.

The accumulator is a signed 32-bit counter guarded by a spinlock. It saturates rather than wrapping, so a knob spun for a very long time between reads reports `INT_MAX` instead of nonsense.

Multiple readers are permitted but not useful: whichever one wakes up first consumes the delta.

## Example

A minimal program that tracks absolute position:

```c
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int main(void)
{
    int fd = open("/dev/rotenc0", O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    long position = 0;

    for (;;) {
        int delta;
        ssize_t n = read(fd, &delta, sizeof delta);

        if (n < 0) {
            perror("read");
            break;
        }
        if (n != sizeof delta) {
            fprintf(stderr, "short read: %zd\n", n);
            break;
        }

        position += delta;
        printf("delta %+d  position %ld\n", delta, position);
        fflush(stdout);
    }

    close(fd);
    return 0;
}
```

Build and run:

```sh
cc -Wall -o knob examples/knob.c
sudo ./knob
```

Turn the encoder and you'll see one line per burst of movement. Spin it quickly and you'll get larger deltas in fewer lines — that's the coalescing working as intended, not lost steps.

## Notes on wiring

Connect A and B to the two GPIOs and the common pin to ground. With `pullup=1` (the default) no external resistors are needed. Mechanical encoders bounce badly; if you see jitter at rest, raise `debounce_us` to 1000 or 2000. Optical encoders don't need debouncing at all — set it to `0` and save yourself the latency.

## License

GPL-2.0. It's a kernel module; it wasn't going to be anything else.
