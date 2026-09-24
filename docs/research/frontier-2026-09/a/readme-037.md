# fitreader

A small C library for reading and writing Garmin FIT files, the binary format GPS fitness devices use to store activity data like heart rate, GPS tracks and lap splits.

fitreader is written in C99, needs nothing beyond the standard library, and never allocates behind your back. You read a file one message at a time, and each message is decoded into a plain struct that you can inspect and then throw away.

## Building

You need a C99 compiler and CMake 3.15 or newer.

```sh
git clone https://github.com/example/fitreader.git
cd fitreader
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
sudo cmake --install build
```

This installs `libfitreader.a`, the shared library, `fitreader.h`, and a `pkg-config` file. To link against it:

```sh
cc myapp.c $(pkg-config --cflags --libs fitreader) -o myapp
```

To vendor the library instead, copy `src/` and `include/` into your project. There is no generated code and no configure step.

Build options:

| Option | Default | Effect |
|---|---|---|
| `FIT_BUILD_SHARED` | `ON` | Build a shared library alongside the static one |
| `FIT_ENABLE_WRITER` | `ON` | Compile the encoder (`fit_writer_*`) |
| `FIT_CHECK_CRC` | `ON` | Validate header and file CRCs on read |

## API overview

Everything lives in `fitreader.h`. Reading follows three steps: open, iterate, close.

```c
fit_reader *fit_open(const char *path, fit_error *err);
fit_reader *fit_open_mem(const uint8_t *buf, size_t len, fit_error *err);

int  fit_next(fit_reader *r, fit_message *msg);   /* 1 = got message, 0 = EOF, -1 = error */
void fit_close(fit_reader *r);

const fit_field *fit_get_field(const fit_message *msg, uint8_t field_num);
double           fit_field_value(const fit_field *f);   /* applies scale and offset */
const char      *fit_strerror(fit_error err);
```

Each `fit_message` has a `global_num`, which is the FIT profile message number (for example `FIT_MESG_RECORD` or `FIT_MESG_LAP`), along with the decoded fields and a timestamp when one is present. Compressed-timestamp headers are expanded for you, so `msg.timestamp` always holds an absolute FIT timestamp, counted in seconds since 1989-12-31 00:00 UTC. Use `fit_time_to_unix()` to convert it.

The `fit_message` belongs to the reader and is only valid until the next call to `fit_next()`. Copy out anything you want to keep.

Definition messages are handled internally and never reach your code. Developer fields are exposed through `msg.dev_fields` if you need them.

## Example: extracting a heart rate stream

```c
#include <stdio.h>
#include <fitreader.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s activity.fit\n", argv[0]);
        return 1;
    }

    fit_error err;
    fit_reader *r = fit_open(argv[1], &err);
    if (!r) {
        fprintf(stderr, "open failed: %s\n", fit_strerror(err));
        return 1;
    }

    fit_message msg;
    int rc;
    while ((rc = fit_next(r, &msg)) == 1) {
        if (msg.global_num != FIT_MESG_RECORD)
            continue;

        const fit_field *hr = fit_get_field(&msg, FIT_RECORD_HEART_RATE);
        if (!hr || fit_field_is_invalid(hr))
            continue;   /* sensor dropout or no HR strap */

        printf("%ld,%u\n",
               (long)fit_time_to_unix(msg.timestamp),
               (unsigned)fit_field_value(hr));
    }

    if (rc < 0)
        fprintf(stderr, "read error: %s\n", fit_strerror(fit_last_error(r)));

    fit_close(r);
    return rc < 0;
}
```

This prints `unix_time,bpm` as CSV, one line per record message. Samples where the device wrote the FIT "invalid" sentinel (`0xFF` for heart rate) are skipped.

## Writing files

The writer mirrors the reader. You call `fit_writer_open()`, then `fit_write_message()` for each message, then `fit_writer_close()`, which patches the header and appends the CRC. Definition messages are emitted automatically the first time each message layout appears. See `examples/write_activity.c` for a complete example.

## Supported message types

fitreader decodes any well-formed FIT file at the binary level. Unknown messages are still returned, with raw field values and no names or scaling. The table below lists which messages have full profile support, meaning named field constants, scale/offset and enum types.

**Supported**

- `file_id`, `file_creator`, `device_info`
- `activity`, `session`, `lap`, `record`, `event`
- `length` (pool swims), `hrv`
- `sport`, `user_profile`, `zones_target`
- `developer_data_id`, `field_description`

**Not yet supported** (returned raw)

- Workout and course files: `workout`, `workout_step`, `course`, `course_point`
- Settings and monitoring: `device_settings`, `monitoring`, `monitoring_info`
- `segment_*` messages, `totals`, `weight_scale`, `blood_pressure`
- Most manufacturer-specific messages (global numbers 0xFF00 and above)

Chained FIT files, where several FIT files are concatenated, are read in full. `fit_next()` moves into the next file on its own, and `fit_file_index()` tells you which file the current message came from.

## License

MIT. See `LICENSE`. FIT is a format defined by Garmin; this project is not affiliated with or endorsed by Garmin.
