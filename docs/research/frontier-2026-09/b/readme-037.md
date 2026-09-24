# fitreader

A small C library for reading and writing Garmin FIT files, the binary format used by GPS fitness devices to record activities: heart rate, GPS tracks, lap splits, power, cadence and so on.

fitreader is written in C99 with no dependencies beyond the standard library. It decodes the FIT header, definition messages and data messages, resolves field types against the global FIT profile, and hands you plain structs. It does not allocate for each record; you supply a buffer and a callback.

## Building

The build uses CMake:

```sh
git clone https://github.com/example/fitreader.git
cd fitreader
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

This installs `libfitreader.a`, `libfitreader.so` and the `fitreader.h` header. To build and run the test suite:

```sh
cmake -B build -DFITREADER_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

If you would rather not use CMake, the library is two source files (`fit.c` and `fit_profile.c`) plus the header; add them to your project directly.

## Public API

Everything is declared in `fitreader.h`. The core types are:

```c
typedef struct fit_file fit_file_t;

typedef struct {
    uint16_t global_num;   /* FIT global message number, e.g. FIT_MSG_RECORD */
    uint32_t timestamp;    /* seconds since the FIT epoch, 0 if absent */
    int      field_count;
    fit_field_t fields[FIT_MAX_FIELDS];
} fit_message_t;

typedef struct {
    uint8_t  def_num;      /* field definition number within the message */
    uint8_t  base_type;    /* FIT_UINT8, FIT_SINT32, FIT_STRING, ... */
    bool     valid;        /* false when the field holds the invalid sentinel */
    union {
        int64_t     i;
        double      f;
        const char *s;
    } value;
} fit_field_t;
```

Opening and reading a file:

```c
fit_file_t *fit_open(const char *path, fit_error_t *err);
fit_file_t *fit_open_mem(const void *buf, size_t len, fit_error_t *err);
int         fit_next(fit_file_t *f, fit_message_t *out);   /* 1 = message, 0 = EOF, <0 = error */
void        fit_close(fit_file_t *f);
const char *fit_strerror(fit_error_t err);
```

`fit_next` fills `out` with the next data message in file order. Definition messages are consumed internally. Fields are already scaled and offset according to the profile, so a heart rate arrives as beats per minute and a latitude as degrees, not semicircles. Use `fit_field_get(&msg, FIT_FIELD_RECORD_HEART_RATE)` to look up a field by its profile definition number; it returns `NULL` if the field is not present in this message.

Writing:

```c
fit_writer_t *fit_writer_open(const char *path, fit_error_t *err);
int           fit_writer_put(fit_writer_t *w, const fit_message_t *msg);
int           fit_writer_close(fit_writer_t *w);   /* writes the CRC and header size */
```

The writer emits a definition message automatically the first time it sees a new message layout.

## Example: extracting a heart rate stream

```c
#include <stdio.h>
#include <fitreader.h>

int main(int argc, char **argv) {
    fit_error_t err;
    fit_file_t *f = fit_open(argv[1], &err);
    if (!f) {
        fprintf(stderr, "open failed: %s\n", fit_strerror(err));
        return 1;
    }

    fit_message_t msg;
    int rc;
    while ((rc = fit_next(f, &msg)) > 0) {
        if (msg.global_num != FIT_MSG_RECORD)
            continue;
        const fit_field_t *hr = fit_field_get(&msg, FIT_FIELD_RECORD_HEART_RATE);
        if (hr && hr->valid)
            printf("%u,%lld\n", msg.timestamp, (long long)hr->value.i);
    }
    if (rc < 0)
        fprintf(stderr, "read error: %s\n", fit_strerror(fit_last_error(f)));

    fit_close(f);
    return rc < 0;
}
```

Compile with `cc hr.c -lfitreader -o hr`. The output is one `timestamp,bpm` line per record message that carried a heart rate value.

## Supported message types

Fully decoded, with named field constants in the header:

- `file_id`, `file_creator`
- `record` (the per-second sample stream)
- `lap`, `session`, `activity`
- `event`
- `device_info`
- `hrv`
- `length` (pool swimming)

Passed through as raw fields but without named constants or profile scaling (you get the base type and the unscaled integer):

- `workout`, `workout_step`
- `course`, `course_point`
- `monitoring`, `monitoring_info`
- `segment_lap`
- Any manufacturer-specific message (global numbers 0xFF00 and above)

Not supported:

- Compressed timestamp headers are decoded, but chained FIT files (multiple FIT files concatenated in one `.fit`) stop at the first file's CRC.
- Developer data fields (`field_description` messages) are skipped. Their presence does not break decoding of the standard fields in the same message.
- Writing files that use the `accumulated` field variants is untested.

If you need a message type that is in the pass-through list, adding it is mostly a matter of extending the tables in `fit_profile.c`; pull requests welcome.

## License

MIT. The FIT protocol is a Garmin specification; this library is an independent implementation and is not affiliated with Garmin.
