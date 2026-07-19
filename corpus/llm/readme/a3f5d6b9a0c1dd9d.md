**wavtag: A small C library for editing WAV metadata**

wavtag is a lightweight, portable library for reading and modifying metadata chunks in WAV audio files without altering the underlying audio data. It supports common tags like artist, title, and comment.

### Build Instructions

To build wavtag, follow these steps:

1. Clone this repository: `git clone https://github.com/your-repo/wavtag.git`
2. Navigate to the project directory: `cd wavtag`
3. Run the following command to compile the library using make:
```bash
make
```
This will create a shared library (`libwavtag.so`) in the `build` directory.

### Public API

The public API consists of three main functions for opening, reading, and writing metadata:

#### Opening a file
```c
// Open a WAV file for reading/writing tags
wavtag_file_t* wavtag_open(const char* filename);
```
This function returns a handle to the opened file.

#### Reading tags
```c
// Read metadata from the current position in the file
int wavtag_read_tags(wavtag_file_t* file, struct tag_info* tags);
```
`tags` is an array of `struct tag_info` structures, where each element represents a single tag. The function returns the number of tags read.

#### Writing modified tags back
```c
// Write metadata to the current position in the file
int wavtag_write_tags(wavtag_file_t* file, const struct tag_info* tags);
```
This function writes the provided `tags` array to the file.

### Byte-order handling

wavtag uses a portable byte-order management mechanism to ensure compatibility across platforms. When reading or writing metadata, the library automatically converts between host and WAV file byte orders (usually big-endian).

### Usage example
```c
#include <stdio.h>
#include <stdlib.h>
#include "wavtag.h"

int main() {
    // Open a WAV file for editing tags
    wavtag_file_t* file = wavtag_open("example.wav");

    // Read existing metadata
    struct tag_info tags[3];
    int num_tags = wavtag_read_tags(file, tags);

    // Modify a tag (e.g., change the title)
    tags[0].name = "TITLE";
    strcpy(tags[0].value, "New Title");

    // Write modified metadata back to the file
    int result = wavtag_write_tags(file, tags);
    if (result != num_tags) {
        printf("Error writing tags: %d\n", result);
        return 1;
    }

    // Close the file
    wavtag_close(file);

    return 0;
}
```
This example demonstrates how to read and modify metadata in a WAV file using wavtag.

Remember to link against the compiled library when building your application:
```bash
gcc -o your_app your_source.c build/libwavtag.so
```
