```markdown
# multitail-go: A Multi-Log Tailer in Go

`multitail-go` is a command-line tool written in Go that allows you to tail multiple log files simultaneously.  It's designed for situations where you need to monitor several related logs at once, such as an application server and its database logs, or different microservices logging to distinct files. It prefixes each line with a color-coded label indicating the source file, making it easy to quickly distinguish between them.

## Installation

The easiest way to install `multitail-go` is using Go's built-in tooling:

```bash
go install github.com/yourusername/multitail-go@latest  # Replace yourusername with the actual repository owner
```

This will download the source code and compile it into a binary named `multitail-go` (or `multitail-go.exe` on Windows) in your `$GOPATH/bin` directory (or `$GOBIN` if set). Make sure that `$GOPATH/bin` is in your system's PATH environment variable so you can execute `multitail-go` from anywhere.

If you prefer a specific version, replace `@latest` with the desired tag or commit hash:

```bash
go install github.com/yourusername/multitail-go@v1.2.3
```

You can also download pre-built binaries for various platforms from the [releases page](https://github.com/yourusername/multitail-go/releases) (replace `yourusername` with your actual repository).  Extract the archive and place the binary in a directory included in your PATH.

## Basic Usage

The simplest usage involves providing a list of log file paths as arguments:

```bash
multitail-go app.log database.log error.log
```

This command will tail `app.log`, `database.log`, and `error.log` simultaneously, displaying each line prefixed with a color-coded label indicating its source file (e.g., "[APP] Log message", "[DB] Connection successful", "[ERR] Fatal error").  The colors are assigned sequentially to the files listed. See the "Color Assignment" section below for more details.

## Glob Pattern Support for Rotating Logs

`multitail-go` supports glob patterns, which are extremely useful when dealing with rotating log files (e.g., `access.log.1`, `access.log.2`).  You can use wildcards like `*` and `?`:

```bash
multitail-go access.log.* error.log.bak *.debug.log
```

This command will tail:

*   `access.log` (the current log file)
*   All files matching the pattern `access.log.*` (e.g., `access.log.1`, `access.log.2`, etc.)
*   `error.log.bak`
*   All files matching the pattern `*.debug.log`

**Important:**  The order in which globbed files are tailed is determined by your operating system's file sorting algorithm (typically lexicographical). If you need a specific ordering, consider explicitly listing the files.

## Filtering Lines with Regular Expressions

You can filter lines based on regular expressions using the `-f` or `--filter` flag followed by a regex pattern:

```bash
multitail-go -f "ERROR" app.log database.log
```

This command will only display lines from `app.log` and `database.log` that contain the string "ERROR".  The regular expression is applied to each line before it's displayed.

You can specify multiple filters:

```bash
multitail-go -f "ERROR" -f "WARN" app.log database.log
```

This will display lines containing either "ERROR" or "WARN".

**Note:**  The regular expression matching is performed using the Go `regexp` package, which supports standard POSIX extended regular expressions. Escaping special characters may be required depending on your regex pattern.

## Color Assignment

By default, `multitail-go` assigns colors to source files sequentially based on their order in the command line arguments.  The first file receives the first color available, the second file receives the next color, and so on. A set of predefined colors is used:

*   File 1: Red
*   File 2: Green
*   File 3: Yellow
*   File 4: Blue
*   File 5: Magenta
*   File 6: Cyan
*   ... (and repeats)

You can customize this behavior using the `--color` flag followed by a comma-separated list of color names. Valid color names are: `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`. For example:

```bash
multitail-go --color green,blue,red app.log database.log error.log
```

This would assign the colors as follows:

*   `app.log`: Green
*   `database.log`: Blue
*   `error.log`: Red

If you provide fewer color names than source files, the color assignment wraps around and repeats from the beginning of the list. For example if `--color green,blue` is provided with 3 source files:

*   File 1: Green
*   File 2: Blue
*   File 3: Green



## Advanced Options (Future Considerations)

* **Custom Prefixes:**  Allow users to specify custom prefixes instead of the default "[FILE]".
* **Timestamp Formatting:** Add options for customizing timestamp format.
* **Sorting by Time:**  Sort lines from multiple files based on their timestamps.
* **Highlighting:** Provide syntax highlighting capabilities for different log formats.


## Reporting Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/yourusername/multitail-go). Remember to replace `yourusername` with your actual repository name.

## Contributing

Contributions are welcome!  Feel free to fork the repository and submit pull requests.  Please follow the Go coding style guidelines.
```
