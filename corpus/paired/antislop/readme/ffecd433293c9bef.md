```markdown
# multitail-go: Watch Multiple Logs Simultaneously

`multitail-go` is a command-line tool written in Go that allows you to concurrently watch multiple log files.  It displays lines from each file, prefixed with a color-coded label indicating the source filename. This is extremely useful for observing application behavior by monitoring both your application server and its dependent services (like databases) simultaneously – providing valuable context during debugging or performance analysis.

## Installation

The easiest way to install `multitail-go` is using Go's built-in package manager:

```bash
go install github.com/your-github-username/multitail-go@latest  # Replace your-github-username
```

This will download the source code and compile it into a binary named `multitail-go` (or similar, depending on your OS) in your `$GOPATH/bin` directory. Ensure that this directory is included in your system's PATH environment variable so you can execute `multitail-go` from anywhere.

If the installation fails or you prefer a different approach, you can download pre-built binaries for various platforms from the [Releases page](https://github.com/your-github-username/multitail-go/releases) (replace with your actual GitHub repository).  Extract the downloaded archive and place the `multitail-go` executable in a directory on your PATH.

## Basic Usage

The core functionality of `multitail-go` is to monitor multiple files concurrently. Here's a simple example:

```bash
multitail-go app.log database.log error.log
```

This command will display lines from `app.log`, `database.log`, and `error.log` in the terminal, each prefixed with a different color to identify its source file. The default colors are assigned sequentially (see "Color Assignment" below).

You can specify as many files as needed:

```bash
multitail-go server1.log server2.log worker.log audit.txt access.json
```

## Glob Pattern Support for Rotating Logs

Dealing with log rotation is common in production environments. `multitail-go` supports glob patterns to handle this gracefully, allowing you to monitor a series of rotated log files:

```bash
multitail-go application/*.log
```

This will watch all `.log` files within the `application/` directory.  The behavior with wildcards is standard shell expansion - typically performed by your shell before being passed to `multitail-go`.

More complex patterns are supported, including:

*   `*.gz`: Matches compressed log files.
*   `access_???.log`: Matches `access_001.log`, `access_002.log`, and so on.
*   `app_[0-9]*.log`: Matches `app_1.log`, `app_10.log`, etc.

The expansion happens *before* execution, meaning it relies on your shell's glob matching capabilities (bash, zsh, powershell, etc.).  Be mindful of this when crafting your patterns to accurately target the intended log files.

## Filtering Lines by Regular Expression

Sometimes you only want to see certain lines from a given log file. `multitail-go` provides filtering functionality using regular expressions. To filter, use the `-f`/`--filter` flag followed by a regex pattern:

```bash
multitail-go -f "ERROR" app.log
```

This will only display lines from `app.log` that contain the string "ERROR". The provided expression is treated as a regular expression (using Go's standard `regexp` package).

You can use different flags for more sophisticated regex matching:

```bash
multitail-go -f "(?i)critical error" database.log  # Case-insensitive critical errors
```

Multiple filters can be applied to each file, allowing very specific monitoring requirements.

## Color Assignment

`multitail-go` automatically assigns colors to different log files based on their order in the command line arguments. The first file is assigned one color, the second another, and so on.

The default color sequence follows a simple pattern:

1.  Red
2.  Green
3.  Yellow
4.  Blue
5.  Magenta
6.  Cyan
7.  White
8. ... (repeats)

If you provide more than 8 files, the colors cycle through this set.

You can customize the color scheme using the `--colors` flag, providing a comma-separated list of color names. The available colors are: `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, and `white`.  If the number of provided colors is less than the arguments passed to `multitail-go`, then it will cycle through the provided colors.

Example:

```bash
multitail-go --colors "red,green" app.log database.log error.log critical.log
```

In this case, `app.log` would be displayed in red and `database.log` in green.  The remaining files (`error.log` and `critical.log`) will reuse these colors (red for `error.log`, green for `critical.log`).

## Command Line Options

Here's a summary of the available command line options:

*   `-f <regex>` / `--filter <regex>`: Filter lines matching the given regular expression.
*   `--colors <color1,color2,...>`: Specify custom colors for files (limited to red, green, yellow, blue, magenta, cyan, and white).
*   `-h` / `--help`: Display this help message.
*   `-v` / `--version`: Print the version information.

## Contributing

Contributions are welcome! Please submit bug reports or feature requests through GitHub issues.  If you'd like to contribute code, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Write clear and concise tests.
4.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).  See the LICENSE file for details.
