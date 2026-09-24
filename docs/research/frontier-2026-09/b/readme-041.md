# changelog2rss

changelog2rss converts a Markdown changelog in the [Keep a Changelog](https://keepachangelog.com/) format into a valid RSS 2.0 feed. Point it at your `CHANGELOG.md`, publish the output alongside your docs, and users can subscribe to your release notes in any feed reader without watching a repository or joining a mailing list.

## Installation

```sh
pip install changelog2rss
```

Or, if you use pipx:

```sh
pipx install changelog2rss
```

Python 3.9 or later is required.

## Basic usage

```sh
changelog2rss CHANGELOG.md -o feed.xml
```

This parses `CHANGELOG.md` and writes `feed.xml`. If `-o` is omitted the feed is written to stdout, which is convenient in a pipeline:

```sh
changelog2rss CHANGELOG.md > public/changelog.xml
```

Run it as part of your docs or release build so the feed updates whenever the changelog does. The tool exits non-zero and prints a message if the file has no version headings it can recognise.

## How entries are parsed

Keep a Changelog files use a second-level heading per release:

```markdown
## [1.4.0] - 2026-03-18
### Added
- Export to CSV.
### Fixed
- Crash when the config file is empty.

## [1.3.2] - 2026-02-02
...
```

Each `## [version] - date` heading becomes one feed item:

- The version (the text inside the brackets) becomes the item **title**, prefixed with the feed title, for example `myproject 1.4.0`.
- The date must be ISO 8601 (`YYYY-MM-DD`) and becomes the item's **pubDate**. Headings without a date are given the file's modification time and a warning is printed.
- Everything between that heading and the next `##` heading is rendered from Markdown to HTML and becomes the item **description**, so the Added, Changed, Fixed and other subsections appear with their lists intact.
- If the changelog has a link reference for the version at the bottom of the file (`[1.4.0]: https://github.com/.../compare/v1.3.2...v1.4.0`) that URL becomes the item **link** and **guid**. Otherwise the link is the site URL with `#version` appended.

The `## [Unreleased]` section is skipped by default so subscribers are not notified of work in progress. Pass `--include-unreleased` to keep it.

Items are emitted newest first, matching the file order. Use `--limit 20` to cap the number of items in the feed; older releases are still in the changelog, but a feed with three hundred entries is unkind to readers.

## Setting the feed title and site link

RSS requires a channel title and link. By default the title is taken from the first `#` heading in the file (usually just "Changelog") and the link is empty, which validators complain about. Set them explicitly:

```sh
changelog2rss CHANGELOG.md -o feed.xml \
  --title "myproject releases" \
  --link https://myproject.example.com/
```

`--link` is also used as the base for per-item links when no compare URL is available. An optional `--description` sets the channel description; it defaults to "Release notes for <title>".

## Validation

The output is plain RSS 2.0 with the `atom:link rel="self"` element included when `--feed-url` is given, which is what most validators want to see. Run the result through the W3C feed validator once after setup to be sure.

## License

MIT.
