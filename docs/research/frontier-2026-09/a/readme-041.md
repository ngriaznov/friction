# changelog2rss

Turn your `CHANGELOG.md` into an RSS feed.

If your project keeps a changelog in the [Keep a Changelog](https://keepachangelog.com/) format, changelog2rss converts it into a valid RSS 2.0 feed, with one item per release. Users can subscribe to it and find out about new versions without watching the repository.

## Installation

```sh
pip install changelog2rss
```

Requires Python 3.9 or later. It installs a single `changelog2rss` command.

## Usage

```sh
changelog2rss CHANGELOG.md feed.xml
```

That's all you need in most cases. The command overwrites `feed.xml` every time it runs, so it fits well at the end of a release script or CI job that publishes the file together with your docs:

```yaml
# GitHub Actions example
- run: pip install changelog2rss
- run: changelog2rss CHANGELOG.md site/releases.xml --title "MyApp releases" --link https://myapp.dev
```

Use `-` as the output path to write to standard output.

## How the changelog is parsed

changelog2rss expects the structure Keep a Changelog describes:

```markdown
## [Unreleased]

## [1.4.0] - 2026-03-12
### Added
- Export to CSV.

### Fixed
- Crash when the config file is empty.

## [1.3.2] - 2026-02-01
...

[1.4.0]: https://github.com/you/myapp/compare/v1.3.2...v1.4.0
```

Each level-2 heading becomes one feed item:

- **Title.** The version number, taken from inside the brackets, becomes `1.4.0`. The optional `--title` value is added as a prefix, giving something like `MyApp 1.4.0`.
- **Date.** The `YYYY-MM-DD` date after the version becomes the item's `pubDate` at 00:00 UTC. Headings that have no date are skipped with a warning, because RSS readers sort by date.
- **Link.** If the file has a link reference definition for the version at the bottom, like the `[1.4.0]:` line above, that URL is used as the item's `<link>`. If it doesn't, the item links to the feed's site link.
- **Description.** Everything under the heading, up to the next version heading, including the `### Added`, `### Fixed` and other subsections, is rendered to HTML and placed in the item description.
- **GUID.** Built from the site link and version, so feed readers don't show the same release twice when the feed is regenerated.

The `[Unreleased]` section is always left out. Headings marked `[YANKED]` are included, with "(yanked)" added to the title.

## Setting the feed title and link

RSS requires a channel title and link. Set them with:

```sh
changelog2rss CHANGELOG.md feed.xml \
  --title "MyApp releases" \
  --link https://myapp.dev
```

If you leave them out, the title defaults to the changelog's top-level `# ` heading and the link defaults to an empty string. The feed will still be generated, but some validators will warn about the missing link. You can also set `--description` for the channel description and `--limit N` to include only the N most recent releases.

## License

MIT
