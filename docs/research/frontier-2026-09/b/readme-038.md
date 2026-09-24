# csssweep

csssweep is a static analysis tool that finds CSS selectors which never match anything in your site's rendered HTML. Point it at a build output directory, and it crawls every HTML file, parses every stylesheet those files reference, and reports the rules that would never apply. It is meant for trimming dead styles before a redesign or a framework migration, when nobody remembers which of the 3,000 rules in `legacy.css` still do anything.

It works on static output only. It does not run JavaScript, which keeps it fast and predictable but also means it has some known blind spots, described below.

## Installation

```sh
npm install -g csssweep
```

Or run it without installing:

```sh
npx csssweep ./dist
```

Node 18 or later is required.

## Basic usage

Build your site, then run csssweep against the output:

```sh
csssweep ./dist
```

csssweep walks the directory recursively, collects every `.html` file, and resolves the stylesheets each one links via `<link rel="stylesheet">` and `<style>` blocks. Stylesheets referenced by absolute URL to another host are skipped unless you pass `--fetch-remote`. Every selector in every stylesheet is then tested against every document. A selector counts as used if it matches at least one element in at least one page.

Useful options:

```
--css <glob>       Also analyse stylesheets not linked from any page
--ignore <path>    Path to an ignore-list file (see below)
--format <fmt>     text (default), json, or csv
--min-pages <n>    Only report a selector as used if it matches on at least n pages
--fail-on-unused   Exit with status 1 if any unused selectors are found
```

## Dynamically added classes

Classes that only appear after JavaScript runs (an `.is-open` on a menu, a `.loaded` on an image, framework-generated state classes) will never appear in static HTML, so csssweep would report every rule using them as dead. To handle this, give it an ignore list.

Create a file, conventionally `.csssweepignore`, with one pattern per line:

```
# state classes toggled by JS
.is-open
.is-active
.has-error

# anything the carousel adds
.slick-*

# regex patterns are allowed between slashes
/^\.js-/
```

Then run:

```sh
csssweep ./dist --ignore .csssweepignore
```

A selector is excluded from the report if any of its compound parts matches an ignore pattern. So ignoring `.is-open` also silences `.menu.is-open > ul` and `.is-open .overlay`. Glob-style `*` matches within a class or id name; wrap a pattern in slashes for a full regular expression. csssweep looks for `.csssweepignore` in the current directory automatically if `--ignore` is not given.

## Report format

The default text report groups unused selectors by stylesheet:

```
dist/css/main.css
  142:3   .sidebar-legacy
  210:1   .btn--ghost:hover
  388:5   #promo-banner .close

dist/css/print.css
  12:1    .no-print

Scanned 214 HTML files, 3 stylesheets, 2,918 selectors.
Unused: 217 (7.4%)   Ignored: 41
```

Each line shows the line and column where the rule begins. With `--format json` you get an array of objects with `file`, `line`, `column`, `selector`, and `bytes` (the size of the rule body, handy for sorting by how much you would save). The `csv` format has the same columns and is convenient for pasting into a spreadsheet to review with a team.

## Known false positives

Because csssweep never executes JavaScript, some selectors it reports as unused are actually live:

- **Toggled classes not in the ignore list.** Anything added by `classList.add`, a framework's conditional class binding, or a CMS widget. The ignore list is the fix, but you have to know the class exists.
- **Pseudo-classes that depend on interaction.** `:hover`, `:focus-visible`, `:checked` and similar are evaluated structurally: `.btn:hover` is reported as used if `.btn` matches. But `:target` and `:focus-within` chains that depend on runtime state can be misjudged.
- **Content fetched after load.** Markup rendered from API responses, infinite scroll, or client-side routing is invisible to a static crawl. If most of your pages are client-rendered, csssweep will report far too much and you should pre-render or snapshot the DOM first.
- **Selectors used from JavaScript.** A rule matched by `querySelector('.hook')` for measurement rather than styling is still "dead" as CSS, but deleting it may not be what you want. Grep your scripts before removing anything that looks like a hook class.

Treat the report as a candidate list, not a deletion list.

## License

MIT.
