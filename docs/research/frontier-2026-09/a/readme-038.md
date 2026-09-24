# csssweep

Find the CSS selectors your site never uses.

csssweep crawls a site's built HTML and the stylesheets it links to, and then reports every selector that doesn't match a single element on any page. It's meant for cleaning up before a redesign, when you want to know which styles are safe to delete and which still matter.

It runs on static output only. It doesn't start a browser or execute JavaScript, which keeps it fast but also means it can't see classes that are added at runtime (see [Known false positives](#known-false-positives)).

## Installation

```sh
npm install --save-dev csssweep
```

Or run it once without installing:

```sh
npx csssweep ./dist
```

Requires Node.js 18 or later.

## Usage

Point csssweep at your build output directory:

```sh
csssweep ./dist
```

It finds every `.html` file under that directory, collects the stylesheets linked from those pages as well as any inline `<style>` blocks, and tests each selector against every page's DOM. A selector counts as used if it matches at least one element on at least one page.

Useful flags:

```sh
csssweep ./dist --css "assets/**/*.css"   # only check these stylesheets
csssweep ./dist --format json > report.json
csssweep ./dist --config csssweep.config.json
csssweep ./dist --fail-on-unused          # exit 1 if anything is unused (for CI)
```

Pseudo-classes and pseudo-elements such as `:hover`, `:focus-visible` and `::before` are removed before matching, so `.btn:hover` counts as used whenever `.btn` is. At-rules like `@media` and `@supports` are entered, and each selector inside them is checked as usual.

## Ignoring dynamic classes

Some classes only exist once JavaScript runs, for example `is-open`, `has-error`, or whatever your component library adds. csssweep can't see those, so you tell it about them in a config file:

```json
{
  "ignore": [
    "is-open",
    "has-error",
    "/^js-/",
    "/^swiper-/",
    "#modal-root"
  ],
  "ignoreFiles": ["vendor/**/*.css"]
}
```

Plain entries are matched as exact class names or IDs. Entries wrapped in slashes are treated as regular expressions and tested against each class or ID in the selector. A selector is left out of the report if any part of it matches an ignore entry. `ignoreFiles` skips whole stylesheets, which is handy for third-party CSS you don't control.

## Report format

The default output is grouped by stylesheet and includes line numbers:

```
assets/main.css
  L42   .card--featured
  L118  .sidebar .promo-banner
  L203  #legacy-header nav > ul

assets/forms.css
  L17   .input-group--inline

4 unused selectors in 2 files (of 612 checked, 18 ignored)
```

With `--format json`, each result gives `file`, `line`, `selector`, and `rule`, the full rule text. That makes it straightforward to feed into other scripts or to diff between runs.

## Known false positives

Treat the report as a list of candidates, not a list of things to delete blindly. The most common false positives are:

- **JavaScript-toggled classes.** Classes added by `classList.add()`, framework bindings (`:class`, `className={...}`), or state libraries don't appear in static HTML. If they aren't in your ignore list, they'll be reported.
- **Client-rendered content.** Markup that exists only after hydration, such as modals, dropdown menus and toasts, is invisible to csssweep, and so are the styles for it.
- **Classes built from strings.** Code like `` `btn-${variant}` `` can't be detected at all. Add a regex such as `/^btn-/` to cover them.
- **Pages that aren't in the build.** Error pages, emails, or routes rendered only on the server at request time won't be crawled unless their HTML is in the directory.

Before you delete anything, search your source for the class name.

## License

MIT
