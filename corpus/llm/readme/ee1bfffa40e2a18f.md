# packlite

packlite is a command-line tool that bundles and minifies a project's JavaScript and CSS into a small number of output files. It's built for simple static sites and small multi-page projects — not single-page apps — where you want fast, predictable builds without pulling in a full framework-oriented bundler.

## Why packlite

Heavier bundlers do a lot: code splitting, tree shaking across complex module graphs, plugin ecosystems for every possible asset type. Most of that is overkill for a marketing site, a documentation site, or a handful of static pages with a shared script and stylesheet. packlite skips the parts you don't need and focuses on the two things that actually matter for that kind of project — combining files and shrinking them — so builds finish in a fraction of the time.

## Installation

```
npm install --save-dev packlite
```

A global install also works if you prefer running it outside a project:

```
npm install -g packlite
```

## Configuration

packlite reads a `packlite.config.js` file from your project root. A minimal config lists your entry points:

```js
module.exports = {
  entries: {
    main: ['src/js/app.js', 'src/js/analytics.js'],
    styles: ['src/css/base.css', 'src/css/layout.css'],
  },
  outDir: 'dist',
};
```

Each key in `entries` becomes one output bundle — `dist/main.js` and `dist/styles.css` in the example above. Files within an entry are concatenated in the order listed, then minified.

## Usage

Run a one-off build:

```
npx packlite build
```

For local development, use watch mode to rebuild automatically as files change:

```
npx packlite watch
```

Watch mode rebuilds only the entry whose source files changed, so iteration stays fast even as the project grows.

## Source Maps

Pass `--sourcemaps` to emit a `.map` file alongside each bundle, useful for debugging minified output in the browser:

```
npx packlite build --sourcemaps
```

Source maps are omitted by default to keep production output lean.

## Build Times

On a mid-sized static site (around 40 JS files and 15 CSS files totaling roughly 600KB unminified), packlite's build completes in under a second on a typical laptop. Doing the equivalent by hand — concatenating files with a shell script and running them through a separate minifier binary — takes noticeably longer in practice, mostly due to process startup overhead from invoking a minifier once per file rather than once per bundle. Compared to configuring a general-purpose bundler like webpack or esbuild for the same output, packlite's cold-start time is lower because it isn't resolving a module dependency graph — it simply processes the file lists you give it, in order.

## When Not to Use packlite

If your project needs code splitting, dynamic imports, JSX or TypeScript compilation, or CSS preprocessing, reach for a full bundler instead. packlite intentionally does not handle any of that — it assumes your files are already plain, valid JS and CSS ready to be combined and shrunk.

## License

MIT
