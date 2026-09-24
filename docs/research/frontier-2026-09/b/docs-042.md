# How Packrat Resolves Modules

When Packrat encounters an `import` or `require`, it has a string, called a *specifier*, and a file that contains it, called the *importer*. Resolution is the process of turning that pair into the absolute path of a file on disk. This page explains how Packrat does that, and where its behaviour can be customised.

Packrat follows Node's resolution conventions closely so that code written for Node and code published to npm works without changes. It layers a small number of bundler-specific features on top, chiefly aliasing.

## Three kinds of specifier

Packrat classifies every specifier before doing anything else:

- **Relative specifiers** start with `./` or `../`. They are resolved against the directory containing the importer. `./utils` from `/app/src/index.js` becomes `/app/src/utils`, which is then subject to extension and directory resolution described below.
- **Absolute specifiers** start with `/`. They are used as-is. These are rare in application code and usually appear only in generated files.
- **Bare specifiers** are everything else: `react`, `lodash/debounce`, `@scope/pkg`. They name a package rather than a location, and finding them is the interesting part of the algorithm.

Before classifying, Packrat checks the specifier against the alias table in the config. Aliasing is covered in its own section below, because it can rewrite a specifier of any kind into any other kind.

## Resolving bare specifiers

A bare specifier is split into a **package name** and an optional **subpath**. The package name is the first path segment, or the first two segments when it begins with `@`. So `lodash/debounce` has package `lodash` and subpath `./debounce`, and `@scope/pkg` has package `@scope/pkg` with no subpath.

Packrat then searches for a directory named after the package inside a `node_modules` directory. Once the package directory is found, Packrat determines which file inside it the specifier refers to:

1. If the package's `package.json` has an `exports` field, Packrat uses it and nothing else. The subpath (or `.` when there is none) is looked up in the exports map. Conditional exports are evaluated against Packrat's condition set, which by default is `["import", "module", "browser", "default"]` for a browser target and `["import", "module", "node", "default"]` for a Node target. A subpath not listed in `exports` is an error, even if the file exists.
2. If there is no `exports` field and the specifier has a subpath, the subpath is resolved as a file or directory relative to the package root.
3. If there is no `exports` field and no subpath, Packrat reads the entry-point fields in order: `browser` (for browser targets only), `module`, then `main`. The first one present is used. If none is present, `index.js` in the package root is tried.

Whatever file this produces then goes through the same extension and directory resolution as any other path.

## The node_modules search order

Packrat looks for `node_modules` directories by walking up from the importer's directory toward the filesystem root. For an importer at `/app/src/components/Button.js`, the candidate directories are checked in this order:

1. `/app/src/components/node_modules`
2. `/app/src/node_modules`
3. `/app/node_modules`
4. `/node_modules`

The first directory that contains the package wins; Packrat does not continue upward to look for a "better" match. This is what allows nested dependencies with conflicting versions: a package at `/app/node_modules/foo/node_modules/bar` shadows `/app/node_modules/bar` for imports from inside `foo`, and only for those.

Two config options adjust the search:

- `resolve.modules` adds extra directories to check before the ancestor walk begins. A common use is `["src", "node_modules"]`, which lets application code import `components/Button` without a relative path. Entries that are not absolute are resolved from the project root.
- `resolve.symlinks` (default `true`) controls whether a symlinked package is resolved to its real path before the walk continues. Turning this off keeps the walk rooted in the symlink's location, which matters for monorepos that link packages with `npm link` or workspaces.

## Extensions and directories

Once Packrat has a candidate path, it tries these in order:

1. The path exactly as given, if it names a file.
2. The path with each extension from `resolve.extensions` appended. The default list is `[".js", ".mjs", ".ts", ".tsx", ".jsx", ".json"]`.
3. If the path is a directory, its `package.json` entry fields as described above, then `index` plus each extension.

If none of these produce a file, resolution fails and the build reports an error pointing at the importer and the specifier.

## Aliasing in the config file

The `resolve.alias` table in `packrat.config.js` rewrites specifiers before the normal algorithm runs. It is the primary way to override default resolution.

```js
export default {
  resolve: {
    alias: {
      "@": "./src",
      "lodash": "lodash-es",
      "react-dom$": "preact/compat",
      "legacy-analytics": false,
    },
  },
};
```

Rules for how aliases apply:

- A key matches the whole specifier or a prefix that ends at a `/` boundary. `"@": "./src"` rewrites `@/utils/date` to `./src/utils/date`, but does not touch `@scope/pkg`.
- A key ending in `$` matches the exact specifier only. `"react-dom$"` rewrites `react-dom` but leaves `react-dom/client` alone.
- A relative target is resolved from the config file's directory, not the importer's. The result is treated as an absolute path.
- A bare target such as `lodash-es` re-enters the bare specifier algorithm, so it is found through `node_modules` as usual.
- A target of `false` resolves the specifier to an empty module. This is useful for stubbing out packages that should not appear in a browser bundle.

Aliases are checked in the order they are declared, and the first match is applied once; Packrat does not re-run the alias table on the rewritten result. Aliases apply to every importer, including files inside `node_modules`, unless the alias entry is scoped with an `only` glob.

## Putting it together

For a given import, Packrat: applies any alias, classifies the specifier, locates the package via the `node_modules` walk if it is bare, picks a file inside it using `exports` or entry fields, and finally applies extension and directory resolution. Every step is deterministic given the config and the filesystem, which is what lets Packrat cache resolution results between builds and invalidate them only when a `package.json` or the config changes.
