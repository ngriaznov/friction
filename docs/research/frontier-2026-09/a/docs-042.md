# How Packrat Resolves Modules

Every time Packrat finds an `import` or `require` in your code, it has to turn the string you wrote into a specific file on disk. This process is called **module resolution**. Most of the time it just works, but knowing the steps helps when an import resolves to the wrong file, a package is found twice, or an alias doesn't take effect.

This page explains the algorithm Packrat follows, in the order it applies each step.

## Kinds of specifiers

Packrat sorts every import specifier into one of three kinds before resolving it:

| Kind | Examples | Resolved against |
|---|---|---|
| **Relative** | `./utils`, `../lib/math.js` | The directory of the importing file |
| **Absolute** | `/src/config.js` | The filesystem root, or `resolve.root` if it's set |
| **Bare** | `react`, `lodash/merge`, `@acme/ui/button` | Aliases, then `node_modules` directories |

Relative and absolute specifiers point directly to a location. Bare specifiers name a *package* and require a search. Most of the resolution logic exists to handle them.

## Step 1: Apply aliases

Aliases are checked **first, before any filesystem lookup**, and they apply to every kind of specifier. This is what lets an alias override Packrat's default behavior.

Aliases are defined in `packrat.config.js`:

```js
export default {
  resolve: {
    alias: {
      '@': './src',
      'lodash': 'lodash-es',
      'react$': './vendor/react-shim.js',
    },
  },
};
```

Packrat matches aliases like this:

- **Prefix match (the default).** The key `@` matches `@` and `@/anything`. The matched prefix is replaced, so `@/components/Nav` becomes `./src/components/Nav`.
- **Exact match.** A key ending in `$` matches only that exact specifier. `react$` rewrites `import 'react'`, but it doesn't rewrite `import 'react/jsx-runtime'`.
- **Longest key wins.** When more than one key matches, the longest one is used. That way `@acme/ui` takes precedence over `@acme`.

After rewriting, Packrat classifies the new specifier again and resumes resolution. In the example above, `./src/components/Nav` is now a relative specifier, resolved against the project root where the config file lives. `lodash-es` is still bare, so it moves on to the `node_modules` search.

An alias can also map to `false`. This resolves the import to an empty module, which is useful for leaving Node-only dependencies out of a browser bundle.

Aliases are applied once. Packrat doesn't reapply aliases to the result of an alias, so two aliases that point to each other can't cause an infinite loop.

## Step 2: Split the bare specifier

A bare specifier has two parts: a **package name** and an optional **subpath**.

- `lodash/merge` → package `lodash`, subpath `./merge`
- `@acme/ui/button` → package `@acme/ui`, subpath `./button`. A scoped package name always includes the part after the slash.
- `react` → package `react`, no subpath

## Step 3: Search `node_modules` directories

Packrat looks for the package by walking **up** the directory tree from the importing file. In each directory, it checks for `node_modules/<package-name>`.

For an import in `/project/packages/web/src/app.js`, Packrat checks these locations in order:

1. `/project/packages/web/src/node_modules/react`
2. `/project/packages/web/node_modules/react`
3. `/project/packages/node_modules/react`
4. `/project/node_modules/react`
5. `/node_modules/react`

Packrat uses **the first directory that exists** and stops searching.

Next, it checks any directories listed in `resolve.modules`, in the order they're listed:

```js
resolve: {
  modules: ['./shared/vendor'],
}
```

Directories in `resolve.modules` are checked *after* the upward walk, so they act as a fallback. If you need a directory to take priority over `node_modules`, use an alias instead.

This nearest-first order is why a monorepo can have two copies of the same package. If `packages/web/node_modules/react` and `/project/node_modules/react` both exist, files under `packages/web` get the first copy, and all other files get the second. To force a single copy, alias the package to one path:

```js
alias: { 'react': path.resolve('./node_modules/react') }
```

An alias that resolves to a path skips the `node_modules` search entirely.

## Step 4: Resolve inside the package

When Packrat finds the package directory, it reads the package's `package.json` to find the entry file.

1. **`exports` field.** If `exports` is present, it's authoritative. Packrat matches the subpath (or `.` when there's no subpath) against the `exports` map and chooses a target using the active **conditions**. By default these are `import`, `browser`, and `default` for browser builds, and `import`, `node`, and `default` for Node builds. If a subpath isn't listed in `exports`, resolution fails, even if the file exists on disk. This is intentional: a package's `exports` defines its public API.
2. **Legacy fields.** When there's no `exports`, Packrat checks the fields in `resolve.mainFields` for the package root. The default order is `browser`, `module`, `main`. For a subpath, Packrat joins it to the package directory and continues to step 5.

## Step 5: Resolve the file

Whether the path came from a relative specifier, an alias, or a package entry, Packrat turns it into a real file in this order:

1. The exact path, if it's a file.
2. The path with each extension in `resolve.extensions` appended, in order. The default list is `.tsx`, `.ts`, `.jsx`, `.js`, `.mjs`, `.json`.
3. If the path is a directory, its `package.json` `main` field, and then `index` with each extension in the list.

The first match wins. If nothing matches, Packrat reports an error that lists every location it tried.

## Debugging resolution

To see exactly how a specifier resolves, run:

```sh
packrat resolve react --from src/app.js
```

The output lists each step, including which alias matched, every `node_modules` directory checked, which `exports` condition was chosen, and the final file path.
