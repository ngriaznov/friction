// Generates the per-platform npm packages from the release archives, and
// stamps every manifest with the version CI resolved from Cargo.toml.
//
// The platform packages are generated rather than committed: each is a
// package.json plus one binary, and the only thing that varies is the
// os/cpu pair and which archive the binary came from. Committing five
// near-identical manifests would mean five places to forget to bump.
//
//   node npm/build.mjs <version> <dist-dir> <out-dir>
//
// `dist-dir` holds the extracted release archives, one directory per target.

import { chmodSync, cpSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [version, distDir, outDir] = process.argv.slice(2);
if (!version || !distDir || !outDir) {
  console.error("usage: node npm/build.mjs <version> <dist-dir> <out-dir>");
  process.exit(1);
}

// npm's `os`/`cpu` fields are what make this work: npm installs only the
// package matching the host and silently skips the others, so the wrapper's
// five optional dependencies resolve to exactly one download.
//
// Linux x64 takes the statically linked musl build deliberately — it runs on
// glibc and musl hosts alike, so Alpine and Debian are one package rather than
// two, and npm never has to tell them apart.
const PLATFORMS = [
  { pkg: "@mhriaznov/friction-darwin-arm64", os: "darwin", cpu: "arm64", target: "aarch64-apple-darwin", bin: "friction" },
  { pkg: "@mhriaznov/friction-darwin-x64", os: "darwin", cpu: "x64", target: "x86_64-apple-darwin", bin: "friction" },
  { pkg: "@mhriaznov/friction-linux-x64", os: "linux", cpu: "x64", target: "x86_64-unknown-linux-musl", bin: "friction" },
  { pkg: "@mhriaznov/friction-linux-arm64", os: "linux", cpu: "arm64", target: "aarch64-unknown-linux-gnu", bin: "friction" },
  { pkg: "@mhriaznov/friction-win32-x64", os: "win32", cpu: "x64", target: "x86_64-pc-windows-msvc", bin: "friction.exe" },
];

const REPO = "https://github.com/ngriaznov/friction";

for (const { pkg, os, cpu, target, bin } of PLATFORMS) {
  const root = join(outDir, pkg);
  mkdirSync(join(root, "bin"), { recursive: true });

  const source = join(distDir, `friction-${version}-${target}`, bin);
  const dest = join(root, "bin", bin);
  cpSync(source, dest);
  // npm preserves the mode it finds, and a binary that arrives non-executable
  // fails at spawn time with a message that does not mention permissions.
  chmodSync(dest, 0o755);

  writeFileSync(
    join(root, "package.json"),
    `${JSON.stringify(
      {
        name: pkg,
        version,
        description: `The ${os}-${cpu} binary for friction. Installed automatically as an optional dependency of friction-cli; not meant to be depended on directly.`,
        license: "MIT OR Apache-2.0",
        repository: { type: "git", url: `git+${REPO}.git` },
        homepage: `${REPO}#readme`,
        os: [os],
        cpu: [cpu],
        files: [`bin/${bin}`],
        preferUnplugged: true,
      },
      null,
      2
    )}\n`
  );

  console.log(`built ${pkg} (${os}/${cpu}) from ${target}`);
}

// Stamp the wrapper and its optional dependencies with the same version, so a
// release can never ship a wrapper pointing at a version of itself that does
// not exist.
const wrapperDir = join(outDir, "friction-cli");
mkdirSync(wrapperDir, { recursive: true });
cpSync(join("npm", "friction-cli"), wrapperDir, { recursive: true });

const manifestPath = join(wrapperDir, "package.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
manifest.version = version;
manifest.optionalDependencies = Object.fromEntries(
  PLATFORMS.map(({ pkg }) => [pkg, version])
);
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
cpSync("README.md", join(wrapperDir, "README.md"));

console.log(`built friction-cli ${version} with ${PLATFORMS.length} optional dependencies`);
