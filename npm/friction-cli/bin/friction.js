#!/usr/bin/env node
// Resolves the platform-specific package npm already installed and hands the
// process over to the real binary.
//
// There is deliberately no postinstall script and nothing is downloaded at
// install time. The binary arrives as an ordinary optional dependency, which
// means npm checks its integrity, the lockfile pins it, `--ignore-scripts`
// installs work, and so do offline and air-gapped installs. A postinstall
// downloader would break every one of those.
//
// npm picks the right platform package from the `os`/`cpu` fields on each one,
// skipping the rest. So exactly one should be present, and this file's only job
// is to find it and exec.

"use strict";

const { spawnSync } = require("node:child_process");

// Each supported platform's package name and the binary inside it. Linux x64
// deliberately maps to the statically linked musl build: it runs on glibc and
// musl hosts alike, so one package covers Alpine and Debian without npm needing
// to distinguish them (the `libc` manifest field is too new to rely on).
const PLATFORMS = {
  "darwin arm64": ["friction-cli-darwin-arm64", "friction"],
  "darwin x64": ["friction-cli-darwin-x64", "friction"],
  "linux x64": ["friction-cli-linux-x64", "friction"],
  "linux arm64": ["friction-cli-linux-arm64", "friction"],
  "win32 x64": ["friction-cli-win32-x64", "friction.exe"],
};

function resolveBinary() {
  const key = `${process.platform} ${process.arch}`;
  const entry = PLATFORMS[key];

  if (!entry) {
    const supported = Object.keys(PLATFORMS).join(", ");
    throw new Error(
      `friction has no prebuilt binary for ${key}.\n` +
        `Supported: ${supported}.\n` +
        `Build from source instead: cargo install --git https://github.com/ngriaznov/friction friction-cli`
    );
  }

  const [pkg, binary] = entry;
  try {
    return require.resolve(`${pkg}/bin/${binary}`);
  } catch {
    throw new Error(
      `friction is installed but its ${key} binary package (${pkg}) is missing.\n` +
        `That usually means the install ran with optional dependencies disabled.\n` +
        `Reinstall without --no-optional, or run: npm install ${pkg}`
    );
  }
}

let binaryPath;
try {
  binaryPath = resolveBinary();
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exit(1);
}

// `stdio: "inherit"` keeps friction's own contract intact: fixed text on
// stdout, the summary on stderr, and a pipe stays a pipe. Exit codes are
// forwarded unchanged, since callers gate on them.
const result = spawnSync(binaryPath, process.argv.slice(2), { stdio: "inherit" });

if (result.error) {
  process.stderr.write(`failed to run ${binaryPath}: ${result.error.message}\n`);
  process.exit(1);
}
// A signal-terminated child has a null status; report it the way a shell would.
process.exit(result.status === null ? 1 : result.status);
