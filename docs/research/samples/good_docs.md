# Plugout

Plugout is a macOS app for cleaning up audio plugins. It scans the places plugins install themselves, groups the many duplicate formats into a single entry per plugin, and lets you remove the ones you no longer use along with the stray files their installers leave behind.

## Why it exists

Installing a plugin on a Mac rarely means installing one thing. A single vendor typically drops several builds of the same plugin — an AU for Logic, a VST3 for most modern hosts, often a VST2, a CLAP, and an AAX for Pro Tools — and each format goes into a different system directory. After a few years of trying demos, buying bundles, and updating what you already own, a plugin folder can hold several hundred bundles.

That clutter has real costs. Most DAWs rescan every installed plugin at launch, so a bloated folder means slower startup. And when you do want a plugin gone, vendor uninstallers are inconsistent: some miss files, many don't exist at all, and the leftover presets, preferences, and caches accumulate quietly in your Library.

## What it does

Plugout scans both user and system plugin locations and builds a single inventory, showing one row per plugin instead of one row per installed format. From there you can remove a plugin entirely, or remove individual formats while keeping the ones you still load — dropping an unused VST2, say, but keeping the AU.

Beyond the plugins themselves, Plugout finds the pieces installers scatter elsewhere: companion standalone apps in `/Applications`, and the support files tied to each installer. Removal is reversible by default. Items are moved to the Trash rather than deleted outright, and when a plugin spans several protected directories, Plugout asks for administrator access once rather than repeatedly.

## Features

- Streaming scan that shows results as they are found, without waiting for the full pass to finish
- Automatic merging of a vendor's naming variants and related products into one entry
- Optional usage tracking that reads REAPER, Ableton, Studio One, and Logic projects to show which plugins you actually open
- Semantic search backed by an on-device embedding model, so searching "reverb" finds relevant processors by function rather than by name
- Full keyboard navigation
- Inventory export to CSV or JSON
- Receipt-aware removal that uses macOS package receipts to delete only the files a given installer wrote

## Built with

The backend is written in Rust, the interface in React, and the two are packaged as a desktop app with Tauri. The design goal throughout is to handle the real, messy state of a working producer's plugin collection rather than an idealized one.
