# Plugout

Plugout cleans up audio plugins on macOS. It finds every stray copy and clears out what you stopped using.

## The problem

Install one plugin and the vendor quietly drops several copies of it onto your disk.

You get an AU.

Then a VST3. Sometimes a VST2 shows up, and a CLAP, and even an AAX, each in a folder you never open.

Each hides somewhere else.

Do that for a decade and you are suddenly facing many hundreds of stray plugin bundles.

It adds up.

Your DAW rescans everything at launch. Startup crawls.

Removal is worse.

The vendor's uninstaller skips half the files it dropped, or never shipped at all, so the leftovers just rot in your Library.

## What it does

It scans everywhere.

One row per plugin. Not one per file.

Whole plugin gone?

One click.

Prefer to drop the aging VST2 you never touch while keeping the AU build that still loads in every session you own?

That works.

It hunts down the companion apps that ride along too, like the standalone build a vendor quietly tucks into your Applications folder.

Nothing dies on the spot. Everything you delete moves to the Trash, where it waits, fully recoverable, until the day you finally decide you are ready to let it go.

Undo is easy.

## Features

- Streams results instantly
- Merges naming variants
- Reads your saved sessions to reveal which plugins you actually reach for across a project
- Finds processors by meaning, so a search for "reverb" surfaces the right ones
- Driven fully by the keyboard
- Exports to CSV or JSON
- Removes only what a plugin's own installer receipt says it wrote to your disk

## Built with

Rust powers the backend.

React runs the interface.

Tauri wraps it into one small app. The point is to tame the chaos a producer's plugin folder becomes after years of downloads and trials.
