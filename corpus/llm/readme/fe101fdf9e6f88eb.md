# thesismake

An incremental Makefile build for LaTeX theses. Only the chapters you actually edited get recompiled, and the bibliography and table of contents are regenerated only when their inputs change — not on every run.

A full thesis build with BibTeX and two or three LaTeX passes takes a while. `thesismake` tracks the auxiliary files that determine whether those passes are really needed, so a typo fix in chapter 4 costs you a couple of seconds instead of a minute.

## Project layout

The Makefile expects this structure and discovers files by globbing, so it does not need a hand-maintained list:

```
thesis/
  Makefile          # copied from this repo, unmodified
  main.tex          # \documentclass, preamble, \include lines
  chapters/
    01-intro.tex
    02-background.tex
    03-method.tex
  references.bib
  figures/
    setup.pdf
    results.png
  build/            # created automatically; all output lands here
```

`main.tex` is the only file you write `\include` lines in. Everything under `chapters/` and `figures/` is picked up automatically as a dependency.

## Installation

Copy `Makefile` into your thesis directory. There is nothing else to install beyond a TeX distribution (TeX Live 2020 or newer) and GNU Make 4.0+. BSD make is not supported — the Makefile uses order-only prerequisites and `$(shell find ...)`.

## Targets

**`make build`** (also the default target) compiles `main.tex` into `build/main.pdf`. It runs BibTeX only if `references.bib` or the generated `.aux` citations changed, and repeats the LaTeX pass only while the `.aux` and `.toc` files are still changing — so cross-references and the table of contents converge without you guessing at pass counts.

**`make clean`** deletes the entire `build/` directory. Nothing in your source tree is touched.

**`make watch`** rebuilds on save. It uses `fswatch` on macOS and `inotifywait` on Linux; if neither is installed, it falls back to a 2-second polling loop and prints a note saying so.

## Adding a chapter

Create the file and add one line to `main.tex`:

```
\include{chapters/04-results}
```

That's it. The Makefile globs `chapters/*.tex`, so the new file becomes a dependency of the PDF on the next invocation. Prefix filenames with a zero-padded number if you care about the order they appear in `ls` — the actual document order comes from `main.tex`, not from the filenames.

Figures work the same way: drop a PDF or PNG into `figures/` and reference it. No Makefile edit required.

## Notes

If a build seems stuck on stale cross-references, run `make clean build`. This is almost always a sign that a `.aux` file was left inconsistent by an interrupted run.
