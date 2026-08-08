# lumen

lumen is a static site generator built specifically for photography
portfolios. It doesn't try to be a general-purpose site generator with a
photography theme bolted on — the folder layout, build pipeline, and
templates all assume that what you're publishing is a set of image
galleries with a bit of surrounding text.

## Installation

```
gem install lumen
```

## Folder convention

A lumen site is a directory with one subfolder per gallery, plus a handful
of loose markdown files for standalone pages:

```
site/
  lumen.yml
  index.md
  about.md
  galleries/
    iceland-2019/
      images/
        001.jpg
        002.jpg
        003.jpg
      iceland-2019.md
    portraits/
      images/
        ...
      portraits.md
```

Each gallery is an `images/` directory plus a single markdown file in the
gallery folder that shares the gallery's name. That markdown file holds the
gallery's body text — an introduction, notes on the shoot, whatever you
want — and lumen renders it above the image grid. There's no front matter
block to fill in: the gallery's title comes from the folder name (lumen
title-cases it, so `iceland-2019` becomes "Iceland 2019"), and everything
else is inferred from the images themselves or set globally in `lumen.yml`.

Standalone pages like `index.md` and `about.md` at the site root work the
same way, minus the images directory — just a markdown file, no front
matter.

## Building the site

```
lumen build
```

This reads `lumen.yml`, walks the `galleries/` directory, and writes the
finished site to the configured output directory (`_site` by default).
Each build regenerates thumbnails only for images that are new or have
changed since the last build, so incremental builds on a large portfolio
are fast.

## Local preview

```
lumen serve
```

starts a local server on `http://localhost:4000` and rebuilds
automatically when a file under `galleries/` or a template changes. Useful
for checking gallery ordering and cropping before publishing.

## Thumbnails and EXIF captions

lumen generates two derivative sizes per image at build time — a grid
thumbnail and a lightbox-sized version — using whatever image processing
backend is available on the system (ImageMagick or libvips, auto-detected).
Neither original files nor derivatives are checked into your gallery
folders; both live under the output directory.

If an image has EXIF data, lumen will pull the capture date and, if
present, the `ImageDescription` or `UserComment` field to use as a caption
under the thumbnail. You can override any auto-extracted caption for a
specific image by adding a line to the gallery's markdown file:

```
caption[003.jpg]: Reynisfjara black sand beach, just after sunrise
```

Images without EXIF data or an explicit caption are shown without one.

## Configuration

Everything site-wide lives in a single `lumen.yml` at the site root:

```yaml
title: Jordan Reyes Photography
theme: minimal
output_dir: _site
galleries_dir: galleries
thumbnail_width: 480
lightbox_width: 1600
```

`theme` selects one of lumen's bundled themes (`minimal`, `grid`, or
`darkroom`); themes can be overridden on a per-file basis by dropping a
same-named template into a `_templates/` directory at the site root.

## License

MIT
