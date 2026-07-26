# phodedupe

`phodedupe` is a command-line tool that scans one or more directories for duplicate photos using perceptual hashing, so resized, re-compressed, or slightly cropped copies of the same image are caught even when their bytes don't match at all.

## Why perceptual hashing instead of a checksum

A plain checksum (MD5, SHA-1, and the like) only tells you when two files are byte-identical. That's fine for finding exact copies, but it misses the much more common case: the same photo saved twice at different JPEG quality levels, exported at a different resolution, or re-uploaded by a service that strips metadata. All of those produce a completely different checksum despite being visually the same image. `phodedupe` instead computes a perceptual hash for each image — a compact fingerprint derived from the image's visual structure — and compares hashes by Hamming distance. Two images that look alike produce hashes that differ in only a few bits, even if their underlying files are nothing alike.

## Installation

```
pip install phodedupe
```

Requires Python 3.9+. Pillow is pulled in automatically for image decoding.

## Basic usage

Scan a folder and generate a report of duplicate groups:

```
phodedupe scan ~/Pictures
```

This walks the directory recursively, hashes every supported image (JPEG, PNG, HEIC, WebP), and groups files whose hashes fall within the configured similarity threshold. Nothing is deleted at this stage — `phodedupe` only ever reports.

Review the report before doing anything destructive:

```
phodedupe report --format html > duplicates.html
```

Open the file in a browser to see each duplicate group side by side with thumbnails, file paths, and sizes.

## Dry runs

Once you're ready to clean up a scanned library, `--dry-run` shows exactly what would be deleted without touching anything:

```
phodedupe clean ~/Pictures --dry-run
```

Drop the flag to actually remove the duplicates, keeping the largest file in each group by default.

## Configuration

`phodedupe` supports a few tunables, either as CLI flags or in a `phodedupe.toml` config file:

- `hash_algorithm`: `phash` (default), `dhash`, or `ahash`. pHash is the most robust to compression artifacts; dHash is faster on very large libraries.
- `threshold`: maximum Hamming distance for two images to be considered duplicates. Default is `8`; lower values require closer visual similarity.
- `min_size`: skip files below a given resolution, useful for ignoring thumbnails and icons picked up in a scan.

Example config:

```toml
hash_algorithm = "phash"
threshold = 6
min_size = "200x200"
```

## A note on burst shots

Perceptual hashing will sometimes group near-identical burst-mode shots together, since they can differ by only a few pixels. Review groups before deleting rather than trusting the tool blindly — that's exactly what the report and dry-run steps are for.
