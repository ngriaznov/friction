# mbox2maildir

A single-file Perl script that converts an mbox archive into a Maildir tree, preserving message flags and delivery dates where the mbox actually records them. Written for moving decade-old mail out of Thunderbird or Mutt and into a client that expects Maildir.

No dependencies beyond core Perl (5.10 or later).

## Usage

```
perl mbox2maildir.pl /path/to/archive.mbox /path/to/output/Maildir
```

The output path is created if it does not exist, along with the standard `cur/`, `new/`, and `tmp/` subdirectories. If the directory already exists and contains messages, the script appends rather than overwriting.

Options:

- `--dry-run` — parse and report counts without writing files
- `--verbose` — print one line per message
- `--from-mangle` — unescape `>From ` lines at the start of message bodies

## Flag mapping

Read/replied/flagged status is stored in the `Status` and `X-Status` headers that most mbox-writing clients emit. The script maps them onto Maildir filename suffixes:

| Source header | Maildir flag | Meaning |
|---|---|---|
| `Status: R` | `S` | Seen |
| `Status: O` | (none) | Old, delivered to `cur/` |
| `X-Status: A` | `R` | Replied |
| `X-Status: F` | `F` | Flagged |
| `X-Status: D` | `T` | Trashed |
| `X-Status: T` | `D` | Draft |

Messages with no `Status` header are treated as unseen and written to `new/`. Everything else lands in `cur/` with the flags appended after `:2,`.

Delivery dates come from the mbox `From ` separator line when present, falling back to the `Date:` header, and finally to the file's mtime. The chosen timestamp is applied to the output file with `utime`.

## Duplicate message IDs

Messages sharing a `Message-ID` are all written out — nothing is silently dropped. Filenames are unique by construction (timestamp, PID, and a counter), so duplicates coexist. Pass `--report-dupes` to get a summary at the end listing each repeated ID and how many copies were written, which is usually enough to decide whether to run a deduplicator afterwards.

## Back up first

Make a copy of the mbox before converting anything large. The script opens the source read-only and has no code path that writes to it, but a conversion that dies partway through leaves a half-populated Maildir, and recovering from that is far easier when the original is untouched. Verify message counts before deleting anything.
