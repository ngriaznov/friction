# gedcom2svg

Make ancestor charts from GEDCOM files.

gedcom2svg reads a GEDCOM file exported from your family tree software, such as Gramps, Family Tree Maker, RootsMagic, MacFamilyTree or an online service, and draws an ancestor (pedigree) chart for one person as an SVG image. The chart is laid out one generation at a time: the starting person, then their parents, then grandparents, and so on.

SVG scales to any size without losing sharpness. It prints cleanly on a large-format plotter, opens in any browser, and can be edited in Inkscape or Illustrator if you want to add decoration afterwards.

## Installation

gedcom2svg needs Python 3.10 or later.

```sh
pip install gedcom2svg
```

Or with pipx, which keeps it in its own environment:

```sh
pipx install gedcom2svg
```

Check the installation:

```sh
gedcom2svg --version
```

It has no dependencies outside Python. Text is measured with a bundled copy of the font metrics, so you don't need a graphics library.

## Basic usage

Every chart needs a GEDCOM file and a starting person:

```sh
gedcom2svg family.ged --start @I42@ -o chart.svg
```

### Finding the starting individual's ID

Each person in a GEDCOM file has a cross-reference ID, such as `@I42@` or `@P1023@`. Your family tree software sets it when exporting, and it isn't the same as the person's name. To look it up, use the `list` subcommand:

```sh
gedcom2svg list family.ged --search "Kowalski"
```

```
@I42@    Jan Kowalski        b. 1911  d. 1987
@I43@    Maria Kowalski      b. 1915  d. 2002
@I118@   Stanisław Kowalski  b. 1880  d. 1944
```

`--search` does a case-insensitive match against given names and surnames, and ignores diacritics, so `kowalski` and `Kowalskí` find the same people. If you leave out `--search`, every individual in the file is listed.

You can write the ID with or without the surrounding `@` characters: `--start I42` works just as well.

## Layout options

### Number of generations

```sh
gedcom2svg family.ged --start @I42@ --generations 6 -o chart.svg
```

`--generations` (short form `-g`) sets how many generations to draw, counting the starting person as generation 1. The default is 5. Every extra generation doubles the number of boxes in the outermost column, so the chart grows quickly:

| Generations | Boxes in last generation | Total boxes |
|---|---|---|
| 4 | 8 | 15 |
| 5 | 16 | 31 |
| 6 | 32 | 63 |
| 8 | 128 | 255 |

When an ancestor is unknown, their box is left out, and so is everything behind it, so charts for families with gaps come out smaller. Use `--show-unknown` to draw empty placeholder boxes instead. This helps when you print a chart to fill in by hand.

### Orientation

```sh
gedcom2svg family.ged --start @I42@ --orientation horizontal -o chart.svg
```

| Value | Layout |
|---|---|
| `horizontal` (default) | The starting person is on the left and ancestors extend to the right. Fathers are above mothers. |
| `vertical` | The starting person is at the bottom and ancestors extend upward. Fathers are to the left of mothers. |
| `fan` | A semicircular fan chart. Each generation is a ring, split into equal wedges. |

Horizontal suits wide paper or screens. Vertical is better for tall posters. Fan charts are compact and look good framed, but names in the outer rings are rotated and set smaller, so for anything beyond six generations horizontal is usually easier to read.

### Other display options

- `--fields name,dates,places` controls what goes in each box. Available fields are `name`, `dates`, `places`, `id` and `occupation`. The default is `name,dates`.
- `--date-format` accepts `year` (default), `full` or `iso`.
- `--box-width` and `--font-size` fine-tune the appearance, in SVG user units.
- `--color-by` accepts `none`, `generation` or `line`. With `line`, the paternal and maternal lines are shaded differently.
- `--title "The Kowalski Family"` adds a heading above the chart.

## Non-ASCII names

Genealogy data is full of names like Łukasz, Søren, Zoë, Nguyễn, Ødegård, or names written in Cyrillic, Greek or Hebrew. gedcom2svg handles them as follows:

- **Encoding detection.** The `CHAR` tag in the GEDCOM header is read and the file is decoded to match. Supported encodings are `UTF-8`, `UNICODE` (UTF-16), `ASCII`, and `ANSEL`, the legacy genealogy encoding that many older programs still export. ANSEL's combining diacritics are converted to precomposed Unicode characters, so `e` followed by a combining acute accent becomes `é`.
- **Mislabelled files.** Some programs write `CHAR ANSI` or `CHAR ASCII` but then fill the file with Windows-1252 or UTF-8 bytes. If a file doesn't decode cleanly with the declared encoding, gedcom2svg tries UTF-8 and then Windows-1252, and warns you about which one it used. Use `--encoding` to force a specific encoding.
- **Output.** The SVG is always written as UTF-8, and characters are kept as they are, not escaped.
- **Fonts.** The default font stack is `"Noto Sans", "DejaVu Sans", sans-serif`, which covers most Latin, Cyrillic and Greek text. For other scripts, set `--font` to a font that covers them, for example `--font "Noto Sans Hebrew"`. The font has to be installed on whatever machine opens the SVG. To make a chart look the same everywhere, use `--embed-font path/to/font.ttf`, which embeds a subset of the font in the file.
- **Right-to-left names.** Names in Hebrew and Arabic are marked with `direction="rtl"`, so browsers render them correctly. Boxes that mix scripts are handled by the viewer's bidi algorithm.

## GEDCOM version compatibility

| Version | Status |
|---|---|
| GEDCOM 5.5.1 | Fully supported. This is what most software exports. |
| GEDCOM 5.5 | Fully supported. |
| GEDCOM 5.5.5 | Supported. |
| GEDCOM 7.0 | Partially supported. Individuals, families, names and dates are read correctly. `SNOTE`, multimedia and extension schemas are ignored. |
| GEDCOM 4.x and older | Not supported. Re-export from your software as 5.5.1. |

Some things to be aware of:

- Only `INDI` and `FAM` records and their `NAME`, `SEX`, `BIRT`, `DEAT`, `FAMC` and `FAMS` tags are needed to build the chart. Unrecognised and vendor-specific tags (those starting with `_`) are skipped quietly.
- If a person has several `FAMC` records, for example birth and adoptive parents, the family marked `PEDI birth` is used. If there is no `PEDI` tag, the first one is used. `--prefer-pedigree adopted` switches this behaviour.
- Date phrases such as `ABT 1850`, `BEF 1900` or `BET 1820 AND 1825` are displayed in abbreviated form (`c. 1850`, `< 1900`, `1820–1825`).
- Files from some online services use non-standard structures. If a chart comes out wrong, run the file through `gedcom2svg validate family.ged`. It lists structural problems such as broken cross-references and parents missing from families.

## License

GPL-3.0-or-later. See `LICENSE`.
