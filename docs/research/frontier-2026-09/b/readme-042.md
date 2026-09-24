# gedcom2svg

gedcom2svg reads a GEDCOM file exported from family tree software and draws an ancestor chart as an SVG image. Pick a starting person, and the tool lays out their parents, grandparents and so on, one generation per column or row, with names, life dates and connecting lines. The result is a single scalable file you can print at any size, open in a browser, or drop into Inkscape for further editing.

It is a command-line tool with no GUI and no database. It reads one file, writes one file, and is happy running in a script.

## Installation

```sh
pip install gedcom2svg
```

Python 3.9 or later is required. There are no compiled dependencies; the GEDCOM parser and the SVG writer are both pure Python.

To check the install:

```sh
gedcom2svg --version
```

## Basic usage

Every person in a GEDCOM file has an identifier, usually of the form `@I123@`. You need the ID of the person the chart should start from. To find it, list the individuals in the file:

```sh
gedcom2svg family.ged --list
```

```
@I1@    Margaret Anne HOLLOWAY    1902-1987
@I2@    Thomas HOLLOWAY           1871-1940
@I3@    Eliza JENKINS             1875-1951
...
```

`--list` accepts a `--search` filter so you can narrow it down by name:

```sh
gedcom2svg family.ged --list --search holloway
```

Then render the chart:

```sh
gedcom2svg family.ged --root @I1@ -o holloway.svg
```

The `@` signs are optional; `--root I1` works too. If `-o` is omitted, the output file is named after the root person, so the command above would write `Margaret_Anne_HOLLOWAY.svg`.

Each box in the chart shows the person's name on the first line and birth and death years on the second (`1902-1987`, or `b. 1902` when the death is unknown, or `1902-` for someone the file marks as living). Pass `--show-places` to include birth and death places on a third line, and `--show-ids` to print the GEDCOM ID in small text in the corner of each box, which helps when you are cross-referencing back to the source file.

## Layout options

### Number of generations

By default the chart shows five generations: the root person, parents, grandparents, great-grandparents and great-great-grandparents. Change it with `--generations`:

```sh
gedcom2svg family.ged --root @I1@ --generations 7 -o wide.svg
```

An ancestor chart doubles in width at every generation, so seven generations means up to 64 boxes in the outermost column and a wide image. Above eight or so the boxes get small enough that you will want to print on A2 or split the chart. Where an ancestor is not recorded in the file, the branch simply stops and the space is left empty rather than collapsed, so the positions stay predictable across charts of the same family.

### Orientation

`--orientation` takes one of four values:

- `right` (default): the root person is on the left and ancestors extend to the right, one generation per column. This is the classic pedigree layout.
- `left`: mirrored, with the root on the right.
- `up`: the root is at the bottom and generations stack upward, like a tree.
- `down`: the root is at the top and generations descend below it.

Horizontal layouts read more naturally for most people and fit long names better; vertical layouts suit portrait paper and taller images for a web page.

### Sizing and style

- `--box-width` and `--box-height` set the size of each person box in pixels (default 180 by 48).
- `--gap` sets the spacing between generations.
- `--font` sets the font family used in the SVG text elements. It defaults to `sans-serif`; the file does not embed the font, so the viewer needs to have it.
- `--css path.css` injects a stylesheet into the SVG. Boxes carry classes `person`, `male`, `female`, `unknown` and `root`, and lines carry `link`, so you can restyle without touching the tool.
- `--no-dates` omits the date line.

## Names with non-ASCII characters

Genealogy data is full of names like Søren, Zoë, Müller, Łukasz and 田中. gedcom2svg handles these correctly as long as it can decode the file, so the important step is reading the input in the right encoding.

GEDCOM 5.5.1 files declare their encoding in the header (`1 CHAR UTF-8`, `1 CHAR ANSEL`, `1 CHAR ANSI`). gedcom2svg honours that declaration. ANSEL, an old library encoding used by many desktop programs, is converted using a built-in table covering the diacritics that appear in practice. Files that claim `ANSI` are treated as Windows-1252. If a file has no `CHAR` line, UTF-8 is tried first, then Windows-1252. You can override the guess with `--encoding`:

```sh
gedcom2svg family.ged --root @I1@ --encoding cp1250 -o chart.svg
```

Once decoded, names are written into the SVG as UTF-8 with proper XML escaping, so nothing is transliterated or dropped. Two things are worth knowing:

- Text width is estimated from character counts to size boxes and centre labels. For CJK names, where each character is wider than a Latin letter, pass a larger `--box-width` or the label may overflow the box. A future version will measure width per script.
- Output file names derived from the person's name (when `-o` is omitted) keep non-ASCII characters as-is. If your filesystem or a downstream tool objects, give an explicit `-o`.

The GEDCOM `NAME` tag stores surnames between slashes (`Margaret Anne /Holloway/`). gedcom2svg strips the slashes and, by default, prints the surname in capitals as most genealogists expect. Use `--surname-case as-is` to keep the original casing.

## GEDCOM version compatibility

gedcom2svg targets GEDCOM 5.5 and 5.5.1, which is what nearly every desktop and web genealogy program exports. Files from Gramps, Ancestry, FamilySearch, MyHeritage, RootsMagic, Family Tree Maker and Legacy have all been tested.

GEDCOM 7.0 files are accepted with some limitations. The core structure the tool needs (`INDI`, `FAM`, `FAMC`, `HUSB`, `WIFE`, `NAME`, `BIRT`, `DEAT`, `DATE`, `PLAC`) is close enough between versions that charts render correctly. The new 7.0 date formats and the `SEX` enumeration values are understood. Extension tags and `SCHMA` declarations are ignored. If you hit a 7.0 file that fails to parse, please open an issue with a sample.

Pre-5.5 files and non-standard dialects with vendor tags are read on a best-effort basis: unknown tags are skipped rather than causing an error. GEDCOM-X, the JSON format, is not supported; export as GEDCOM 5.5.1 instead.

A file with structural problems (a `FAMC` pointing at a missing family, a person with two `FAMC` records) produces a warning listing the affected IDs, and the chart is drawn with whatever is resolvable.

## Examples

Four-generation chart, vertical, with places, for a web page:

```sh
gedcom2svg family.ged --root @I1@ --generations 4 --orientation up \
  --show-places --box-width 220 -o holloway-web.svg
```

Convert the result to PDF for printing with any SVG-aware tool, for instance:

```sh
rsvg-convert -f pdf holloway-web.svg -o holloway.pdf
```

## License

MIT.
