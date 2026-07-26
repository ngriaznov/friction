# isbnparse

A small Go library for parsing and validating book and serial identifiers: ISBN-10, ISBN-13, and ISSN. It verifies checksums and converts between the ISBN-10 and ISBN-13 forms. No dependencies outside the standard library.

## Install

```
go get github.com/tsuchida/isbnparse
```

Requires Go 1.21 or newer.

## Usage

```go
package main

import (
	"fmt"
	"log"

	"github.com/tsuchida/isbnparse"
)

func main() {
	id, err := isbnparse.Parse("0-306-40615-2")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(id.Kind())        // ISBN10
	fmt.Println(id.Normalized())  // 0306406152
	fmt.Println(id.Hyphenated())  // 0-306-40615-2

	thirteen, err := id.ToISBN13()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(thirteen.Normalized()) // 9780306406157
}
```

`Parse` sniffs the identifier type from its length and shape. If you already know what you expect, use `ParseISBN10`, `ParseISBN13`, or `ParseISSN` — they reject anything else rather than guessing, which is usually what you want when validating a form field.

For a boolean check with no allocation, `isbnparse.Valid(s)` returns just `true` or `false`.

## Conversion

`ToISBN13` prepends the `978` prefix, drops the old check digit, and recomputes the ISBN-13 checksum. It always succeeds on a valid ISBN-10.

`ToISBN10` goes the other way, but only for ISBN-13s in the `978` range. A `979`-prefixed ISBN-13 has no ISBN-10 equivalent — the numbering space simply doesn't map — and the call returns `ErrNoISBN10Form`. Handle it; it's not a bug, and it's increasingly common with newer publications.

Both methods return a new `Identifier`. Nothing is mutated in place.

## Errors

Every failure is one of these sentinel values, so you can match with `errors.Is`:

| Error | Meaning |
| --- | --- |
| `ErrLength` | Wrong number of digits after stripping separators |
| `ErrCharacter` | A character that isn't a digit, a permitted `X`, or a separator |
| `ErrCheckDigit` | Well-formed, but the checksum doesn't match |
| `ErrMisplacedX` | `X` appearing somewhere other than the final position |
| `ErrNoISBN10Form` | `ToISBN10` on a non-`978` ISBN-13 |
| `ErrEmpty` | Empty or whitespace-only input |

`ErrCharacter` and `ErrCheckDigit` are distinguished deliberately: the first usually means the user mistyped or pasted junk, the second usually means a single transposed digit. Different messages are worth showing.

Errors wrap a `*ParseError` carrying the original input and, where applicable, the byte offset of the offending character. Type-assert if you want to underline the bad position in a UI.

## Hyphens and whitespace

Input is normalized before validation. Hyphens, en dashes, spaces, and a leading `ISBN`/`ISBN-13`/`ISSN` label are stripped, so all of these parse identically:

```
9780306406157
978-0-306-40615-7
978 0 306 40615 7
ISBN 978-0-306-40615-7
```

Normalization is permissive about *where* hyphens fall, because correct ISBN group hyphenation depends on registration-group ranges that change over time. Validating hyphen placement would mean shipping and maintaining that range table, and the checksum doesn't depend on it.

`Normalized()` gives you the bare digits — that's what you should store and index on. `Hyphenated()` reproduces the hyphenation exactly as it appeared in the input; if the input had none, you get the bare digits back. There is no "re-hyphenate correctly" function, for the reason above.

`X` is accepted in the check position of ISBN-10 and ISSN, in either case, and is emitted uppercase.

## License

MIT.
