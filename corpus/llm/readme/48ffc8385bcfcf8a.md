# csvtype

Read a CSV, infer a column schema by sampling rows, then iterate records as typed values instead of `&str`.

The `csv` crate gives you strings. Most of the time what you actually want is `i64`, `f64`, `NaiveDate`, `bool`, or `String`, and you end up writing the same block of `parse()` calls and match arms for every file. `csvtype` reads a sample of the file, decides what each column looks like, and hands back a `TypedRecord` you can index by name.

## Install

```toml
[dependencies]
csvtype = "0.4"
```

Minimum supported Rust version is 1.70.

## Example

```rust
use csvtype::{Reader, ColumnType, Value};

fn main() -> Result<(), csvtype::Error> {
    let mut reader = Reader::from_path("sales.csv")?;
    let schema = reader.infer_schema()?;

    for column in schema.columns() {
        println!("{} -> {:?}", column.name(), column.kind());
    }
    // region -> String
    // units -> Integer
    // unit_price -> Float
    // closed_on -> Date
    // refunded -> Boolean

    for record in reader.typed_records() {
        let record = record?;
        let units = record.get("units")?.as_integer()?;
        let price = record.get("unit_price")?.as_float()?;

        if let Value::Date(d) = record.get("closed_on")? {
            println!("{d}: {:.2}", units as f64 * price);
        }
    }

    Ok(())
}
```

`infer_schema` reads from the underlying stream and buffers what it consumes, so calling it before `typed_records()` does not skip rows. You can also call `Reader::from_reader` with anything implementing `io::Read`, though inference on a non-seekable stream is limited to whatever fits in the sample buffer.

## Sampling and accuracy

By default `csvtype` samples the first 1,000 data rows. Inference is conservative: a column is only assigned a narrow type if every sampled value parses as that type, and empty cells are ignored rather than counted as failures. Types widen in a fixed order — `Integer` widens to `Float`, and anything unresolvable falls back to `String`.

Sample size is the main accuracy lever, and the failure mode is worth understanding. A column of order IDs that looks like `10024`, `10025`, ... for the first 900 rows and then hits `10026-B` on row 40,000 will be inferred as `Integer`, and iteration will return a `TypeMismatch` error when it reaches that row. Inference errors do not surface until you read the offending row.

Raise the sample, or read the whole file, when you do not trust the head of the data to be representative:

```rust
use csvtype::{Reader, SampleSize};

let mut reader = Reader::from_path("sales.csv")?
    .with_sample(SampleSize::Rows(50_000));

// Or scan everything. Costs a full pass; buffers the file.
let mut reader = Reader::from_path("sales.csv")?
    .with_sample(SampleSize::All);
```

`SampleSize::Random(n)` reservoir-samples across the file instead of taking a prefix. It requires a seekable source but is usually the better default for sorted or grouped data.

## Overriding a column

When you already know the type, say so. Overrides are applied before inference runs, so the column is not sampled at all.

```rust
use csvtype::{Reader, ColumnType};

let mut reader = Reader::from_path("sales.csv")?
    .with_column_type("order_id", ColumnType::String)
    .with_column_type("closed_on", ColumnType::DateFormat("%d/%m/%Y"));
```

Unknown column names in an override are an error at `infer_schema()` time, not silently ignored — a typo in a header name should not quietly cost you a type.

Date parsing uses `chrono` and tries ISO 8601 first, then a short list of common formats. If your dates are ambiguous (`03/04/2024` could be either), pin the format explicitly with `DateFormat`. `csvtype` will not guess between day-first and month-first.

## License

MIT OR Apache-2.0
