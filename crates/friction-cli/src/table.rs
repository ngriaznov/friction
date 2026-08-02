//! Plain-text table rendering for `check`'s per-metric table.
//!
//! Every value is formatted with a fixed decimal precision
//! ([`FLOAT_PRECISION`]) so two runs over the same input produce
//! byte-identical text output — no locale-, platform-, or
//! `Display`-impl-dependent float formatting.

use std::fmt::Write as _;

/// Decimal places every floating-point table cell is formatted with.
const FLOAT_PRECISION: usize = 4;

/// One row of `check`'s per-metric table: a metric's value, this genre's
/// envelope band for it (if any), and whether the value falls inside that
/// band.
pub struct MetricTableRow<'a> {
    pub name: &'a str,
    pub value: f64,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub in_envelope: Option<bool>,
}

/// Renders `rows` as a fixed-width, deterministic plain-text table:
/// `METRIC | VALUE | BAND | STATUS`, column widths sized to the longest
/// cell in each column (never fewer than its header's own width).
///
/// A row with no band (`lo`/`hi`/`in_envelope` all `None`) prints `n/a`
/// for both the band and status cells, rather than a blank: every cell
/// in the output is always non-empty.
#[must_use]
pub fn render_metric_table(rows: &[MetricTableRow<'_>]) -> String {
    let name_header = "METRIC";
    let value_header = "VALUE";
    let band_header = "BAND";
    let status_header = "STATUS";

    let format_value = |v: f64| format!("{v:.FLOAT_PRECISION$}");
    let format_band = |row: &MetricTableRow<'_>| match (row.lo, row.hi) {
        (Some(lo), Some(hi)) => format!("[{}, {}]", format_value(lo), format_value(hi)),
        _ => "n/a".to_string(),
    };
    let format_status = |row: &MetricTableRow<'_>| match row.in_envelope {
        Some(true) => "in".to_string(),
        Some(false) => "OUT".to_string(),
        None => "n/a".to_string(),
    };

    let name_width = rows
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(0)
        .max(name_header.len());
    let value_width = rows
        .iter()
        .map(|r| format_value(r.value).len())
        .max()
        .unwrap_or(0)
        .max(value_header.len());
    let band_width = rows
        .iter()
        .map(|r| format_band(r).len())
        .max()
        .unwrap_or(0)
        .max(band_header.len());
    let status_width = status_header.len().max(3); // fits "OUT"

    let mut out = String::new();
    let _ = writeln!(
        out,
        "{name_header:name_width$}  {value_header:>value_width$}  {band_header:band_width$}  \
         {status_header:status_width$}"
    );
    for row in rows {
        let value = format_value(row.value);
        let band = format_band(row);
        let status = format_status(row);
        let _ = writeln!(
            out,
            "{name:name_width$}  {value:>value_width$}  {band:band_width$}  {status:status_width$}",
            name = row.name,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_metric_table_marks_in_out_and_missing_bands() {
        let rows = vec![
            MetricTableRow {
                name: "triad_rate",
                value: 0.1234,
                lo: Some(0.0),
                hi: Some(0.5),
                in_envelope: Some(true),
            },
            MetricTableRow {
                name: "em_dash_density",
                value: 12.0,
                lo: Some(0.0),
                hi: Some(5.0),
                in_envelope: Some(false),
            },
            MetricTableRow {
                name: "unbanded_metric",
                value: 1.0,
                lo: None,
                hi: None,
                in_envelope: None,
            },
        ];
        let table = render_metric_table(&rows);
        assert!(table.contains("triad_rate"));
        assert!(table.contains("0.1234"));
        assert!(table.contains("in"));
        assert!(table.contains("OUT"));
        assert!(table.contains("n/a"));
    }

    #[test]
    fn render_metric_table_is_deterministic_across_calls() {
        let rows = vec![MetricTableRow {
            name: "x",
            value: 1.0,
            lo: Some(0.0),
            hi: Some(2.0),
            in_envelope: Some(true),
        }];
        assert_eq!(render_metric_table(&rows), render_metric_table(&rows));
    }
}
