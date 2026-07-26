# invoicer4j

Generate PDF invoices from an XML layout template and a plain Java data object. invoicer4j is a thin layer over [PdfBoxen](https://example.invalid/pdfboxen) that handles the parts of invoice generation that are tedious rather than interesting: laying out a line-item table that grows, formatting money in the right locale, and getting totals to line up on the right margin.

It does not talk to your accounting system, generate invoice numbers, or store anything. You hand it data, it hands you bytes.

## Installation

```xml
<dependency>
  <groupId>io.invoicer4j</groupId>
  <artifactId>invoicer4j-core</artifactId>
  <version>2.4.1</version>
</dependency>
```

Java 17 or later. The only transitive dependency is PdfBoxen 3.x.

## Minimal example

```java
Invoice invoice = Invoice.builder()
    .number("2024-0417")
    .issued(LocalDate.of(2024, 4, 17))
    .due(LocalDate.of(2024, 5, 17))
    .currency(Currency.getInstance("EUR"))
    .locale(Locale.GERMANY)
    .seller(Party.of("Kestrel Werkstatt GmbH", "Ritterstraße 12, 10969 Berlin"))
    .buyer(Party.of("Halden Design AB", "Storgatan 4, 411 24 Göteborg"))
    .line("Bearing housing, machined", 12, new BigDecimal("48.50"))
    .line("Surface treatment", 12, new BigDecimal("6.25"))
    .line("Freight", 1, new BigDecimal("140.00"))
    .taxRate(new BigDecimal("0.19"))
    .build();

Template template = Template.fromClasspath("/templates/standard.xml");

try (OutputStream out = Files.newOutputStream(Path.of("2024-0417.pdf"))) {
    new InvoiceRenderer(template).render(invoice, out);
}
```

`Invoice` is immutable and its builder validates on `build()` — a missing currency or a line with a negative quantity throws `InvoiceValidationException` there rather than at render time.

## Templates

A template is an XML file describing a page as a stack of blocks. There is no expression language and no conditionals; the goal is that a designer can edit one without learning a templating dialect.

```xml
<template page="A4" margin="20mm">
  <text field="seller.name" size="14" weight="bold"/>
  <text field="seller.address" size="9" color="#555555"/>
  <spacer height="12mm"/>

  <row>
    <text field="buyer.name" size="10"/>
    <text field="number" size="10" align="right" label="Invoice"/>
  </row>
  <spacer height="8mm"/>

  <table field="lines">
    <column field="description" width="55%" heading="Description"/>
    <column field="quantity"    width="10%" heading="Qty" align="right"/>
    <column field="unitPrice"   width="17%" heading="Unit" align="right" format="money"/>
    <column field="amount"      width="18%" heading="Amount" align="right" format="money"/>
  </table>

  <totals fields="subtotal,tax,total"/>
  <footer field="paymentTerms" size="8"/>
</template>
```

`field` attributes are resolved against the `Invoice` object by property path. Unknown paths fail at template load, not at render, so a typo shows up the first time you construct the `Template` rather than on the invoice you already emailed.

`format="money"` applies the invoice's currency and locale. `format="date"` and `format="percent"` are also available. Widths in a `<table>` must sum to 100%.

Custom fields go in the invoice's `extras` map and are addressed as `extras.yourKey`.

## Currencies and number formatting

Anything `java.util.Currency` recognizes is supported. Formatting is delegated to `NumberFormat` for the invoice locale, which gets you the correct decimal separator, grouping separator, symbol placement, and fraction digits — including the zero-decimal cases like JPY and KRW, where invoicer4j will also refuse to accept line amounts with a fractional part.

Rounding uses `RoundingMode.HALF_UP` on each line and again on the tax figure. If your jurisdiction requires banker's rounding, pass a `RoundingMode` to `Invoice.builder().rounding(...)`.

The invoice locale and the currency are independent on purpose. A German company invoicing in USD sets `Locale.GERMANY` with `Currency.getInstance("USD")` and gets `1.234,56 $`.

## Known limitations

**Multi-page invoices are partially supported.** A `<table>` that overflows will continue onto a second page and repeat its heading row, but the `<totals>` and `<footer>` blocks are only ever drawn on the last page, and any block above the table is drawn only on the first. There is no page numbering and no "continued" marker. Invoices past roughly 45 line items start to look wrong. Tracked in #112.

**No right-to-left text.** Arabic and Hebrew strings render in logical order, which is to say incorrectly.

**Fonts must be embedded manually** via `Template.withFont(...)` if you need glyphs outside Latin-1. The bundled default is Helvetica, which will silently substitute a box for anything it cannot draw.

## License

Apache 2.0.
