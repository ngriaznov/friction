# invoicer4j

invoicer4j is a Java library for generating PDF invoices from a template
plus a plain data object. It's built on top of an existing PDF layout
engine (iText under the hood) so it doesn't have to solve PDF rendering
itself — invoicer4j's job is turning a template definition and an invoice
object into layout instructions.

## Maven

```xml
<dependency>
  <groupId>com.invoicer4j</groupId>
  <artifactId>invoicer4j-core</artifactId>
  <version>1.6.2</version>
</dependency>
```

## Minimal example

```java
Invoice invoice = new Invoice.Builder()
    .invoiceNumber("INV-2024-0042")
    .issueDate(LocalDate.of(2024, 3, 1))
    .dueDate(LocalDate.of(2024, 3, 31))
    .billTo(new Party("Acme Retail Ltd", "12 Market St, London"))
    .billFrom(new Party("Northwind Supply Co", "45 Harbor Rd, Bristol"))
    .currency(Currency.getInstance("GBP"))
    .locale(Locale.UK)
    .addLineItem(new LineItem("Widget, small", 4, new BigDecimal("2.50")))
    .addLineItem(new LineItem("Widget, large", 2, new BigDecimal("6.00")))
    .build();

InvoiceRenderer renderer = new InvoiceRenderer(Template.load("templates/standard.xml"));
renderer.render(invoice, new File("invoice-0042.pdf"));
```

`LineItem` takes a description, quantity, and unit price; totals, tax
lines (if the template defines any), and the grand total are computed by
`InvoiceRenderer` from the line items rather than being supplied directly.

## Templates

Templates are a simple XML layout format — not a general-purpose
templating language, just enough markup to describe where the standard
invoice elements go and how they're styled:

```xml
<template>
  <header>
    <logo src="assets/logo.png" width="120"/>
    <field name="invoiceNumber" label="Invoice #"/>
    <field name="issueDate" label="Date"/>
  </header>
  <parties>
    <billFrom/>
    <billTo/>
  </parties>
  <lineItemsTable>
    <column field="description" label="Description" width="50%"/>
    <column field="quantity" label="Qty" width="15%"/>
    <column field="unitPrice" label="Unit Price" width="15%"/>
    <column field="lineTotal" label="Total" width="20%"/>
  </lineItemsTable>
  <totals showTax="true"/>
  <footer text="Payment due within 30 days."/>
</template>
```

`Template.load` reads this from either the classpath or the filesystem.
Fonts, page size, and margins are set at the top of the `<template>`
element via attributes; anything not specified falls back to invoicer4j's
built-in defaults (A4, Helvetica, 20mm margins).

## Currencies and locale formatting

Line item amounts and totals are formatted using `java.text.NumberFormat`
seeded from the `Locale` set on the invoice, so `1234.5` renders as
`£1,234.50` for `Locale.UK` with `Currency.getInstance("GBP")`, or
`1.234,50 €` for `Locale.GERMANY` with `Currency.getInstance("EUR")`. Any
`Currency` and `Locale` combination supported by the JDK works; invoicer4j
doesn't maintain its own currency formatting tables.

## Known limitations

- **Multi-page invoices are basic.** If the line items table overflows a
  page, invoicer4j will continue it onto a new page, but header and totals
  placement on continuation pages is fixed — there's currently no way to
  customize a distinct "continued" header layout per template. For
  invoices that reliably run long, keep an eye on how the overflow looks
  in your chosen template before relying on it in production.
- **No built-in tax rate logic.** invoicer4j will lay out a tax line if
  the template asks for one and the invoice object supplies a tax amount,
  but it doesn't compute tax rates itself — that calculation is the
  caller's responsibility.
- **Templates are layout-only.** There's no conditional logic or looping
  construct beyond the line items table; anything more dynamic than the
  built-in elements needs to be handled in Java before building the
  `Invoice` object.

## License

Apache-2.0
