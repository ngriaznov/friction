**invoicer4j**
================

A Java library for generating PDF invoices from a template plus a data object.

### Maven Dependency Coordinates

Add the following dependency to your `pom.xml` file:
```xml
<dependency>
    <groupId>com.example</groupId>
    <artifactId>invoicer4j</artifactId>
    <version>1.0.0</version>
</dependency>
```
Replace `com.example` with the actual group ID of your project.

### Minimal Code Sample

Here's a minimal example demonstrating how to construct an invoice with line items and render it to a file:
```java
import org.invoicer4j.Invoice;
import org.invoicer4j.Template;
import org.invoicer4j.Data;

public class Main {
    public static void main(String[] args) throws Exception {
        // Define the template (see below for details)
        Template template = new Template("path/to/template.xml");

        // Create an invoice data object
        Data data = new Data();
        data.setInvoiceNumber("INV-001");
        data.setCustomerName("John Doe");
        data.setBillingAddress("123 Main St, Anytown USA");
        data.addLineItem(new LineItem("Product A", 10.99));
        data.addLineItem(new LineItem("Product B", 5.49));

        // Create an invoice instance
        Invoice invoice = new Invoice(template, data);

        // Render the invoice to a file
        invoice.renderToFile("path/to/output.pdf");
    }
}
```
### Template Definition

Templates are defined using a simple XML layout format. The following is an example template:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<template>
    <header>
        <title>Invoice</title>
        <date>${invoice.date}</date>
    </header>
    <customer>
        <name>${invoice.customerName}</name>
        <address>${invoice.billingAddress}</address>
    </customer>
    <items>
        ${invoice.lineItems}
    </items>
    <footer>
        <total>${invoice.total}</total>
        <paymentTerms>Net 30 days</paymentTerms>
    </footer>
</template>
```
In this example, the template uses placeholders for dynamic values (e.g. `${invoice.date}`). These placeholders are replaced with actual values when rendering the invoice.

### Supported Currencies and Locale-Aware Number Formatting

invoicer4j supports a wide range of currencies and locale-aware number formatting. By default, the library will use the system's default locale to format numbers and dates. You can override this behavior by setting the `locale` property on the `Data` object:
```java
data.setLocale(Locale.US); // Use US English locale for formatting
```
### Known Limitations

Multi-page invoices are not currently supported. If an invoice requires multiple pages, it will be rendered as a single page with truncated content.

### API Documentation

For more information on the invoicer4j API, please refer to the [Javadoc documentation](https://example.com/invoicer4j/javadoc).

### Contributing

Contributions are welcome! Please submit pull requests or issues through the project's GitHub repository: <https://github.com/example/invoicer4j>
