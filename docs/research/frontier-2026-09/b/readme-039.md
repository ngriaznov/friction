# sqlsplit

sqlsplit takes a large SQL dump and splits it into one file per table. When you have a 40 GB `mysqldump` of the whole database and only need to restore `orders` and `customers`, this saves you from loading everything or hand-editing a file too big for any editor.

It streams the input, so memory use stays flat regardless of dump size.

## Installation

With pip:

```sh
pip install sqlsplit
```

Or download a prebuilt binary from the releases page and put it somewhere on your `PATH`. There are no runtime dependencies.

## Basic usage

```sh
sqlsplit backup.sql -o tables/
```

This reads `backup.sql` and writes one file per table into `tables/`, creating the directory if needed:

```
tables/
  _header.sql
  customers.sql
  orders.sql
  order_items.sql
  _footer.sql
```

`_header.sql` holds everything that appears before the first table (session settings such as `SET NAMES`, `SET FOREIGN_KEY_CHECKS=0`, and so on) and `_footer.sql` holds anything after the last one. To restore a single table with its original settings:

```sh
cat tables/_header.sql tables/orders.sql tables/_footer.sql | mysql mydb
```

Input from stdin is supported, so you can decompress on the fly:

```sh
zcat backup.sql.gz | sqlsplit - -o tables/
```

Use `--tables orders,customers` to extract only the named tables and skip writing the rest.

## How table boundaries are detected

sqlsplit scans the dump line by line looking for statements that name a table. A new table's file starts at the first of:

- a `DROP TABLE IF EXISTS \`name\`` statement
- a `CREATE TABLE \`name\`` statement
- an `INSERT INTO \`name\`` statement whose table has not been seen yet

Everything from that point until the next boundary goes into that table's file, including `LOCK TABLES`, `ALTER TABLE ... ADD INDEX` and the `UNLOCK TABLES` that dump tools typically emit after the inserts. Comment blocks like `-- Table structure for table \`orders\`` are treated as part of the table that follows them.

The scanner understands the quoting rules of MySQL, MariaDB and PostgreSQL (`pg_dump` plain-text format), so a string literal containing the text `CREATE TABLE` inside an `INSERT` will not trigger a split. Multi-line `INSERT` statements and extended inserts are handled without buffering the whole statement.

## Compressing the output

Add `-z` or `--gzip` to compress each output file as it is written:

```sh
sqlsplit backup.sql -o tables/ --gzip
```

Files are then named `orders.sql.gz` and so on, and can be restored with `zcat tables/orders.sql.gz | mysql mydb`. Compression level defaults to 6; set it with `--gzip-level 9`.

## License

MIT.
