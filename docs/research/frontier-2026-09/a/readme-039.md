# sqlsplit

Split a large SQL dump into one file per table.

If you only need to restore three tables from a 40 GB backup, you shouldn't have to load the whole thing. sqlsplit streams through the dump once and writes each table's schema and data to a separate `.sql` file, which you can then restore on its own.

## Installation

Download a prebuilt binary from the [releases page](https://github.com/example/sqlsplit/releases), or install with Go:

```sh
go install github.com/example/sqlsplit@latest
```

## Usage

```sh
sqlsplit backup.sql ./tables
```

This produces:

```
tables/
  _header.sql
  customers.sql
  orders.sql
  order_items.sql
  ...
```

Restore only what you need:

```sh
mysql mydb < tables/_header.sql
mysql mydb < tables/orders.sql
```

The dump can also be read from standard input, which is useful for compressed backups:

```sh
zcat backup.sql.gz | sqlsplit - ./tables
```

Memory use stays flat no matter how big the dump is, because sqlsplit never holds more than one statement in memory.

## How table boundaries are detected

sqlsplit reads the dump statement by statement. It keeps track of quoted strings, comments and escaped characters, so a semicolon inside a string value won't end a statement early.

- A `CREATE TABLE` statement starts a new table. The table's output file is opened, and the `CREATE TABLE` along with any `DROP TABLE IF EXISTS` just before it is written there.
- An `INSERT INTO <table>` statement is appended to that table's file, whatever order the statements appear in.
- Related statements such as `ALTER TABLE`, `CREATE INDEX` and `LOCK/UNLOCK TABLES` are routed to the file of the table they reference.
- Anything that doesn't belong to a table, such as `SET` statements, character set settings and the dump header, goes into `_header.sql`. Restore it first so each table file runs with the same session settings as the original dump.

Table names are unquoted before use, whether they were written with backticks or double quotes, and are sanitised into safe filenames. Output from both MySQL (`mysqldump`) and PostgreSQL (`pg_dump` with `--inserts`) is supported.

## Compressing output

Add `--gzip` to compress each file as it's written:

```sh
sqlsplit --gzip backup.sql ./tables
```

Each file gets a `.sql.gz` extension and can be restored with:

```sh
gunzip -c tables/orders.sql.gz | mysql mydb
```

Use `--gzip-level 1-9` to trade speed against size (the default is 6).

## License

MIT
