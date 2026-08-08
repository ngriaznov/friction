# tasq

tasq is a command-line todo manager that stores your tasks as plain lines
of text in a file. No database, no binary format — just a text file you
can open in any editor, grep, or check into version control and get a
sane diff every time.

## Usage

Add a task:

```
tasq add "renew car registration @errands due:2024-03-15"
```

Mark it done:

```
tasq done 3
```

(task numbers come from `tasq list`, and refer to the current position in
the list, not a permanent ID)

List everything:

```
tasq list
```

List only what's still open:

```
tasq list --open
```

## Tags

Prefix a word with `@` anywhere in the task text to tag it:

```
tasq add "fix the leaky faucet @home @urgent"
```

A task can have any number of tags. Filter by one:

```
tasq list --tag home
```

## Due dates

Add `due:YYYY-MM-DD` anywhere in the task text to give it a due date:

```
tasq add "submit expense report due:2024-03-01"
```

Filter tasks due on or before a date:

```
tasq list --due 2024-03-01
```

Or just see what's overdue as of today:

```
tasq list --overdue
```

## The task file

By default tasq reads and writes `~/.tasq/tasks.txt`. Each line is one
task; completed tasks are prefixed with `x ` rather than being deleted, so
your history stays in the file (and in your git log, if you're tracking
it). A line looks like:

```
x fix the leaky faucet @home @urgent
submit expense report due:2024-03-01
```

To point tasq at a different file — useful if you keep separate task
files per project, or sync the file via a specific git repo path — set
`TASQ_FILE`:

```
export TASQ_FILE=~/work/project-a/tasks.txt
```

Every tasq command respects `TASQ_FILE` if it's set, falling back to the
default path otherwise.

## License

MIT
