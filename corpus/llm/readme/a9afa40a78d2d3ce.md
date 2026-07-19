# Tasq: A Command-Line Todo Manager

Tasq is a simple, command-line todo manager that stores tasks as plain lines of text in a file, making it easy to manage your to-do list while preserving readability and diffability in version control.

## Getting Started

To start using Tasq, run the following command:

```bash
tasq init
```

This will create a new `tasks.txt` file in your current working directory. You can then use the following commands to add, complete, and list tasks:

### Adding Tasks

Use the `add` command followed by a task description:

```bash
tasq add @tag:Due Date Task Description
```

Example:
```bash
tasq add @urgent:2023-12-31 Buy milk
```

### Completing Tasks

Use the `complete` command with the task ID (1-based index):

```bash
tasq complete 1
```

### Listing Tasks

Use the `list` command without any arguments to view all tasks:

```bash
tasq list
```

You can also filter the list by tag using the `-t` option and due date using the `-d` option. For example, to list only urgent tasks with a due date on December 31st:

```bash
tasq -td:2023-12-31 list
```

## Tag Syntax

Tasks are tagged with an `@` prefix followed by a colon and the tag name. For example, `@urgent`, `@low`, or `@done`. This syntax allows you to filter tasks based on their tags.

## Due Date Syntax

Due dates are specified in YYYY-MM-DD format, preceded by a colon and a space. For example: `2023-12-31`.

## Task File Location

By default, Tasq stores its task file in the current working directory. If you want to store it elsewhere, set the `TASQ_TASKS_DIR` environment variable:

```bash
export TASQ_TASKS_DIR=/path/to/your/tasks/file
tasq init
```

This will create a new tasks file in the specified location.

## Installation

To install Tasq, save this README as a file (e.g., `tasq.md`) and run the following command from your terminal:

```bash
curl -o tasq https://raw.githubusercontent.com/tuupola/tasq/master/TASQ.md
```

Then, make the script executable:

```bash
chmod +x tasq
```

Finally, link the script to an alias (optional):

```bash
ln -s tasq tasq
```

This will allow you to run Tasq using its alias.
