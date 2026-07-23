```markdown
# backupctl - LAMP Stack Automated Backups

backupctl is a collection of bash scripts designed to automate nightly backups of your LAMP (Linux, Apache, MySQL, PHP) stack server.  It provides a streamlined and reliable way to protect your critical data through regularly scheduled database dumps, web root snapshots using `rsync`, and automated cleanup of old archives.

## Requirements & Dependencies

* **Bash:** The scripts are written in bash and require execution capabilities.
* **MySQL Client (`mysql`):** Required for creating MySQL backups.  Ensure the client version is compatible with your server's database version.
* **Rsync:** Used to efficiently copy the web root files.
* **gzip/tar (optional):** Utilized for compressing archives and improving storage efficiency.

## Configuration - Environment Variables

The following environment variables *must* be set before running `backupctl`. It is recommended to define them in a `.env` file (see usage example below) or directly in your execution context:

* **`DB_HOST`:** MySQL hostname or IP address (e.g., `localhost`).
* **`DB_USER`:** MySQL username for backup operations. This user should have appropriate SELECT privileges on all databases you wish to back up.
* **`DB_PASSWORD`:** MySQL password for the specified user.
* **`WEBROOT`:** The absolute path to your web server's document root (e.g., `/var/www/html`).
* **`BACKUP_DIR`:**  The base directory where all backups will be stored. This should be an accessible location with sufficient storage space.  (default: `/opt/backups`)

**Example `.env` file:**

```dotenv
DB_HOST=localhost
DB_USER=backupuser
DB_PASSWORD=securepassword
WEBROOT=/var/www/html
BACKUP_DIR=/opt/backups
```

You can then source this file before execution using: `source .env`

## Execution & Backup Strategy

The scripts perform the following actions nightly:

1. **MySQL Database Dump:** Creates a compressed SQL dump of all databases on the server. The script uses `mysqldump` to create the backup and pipes it directly to `gzip` for efficient storage.
2. **Web Root Rsync:** Uses `rsync` to copy your web root directory (specified by `WEBROOT`) into the `BACKUP_DIR`.  This ensures a full snapshot of all website files is captured.
3. **Archive Rotation:** Removes backups older than the configured retention period.

## Installation & Scheduling - The Cron Entry

Add the following cron entry to schedule nightly execution:

```crontab
0 2 * * * /path/to/backupctl.sh >/dev/null 2>&1
```

This will execute `backupctl.sh` at 2:00 AM every night. Replace `/path/to/backupctl.sh` with the actual path to your script.  Redirecting output prevents unnecessary emails and keeps logs clean.

## Backup Location & Retention

* **Backup Directory:** The primary location for backups is defined by `BACKUP_DIR`.
* **Retention Policy:** Currently, retention is implicitly controlled through archive deletion in `backupctl.sh`, retaining the newest 7 days’ worth of database dumps and webroot archives by default.  Customization would require editing this script directly to adjust the number of retained backup rotations.

## Restoration - The Restore Script (`restorectl.sh`)

A corresponding restoration script, `restorectl.sh`, is provided for recovering from failures. It reverses the process:

1. **Database Restore:** Extracts a specified database dump and restores it using MySQL.
2. **Web Root Restore:** Copies files from a selected archive within the `BACKUP_DIR` back to your web root (`WEBROOT`).  This requires providing the exact filename of the backup archive you wish to restore.

**Usage:** `./restorectl.sh <backup_archive_name>` (e.g., `./restorectl.sh 2023-10-27.tar.gz`)

## Important Considerations

* **Testing:** Regularly test both the backup and restore process!
* **Permissions:** Ensure proper permissions on all directories involved in backups, especially `BACKUP_DIR`.
* **Security:** Protect your `.env` file (if used) as it contains sensitive database credentials.
* **Monitoring:** Implement monitoring to verify that backups are completing successfully each night.
```
