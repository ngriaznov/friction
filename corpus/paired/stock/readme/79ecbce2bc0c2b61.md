```markdown
# backupctl - Automated LAMP Stack Backups

`backupctl` is a collection of bash scripts designed to automate nightly backups of a typical LAMP (Linux Apache MySQL PHP) stack server. It handles MySQL dumps, web root synchronization using `rsync`, and archive rotation for efficient storage management.  This solution prioritizes simplicity and ease of configuration for smaller deployments or environments where full-blown backup solutions are overkill.

## Requirements & Dependencies

* **Bash:**  The scripts are written in bash and require a compatible shell environment.
* **MySQL Client:** `mysqldump` must be installed on the server.
* **rsync:** Required for synchronizing the web root directory.
* **gzip:** Used for compressing backups.
* **cron:** For scheduling nightly backup execution.

## Configuration - Environment Variables

The scripts rely heavily on environment variables for configuration.  **Do not hardcode sensitive information within the scripts themselves.** Set these variables in your `.bashrc`, `.profile`, or a dedicated environment file sourced before running `backupctl`.

* **`DB_USER`:** MySQL username (e.g., "root").
* **`DB_PASSWORD`:**  MySQL password.
* **`DB_NAME`:** Name of the database to be backed up (e.g., "my_database").
* **`WEB_ROOT`:** Absolute path to your web root directory (e.g., "/var/www/html").
* **`BACKUP_DIR`:**  Absolute path where backups will be stored (e.g., "/opt/backups/lamp"). This directory *must exist*.
* **`RETENTION_DAYS`:** Number of days to retain old backups. Older backups than this value will be automatically deleted. Default: 7

## Script Overview & Backup Process

The `backupctl` scripts perform the following actions nightly:

1. **MySQL Dump:**  Creates a compressed SQL dump of your specified database using `mysqldump`.
2. **Web Root Sync:**  Synchronizes the contents of your web root directory to the backup location, preserving permissions and timestamps using `rsync`.
3. **Archive Rotation:** Deletes backups older than `RETENTION_DAYS`.

## Installation & Scheduling (Cron)

1. **Place Scripts:** Copy the scripts into a suitable location on your server (e.g., `/opt/backupctl`). Make sure they are executable: `chmod +x /opt/backupctl/*`
2. **Set Environment Variables:** As described above, set the required environment variables.
3. **Create Cron Entry:** Add the following line to your crontab (using `crontab -e`):

   ```cron
   0 2 * * * /opt/backupctl/backup.sh > /dev/null 2>&1
   ```

   This will run the `backup.sh` script at 2:00 AM every night.  Adjust the timing as needed.

## Backup Location & Retention

* **Backup Location:** Backups are stored in the directory specified by the `BACKUP_DIR` environment variable. Each backup includes a timestamped MySQL dump (`*.sql.gz`) and a synchronized web root (`webroot`).
* **Retention:**  Old backups older than the value defined in `RETENTION_DAYS` are automatically deleted during each nightly run.

## Restore Script (restore.sh)

A basic restore script, `restore.sh`, is provided to reverse the backup process.  **Use with caution and test thoroughly on a non-production environment first.**

1. **Navigate:** Change directory to the location of the backup you want to restore from.
2. **MySQL Restore:** Execute the following command: `mysql -u <DB_USER> -p<DB_PASSWORD> < database_name.sql.gz` (replace `<database_name.sql.gz>` with the name of the SQL dump file).
3. **Web Root Restore:** Replace the contents of your web root directory with the files from the `webroot` archive within the backup folder using a method such as `rsync`.


## Disclaimer

This script is provided "as is" and without any warranty. Use it at your own risk.  Always test backups and restore procedures regularly to ensure data integrity and recoverability.
```
