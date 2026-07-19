## backupctl - Automated LAMP Stack Backups

**backupctl** is a collection of bash scripts designed to automate nightly backups of your LAMP stack – MySQL databases, web root files via rsync, and archive rotation. This provides a simple and reliable way to protect your website and data.  This README outlines installation, configuration, usage, and the restore process.

**Prerequisites:**

*   Bash shell
*   `rsync` utility installed on both the server and destination (e.g., external drive or cloud storage)
*   `mysql` client installed on the server.


**Environment Variables:**

To avoid hardcoding sensitive credentials, `backupctl` relies on environment variables:

*   `MYSQL_USER`: MySQL username with appropriate privileges to dump databases.
*   `MYSQL_PASSWORD`: Password for the MySQL user.
*   `WEBROOT_SOURCE`:  The absolute path to your web root directory (e.g., `/var/www/html`).
*   `WEBROOT_DESTINATION`: The absolute path where backups of the webroot are stored (e.g., `/mnt/backup/webroot`).
*   `BACKUP_DIR`: The base directory where all backup archives will be stored (e.g., `/mnt/backup`).
*   `RETENTION_DAYS`: Number of days to retain backups. Backups older than this will be automatically deleted. Defaults to 7 days.

**Installation & Cron Entry:**

1.  Download the `backupctl` scripts from [Insert Download Link Here].
2.  Place the scripts in a suitable location on your server (e.g., `/usr/local/bin`).
3.  Create a cron job to run the backup script nightly:

    ```bash
    crontab -e
    ```

    Add the following line to schedule execution at 2:00 AM daily:

    ```
    0 2 * * * /path/to/backupctl.sh
    ```

**Backup Process:**

The `backupctl.sh` script performs the following actions:

*   Dumps MySQL databases using `mysqldump`. Dumps are stored in `$BACKUP_DIR/mysql`.
*   Uses `rsync` to create a mirror of your webroot directory in `$BACKUP_DIR/webroot`.
*   Rotates old archives, deleting backups older than `$RETENTION_DAYS` days.

**Backup Location:**

Backups are stored within the `$BACKUP_DIR` directory (e.g., `/mnt/backup`). The script creates subdirectories for `mysql` and `webroot` within this base location to organize backups.

**Retention Configuration:**

The `RETENTION_DAYS` variable controls how long backup archives are kept.  Adjust this value according to your business needs.


**Restore Script (reversectl.sh):**

A separate script, `reversectl.sh`, is provided for restoring backups. This script reverses the backup process – it's crucial to use this in a controlled environment. The script uses the same variables defined during the backup process.  Instructions on how to use reversectl.sh are detailed within the script itself.

**Important Notes:**

*   Regularly test your restore procedures to ensure backups are valid and you know the restoration steps.
*   Secure your `backupctl` scripts and any configuration files containing sensitive credentials.
*   Adapt this script to suit your specific needs – modify database names, webroot paths, and retention policies as required.

---

**Disclaimer:** This README provides a basic framework for automated backups.  Ensure you thoroughly understand the scripts and their functionality before deploying them in a production environment. Implement appropriate security measures and regularly test your backup and restore processes.
