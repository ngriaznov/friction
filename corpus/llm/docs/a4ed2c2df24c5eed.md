### How-To Guide: Setting Up Automatic Database Backups for Corkboard

Corkboard is a versatile self-hosted note-taking application that allows users to manage their notes in an organized manner. To ensure your data remains safe and can be recovered if necessary, setting up automatic database backups is crucial. This guide will walk you through the process of configuring automatic backups using cron jobs on a Linux server, storing these backups securely, and restoring from them when needed.

#### Prerequisites

- A running instance of Corkboard installed on a Linux server.
- Basic knowledge of command-line operations.
- Administrative access to your server.

### Step 1: Locate the Database Configuration

First, locate where the database configuration for Corkboard is stored. This can typically be found in a file named `config.php` within the root directory of your Corkboard installation or possibly in another location depending on how it was set up.

```bash
cd /path/to/corkboard/
nano config.php
```

Look for lines that define database settings such as:

```php
$DB_HOST = 'localhost';
$DB_USER = 'corkboard_user';
$DB_PASS = 'password';
$DB_NAME = 'corkboard_db';
```

Note the database name, which is crucial for backup and restore operations.

### Step 2: Create a Backup Script

Create a simple shell script to handle the database backup process. This script will use `mysqldump` to create a backup of your database.

```bash
nano /path/to/corkboard/backup.sh
```

Add the following content:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d%H%M%S)
DB_USER="corkboard_user"
DB_PASS="password" # Replace with actual password
DB_NAME="corkboard_db"
BACKUP_DIR="/path/to/corkboard/backups"
MYSQLDUMP_PATH="/usr/bin/mysqldump"

mkdir -p $BACKUP_DIR

$MYSQLDUMP_PATH --user=$DB_USER --password=$DB_PASS $DB_NAME > $BACKUP_DIR/backup_$DATE.sql
```

Make the script executable:

```bash
chmod +x /path/to/corkboard/backup.sh
```

### Step 3: Configure Cron Jobs for Automatic Backups

Cron jobs allow you to schedule tasks at specific times. To set up a cron job, follow these steps:

1. Open your crontab configuration file using `crontab -e`.

2. Add the following line to run the backup script every day at 3 AM (adjust as needed):

   ```bash
   0 3 * * * /path/to/corkboard/backup.sh > /dev/null 2>&1
   ```

   This cron job will run `/path/to/corkboard/backup.sh` and redirect both stdout and stderr to null, minimizing clutter in your logs.

3. Save the file and exit the editor.

### Step 4: Verify Backups

To ensure that backups are working correctly:

- Check if files are being created in the backup directory.
  
```bash
ls /path/to/corkboard/backups/
```

- Attempt to restore a sample backup to verify its integrity. (This step is optional but recommended for peace of mind.)

### Step 5: Restore from Backup

In case you need to recover data, follow these steps:

1. Ensure your server has access to the latest backup files.
2. Stop any running services that might interfere with database operations.

   ```bash
   service mysql stop
   ```

3. Use `mysql` command to restore the database from a specific backup file. Replace `/path/to/backup.sql` and `corkboard_db` accordingly:

   ```bash
   mysql -u corkboard_user -p corkboard_db < /path/to/backup.sql
   ```

4. Once restored, start services back up.

   ```bash
   service mysql start
   ```

5. Verify that Corkboard is functioning as expected with the new data.

### Conclusion

By following these steps, you can set up reliable automatic backups for your Corkboard installation. This ensures that even if something goes wrong, you can recover your valuable notes and data quickly and easily. Regularly reviewing and updating your backup strategy can further enhance data security.
