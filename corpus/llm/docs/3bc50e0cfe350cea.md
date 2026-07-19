## Getting Started with dedupe-shots: Your Local Photo Library De-Duper

`dedupe-shots` is a powerful command-line tool designed to identify and remove duplicate photos within your local photo library using perceptual hashing – meaning it focuses on *content* rather than just filenames or metadata. This helps catch subtle variations like slightly different lighting, compression artifacts, or minor edits that traditional methods would miss.

**1. Installation:**

`dedupe-shots` is written in Python and can be installed easily using `pip`:

```bash
pip install dedupe-shots
```

This will install the tool along with its dependencies. Make sure you have Python 3.7 or higher installed on your system.  You may need to use `sudo pip install dedupe-shots` if you encounter permission issues.

**2. Basic Usage - Scanning Your Library:**

The core command for scanning your photo library is:

```bash
dedupe-shots scan <directory>
```

Replace `<directory>` with the path to the directory containing your photos (e.g., `/home/user/Pictures`).  `dedupe-shots` will then recursively search this directory and its subdirectories, comparing images based on perceptual hashes. 

The first time you run this command, it may take a significant amount of time – potentially several hours – depending on the size of your library. Be patient! `dedupe-shots` builds a database of hashes for all identified images.

**3.  Identifying Matches - The Preview Mode:**

After the scan completes, `dedupe-shots` will display a list of potential duplicate matches. It’s *crucial* to review these carefully before deleting anything. Use the `--preview` flag:

```bash
dedupe-shots preview <directory>
```

This command doesn't perform deletions; instead, it shows you detailed information about each match pair.  You will see:

* **Image Path:** The full path to the image file.
* **Hash Value:** The perceptual hash value used for comparison.
* **Similarity Score:** A numerical score indicating how similar the images are (higher is more similar). This provides an estimate of the visual difference between the files.
* **Difference Details:** Highlights subtle differences like brightness, contrast, and color variations.

**4.  Reviewing Matches & Selecting Deletions:**

The `preview` output will present matches in pairs. Carefully examine each pair using the similarity score and difference details to determine if they are truly duplicates. 

To mark a match for deletion, use the `--delete` flag *after* reviewing the preview:

```bash
dedupe-shots delete --preview <directory>
```

This command will then prompt you to confirm which matches you want to permanently remove.  You can review the list again and select specific pairs by typing their index number (starting from 1) or use the `--all` flag to delete all identified duplicates. **Use this with extreme caution!**

**5.  Important Options & Flags:**

*   `-h, --help`: Displays a help message listing available options.
*   `--quiet`: Suppresses verbose output during scanning.
*   `--force`: Automatically deletes matches without prompting for confirmation (use with *extreme* caution!).
*   `--dry-run`:  Simulates the deletion process without actually deleting anything – useful for testing your commands.

**6.  Safety First!**

Before running `dedupe-shots`, it’s highly recommended to:

* **Back up your photo library.** This is a critical safeguard against accidental data loss.
* **Start with small test directories.** Experiment on a smaller subset of your photos before scanning your entire library. 
* **Carefully review all identified matches.**  Don't blindly delete files based solely on the tool’s output.

**Documentation:**

For more detailed information, including advanced options and troubleshooting tips, please refer to the comprehensive documentation available at [Insert Placeholder URL Here - Replace with actual URL]. Happy de-duping!
