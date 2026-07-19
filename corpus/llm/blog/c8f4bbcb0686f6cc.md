# Preventing Hardcoded Secrets with a Custom Git Pre-Commit Hook

## The Motivating Incident: A Lesson in Security Through Experience

It was a typical Friday afternoon when I received an email from my colleague, Sarah. She had just discovered some sensitive information—API keys and AWS secrets—hardcoded in our codebase, ready to be committed. The realization hit hard: these credentials were not only exposed but could potentially fall into the wrong hands if they got merged into production.

This incident was a wake-up call for all of us on the team. We knew that developers often forget to remove sensitive information from their commits, especially when working late or under tight deadlines. To prevent such mistakes in the future and ensure our codebase remains secure, I decided to write a custom Git pre-commit hook. This tool would automatically check for hardcoded secrets before any commit could be made.

## The Custom Pre-Commit Hook

### Setting Up the Environment

Firstly, we need to understand that a pre-commit hook runs automatically every time a developer attempts to make a commit but before the commit is actually made. This gives us an opportunity to run checks on the code and prevent sensitive information from being committed.

To create our custom pre-commit hook, I followed these steps:

1. **Locate the .git/hooks Directory**: Every Git repository has a `.git` directory containing various hooks. The `pre-commit` file is located inside the `hooks` folder under `.git`.

2. **Create or Edit the Pre-Commit Hook File**: If the pre-commit hook does not already exist, you can create it by running:
   ```bash
   touch .git/hooks/pre-commit
   ```
   Then give it executable permissions using:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

3. **Write the Script**: I wrote a simple bash script to check for hardcoded secrets in our codebase.

### The Script

Here's what my pre-commit hook script looked like:

```bash
#!/bin/bash

# Define patterns to search for API keys and AWS secrets
PATTERNS=("API_KEY=" "AWS_SECRET_ACCESS_KEY=")

# Check if any of the patterns are present in the staged files
for pattern in "${PATTERNS[@]}"; do
  git diff --cached | grep -Fq "$pattern" && {
    echo "Error: Hardcoded API key or AWS secret detected."
    exit 1
  }
done

echo "Commit successful! No hardcoded secrets found."
```

### How It Works

- **Pattern Matching**: The script defines a list of patterns that are commonly used for storing API keys and AWS access keys. These can be customized based on the specific needs of your project.
  
- **Checking Staged Changes**: Using `git diff --cached`, we compare the staged changes against these patterns. If any pattern is found, it triggers an error message.

- **User Feedback**: The script provides clear feedback to the user about what went wrong and why their commit was blocked.

### Implementation

To implement this hook, you simply need to save this script in the `.git/hooks/pre-commit` file and make sure it has execute permissions. Once set up, every time a developer tries to commit changes, they will be prompted by Git if any hardcoded secrets are found.

## Conclusion

By automating the process of checking for hardcoded secrets with a custom pre-commit hook, we can significantly reduce the risk of sensitive information being committed and exposed. This not only helps maintain the security of our project but also ensures that developers remain vigilant about these issues during their daily workflow.

Incorporating such hooks is just one part of a comprehensive strategy to secure your codebase. Regular security reviews, access controls, and continuous monitoring are equally important. But starting with this small step can make a big difference in safeguarding your project from potential threats.

Happy coding!
