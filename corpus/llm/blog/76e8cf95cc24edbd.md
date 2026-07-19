**The Great API Key Debacle: How I Learned to Catch Secrets Before They Slip Away**

It's happened to all of us: we're working on a project, making good progress, and then suddenly our code breaks because some dependency is no longer available or an API key has changed. It's frustrating, time-consuming, and can even lead to public embarrassment when your app crashes due to hardcoded credentials.

I experienced this firsthand a few months ago when I was working on a side project that used the Stripe payment gateway. I had hard-coded my Stripe secret key directly into the code, thinking it would be temporary. Fast forward a few weeks, and I realized I needed to switch to a different plan for production. But instead of being able to easily swap out the key, I found myself frantically scouring through my codebase trying to track down every instance where the key was hardcoded.

That's when I knew I had to take action. I couldn't be the only one who'd fallen victim to this common pitfall. So, I decided to write a custom Git pre-commit hook that would catch hardcoded API keys and AWS secrets before they ever made it into my codebase.

Here's how it works:

1. The script uses `git diff --cached` to compare the staged changes against the last commit.
2. It then uses regular expressions to search for any matches of known API keys or AWS secret patterns in the changed files.
3. If a match is found, the hook outputs an error message with details on what was found and how to fix it.

The script itself is pretty straightforward:
```bash
#!/bin/bash

# List of known API keys to look for (update as needed)
API_KEYS=( "STRIPE_SECRET_KEY" "PAYPAL_API_KEY" )

# List of known AWS secret patterns (update as needed)
AWS_SECRETS=( "/aws/secret/([a-zA-Z0-9]+)/" )

# Get the list of staged files
files=$(git diff --cached --name-only)

# Iterate over each file
for file in $files; do
  # Use regular expressions to search for API keys or AWS secrets
  if grep -qFf <(echo "${API_KEYS[@]}") "$file"; then
    echo "Error: Hardcoded API key found in $file"
    exit 1
  fi

  if grep -qFf <(echo "${AWS_SECRETS[@]}") "$file"; then
    echo "Error: AWS secret pattern found in $file"
    exit 1
  fi
done
```
To use this script, simply save it as `.git/hooks/pre-commit` and make sure it's executable with `chmod +x .git/hooks/pre-commit`. From there, the hook will automatically run whenever you try to commit changes.

This has already saved me a ton of headaches and stress. And I'm confident that it'll do the same for others who've suffered through the Great API Key Debacle.

**UPDATE**: If you're interested in using this script as-is or customizing it to fit your needs, feel free to fork this repository on GitHub: [github.com/yourusername/git-pre-commit-hook](https://github.com/yourusername/git-pre-commit-hook)

Thanks for reading, and I'd love to hear about any other preventative measures you've put in place to avoid these kinds of mistakes!
