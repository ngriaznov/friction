# Configuring Scraplink: Expiration, Highlighting Themes, and Token-Restricted Uploads

Scraplink is configured through a single `scraplink.toml` file, usually at `/etc/scraplink/scraplink.toml`. After editing it, restart the service (`systemctl restart scraplink`) or send `SIGHUP` to reload without dropping connections. Run `scraplink check-config` first to catch typos.

This guide covers three common adjustments: how long pastes live, which syntax highlighting themes are offered, and locking uploads down to holders of an API token.

## Set an expiration policy

By default Scraplink keeps pastes forever. Most self-hosters want a default lifetime plus a cap on what users can request.

```toml
[expiration]
default = "7d"        # applied when the client does not specify one
max = "90d"           # longest a client may request
allow_never = false   # reject requests for a permanent paste
options = ["10m", "1h", "1d", "7d", "30d", "90d"]
```

- `default` is used by the web form and by API clients that omit `expires`.
- `max` is enforced server-side. A request above it is clamped to `max`, not rejected, so older clients keep working.
- `options` controls the dropdown in the web interface. Values must be less than or equal to `max`.
- Setting `allow_never = true` adds a **Never** option and lets API clients send `expires = "never"`.

Expired pastes are removed by the built-in sweeper, which runs every 15 minutes. To change the interval, set `sweep_interval = "5m"` under `[expiration]`. Deleted pastes return HTTP 410 until the sweeper runs, and 404 after.

Existing pastes are not affected by a changed `default`; it only applies to new pastes.

## Enable syntax highlighting themes

Scraplink ships with a set of highlighting themes and lets viewers pick one from the paste page. Choose which themes to offer and which is the default:

```toml
[highlighting]
default_theme = "github-light"
themes = ["github-light", "github-dark", "solarized-light", "solarized-dark", "monokai"]
auto_dark = true
line_numbers = true
```

- `themes` lists what appears in the viewer's theme picker. Run `scraplink themes list` to see every bundled theme name.
- `auto_dark = true` switches to the first theme in `themes` whose name contains `dark` when the viewer's browser prefers a dark colour scheme.
- A viewer's chosen theme is stored in a cookie and remembered on later visits.

To add your own theme, drop a CSS file into `/var/lib/scraplink/themes/` named after the theme (for example `club-blue.css`) and add `"club-blue"` to `themes`. Scraplink uses standard `hljs-` class names, so any highlight.js-compatible stylesheet works.

Language detection is automatic, but clients can force a language with the `lang` field or by using a file extension in the paste title, such as `deploy.sh`.

## Restrict uploads by API token

For a private instance, require a token on every upload while leaving reading open.

1. Enable token enforcement:

   ```toml
   [uploads]
   require_token = true
   anonymous_read = true   # pastes remain viewable without a token
   ```

2. Create a token for each person or script:

   ```
   scraplink token create --name "ci-deploy" --expires 180d
   scraplink token create --name "alice"
   ```

   The command prints the token once. Store it somewhere safe; Scraplink keeps only a hash.

3. Clients send the token in an `Authorization: Bearer <token>` header, or as the `token` field of a multipart upload. The web form shows a token field when `require_token` is on.

Useful token commands:

- `scraplink token list` shows names, creation dates, expiry, and last use.
- `scraplink token revoke <name>` disables a token immediately.
- `scraplink token limit <name> --per-hour 60` applies a per-token rate limit on uploads.

Each paste records which token created it, so `scraplink paste list --token alice` lets you audit or bulk-delete one person's uploads.

## Verify the configuration

After restarting, confirm the settings took effect:

```
scraplink config show
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://paste.example.org/api/paste -d "hello"
```

The second command should return `401` when `require_token` is enabled and no token is supplied.
