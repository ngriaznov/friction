# Configure Expiration, Syntax Highlighting, and Upload Restrictions in Scraplink

This guide explains how to set three common options on a self-hosted Scraplink instance: how long pastes last, which syntax highlighting themes users can choose, and which clients are allowed to upload.

All three are set in `scraplink.yaml`. Its location depends on how you installed Scraplink:

- Package install: `/etc/scraplink/scraplink.yaml`
- Docker: the file mounted at `/config/scraplink.yaml`

After editing the file, restart the service so your changes take effect:

```sh
sudo systemctl restart scraplink
# or
docker compose restart scraplink
```

## Set an expiration policy

By default, pastes never expire. To keep storage under control, set a default lifetime and a maximum lifetime.

```yaml
expiration:
  default: 7d
  max: 90d
  allow_never: false
  options: [10m, 1h, 1d, 7d, 30d, 90d]
  purge_interval: 1h
```

- `default` is used when an uploader doesn't specify an expiration.
- `max` caps any requested lifetime. Longer requests are shortened to this value, and the response includes a warning.
- `allow_never` controls whether users can create permanent pastes. Set it to `false` to require every paste to expire.
- `options` lists the choices shown in the web UI's expiration dropdown. Each value must be less than or equal to `max`.
- `purge_interval` sets how often the background job deletes expired pastes. An expired paste returns `404` immediately, even before the purge job removes it from storage.

Durations accept `m` (minutes), `h` (hours), and `d` (days).

**Burn after reading.** To let users create pastes that are deleted after they're viewed once, add:

```yaml
expiration:
  allow_burn_after_read: true
```

## Enable syntax highlighting themes

Scraplink ships with several built-in themes. Choose which ones are available and set a default:

```yaml
highlighting:
  enabled: true
  default_theme: github-light
  dark_default_theme: github-dark
  themes:
    - github-light
    - github-dark
    - solarized-light
    - solarized-dark
    - monokai
  auto_detect_language: true
```

- `default_theme` and `dark_default_theme` apply based on the viewer's system color scheme. A viewer can change the theme from the paste view, and their choice is stored in a cookie.
- `themes` controls which themes appear in the theme picker. Run `scraplink themes list` to see all built-in theme names.
- `auto_detect_language` guesses the language when the uploader doesn't specify one. Set it to `false` to show undetected pastes as plain text.

**Adding a custom theme.** Put a CSS file in the `themes/` directory next to your config file, such as `themes/company.css`, and add `company` to the `themes` list. Scraplink loads it at startup.

## Restrict uploads by API token

By default, anyone who can reach your instance can create pastes. To require an API token for uploads while still letting anyone view pastes, set:

```yaml
auth:
  require_token_for_upload: true
  allow_anonymous_view: true
```

Then create tokens with the CLI:

```sh
scraplink token create --name ci-bot --max-size 1MB --max-expiration 1d
scraplink token create --name alice
```

The token appears only once, so store it somewhere safe. The optional flags limit what a token can do:

- `--max-size` sets the largest paste the token can upload.
- `--max-expiration` overrides the global `max` for that token.
- `--rate-limit` sets the uploads allowed per minute, for example `--rate-limit 30`.

Clients send the token in the `Authorization` header:

```sh
curl -H "Authorization: Bearer $SCRAPLINK_TOKEN" \
     --data-binary @build.log \
     "https://paste.example.com/api/pastes?expires=1d&lang=text"
```

A request without a valid token receives `401 Unauthorized`.

To list or revoke tokens:

```sh
scraplink token list
scraplink token revoke ci-bot
```

A revoked token stops working immediately. Pastes that were already created with it stay available until they expire.

**Web UI uploads.** When tokens are required, users who upload through the browser are asked to enter a token once, and it's stored in their browser. If you'd rather allow browser uploads only from your internal network, set `auth.upload_allowlist` to a list of CIDR ranges. Requests from those ranges don't need a token.

## Verify your configuration

Run the built-in check before restarting:

```sh
scraplink config check
```

The check reports invalid durations, unknown theme names, and `options` values that exceed `max`.
