# Running Quillfeed behind a reverse proxy with HTTPS

Quillfeed listens on plain HTTP and has no built-in TLS. The supported way to expose it to the internet is to put a reverse proxy in front of it and let the proxy terminate TLS. This guide sets up Quillfeed and Caddy with Docker Compose, obtains a certificate from Let's Encrypt, and covers the settings Quillfeed needs in order to generate correct links behind a proxy.

You will need a server with Docker and the Compose plugin installed, a domain name whose A record already points at that server, and ports 80 and 443 free.

## 1. Create the project directory

```
mkdir -p /srv/quillfeed && cd /srv/quillfeed
mkdir data caddy-data
```

`data/` holds the SQLite database and downloaded article content. `caddy-data/` holds the certificates and account key, so it must survive container restarts — otherwise you will re-request a certificate on every deploy and hit Let's Encrypt's rate limits.

## 2. Write the compose file

Create `docker-compose.yml`:

```yaml
services:
  quillfeed:
    image: quillfeed/quillfeed:2.4
    restart: unless-stopped
    volumes:
      - ./data:/var/lib/quillfeed
    environment:
      QUILLFEED_BASE_URL: https://feeds.example.com
      QUILLFEED_TRUSTED_PROXIES: 172.16.0.0/12
      QUILLFEED_SECRET_KEY_FILE: /run/secrets/session_key
      QUILLFEED_POLL_INTERVAL: 30m
      QUILLFEED_DB_PATH: /var/lib/quillfeed/quillfeed.db
    secrets:
      - session_key
    expose:
      - "8480"

  caddy:
    image: caddy:2
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ./caddy-data:/data
    depends_on:
      - quillfeed

secrets:
  session_key:
    file: ./session_key
```

Note that Quillfeed is published with `expose`, not `ports`. The application should only be reachable from the Compose network; if you bind it to a host port as well, anyone can reach it over unencrypted HTTP and bypass the proxy entirely.

Generate the session key before the first start:

```
openssl rand -hex 32 > session_key
chmod 600 session_key
```

## 3. Environment variables that matter behind a proxy

| Variable | Purpose |
| --- | --- |
| `QUILLFEED_BASE_URL` | The public URL. Used to build absolute links in the OPML export, password-reset mails, and the Fever-compatible API. Set it to the `https://` address, not the internal one. |
| `QUILLFEED_TRUSTED_PROXIES` | CIDR ranges Quillfeed will accept `X-Forwarded-For` and `X-Forwarded-Proto` from. Leave it unset and Quillfeed logs every client as the proxy's address and refuses to set secure cookies. |
| `QUILLFEED_SECRET_KEY_FILE` | Path to the session signing key. Changing it invalidates all sessions. |
| `QUILLFEED_POLL_INTERVAL` | How often feeds are refreshed. Anything under `15m` is discourteous to small publishers. |

## 4. Write the Caddyfile

```
feeds.example.com {
    encode gzip
    reverse_proxy quillfeed:8480 {
        header_up X-Forwarded-Proto {scheme}
    }
    request_body {
        max_size 20MB
    }
}
```

Caddy sets `X-Forwarded-For` and `X-Forwarded-Host` on its own; the explicit `X-Forwarded-Proto` line is there so Quillfeed knows the outside connection was TLS and marks its session cookie `Secure`. The `max_size` bump matters if you plan to import a large OPML file — the default 10 MB is enough for most people but not for a decade of accumulated subscriptions.

## 5. Start it and watch the certificate get issued

```
docker compose up -d
docker compose logs -f caddy
```

Caddy solves the HTTP-01 challenge on port 80 and then serves HTTPS. Look for a line containing `certificate obtained successfully`. If you instead see a challenge failure, the usual causes are DNS not yet propagated, port 80 blocked by a cloud firewall, or another web server already bound to it. Fix the cause and run `docker compose restart caddy`; Caddy retries with backoff on its own, so you may simply need to wait a minute.

Renewal is automatic and needs no cron entry. Caddy checks twice a day and renews at roughly 30 days remaining.

## 6. First login

Open `https://feeds.example.com`. Quillfeed shows a one-time setup form on the first request and creates the initial administrator account. That form is disabled permanently once an account exists, but it is open to anyone until you use it — so complete this step immediately after the certificate is issued, not the next morning.

Verify the proxy setup from **Settings → Diagnostics**. The page reports the client address Quillfeed sees and whether it considers the connection secure. If it shows the Docker gateway address instead of your own, widen `QUILLFEED_TRUSTED_PROXIES` to include the Compose network's subnet.
