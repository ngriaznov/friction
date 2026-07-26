# A URL shortener in a weekend, analytics included

I have a newsletter with about four hundred subscribers and no idea which links anyone clicks. Every hosted shortener I looked at either wanted eight dollars a month or wanted to know a lot more about my readers than I did. So I gave myself Saturday and Sunday to build one.

## Stack, chosen for speed of writing rather than speed of running

Go for the server, SQLite for storage, and server-rendered HTML with a single small chart drawn by hand in SVG. No frontend build step, no npm, no framework. The whole thing compiles to one binary that I scp to a VPS and run behind Caddy.

That combination isn't the fastest possible thing, but it removed every category of problem that eats weekend projects. There was no bundler to configure, no migration tool to set up, no ORM to fight. `database/sql` plus `modernc.org/sqlite` gets you a working data layer in about fifteen lines, and SQLite means the database is a file I can copy with `scp` when I want a backup.

## Slugs

The links table is boring: id, slug, target URL, created timestamp. Slugs are six characters from a 32-character alphabet with the ambiguous ones removed — no `0`, `O`, `1`, `l`. That's a billion possibilities, which is a comically large space for four hundred readers, but it means I can generate randomly and just retry on the unique-index collision rather than maintaining a counter.

I also allowed custom slugs, which took ten minutes and turned out to be the feature I actually use. `/nov-tools` reads better in an email than `/k7fq2p`.

## The redirect path

This is the only piece where performance matters, since it sits between the reader and the thing they wanted. One indexed lookup by slug, then a 302. I used 302 rather than 301 deliberately: a permanent redirect gets cached by the browser and I stop seeing repeat clicks.

The click is recorded *after* the redirect is written, on a background goroutine feeding a buffered channel. If the analytics writer falls behind, clicks get dropped rather than the redirect getting slow. For a personal newsletter that tradeoff is obviously correct, and it meant I never had to think about write contention on the SQLite file.

## What I actually record

Timestamp, slug, referrer, and a coarse user-agent bucket (mobile / desktop / bot). No IP address, no cookie, no fingerprint. Partly that's principle and partly it's that storing IPs would have made me think about retention policies on a Sunday afternoon.

Bot filtering turned out to matter more than I expected. My first day of data showed a link with 60 clicks that I knew had been sent to twelve people. Most of it was link-prefetching by mail clients and security scanners hitting every URL in the message. I now drop anything whose user-agent matches a short list, and anything that arrives within two seconds of a link's first-ever click, which catches the scanners that lie about their agent.

## The dashboard

One page. A table of links sorted by clicks in the last 30 days, and for a selected link, a bar-per-day chart and a referrer breakdown.

The chart is generated server-side as inline SVG — a `<rect>` per day, scaled by the max value in the window. Maybe fifty lines of Go template. I keep expecting to regret not using a charting library, and I keep not regretting it, because the chart does exactly one thing and I never have to upgrade it.

The referrer breakdown collapses to registrable domain, so all the `mail.google.com` variants land in one row. About 40% of clicks report no referrer at all, which is normal for links opened from email clients, and I show that honestly as "direct / unknown" rather than hiding it.

## Would I do it again

Two days, one binary, zero monthly cost. The one thing I'd change is starting with the bot filter instead of bolting it on — my first week of numbers is garbage and I can't retroactively clean it.
