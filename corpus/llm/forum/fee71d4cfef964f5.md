Okay, here's a StackExchange-style answer detailing how to write a systemd unit file for a custom long-running daemon, incorporating your requested elements:

---

**Subject: Creating a Robust Systemd Unit File for Your Long-Running Daemon**

**Question:** I’m developing a daemon that needs to run reliably and automatically restart if it crashes.  I've heard about `Restart=on-failure`, `WatchdogSec` with `sd_notify keepalives`, and the differences between `Type=simple` and `Type=notify`. Where do I start, and what’s the best approach for ensuring my daemon stays alive?

**Answer:**

You've identified key concepts for creating a resilient systemd service! Let’s break this down. Systemd is designed to manage services reliably, and these features are crucial for that purpose.

**1. Restart Policies: `Restart=on-failure`**

This is your starting point.  `Restart=on-failure` tells systemd to automatically restart your daemon if it exits with a non-zero exit code (indicating an error). It's generally recommended as a baseline. However, be aware that overly aggressive restarts can mask underlying issues. Consider carefully whether you want *all* failures to trigger a restart or only specific ones.

**2. WatchdogSec and `sd_notify keepalives`**

This is where things get more sophisticated.  The `WatchdogSec` directive monitors the daemon's health. If the daemon doesn’t send a notification (using `sd_notify`) within the specified time (default 60 seconds), systemd interprets this as a failure and restarts the service.

To use `sd_notify`, your daemon *must* periodically call `sd_notify(3)` to tell systemd it's still alive. This is generally done in a loop within your daemon’s main execution thread.  This keeps the watchdog informed that the service is actively running and prevents unnecessary restarts due to inactivity.  The `keepalives` option, when combined with `WatchdogSec`, provides an extra layer of protection against hung processes.

**3. Unit Type: `Type=simple` vs. `Type=notify`**

*   **`Type=simple`:** This is the simplest type and assumes your daemon just starts and stops – it doesn’t provide any feedback to systemd about its state. It's suitable for basic daemons but lacks sophisticated monitoring capabilities. You’ll need to rely solely on exit codes for restart logic.

*   **`Type=notify`:** This is the *recommended* type for most long-running services. Your daemon must call `sd_notify(3)` to signal its state changes (e.g., starting, stopping, running).  Systemd uses this notification mechanism to monitor the service and react accordingly. It allows integration with features like WatchdogSec.

**Example Unit File (`mydaemon.service`)**

```systemd
[Unit]
Description=My Long-Running Daemon
After=network.target  # Start after networking is available (adjust as needed)

[Service]
Type=notify          # Use the notify type for proper monitoring
Restart=on-failure   # Restart if it exits with an error
WatchdogSec=60       # Monitor health every 60 seconds
ExecStart=/usr/local/bin/mydaemon  # Path to your daemon executable
User=myuser           # Run as a specific user (important for security)
Group=mygroup         # Run as a specific group

[Install]
WantedBy=multi-user.target  # Start during normal multi-user operation
```

**Important Considerations:**

*   **Logging:** Ensure your daemon logs its activity effectively to help with debugging restarts. Use systemd’s journalctl command to view these logs.
*   **Error Handling:** Implement robust error handling within your daemon itself – don't just rely on `Restart=on-failure`.  Handle exceptions gracefully and log relevant information.
*   **Security:** Always run services as a non-root user with minimal privileges.

To learn more, consult the official systemd documentation: [https://www.freedesktop.org/wiki/Project/systemd/](https://www.freedesktop.org/wiki/Project/systemd/)


---

Do you want me to elaborate on any specific aspect of this answer (e.g., providing a snippet of code for the `sd_notify` call, or discussing more advanced systemd features)?
