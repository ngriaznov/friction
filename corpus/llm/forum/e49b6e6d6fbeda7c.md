**Writing a systemd Unit File for a Custom Long-Running Daemon**

Systemd provides a powerful way to manage system services and daemons on Linux systems. When writing a custom long-running daemon, it's essential to create a proper systemd unit file to ensure the service can be started, stopped, and restarted correctly. This answer will cover three crucial aspects of systemd unit files: `Restart=on-failure`, `WatchdogSec` with `sd_notify` keepalives, and the difference between `Type=simple` and `Type=notify`.

**Restart=on-failure**

The `Restart=on-failure` option tells systemd to restart the service if it exits due to a failure. A failure is defined as any exit status other than 0 (success). This ensures that even if your daemon crashes or encounters an unexpected error, it will be automatically restarted by systemd.

To use this feature, simply add `Restart=on-failure` to the `[Service]` section of your unit file:
```properties
[Unit]
Description=My Custom Daemon

[Service]
Restart=on-failure
ExecStart=/usr/bin/my-daemon
```
**WatchdogSec with sd_notify keepalives**

`sd_notify` is a mechanism for daemons to send notifications to systemd about their status. When used in conjunction with `WatchdogSec`, it allows the daemon to periodically notify systemd that it's still running, preventing systemd from thinking the service has failed.

Set `WatchdogSec` to a value representing the interval between keepalive messages (in seconds). Then, use `sd_notify` to send a "STATUS=1" message at regular intervals:
```properties
[Unit]
Description=My Custom Daemon

[Service]
Restart=on-failure
ExecStart=/usr/bin/my-daemon
WatchdogSec=10
```
In your daemon's code, add the following `sd_notify` calls:
```c
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// ...

int main() {
    // ...
    while (1) {
        // Do some work...
        sd_notify(0, "STATUS=1");
        sleep(WATCHDOG_SEC);
    }
    return 0;
}
```
**Type=simple vs Type=notify**

`Type=simple` is the default behavior for systemd services. It expects the service to notify systemd immediately after startup.

On the other hand, `Type=notify` tells systemd to wait for a notification from the service before considering it up and running.

Choose `Type=notify` if your daemon requires some setup or initialization time before it's ready to serve requests:
```properties
[Unit]
Description=My Custom Daemon

[Service]
Restart=on-failure
ExecStart=/usr/bin/my-daemon
Type=notify
```
In your daemon's code, add the following `sd_notify` call after startup is complete:
```c
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int main() {
    // ...
    sd_notify(0, "READY=1");
    return 0;
}
```
**Example Unit File**

Here's a complete example unit file that incorporates all the concepts discussed above:
```properties
[Unit]
Description=My Custom Daemon

[Service]
Restart=on-failure
ExecStart=/usr/bin/my-daemon
WatchdogSec=10
Type=notify

[Install]
WantedBy=default.target
```
Remember to replace `/usr/bin/my-daemon` with the actual path to your custom daemon executable.

In conclusion, writing a systemd unit file for a custom long-running daemon requires attention to detail and understanding of the various options available. By setting `Restart=on-failure`, using `WatchdogSec` with `sd_notify` keepalives, and choosing between `Type=simple` and `Type=notify`, you can ensure your service is properly managed by systemd.
