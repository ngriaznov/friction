# How to Configure a Reverse Proxy with Wharfgate for Routing Traffic Based on URL Path Prefix

Wharfgate is an efficient, lightweight load balancer that enables you to route traffic to multiple backend services based on various criteria, such as URL path prefixes. This guide will walk you through setting up Wharfgate to direct incoming requests to three specific backend services, depending on the requested URL prefix.

## Prerequisites

Before proceeding, ensure you have:

- Wharfgate installed and running.
- Three backend services accessible at known URLs (e.g., `http://backend1.example.com`, `http://backend2.example.com`, `http://backend3.example.com`).
- Basic knowledge of network configurations and command-line operations.

## Step 1: Install and Configure Wharfgate

1. **Download and Install**: Obtain the latest version of Wharfgate from its official repository or website.
2. **Run Wharfgate**: Start the service with default settings, ensuring it listens on your desired network interface (e.g., `0.0.0.0` for all interfaces).
3. **Set Up Logging and Debugging**: Enable logging to monitor operations and debugging if you encounter issues.

## Step 2: Create a Configuration File

Create a configuration file named `wharfgate.conf` in the Wharfgate's config directory (typically `/etc/wharfgate/`). Below is an example configuration tailored for routing based on URL path prefixes:

```plaintext
# wharfgate.conf

listen 80 # Listen on port 80, adjust if necessary.

backend1 {
    url_pattern ^https?://backend1\.example\.com/
    server_address http://backend1.example.com
}

backend2 {
    url_pattern ^https?://backend2\.example\.com/
    server_address http://backend2.example.com
}

backend3 {
    url_pattern ^https?://backend3\.example\.com/
    server_address http://backend3.example.com
}
```

### Explanation of Configuration

- **listen**: Specifies the port on which Wharfgate should listen for incoming requests. Here, it's set to `80` (HTTP). Adjust if you're using HTTPS or another protocol.
- **backend1, backend2, backend3**: Define individual backend services with their respective URL patterns and server addresses.

## Step 3: Set Up Health Checks

Health checks are crucial for ensuring that only healthy backend servers receive traffic. Configure health check endpoints on your backend services if they don't already exist. For example:

- Backend1 should respond to `/health` at `http://backend1.example.com/health`.
- Similarly, set up health endpoints for Backend2 and Backend3.

Configure Wharfgate to use these health check endpoints during service registration by adding a `health_check_url` directive under each backend configuration block in the `wharfgate.conf`:

```plaintext
backend1 {
    url_pattern ^https?://backend1\.example\.com/
    server_address http://backend1.example.com
    health_check_url http://backend1.example.com/health
}
```

## Step 4: Restart Wharfgate

After saving your `wharfgate.conf`, restart the Wharfgate service to apply changes:

```bash
sudo systemctl restart wharfgate.service
```

or use the appropriate command for your system.

## Step 5: Test Configuration

Use a tool like `curl` or Postman to test that requests are correctly routed based on URL paths. For example, sending a request to `http://your-wharfgate-ip/abc` should route to Backend1 if its URL pattern matches the path prefix.

```bash
curl http://your-wharfgate-ip/abc
```

Ensure responses come from the correct backend service and include health check status updates in logs for monitoring purposes.

## Additional Notes

- **Security**: Consider implementing SSL/TLS termination at Wharfgate to secure traffic between clients and backends.
- **Scalability**: For high availability, deploy multiple instances of Wharfgate behind a load balancer or use an autoscaling mechanism based on request volume.
- **Monitoring**: Utilize logging and monitoring tools (e.g., Prometheus, Grafana) to track performance metrics such as response times and error rates.

By following these steps, you can effectively configure Wharfgate to route traffic to three backend services based on URL path prefixes. This setup not only optimizes resource utilization but also enhances the scalability and reliability of your application infrastructure.
