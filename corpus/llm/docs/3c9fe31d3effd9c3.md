## Setting Up a Lightweight Reverse Proxy & Load Balancer with Wharfgate

Wharfgate is a powerful yet lightweight reverse proxy and load balancer written in Go. It's perfect for small to medium-sized deployments where you need robust routing, health checks, and SSL termination without the overhead of more complex solutions like Nginx or HAProxy. This guide will walk you through setting up Wharfgate to route traffic to three backend services based on URL path prefixes.

**Understanding the Goal:**

We’ll create a setup where Wharfgate acts as the single entry point for all incoming requests. Based on the requested URL prefix, it will intelligently forward those requests to one of three backend services:

* **Service A (http://service-a.example.com):**  Handles requests starting with `/api/v1`
* **Service B (http://service-b.example.com):** Handles requests starting with `/app`
* **Service C (http://service-c.example.com):** Handles all other requests


**Prerequisites:**

* **A Server:** You’ll need a server (e.g., Ubuntu, Debian, CentOS) to run Wharfgate on.  This server needs network connectivity to reach your backend services.
* **Docker & Docker Compose:** We'll use Docker and Docker Compose for easy deployment and configuration management. Ensure they are installed.
* **Basic Linux Command Line Knowledge:** Familiarity with navigating the command line is necessary.

**Step 1: Create a Wharfgate Configuration File (wharfgate.yml)**

This file defines the routing rules and other settings for Wharfgate. Here’s an example:

```yaml
# wharfgate.yml
server: "0.0.0.0"
port: 8080

log:
  level: INFO
  format: text

routes:
  - path: "/api/v1"
    service: service_a
    weight: 100  # Optional - for weighted load balancing

  - path: "/app"
    service: service_b
    weight: 100

  - path: "/"   # Catch-all route – important!
    service: service_c
    weight: 100

services:
  service_a:
    url: "http://service-a.example.com"
    healthy_check_path: "/health" # Define health check endpoint for service A
    timeout: 5s

  service_b:
    url: "http://service-b.example.com"
    healthy_check_path: "/health" # Define health check endpoint for service B
    timeout: 5s

  service_c:
    url: "http://service-c.example.com"
    healthy_check_path: "/health" # Define health check endpoint for service C
    timeout: 5s
```

**Explanation:**

* **`server`, `port`, `log`:**  These settings configure Wharfgate's basic operation.
* **`routes`:** This is the core of the configuration. Each route defines a matching URL prefix and the corresponding backend service to forward requests to.
    * `path`: The URL path to match (e.g., `/api/v1`).
    * `service`: The name of the service defined in the `services` section.
    * `weight`:  (Optional) An integer value for weighted load balancing. Routes with higher weights will receive more traffic.
* **`services`:** This section defines each backend service Wharfgate will use.
    * `url`: The URL of the backend service.
    * `healthy_check_path`: The path Wharfgate will use to check the health of the service (more on this below).
    * `timeout`:  The timeout for requests to the backend service (in seconds).


**Step 2: Deploying Wharfgate with Docker Compose**

Create a directory for your Wharfgate setup and place the `wharfgate.yml` file inside it. Then, create a `docker-compose.yml` file:

```yaml
version: "3.7"
services:
  wharfgate:
    image: jpmurphy/wharfgate:latest # Use the latest Wharfgate image
    ports:
      - "8080:8080"
    volumes:
      - ./wharfgate.yml:/usr/local/bin/wharfgate.yml
    command: wharfgate -config /usr/local/bin/wharfgate.yml

```

**Explanation:**

* **`version`**: Specifies the Docker Compose file version.
* **`wharfgate`**: Defines the Wharfgate service.
    * `image`: Uses the official Wharfgate image from Docker Hub.
    * `ports`:  Maps port 8080 on the host to port 8080 inside the Wharfgate container.
    * `volumes`: Mounts your `wharfgate.yml` file into the Wharfgate container, allowing it to read the configuration.
    * `command`: Runs the Wharfgate executable with your configuration file.

**Step 3: Start Wharfgate**

Navigate to the directory containing your `docker-compose.yml` file in your terminal and run:

```bash
docker-compose up -d
```

This command will download the Wharfgate image (if it's not already present) and start the Wharfgate container in detached mode (`-d`).

**Step 4:  Health Checks & Backend Service Configuration**

Crucially, each backend service needs a `/health` endpoint that Wharfgate can use to check its status. This endpoint should return a 200 OK response if the service is healthy and a different code (e.g., 503 Service Unavailable) if it's not.  For example:

* **service-a.example.com:** Implement a `/health` endpoint that checks database connectivity, etc.
* **service-b.example.com:** Similarly implement a `/health` endpoint.
* **service-c.example.com:** This might just be a placeholder or a simple health check.

**Important Note:** The `healthy_check_path` in the `wharfgate.yml` file *must* match the path of your service's health check endpoint.



**Testing:**

Once Wharfgate is running, you can access it by navigating to `http://localhost:8080` in your browser.  Try accessing different URLs – `/api/v1`, `/app`, and anything else – to verify that traffic is routed correctly to the appropriate backend service.


This guide provides a basic setup for Wharfgate. You can customize it further by adding more routes, configuring SSL termination, adjusting timeouts, and implementing more sophisticated health checks.  Wharfgate's documentation ([https://github.com/jpmurphy/wharfgate](https://github.com/jpmurphy/wharfgate)) offers detailed information on all its features and configuration options.
