# Container deployment guide

DemandIQ can run as a single Streamlit container. This guide covers the reproducible local deployment path and the checks to perform before exposing the app beyond a trusted machine or network.

## Requirements

- Docker Engine
- Docker Compose v2 (the `docker compose` command)
- Port `8501` available on the host

No API key, database, or external service is required for the demo workflow.

## Start the application

From the repository root:

```bash
docker compose up --build --detach
```

Open the dashboard at `http://localhost:8501`. Check container state and logs with:

```bash
docker compose ps
docker compose logs --follow demandiq
```

The Streamlit health endpoint is `http://localhost:8501/_stcore/health`; it should return a successful HTTP response once startup completes. The image also declares a Docker health check for this endpoint. GitHub Actions builds the image, starts the container as its non-root application user, and probes this endpoint before the workflow can pass.

Stop the application with:

```bash
docker compose down
```

## Operational boundaries

- The current app is a single-user portfolio MVP. It does not provide user authentication, role-based access, or tenant isolation.
- Uploaded CSV data and analysis state are handled by the running Streamlit process; there is no database or durable upload storage. Restarting or replacing the container does not preserve an uploaded session.
- The Compose file publishes port `8501` on the host. Keep the service on a trusted local network unless access controls are added.
- Do not expose the dashboard directly to the public internet. A public deployment needs TLS, authentication, network ingress restrictions, resource limits, monitoring, and an explicit data-retention policy.
- Forecasts and replenishment quantities are decision-support estimates. Validate stock, open purchase orders, pack sizes, supplier constraints, and demand assumptions before acting on them.

## Moving to a container platform

The Dockerfile is the portable deployment artifact. For a hosted container platform, build and publish the image to a registry, configure the service to listen on port `8501`, and use `/_stcore/health` as the HTTP health-check path. Configure TLS and authentication at the platform or reverse-proxy layer before making the service reachable outside a trusted environment. Keep credentials and platform-specific configuration outside the image.

A successful local container smoke test proves that the image starts and serves its health endpoint; it does not by itself establish production security, availability, scalability, or forecast validity.
