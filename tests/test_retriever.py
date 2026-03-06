# infra/nginx.conf
# ─────────────────────────────────────────────────────────────────────────────
# WHY NGINX (and not AWS ALB alone, or Traefik)?
#
# - AWS ALB: handles TLS and load balancing across instances, but doesn't do
#   per-instance rate limiting, request buffering, or connection keep-alive
#   tuning. We use both: ALB in front, Nginx per-instance.
#
# - Traefik: excellent for K8s service mesh. On a single EC2 with Docker
#   Compose, Nginx has a lower memory footprint and simpler config.
#
# - Nginx: 20 years of production hardening. Connection pooling to upstream
#   (keepalive 32) means uvicorn workers don't pay TCP handshake overhead on
#   every request.
# ─────────────────────────────────────────────────────────────────────────────

worker_processes auto;  # One worker per CPU core
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;          # Linux epoll: most efficient I/O multiplexer on EC2
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Log format includes upstream response time for latency monitoring
    log_format main '$remote_addr - $request_time $upstream_response_time '
                    '"$request" $status $body_bytes_sent';
    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    keepalive_timeout 65;
    gzip on;

    # Rate limiting: 100 requests/minute per IP
    # Why limit_req_zone on $binary_remote_addr?
    # $binary_remote_addr uses 4 bytes (vs 15 for $remote_addr string),
    # reducing memory usage of the shared zone by ~75%.
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;

    # Upstream: the FastAPI app running in Docker on port 8000
    # keepalive 32: maintain 32 persistent connections to uvicorn workers,
    # avoiding TCP handshake overhead per request (~1ms saved per request)
    upstream rag_app {
        server app:8000;
        keepalive 32;
    }

    server {
        listen 80;
        server_name _;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Health check endpoint — bypasses rate limiting for load balancer polls
        location /health {
            proxy_pass http://rag_app;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }

        # API endpoints
        location / {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://rag_app;
            proxy_http_version 1.1;
            proxy_set_header Connection "";           # Required for keepalive
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # Timeouts: CLIP embedding takes up to 500ms; agent call up to 30s
            proxy_read_timeout 60s;
            proxy_connect_timeout 5s;

            # Buffer settings: allow Nginx to buffer the full request body
            # before forwarding to uvicorn — prevents slow-client attacks
            proxy_request_buffering on;
            client_max_body_size 10M;  # Max image upload size
        }
    }
}