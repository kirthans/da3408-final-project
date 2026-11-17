#!/bin/sh
set -eu
# Restrict configuration strings to URL characters before writing JavaScript.
api_base=${API_BASE_URL:-/api}
upstream=${API_UPSTREAM:-http://api:8000/}
case "$api_base" in *[!a-zA-Z0-9_./:~-]*) echo 'Invalid API_BASE_URL' >&2; exit 1;; esac
case "$upstream" in http://*|https://*) ;; *) echo 'API_UPSTREAM must be an HTTP URL' >&2; exit 1;; esac
case "$upstream" in *[!a-zA-Z0-9_./:~-]*) echo 'Invalid API_UPSTREAM' >&2; exit 1;; esac
upstream=${upstream%/}
mlflow=${MLFLOW_PORT:-5000}; grafana=${GRAFANA_PORT:-3000}; prometheus=${PROMETHEUS_PORT:-9090}
for port in "$mlflow" "$grafana" "$prometheus"; do
    case "$port" in ''|*[!0-9]*) echo 'Service ports must be numeric' >&2; exit 1;; esac
    [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || exit 1
done
printf 'window.ELO_CONFIG = {API_BASE_URL:"%s",MLFLOW_PORT:%s,GRAFANA_PORT:%s,PROMETHEUS_PORT:%s};\n' "$api_base" "$mlflow" "$grafana" "$prometheus" > /usr/share/nginx/html/config.js
# Runtime DNS lookup keeps the static app healthy even before the API exists.
cat > /etc/nginx/conf.d/default.conf <<CONF
server {
    listen 8080;
CONF
if [ -r /etc/nginx/certs/localhost.pem ] && [ -r /etc/nginx/certs/localhost-key.pem ]; then
    cat >> /etc/nginx/conf.d/default.conf <<'TLS'
    listen 8443 ssl;
    ssl_certificate /etc/nginx/certs/localhost.pem;
    ssl_certificate_key /etc/nginx/certs/localhost-key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
TLS
else
    echo 'WARNING: TLS certificates missing; HTTPS server disabled. HTTP remains available on 8080.' >&2
fi
cat >> /etc/nginx/conf.d/default.conf <<CONF
    root /usr/share/nginx/html;
    index index.html;
    client_max_body_size 2m;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'" always;
    resolver 127.0.0.11 valid=10s ipv6=off;
    set \$api_upstream "$upstream";
    location /api/ {
        rewrite ^/api/(.*) /\$1 break;
        proxy_pass \$api_upstream\$uri\$is_args\$args;
        proxy_read_timeout 60s;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location = /config.js { expires -1; }
    location /vendor/ { expires 30d; }
    location / { try_files \$uri \$uri/ =404; }
}
CONF
