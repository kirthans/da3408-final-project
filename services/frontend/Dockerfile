FROM nginxinc/nginx-unprivileged:1.28.0-alpine
USER root
COPY services/frontend/nginx.conf /etc/nginx/nginx.conf
COPY services/frontend/site /usr/share/nginx/html
COPY services/frontend/docker-entrypoint.d/40-elo-config.sh /docker-entrypoint.d/40-elo-config.sh
RUN chmod +x /docker-entrypoint.d/40-elo-config.sh \
    && chown -R 101:101 /usr/share/nginx/html /etc/nginx/conf.d
USER 101
EXPOSE 8080 8443
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
