# syntax=docker/dockerfile:1
# TokenSlim — compressing reverse proxy image.
#
#   docker build -t tokenslim .
#   docker run --rm -p 8787:8787 tokenslim
#
# See docs/DOCKER.md for configuration and usage.

# --- Stage 1: build — install the package into an isolated prefix ------------
FROM python:3.12-slim AS build

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

# --- Stage 2: runtime — slim image, non-root user ----------------------------
FROM python:3.12-slim

COPY --from=build /install /usr/local

RUN useradd --create-home --shell /usr/sbin/nologin tokenslim
USER tokenslim

ENV PYTHONUNBUFFERED=1 \
    TOKENSLIM_PROXY_PORT=8787 \
    TOKENSLIM_UPSTREAM=https://api.openai.com

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s CMD \
    python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('TOKENSLIM_PROXY_PORT', '8787') + '/health', timeout=2)" \
    || exit 1

ENTRYPOINT ["tokenslim"]
CMD ["proxy"]
