# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM rust:1.86-bookworm@sha256:300ec56abce8cc9448ddea2172747d048ed902a3090e6b57babb2bf19f754081 AS builder

ARG VCS_REF=unknown

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN MARIAN_MLX_BUILD_GIT_SHA="$VCS_REF" \
    cargo build --locked --release -p marian-server --features cpu && \
    install -D -m 0755 target/release/marian-mlx-server /out/bin/marian-mlx-server

FROM debian:bookworm-slim@sha256:60eac759739651111db372c07be67863818726f754804b8707c90979bda511df AS runtime

ARG VERSION=0.2.0
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Marian MLX CPU" \
      org.opencontainers.image.description="Local Marian translation service using the pure-Rust Q8 CPU backend" \
      org.opencontainers.image.source="https://github.com/malusama/marian-mlx" \
      org.opencontainers.image.url="https://github.com/malusama/marian-mlx" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.revision="$VCS_REF"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gzip util-linux && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 65532 marian-mlx && \
    useradd --uid 65532 --gid 65532 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin marian-mlx && \
    install -d -o 65532 -g 65532 /models /usr/share/licenses/marian-mlx

COPY --from=builder /out/bin/ /usr/local/bin/
COPY --chmod=0755 docker/prepare-model.sh /usr/local/bin/marian-mlx-prepare-model
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/marian-mlx-entrypoint
COPY LICENSE LICENSE-APACHE-2.0 THIRD_PARTY_NOTICES.md /usr/share/licenses/marian-mlx/

ENV MARIAN_MLX_BACKEND=cpu \
    MARIAN_MLX_BIND=0.0.0.0:3000 \
    MARIAN_MLX_MODEL_DIR=/models/en-zh

USER 65532:65532
VOLUME ["/models"]
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/marian-mlx-entrypoint"]
CMD []
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 \
  CMD curl -fsS http://127.0.0.1:3000/readyz || exit 1
