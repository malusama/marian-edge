# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM rust:1.86-bookworm@sha256:300ec56abce8cc9448ddea2172747d048ed902a3090e6b57babb2bf19f754081 AS builder

ARG TARGETARCH
ARG BERGAMOT_REV=9271618ebbdc5d21ac4dc4df9e72beb7ce644774
ARG VCS_REF=unknown
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git libopenblas-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Marian generates a revision header from Git and checks submodules during
# CMake. Keep the metadata, but disable updates after fetching the exact build set.
RUN git clone --filter=blob:none https://github.com/browsermt/bergamot-translator.git bergamot && \
    cd bergamot && \
    git checkout "$BERGAMOT_REV" && \
    git submodule update --init --depth 1 \
      3rd_party/marian-dev 3rd_party/ssplit-cpp && \
    git -C 3rd_party/marian-dev submodule update --init --depth 1 \
      src/3rd_party/intgemm src/3rd_party/ruy \
      src/3rd_party/sentencepiece src/3rd_party/simd_utils && \
    git -C 3rd_party/marian-dev/src/3rd_party/ruy submodule update \
      --init --depth 1 third_party/cpuinfo && \
    git -C 3rd_party/marian-dev config -f .gitmodules \
      --get-regexp '^submodule\..*\.path$' | \
      while read -r key submodule_path; do \
        name=${key#submodule.}; \
        name=${name%.path}; \
        git -C 3rd_party/marian-dev config "submodule.${name}.update" none; \
      done

COPY crates/marian-bergamot/native /src/crates/marian-bergamot/native
RUN case "$TARGETARCH" in \
      arm64) arch=armv8-a ;; \
      amd64) arch=x86-64 ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac && \
    cmake -S /src/crates/marian-bergamot/native -B /build/worker \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/out \
      -DBERGAMOT_SOURCE=/build/bergamot \
      -DBUILD_ARCH="$arch" \
      -DUSE_MKL=OFF \
      -DGIT_SUBMODULE=OFF \
      -DSSPLIT_USE_INTERNAL_PCRE2=ON \
      -DCOMPILE_TESTS=OFF \
      -DCOMPILE_UNIT_TESTS=OFF && \
    cmake --build /build/worker --target marian-mlx-bergamot-worker --parallel "$(nproc)" && \
    cmake --install /build/worker --strip

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN MARIAN_MLX_BUILD_GIT_SHA="$VCS_REF" \
    cargo build --locked --release -p marian-server --features bergamot && \
    install -D -m 0755 target/release/marian-mlx-server /out/bin/marian-mlx-server && \
    install -D -m 0644 /build/bergamot/LICENSE /out/licenses/bergamot-translator/MPL-2.0.txt && \
    install -D -m 0644 /build/bergamot/3rd_party/marian-dev/LICENSE.md /out/licenses/marian/LICENSE.md && \
    install -D -m 0644 /build/bergamot/3rd_party/ssplit-cpp/LICENSE.md /out/licenses/ssplit/LICENSE.md && \
    install -D -m 0644 /build/bergamot/3rd_party/marian-dev/src/3rd_party/sentencepiece/LICENSE /out/licenses/sentencepiece/LICENSE && \
    install -D -m 0644 /build/bergamot/3rd_party/marian-dev/src/3rd_party/ruy/LICENSE /out/licenses/ruy/LICENSE && \
    install -D -m 0644 /build/bergamot/3rd_party/marian-dev/src/3rd_party/ruy/third_party/cpuinfo/LICENSE /out/licenses/cpuinfo/LICENSE && \
    install -D -m 0644 /build/bergamot/3rd_party/marian-dev/src/3rd_party/ruy/third_party/cpuinfo/deps/clog/LICENSE /out/licenses/clog/LICENSE && \
    install -D -m 0644 /build/bergamot/3rd_party/marian-dev/src/3rd_party/intgemm/LICENSE /out/licenses/intgemm/LICENSE && \
    install -D -m 0644 /usr/share/doc/libopenblas-dev/copyright /out/licenses/openblas/copyright && \
    install -D -m 0644 /build/bergamot/3rd_party/ssplit-cpp/src/3rd-party/pcre2-10.39/COPYING /out/licenses/pcre2/COPYING && \
    printf '%s\n' "$BERGAMOT_REV" > /out/licenses/bergamot-translator/SOURCE-REVISION

FROM debian:bookworm-slim@sha256:60eac759739651111db372c07be67863818726f754804b8707c90979bda511df AS runtime

ARG VERSION=0.1.1
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Marian MLX CPU" \
      org.opencontainers.image.description="Local Marian translation service using Bergamot on Linux CPU" \
      org.opencontainers.image.source="https://github.com/malusama/marian-mlx" \
      org.opencontainers.image.url="https://github.com/malusama/marian-mlx" \
      org.opencontainers.image.licenses="MIT AND MPL-2.0" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.revision="$VCS_REF"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libstdc++6 util-linux && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 65532 marian-mlx && \
    useradd --uid 65532 --gid 65532 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin marian-mlx && \
    install -d -o 65532 -g 65532 /models /usr/share/licenses/marian-mlx

COPY --from=builder /out/bin/ /usr/local/bin/
COPY --from=builder /out/licenses/ /usr/share/licenses/marian-mlx/
COPY --chmod=0755 docker/prepare-model.sh /usr/local/bin/marian-mlx-prepare-model
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/marian-mlx-entrypoint
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/share/licenses/marian-mlx/

USER 65532:65532
VOLUME ["/models"]
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/marian-mlx-entrypoint"]
CMD ["--backend", "bergamot", "--bind", "0.0.0.0:3000", "--model-dir", "/models/en-zh"]
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 \
  CMD curl -fsS http://127.0.0.1:3000/readyz || exit 1
