# Immersive Translate setup

Marian MLX implements Immersive Translate's Custom API at `/imme`. Readiness,
identity, translation, and extension URLs must all share one service origin:
only the path changes.

| Deployment | Service listener | Client service origin | Change the client port |
|---|---|---|---|
| native macOS installer | `127.0.0.1:3000` by default | `http://127.0.0.1:3000` | install with `MARIAN_MLX_PORT=3100`; use 3100 everywhere afterward |
| native source build | `127.0.0.1:3000` by default | `http://127.0.0.1:3000` | use `--bind 127.0.0.1:3100` or `MARIAN_MLX_BIND=127.0.0.1:3100` |
| Docker Compose | container `0.0.0.0:3000`; host loopback 3000 by default | `http://127.0.0.1:3000` | use `MARIAN_MLX_HOST_PORT=3100`; the container remains on 3000 |
| direct Docker | container `0.0.0.0:3000` | chosen host mapping | publish `127.0.0.1:3100:3000`; use host port 3100 |

For example, this complete native setup uses port 3100 consistently:

```sh
PORT=3100
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.6.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.6.0 MARIAN_MLX_PORT="$PORT" sh

SERVICE_ORIGIN="http://127.0.0.1:$PORT"
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
# Enter http://127.0.0.1:3100/imme in Immersive Translate.
```

The equivalent Compose setup is:

```sh
MARIAN_MLX_HOST_PORT=3100 docker compose up -d
curl -fsS http://127.0.0.1:3100/readyz
# Enter http://127.0.0.1:3100/imme in Immersive Translate.
```

## Extension settings

The following steps use the default service origin
`http://127.0.0.1:3000`. Substitute the actual host port if it was changed.

1. Confirm `http://127.0.0.1:3000/readyz` returns HTTP 200 and inspect
   `http://127.0.0.1:3000/info`.
2. Open **Immersive Translate > Options > Developer settings** and enable beta
   testing features.
3. Open **Options > General**, choose **Custom API** as the translation
   service, and enter `http://127.0.0.1:3000/imme` as the API URL.
4. Set the page/source language to **English** and the target language to
   **Simplified Chinese** (`zh-CN`).
5. Translate a short English page first. Keep the extension defaults unless
   the service reports overload or timeouts.

Chinese UI labels can vary slightly by extension version: “开发者设置 / Beta
测试功能”, “基本设置 / 翻译服务 / 自定义 API”. The request and response fields
below follow the [official Immersive Translate Custom API
contract](https://immersivetranslate.com/docs/services/custom/).

The server normalizes `en-US`, `en_US`, `zh-CN`, and `zh-Hans` to the model's
primary codes. `auto` uses a small English/CJK detector; explicitly choosing
English is more predictable because this release has only the `en -> zh`
model.

## Verify the exact extension contract

Set `SERVICE_ORIGIN` to the origin actually used by the deployment:

```sh
SERVICE_ORIGIN=${SERVICE_ORIGIN:-http://127.0.0.1:3000}
curl -fsS "$SERVICE_ORIGIN/imme" \
  -H 'content-type: application/json' \
  -d '{
    "source_lang":"en-US",
    "target_lang":"zh-CN",
    "text_list":["Hello world.","This stays on your computer."]
  }'
```

The response shape is:

```json
{
  "translations": [
    {"detected_source_lang": "en", "text": "..."},
    {"detected_source_lang": "en", "text": "..."}
  ]
}
```

The two translation endpoints intentionally have different request shapes:

| Endpoint | Accepted fields | Output budget |
|---|---|---|
| `POST /imme` | optional `source_lang`, required `target_lang`, required `text_list` | the contract does not define `max_output_tokens`; each item starts with the default 512-token budget |
| `POST /translate` | required `text` and `to`, optional `from` and `max_output_tokens`; `source_lang`/`target_lang` are aliases for `from`/`to` | defaults to 512 and is clamped to 1-2,048 |

Do not send `/translate`'s `from`/`to` shape to `/imme`. The complete JSON
request body is limited to 64 KiB, including JSON syntax and every list item.
`text_list` may contain at most 256 nonempty items, and every text passed to
either endpoint must be nonempty. The output budget is a caller ceiling rather
than a promise to generate that many tokens; EOS and backend model/runtime
limits may stop generation earlier.

The Custom API reserves placeholders such as `{0}` and `<b0></b0>`. Current
model-backed FP32 CPU and Metal regressions preserve those placeholder strings
and their order, while an HTTP adapter regression covers the `/imme` shape.
This is compatibility evidence for those tested paths, not a general
translation-quality or Q8 claim.

## CORS

CORS is disabled by default. Extension host permissions are usually enough for
loopback access. If the extension explicitly reports a CORS failure, configure
its exact trusted origin with `MARIAN_MLX_CORS_ORIGIN`. Use `*` only for a
personal service that remains bound and published on loopback.

When CORS is enabled, the server allows GET, POST, and the `Content-Type`
header. Custom browser headers such as `Authorization` are not in the current
allowlist. Re-running the native installer without an explicit port preserves
the previously saved port:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/v0.6.0/scripts/install-macos.sh | \
  MARIAN_MLX_VERSION=v0.6.0 \
  MARIAN_MLX_CORS_ORIGIN='chrome-extension://TRUSTED_EXTENSION_ID' sh
```

## Troubleshooting

- `/readyz` returns 503: the model worker is not ready, is draining, or has
  stopped. Inspect `/info` and service logs. Queue saturation does not change
  `/readyz`.
- `/translate` or `/imme` returns `503` with `Retry-After: 1`: the bounded
  admission queue is full or shutting down. Retry with jitter or lower
  concurrency.
- `422 unsupported direction`: make the source English and target Simplified
  Chinese; Chinese-to-English is not included yet.
- Browser cannot connect: use the host's loopback address, not a
  container-only hostname. Confirm `/readyz` and `/imme` use the same host port;
  a Docker container still listens internally on 3000 even when the host
  publishes 3100.
- Translation differs from another local service: the native direct Metal
  backend uses FP32 weights while the pure-Rust CPU container uses Q8 weights;
  near-tie token choices can differ. The Q8 release gate requires the five
  golden translations but does not claim bit-for-bit FP32 equivalence.
- A long paragraph is returned as one list item: CPU and Metal both use the
  shared tokenizer-aware segmenter, translate bounded chunks, preserve
  separators such as newlines, and reassemble the result in order.
