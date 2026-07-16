# Immersive Translate setup

Marian Edge serves the Immersive Translate Custom API at `/imme`. It does not
open a second listener for the extension: `/readyz`, `/info`, `/translate`, and
`/imme` all use the same service origin.

## Find the service address first

| How the backend was started | Service origin | Extension API URL |
|---|---|---|
| default macOS install or default Compose | `http://127.0.0.1:3000` | `http://127.0.0.1:3000/imme` |
| `--bind 127.0.0.1:3100` | `http://127.0.0.1:3100` | `http://127.0.0.1:3100/imme` |
| native install with `MARIAN_EDGE_PORT=3100` | `http://127.0.0.1:3100` | `http://127.0.0.1:3100/imme` |
| Compose with `MARIAN_EDGE_HOST_PORT=3100` | `http://127.0.0.1:3100` | `http://127.0.0.1:3100/imme` |

Container port 3000 is an internal detail. A client uses the published host
port. If readiness succeeds on 3100, the extension must also use 3100.

Set `SERVICE_ORIGIN` once and use it for every check:

```sh
SERVICE_ORIGIN=http://127.0.0.1:3100  # replace with the running backend
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

For a native installation, the saved port is available at
`~/.local/share/marian-edge/config/port`. For a foreground source build, use
the value passed to `--bind` or `MARIAN_EDGE_BIND`.

## Extension settings

1. Confirm `$SERVICE_ORIGIN/readyz` returns HTTP 200.
2. Open **Immersive Translate > Options > Developer settings** and enable beta
   testing features.
3. Under **Options > General**, select **Custom API**.
4. Enter the service origin followed by `/imme`. For example, a backend on
   3100 uses `http://127.0.0.1:3100/imme`.
5. Select English as the source and Simplified Chinese (`zh-CN`) as the target.

Chinese UI labels vary slightly by extension version. The relevant items are
usually “开发者设置 / Beta 测试功能” and “基本设置 / 翻译服务 / 自定义 API”.

## Test the payload

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

Response:

```json
{
  "translations": [
    {"detected_source_lang": "en", "text": "..."},
    {"detected_source_lang": "en", "text": "..."}
  ]
}
```

The server normalizes `en-US`, `en_US`, `zh-CN`, and `zh-Hans` to `en` and
`zh`. `auto` uses a small English/CJK heuristic; explicitly selecting English
is more predictable because this release contains only an `en -> zh` model.

## Request limits

`/imme` and `/translate` use different JSON shapes:

| Endpoint | Fields | Limits |
|---|---|---|
| `POST /imme` | optional `source_lang`, required `target_lang`, required `text_list` | at most 256 nonempty items; each item starts with a 512-token output budget |
| `POST /translate` | required `text` and `to`; optional `from` and `max_output_tokens` | output budget defaults to 512 and is clamped to 1-2,048 |

The complete JSON request body is limited to 64 KiB. EOS or a backend limit may
end generation before the output budget is reached.

Immersive Translate placeholders such as `{0}` and `<b0></b0>` are covered by
the FP32 CPU and Metal regression tests. That test does not establish general
translation quality or Q8 equivalence.

The payload follows Immersive Translate's
[Custom API documentation](https://immersivetranslate.com/docs/services/custom/).

## CORS

CORS is disabled by default. Extension host permissions are usually enough for
loopback access. If the extension reports a CORS error, configure its exact
trusted origin with `MARIAN_EDGE_CORS_ORIGIN`.

Use `*` only for a personal service that remains bound to loopback. The server
accepts GET, POST, and the `Content-Type` header when CORS is enabled; custom
headers such as `Authorization` are not allowed.

Re-running the native installer without a port argument keeps the saved port:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 \
  MARIAN_EDGE_CORS_ORIGIN='chrome-extension://TRUSTED_EXTENSION_ID' sh
```

## Troubleshooting

- Cannot connect: verify the exact `$SERVICE_ORIGIN/readyz` URL, then use the
  same origin with `/imme`. Do not use a container-only hostname.
- `/readyz` returns 503: the model is starting, draining, or stopped. Check
  `/info` and service logs.
- `/translate` or `/imme` returns 503 with `Retry-After: 1`: the request queue
  is full. Retry with jitter or reduce concurrency.
- `422 unsupported direction`: choose English as the source and Simplified
  Chinese as the target.
- A long paragraph returns as one list item: both backends split oversized text
  internally and then reassemble it, preserving separators such as newlines.
- CPU and Metal translations differ: the native backend uses FP32 weights while
  the published CPU image uses Q8 weights; close token scores can resolve
  differently.
