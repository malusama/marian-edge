# Immersive Translate setup

Marian MLX implements Immersive Translate's Custom API request and response
shape at:

```text
http://127.0.0.1:3000/imme
```

## Extension settings

1. Start the service and confirm `http://127.0.0.1:3000/readyz` returns `200`.
2. Open **Immersive Translate > Options > Developer settings** and enable beta
   testing features.
3. Open **Options > General**, choose **Custom API** as the translation
   service, and enter `http://127.0.0.1:3000/imme` as the API URL.
4. Set the page/source language to **English** and the target language to
   **Simplified Chinese** (`zh-CN`).
5. Translate a short English page first. Keep the extension defaults unless
   the service reports overload or timeouts.

Chinese UI labels can vary slightly by extension version: “开发者设置 / Beta
测试功能”, “基本设置 / 翻译服务 / 自定义 API”. The official Custom API guide
is <https://immersivetranslate.com/docs/services/custom/>.

The server normalizes `en-US`, `en_US`, `zh-CN`, and `zh-Hans` to the model's
primary codes. `auto` uses a small English/CJK detector; explicitly choosing
English is more predictable because this release has only the `en -> zh`
model.

The API is CORS-disabled by default. Extension host permissions are usually
enough for localhost access. If the extension explicitly reports a CORS
failure, configure its exact origin with `MARIAN_MLX_CORS_ORIGIN` or use `*`
only for a loopback-only personal service.

## Verify the exact extension contract

```sh
curl -fsS http://127.0.0.1:3000/imme \
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

## Troubleshooting

- “Service unavailable”: open `/readyz`; a `503` means the model worker is not
  ready or the bounded queue is saturated.
- `422 unsupported direction`: make the source English and target Simplified
  Chinese; Chinese-to-English is not included yet.
- Browser cannot connect: use `127.0.0.1`, not a container-only hostname, and
  confirm the host publishes port 3000.
- Translation differs from another local service: the native direct Metal
  backend uses FP32 weights while the pure-Rust CPU container uses Q8 weights;
  near-tie token choices can differ. The Q8 release gate requires the five
  golden translations but does not claim bit-for-bit FP32 equivalence.
- A long paragraph is returned as one list item: the CPU backend segments it at
  tokenizer-aware sentence boundaries, translates bounded chunks, preserves
  separators such as newlines, and reassembles the result in order.
