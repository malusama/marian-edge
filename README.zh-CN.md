# Marian Edge

Marian Edge 是一个本地英译中 HTTP 服务：

- Apple Silicon Mac 使用原生 Metal 后端；
- Linux AMD64/ARM64 使用纯 Rust Q8 CPU 后端；
- 模型、分词、长文本分段和请求调度都在本机完成。

当前只提供英译中，解码器固定为 `beam=1`。旧的 `marian-mlx` 命令、环境变量
和模型格式名称仅作为升级兼容入口保留。

[English README](README.md)

## 选择运行方式

| 主机 | 推荐方式 | 计算设备 |
|---|---|---|
| Apple Silicon Mac，macOS 14+ | 原生安装 | Metal GPU |
| Linux AMD64/ARM64 | Docker Compose | Q8 CPU |
| Mac 上的 Docker Desktop | Docker Compose | Linux ARM CPU，不使用 Metal |
| macOS/Linux 开发环境 | 源码构建 | Metal 或 CPU |

Docker Desktop 无法把 macOS 的 Metal 设备传进 Linux 容器。在 Mac 上需要 GPU
推理时，请安装原生版本。

## macOS 原生安装

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/main/scripts/install-macos.sh | sh
```

固定安装 v0.7.0：

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 sh
```

安装器不需要 root。它会校验下载内容、在本机转换模型，并注册用户级
LaunchAgent。首次安装需要下载模型，预留至少 750 MB 可用空间。

常用命令：

```sh
~/.local/bin/marian-edgectl status
~/.local/bin/marian-edgectl verify
~/.local/bin/marian-edgectl logs
~/.local/bin/marian-edgectl update
~/.local/bin/marian-edgectl rollback
~/.local/bin/marian-edgectl uninstall          # 保留模型和缓存
~/.local/bin/marian-edgectl uninstall --purge  # 一并删除模型和缓存
```

完整的启动、停止和排错命令见[运维指南](docs/OPERATIONS.md)。

默认监听 `127.0.0.1:3000`。如果要使用 3100，请在安装时指定；后续升级会沿用
保存的端口：

```sh
PORT=3100
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 MARIAN_EDGE_PORT="$PORT" sh

SERVICE_ORIGIN="http://127.0.0.1:$PORT"
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

## Docker CPU

```sh
docker compose pull
docker compose up -d
docker compose ps
```

Compose 默认把容器的 3000 端口映射到宿主机 `127.0.0.1:3000`。改用宿主机
3100 时，容器内部仍然是 3000：

```sh
MARIAN_EDGE_HOST_PORT=3100 docker compose up -d
SERVICE_ORIGIN=http://127.0.0.1:3100
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

不使用 Compose：

```sh
docker run -d --name marian-edge --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-edge-models:/models \
  --read-only --tmpfs /tmp:size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/malusama/marian-edge:cpu-0.7.0
```

镜像不包含模型。首次启动会下载并校验固定版本的 Mozilla 英译中模型，之后从
Docker volume 复用。`MARIAN_EDGE_CPU_THREADS` 可设为 `1`、`2` 或 `4`；请按
实际机器和文本负载测试，不要把线程数直接当成 worker 数量。

## 端口规则

Marian Edge 只有一个 HTTP 监听地址。`/readyz`、`/info`、`/translate` 和
`/imme` 必须使用同一个主机和端口，`/imme` 没有单独的端口。

| 后端实际地址 | 沉浸式翻译 API 地址 |
|---|---|
| `http://127.0.0.1:3000` | `http://127.0.0.1:3000/imme` |
| `http://127.0.0.1:3100` | `http://127.0.0.1:3100/imme` |

例如，源码实例若用 `--bind 127.0.0.1:3100` 启动，就必须在沉浸式翻译里填写
`http://127.0.0.1:3100/imme`，不能照抄默认的 3000。

后面的示例统一使用 `SERVICE_ORIGIN`。请先把它设成当前后端的实际地址：

```sh
SERVICE_ORIGIN=http://127.0.0.1:3100  # 按实际监听端口修改
curl -fsS "$SERVICE_ORIGIN/readyz"
curl -fsS "$SERVICE_ORIGIN/info"
```

## 沉浸式翻译

1. 用上面的 `/readyz` 命令确认服务可用。
2. 打开“沉浸式翻译 -> 设置 -> 开发者设置”，启用 Beta 测试功能。
3. 在“基本设置 -> 翻译服务”中选择“自定义 API”。
4. API 地址填写 **`SERVICE_ORIGIN` 后接 `/imme`**。例如后端在 3100，就填写
   `http://127.0.0.1:3100/imme`。
5. 源语言选择英语，目标语言选择简体中文。

浏览器扩展通常可以直接访问本机地址。如果扩展明确报告 CORS 错误，再配置
`MARIAN_EDGE_CORS_ORIGIN`；不要为排错直接把服务暴露到公网。请求格式和常见
错误见[沉浸式翻译指南](docs/IMMERSIVE_TRANSLATE.md)。

## API

```sh
SERVICE_ORIGIN=${SERVICE_ORIGIN:-http://127.0.0.1:3000}
curl -fsS "$SERVICE_ORIGIN/translate" \
  -H 'content-type: application/json' \
  -d '{"text":"The weather is beautiful today.","from":"en-US","to":"zh-CN"}'
```

```json
{"text":"...","from":"en","to":"zh"}
```

| 接口 | 用途 |
|---|---|
| `POST /translate` | 翻译一段文本 |
| `POST /imme` | 沉浸式翻译的批量格式 |
| `POST /detect` | 简单的英语/CJK 判断 |
| `GET /livez` | 进程存活检查 |
| `GET /readyz` | 模型是否就绪 |
| `GET /health` | 旧客户端兼容接口 |
| `GET /info` | 版本、后端、设备和模型信息 |
| `GET /metrics` | Prometheus 指标 |

`/translate` 接受 `text`、`from`、`to` 和可选的 `max_output_tokens`；默认输出
预算为 512 token。`/imme` 接受 `source_lang`、`target_lang` 和 `text_list`，
每次最多 256 项。完整 JSON 请求体上限为 64 KiB。详细字段和错误码见
[沉浸式翻译指南](docs/IMMERSIVE_TRANSLATE.md)与[运维指南](docs/OPERATIONS.md)。

## 当前范围

- 支持 Apple Silicon direct Metal FP32，以及可选的 mixed-f16 权重存储。
- 支持 Linux AMD64/ARM64 Q8 CPU，以及使用 FP32 模型清单的纯 Rust CPU。
- 支持 SentencePiece、长文本分段、词表 shortlist 和动态批处理。
- 当前只包含英译中模型；`/detect` 不是通用语种识别。
- 当前解码器固定为 `beam=1`（贪心解码），尚未实现 `beam>1` 的束搜索。

beam search 的评估计划见[优化路线图](docs/OPTIMIZATION_ROADMAP.md)。历史性能数据
和测试条件见[性能记录](docs/BENCHMARKS.md)。

## 从源码构建

只测试 HTTP 层：

```sh
make check
cargo run -p marian-server -- --backend echo
```

CPU：

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features cpu
target/release/marian-edge-server --backend cpu --cpu-threads 4 \
  --model-dir models/enzh
```

Apple Silicon Metal：

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features metal
target/release/marian-edge-server --backend metal --model-dir models/enzh
```

源码结构和后端边界见[架构说明](docs/ARCHITECTURE.md)，参数见
[运维指南](docs/OPERATIONS.md)，贡献与发布流程见[CONTRIBUTING](CONTRIBUTING.md)。

## 安全与许可证

示例只监听本机回环地址。服务本身没有鉴权和 TLS，不应直接暴露到公网。安全
问题请按 [SECURITY](SECURITY.md) 私下报告。

服务代码与项目自带的 MSL kernel 使用 MIT 许可证。模型文件不随仓库或镜像
分发；下载脚本从上游获取并校验。详情见[第三方声明](THIRD_PARTY_NOTICES.md)。
