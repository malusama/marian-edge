# Marian MLX

一个本地英译中服务。HTTP 与并发调度层使用 Rust；Apple
Silicon 原生版通过 MLX/Metal 在 Mac GPU 上推理；Linux Docker 版通过官方
Bergamot 在 CPU 上推理，并原生支持 ARM64 的 Ruy/NEON。

[English README](README.md)

## 先选对运行方式

| 主机 | 运行方式 | 实际计算设备 | 启动方式 |
|---|---|---|---|
| Apple Silicon Mac，macOS 14+ | 原生安装 | MLX / Metal GPU | 下方一键安装 |
| Linux AMD64 | Docker | Bergamot CPU | `docker compose up -d` |
| Linux ARM64 | Docker | Bergamot Ruy + NEON CPU | `docker compose up -d` |
| Mac 上的 Docker Desktop | Linux ARM 容器 | **CPU，不是 Metal** | `docker compose up -d` |

Docker 在 Mac 上运行的是 Linux 虚拟机，不能把 macOS Metal GPU 透传给容器。
想用 Mac GPU，必须安装原生版。

## Mac GPU 一键安装

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/main/scripts/install-macos.sh | sh
```

安装器不需要 root，会校验 Release 和所有模型文件；模型由用户机器直接从
Mozilla 存储下载并在本机转换，然后安装用户级 LaunchAgent，默认监听
`127.0.0.1:3000`。首次安装需要约 750 MB 可用空间。

```sh
~/.local/bin/marian-mlxctl status
~/.local/bin/marian-mlxctl verify
~/.local/bin/marian-mlxctl logs
~/.local/bin/marian-mlxctl restart
~/.local/bin/marian-mlxctl update
~/.local/bin/marian-mlxctl uninstall          # 保留模型与缓存
~/.local/bin/marian-mlxctl uninstall --purge  # 全部删除
```

用 `MARIAN_MLX_PORT=3100` 可改端口。安装器不会抢占无关进程的端口；新版本
未通过 `/readyz` 时会自动回滚。

## Docker CPU 一键启动

```sh
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/info
```

也可以直接运行：

```sh
docker run -d --name marian-mlx --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-mlx-models:/models \
  ghcr.io/malusama/marian-mlx:cpu
```

镜像是 AMD64/ARM64 多架构、非 root、CPU-only。镜像不内置模型；第一次启动
时会从 Mozilla 存储直接下载固定的英译中模型，校验压缩前后 SHA-256 后写入
named volume，后续启动直接复用。

CPU 默认只启用 1 个翻译 worker。即使只有 1 个 worker，并发 HTTP 请求仍会先
合并成批次；在我们的 ARM64 真模型冒烟测试中，峰值内存约为 0.4-0.5 GB。
如果机器内存充足且长期有并发流量，可以显式尝试 2 个 worker；同一测试约为
0.7-0.8 GB，因为每个活跃 worker 都会持有独立的模型工作区。

```sh
MARIAN_MLX_CPU_THREADS=2 docker compose up -d
```

小型 ARM 设备、NAS 和 Docker Desktop 建议从 1 开始。worker 越多并不一定
越快，它是吞吐量与内存之间的调优项，应以真实请求压测结果为准。

## 沉浸式翻译设置

1. 先确认 `http://127.0.0.1:3000/readyz` 返回 200。
2. 打开“沉浸式翻译 -> 设置 -> 开发者设置”，启用 Beta 测试功能。
3. 在“基本设置 / General -> 翻译服务”中选择“自定义 API”。
4. API 地址填写 `http://127.0.0.1:3000/imme`。
5. 源语言选择英语，目标语言选择简体中文。

服务默认不开放 CORS。扩展具备 localhost 权限时通常不需要；如果扩展明确报告
CORS 错误，可在只绑定本机的前提下重新安装并显式启用：

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-mlx/main/scripts/install-macos.sh | \
  MARIAN_MLX_CORS_ORIGIN='*' sh
```

Docker 可在 `compose.yaml` 的 `environment` 中增加
`MARIAN_MLX_CORS_ORIGIN: "*"`。更完整的请求样例与排错见
[沉浸式翻译文档](docs/IMMERSIVE_TRANSLATE.md)。

## API

```sh
curl -fsS http://127.0.0.1:3000/translate \
  -H 'content-type: application/json' \
  -d '{"text":"The weather is beautiful today.","from":"en-US","to":"zh-CN"}'
```

```json
{"text":"...","from":"en","to":"zh"}
```

| 接口 | 用途 |
|---|---|
| `POST /translate` | 单条翻译 |
| `POST /imme` | 沉浸式翻译兼容批量格式 |
| `POST /detect` | 简单的英语/CJK 启发式检测，不是通用语种识别 |
| `GET /livez` | 进程与事件循环存活 |
| `GET /readyz` | 模型已加载且调度器可接收请求 |
| `GET /health` | 旧客户端兼容接口 |
| `GET /info` | 版本、提交、后端、设备、精度、模型、运行时间 |
| `GET /metrics` | Prometheus 指标 |

`en-US`、`en_US`、`zh-CN`、`zh-Hans` 会归一化为 `en` 和 `zh`。当前版本只
支持英译中。`max_output_tokens` 仅 MLX 后端支持；Bergamot 使用模型固定的
`max-length-factor`，非默认值会明确报错。

## 当前支持范围

| 能力 | 状态 |
|---|---|
| Apple Silicon / MLX v0.32 / Metal GPU | 支持，已用 Metal Trace 验证 |
| Linux AMD64 / Bergamot int8 CPU | 支持 |
| Linux ARM64 / Ruy + NEON CPU | 支持，已在 ARM64 实机测试 |
| 英译中 `base-memory` 模型 | 支持 |
| Transformer 编码器、SSRU 贪心解码、词表 shortlist | 支持 |
| 有界排队、按形状动态 micro-batch | 支持 |
| 更多语向 | 暂不支持 |
| beam > 1 | 暂不支持；当前发布模型为 beam 1 |
| 通用语种识别 | 不包含 |

## 并发模型

```text
并行 HTTP 请求
      |
      v
Rust / Axum 校验
      |
      v
有界队列 -- 满 --> 503 + Retry-After
      |
      v
按语向和长度归桶的 micro-batch
      |
      v
单一后端 owner 线程
      |
      +--> MLX 图 --> Metal GPU             (Mac 原生)
      |
      +--> 常驻 Bergamot worker --> Ruy CPU (Linux Docker)
```

GPU 状态始终由同一个线程持有；Bergamot worker 会跨请求复用。架构维护说明见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 从源码构建

只检查 Rust 服务层不需要模型或 MLX：

```sh
make check
cargo run -p marian-server -- --backend echo
```

`echo` 只用于 API 开发，生产后端失败时绝不会静默回退到它。

Mac 原生构建需要 Xcode Metal Toolchain、CMake 3.25+、Rust 1.86 和 `uv`：

```sh
xcodebuild -downloadComponent MetalToolchain
git submodule update --init --recursive
scripts/build-mlx.sh
scripts/prepare-enzh-model.sh
scripts/build-release.sh
MARIAN_MLX_METALLIB="$PWD/build/mlx-install/lib/mlx.metallib" \
  target/release/marian-mlx-server --backend mlx --model-dir models/enzh
```

脚本固定 Python 与转换依赖版本，并用 SHA-256 校验 MLX CMake 依赖和模型文件。
模型、转换后权重、缓存和构建产物不会进入 Git。

## 性能

已记录的 M1 短句测试中，FP32 MLX 在并发 32 时达到 536.04 req/s；Bergamot
int8 容器使用默认单 worker 时为 95.61 req/s，并已用 Instruments 捕获 Metal
命令执行。这只代表一台机器和一种句长，完整方法见
[BENCHMARKS](docs/BENCHMARKS.md)。

## 安全、维护与许可证

示例默认只监听 loopback。服务本身没有鉴权和 TLS，不应直接暴露到公网。运维
命令与健康检查见 [OPERATIONS](docs/OPERATIONS.md)，贡献方式见
[CONTRIBUTING](CONTRIBUTING.md)，安全问题按 [SECURITY](SECURITY.md) 私下报告。

Rust/MLX 服务代码采用 MIT。MLX 是 MIT。Docker CPU 后端使用固定版本的
官方 Bergamot（MPL-2.0）。本项目不分发模型文件，脚本只在用户主动运行时从
上游下载。完整依赖与许可证见 [第三方声明](THIRD_PARTY_NOTICES.md)。

Firefox 和 Mozilla 是 Mozilla Foundation 在美国及其他国家/地区的商标。
