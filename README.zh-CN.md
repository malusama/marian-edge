# Marian MLX

一个本地英译中服务。Apple Silicon 原生版由 Rust host 通过 `objc2-metal`
直接驱动 Metal，并在启动时编译内嵌的 MSL compute kernel，不再链接 MLX，
也没有 C++ 推理桥。Linux Docker 和其他跨平台构建使用纯 Rust CPU 引擎，
完整支持 Q8 与 FP32 Transformer/SSRU 计算图、词表 shortlist 和贪心解码。

项目名为兼容性继续保留。分词、长文本分段、调度、模型加载和两套推理 host
都由 Rust 实现；仓库已经删除此前的 Bergamot/C++ runtime。

[English README](README.md)

## 先选对运行方式

| 主机 | 运行方式 | 实际计算设备 | 启动方式 |
|---|---|---|---|
| Apple Silicon Mac，macOS 14+ | 原生单一可执行文件 | 直接使用 Metal GPU | 下方一键安装 |
| Linux AMD64 | Docker | 纯 Rust Q8 CPU | `docker compose up -d` |
| Linux ARM64 | Docker | 纯 Rust Q8 CPU | `docker compose up -d` |
| Mac 上的 Docker Desktop | Linux ARM 容器 | 纯 Rust Q8 CPU，**不是 Metal** | `docker compose up -d` |
| macOS 或 Linux 源码构建 | 原生可执行文件 | 纯 Rust Q8 或 FP32 CPU | `--features cpu -- --backend cpu` |

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

CPU 模型由单一 owner 持有，因此增加计算线程不会加载额外模型副本。并发 HTTP
请求仍会先合并成批次再推理。`MARIAN_MLX_CPU_THREADS` 支持 1、2 或 4，同时
控制 FP32 矩阵乘法以及 Q8 的 rten/精确 AVX2 row-parallel kernel：

```sh
MARIAN_MLX_CPU_THREADS=2 docker compose up -d
```

无论设置几个计算线程，模型都仍只有一个 owner。增加内部计算并行度前应在
目标机器和真实流量上测量。

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
支持英译中。`max_output_tokens` 可用于 direct Metal 和纯 Rust CPU 后端。

## 当前支持范围

| 能力 | 状态 |
|---|---|
| Apple Silicon / direct Metal FP32 | 支持 |
| Linux AMD64 / 纯 Rust Q8 CPU | 支持 |
| Linux ARM64 / 纯 Rust Q8 CPU | 支持，已在 ARM64 实机测试 |
| 跨平台纯 Rust FP32 CPU | 使用 FP32 manifest 时支持 |
| 纯 Rust Q8 Transformer/SSRU 计算图 | 支持；dense 权重保持量化 |
| 纯 Rust SentencePiece 与长文本分段 | 支持 |
| 英译中 `base-memory` 模型 | 支持 |
| Transformer 编码器、SSRU 贪心解码、词表 shortlist | 支持 |
| 有界排队、按形状动态 micro-batch | 支持 |
| 更多语向 | 暂不支持 |
| beam > 1 | 暂不支持；当前发布模型为 beam 1 |
| 通用语种识别 | 不包含 |

`/imme` 中每一项仍对应一个输出项。Metal 单条上限为 4096 token。CPU 引擎会
把每个推理 chunk 控制在 255 个源 piece 加 EOS 以内，并约束 batch 的 padding
attention 工作量，因为 encoder attention 是平方复杂度。更长的文本会在
tokenizer 感知的句子边界分段，之后按原顺序拼回并保留包括换行在内的分隔符。

Q8 后端对 5 条 release golden 全部精确一致。在 200 条差分语料中，与已退役
CPU 参考实现有 164 条输出精确一致；其余多为接近分数下的 token 选择差异，
因此这里不声称逐 token 完全等价。测试中的 80 句重复长文本与换行样例和已
退役长文本基线一致。

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
      +--> Rust host --> 内嵌 MSL --> Metal GPU (Mac 原生)
      |
      +--> 纯 Rust Transformer/SSRU --> Q8/FP32 CPU（跨平台）
```

后端状态始终由同一个 owner 线程持有。CPU dense 计算会根据模型 manifest
使用保持量化的 Q8 权重或 FP32 权重。架构维护说明见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
后续以 M1 为优先级的实测优化任务见
[优化路线图](docs/OPTIMIZATION_ROADMAP.md)。

## 从源码构建

只检查 Rust 服务层不需要模型或 Metal：

```sh
make check
cargo run -p marian-server -- --backend echo
```

`echo` 只用于 API 开发，生产后端失败时绝不会静默回退到它。

纯 Rust CPU 后端根据模型 manifest 自动选择 Q8 或 FP32。Linux 的 `auto` 和
已发布的 `:cpu` 镜像使用这个后端；原生 Metal 路径使用转换后的 FP32 模型。

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features cpu
target/release/marian-mlx-server --backend cpu --cpu-threads 4 \
  --model-dir models/enzh
```

`--backend cpu-q8`、`--backend cpu-fp32` 和 `--backend rust` 都是 `cpu` 的
兼容别名，实际使用 Q8 还是 FP32 仍由 manifest 决定。计算线程数会在推理
启动前固定。

Mac 原生构建需要 macOS SDK/Command Line Tools、Rust 1.86 和 `uv`：

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features metal
target/release/marian-mlx-server --backend metal --model-dir models/enzh
```

旧自动化仍可使用 `mlx` feature 或 `--backend mlx`，它们现在只是 direct Metal
实现的兼容 alias。MSL 源码内嵌在可执行文件里，并在进程启动时通过 Metal
framework 编译，因此不再需要 `libmlx.dylib`、外置 `.metallib`、MLX submodule
或 `scripts/build-mlx.sh`。Mac Release 只发布一个可执行文件；模型目录仍由用户
单独下载和准备。

模型准备脚本固定 Python 与转换依赖版本，并用 SHA-256 校验模型文件。模型、
转换后权重、缓存和构建产物不会进入 Git。

## 性能

此前公开的 M1 数字来自已经移除的 MLX 后端，不能当作 direct Metal 的成绩。
新后端需要重新完成一致性、吞吐、延迟、内存与 Metal Trace 测量后，才能发布
当前性能结论。历史基线和复测要求见 [BENCHMARKS](docs/BENCHMARKS.md)。

## 安全、维护与许可证

示例默认只监听 loopback。服务本身没有鉴权和 TLS，不应直接暴露到公网。运维
命令与健康检查见 [OPERATIONS](docs/OPERATIONS.md)，贡献方式见
[CONTRIBUTING](CONTRIBUTING.md)，安全问题按 [SECURITY](SECURITY.md) 私下报告。

Rust 服务与项目自带的 MSL kernel 采用 MIT。纯 Rust SentencePiece 推理由
Apache-2.0 许可的 `sentencepiece-rust` crate 提供。CPU kernel 使用的 Rust
crate 记录在 `Cargo.lock`。本项目不分发模型文件，脚本只在用户主动运行时从
上游下载。
完整依赖与许可证见 [第三方声明](THIRD_PARTY_NOTICES.md)。

Firefox 和 Mozilla 是 Mozilla Foundation 在美国及其他国家/地区的商标。
