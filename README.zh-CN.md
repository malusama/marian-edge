# Marian Edge

一个本地英译中服务。Apple Silicon 原生版由 Rust host 通过 `objc2-metal`
直接驱动 Metal，并在启动时编译内嵌的 MSL compute kernel，不再链接 MLX，
其中包括融合在线 softmax 的 FlashAttention-style kernel；也没有 C++ 推理桥。
Linux Docker 和其他跨平台构建使用纯 Rust CPU 引擎，
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
  https://raw.githubusercontent.com/malusama/marian-edge/main/scripts/install-macos.sh | sh
```

固定版本安装：

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 sh
```

安装器不需要 root，会校验 Release 和所有模型文件；模型由用户机器直接从
Mozilla 存储下载并在本机转换，然后安装用户级 LaunchAgent，默认监听
`127.0.0.1:3000`。安装器每次运行都要求至少 750 MB 可用空间；首次安装会使用
其中大部分空间，并因下载和本地转换模型耗时更久。`v0.1.1` 仍作为最后一个
历史 MLX/Bergamot 版本保留，但其 runtime 布局不兼容 `v0.2.0` 的 direct Metal
bundle 契约；需要在两种布局间回滚时请使用 `v0.2.1` 或更高版本。

现有 `marian-mlx` 安装可以直接升级。安装器会发现原来的数据与状态目录，等新
服务通过就绪检查后再替换旧 LaunchAgent，继续接受 `MARIAN_MLX_*` 参数，并
保留 `marian-mlxctl` 作为 `marian-edgectl` 的兼容别名。

```sh
~/.local/bin/marian-edgectl status
~/.local/bin/marian-edgectl verify
~/.local/bin/marian-edgectl logs
~/.local/bin/marian-edgectl restart
~/.local/bin/marian-edgectl stop
~/.local/bin/marian-edgectl start
~/.local/bin/marian-edgectl update
~/.local/bin/marian-edgectl rollback
~/.local/bin/marian-edgectl uninstall          # 保留模型与缓存
~/.local/bin/marian-edgectl uninstall --purge  # 全部删除
```

改用 3100 时，要在管道接收端设置端口，之后所有接口（包括沉浸式翻译）都使用
同一个端口：

```sh
PORT=3100
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 MARIAN_EDGE_PORT="$PORT" sh
curl -fsS "http://127.0.0.1:$PORT/readyz"
curl -fsS "http://127.0.0.1:$PORT/info"
# 沉浸式翻译地址：http://127.0.0.1:3100/imme
```

后续升级会保留已保存的端口。安装器不会抢占无关进程的端口；新版本未通过
`/readyz` 时会自动回滚。

## Docker CPU 一键启动

升级时先显式拉取，避免复用本地旧的 `:cpu` 镜像：

```sh
docker compose pull
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/info
```

Compose 内部服务始终使用 3000。若宿主机要发布到 3100，只修改 host 侧，并在
所有客户端地址中使用 3100：

```sh
MARIAN_EDGE_HOST_PORT=3100 docker compose up -d
curl -fsS http://127.0.0.1:3100/readyz
curl -fsS http://127.0.0.1:3100/info
# 沉浸式翻译地址：http://127.0.0.1:3100/imme
```

也可以直接运行：

```sh
docker run -d --name marian-edge --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v marian-edge-models:/models \
  --read-only --tmpfs /tmp:size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/malusama/marian-edge:cpu-0.7.0
```

发布镜像是 AMD64/ARM64 多架构、非 root、CPU-only。镜像不内置模型；第一次
启动时会从 Mozilla 存储直接下载固定的英译中模型，校验压缩前后 SHA-256 后
写入 named volume，后续启动直接复用。

CPU 模型由单一 owner 持有，因此增加计算线程不会加载额外模型副本。并发 HTTP
请求仍会先合并成批次再推理。`MARIAN_EDGE_CPU_THREADS` 支持 1、2 或 4，同时
控制 FP32 矩阵乘法以及 Q8 的 rten/精确 AVX2 row-parallel kernel：

```sh
MARIAN_EDGE_CPU_THREADS=2 docker compose up -d --force-recreate
```

无论设置几个计算线程，模型都仍只有一个 owner。增加内部计算并行度前应在
目标机器和真实流量上测量。

## 沉浸式翻译设置

1. 先确认 `http://127.0.0.1:3000/readyz` 返回 200。
2. 打开“沉浸式翻译 -> 设置 -> 开发者设置”，启用 Beta 测试功能。
3. 在“基本设置 / General -> 翻译服务”中选择“自定义 API”。
4. API 地址填写 `http://127.0.0.1:3000/imme`。
5. 源语言选择英语，目标语言选择简体中文。

以上地址使用默认端口。如果原生安装器或 Compose 改成宿主机 3100，应先检查
`http://127.0.0.1:3100/readyz`，并在扩展里填写
`http://127.0.0.1:3100/imme`；健康检查和扩展地址不能混用两个端口。

服务默认不开放 CORS。扩展具备 loopback 访问权限时通常不需要；如果扩展明确
报告 CORS 错误，优先配置扩展的精确可信 origin。`*` 只适合仍绑定本机的个人
部署：

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/malusama/marian-edge/v0.7.0/scripts/install-macos.sh | \
  MARIAN_EDGE_VERSION=v0.7.0 MARIAN_EDGE_CORS_ORIGIN='*' sh
```

Docker 可在 `compose.yaml` 的 `environment` 中增加
`MARIAN_EDGE_CORS_ORIGIN: "*"`。更完整的请求样例与排错见
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
| `GET /readyz` | 模型 worker 生命周期已就绪；不表示队列仍有空位 |
| `GET /health` | 旧客户端兼容接口 |
| `GET /info` | 版本、提交、后端、设备、精度、attention 模式、模型、运行时间 |
| `GET /metrics` | Prometheus 指标 |

`en-US`、`en_US`、`zh-CN`、`zh-Hans` 会归一化为 `en` 和 `zh`。当前版本只
支持英译中。

| 接口 | 接受的 JSON 字段 | 限制 |
|---|---|---|
| `POST /translate` | `text`、可选 `from`、必填 `to`、可选 `max_output_tokens`；`source_lang`、`target_lang` 分别是 `from`、`to` 的别名 | `max_output_tokens` 默认 512，并收敛到 1-2,048 |
| `POST /imme` | 可选 `source_lang`、必填 `target_lang`、必填 `text_list` | 最多 256 个非空项目；协议未定义 `max_output_tokens`，每项使用默认 512 token 预算 |
| `POST /detect` | 必填 `text`；返回 `{"language":"en"}` 或 `{"language":"zh"}` | 仅提供英语/CJK 启发式检测 |

两套请求字段不要混用。完整 JSON 请求体（包括 JSON 结构和所有列表项目）最大
64 KiB，每个文本项也必须非空。通过 `/translate` 使用时，direct Metal 和纯
Rust CPU 都支持 `max_output_tokens`；它是调用方上限，EOS 与后端模型/runtime
上限可能让生成更早结束。

## 当前支持范围

| 能力 | 状态 |
|---|---|
| Apple Silicon / direct Metal FP32 | 支持 |
| Apple Silicon / direct Metal mixed-f16 存储 | 显式选择；资格语料中与 FP32 精确一致 198/200 条 |
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

`/imme` 中每一项仍对应一个输出项。CPU 与 Metal 后端共用 `marian-core` 的长
文本分段策略，并由各自 tokenizer 提供精确 piece 计数：CPU 每段最多 255 个源
piece 加 EOS，Metal 每段最多 4095 个源 piece 加 EOS。更长的文本会优先在
tokenizer 感知的句子边界分段，之后按原顺序拼回并保留包括换行在内的分隔符；
自动分段不会重置该输入的 `max_output_tokens` 总预算。完整 HTTP JSON 请求体另
受 64 KiB 限制。CPU 还会约束 batch 的 padding attention 工作量，因为
encoder attention 是平方复杂度；融合 Metal attention 不生成平方大小的 score
矩阵。

Q8 后端对 5 条 release golden 全部精确一致。在 200 条差分语料中，与已退役
CPU 参考实现有 164 条输出精确一致；其余多为接近分数下的 token 选择差异，
因此这里不声称逐 token 完全等价。这组 164/200 比较和 80 句重复长文本、换行
测试属于历史工程证据，原始结果 artifact 当前未提交到仓库。

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
target/release/marian-edge-server --backend cpu --cpu-threads 4 \
  --model-dir models/enzh
```

`--backend cpu-q8`、`--backend cpu-fp32` 和 `--backend rust` 都是 `cpu` 的
兼容别名，实际使用 Q8 还是 FP32 仍由 manifest 决定。计算线程数会在推理
启动前固定。

Mac 原生构建需要 macOS SDK/Command Line Tools、Rust 1.86 和 `uv`：

```sh
scripts/prepare-enzh-model.sh
cargo build --locked --release -p marian-server --features metal
target/release/marian-edge-server --backend metal --model-dir models/enzh
```

当前模型默认使用四 query 分块、32 key 流式读取的 FlashAttention-style
实现，通过在线 softmax 避免写出 O(N^2) attention score 矩阵。`/info` 会报告
attention 实现与设备 profile。生产环境保持 `auto` 即可；`classic` 和强制
`flash` 用于兼容性验证和 A/B：

```sh
MARIAN_EDGE_METAL_ATTENTION=classic \
  target/release/marian-edge-server --backend metal --model-dir models/enzh

MARIAN_EDGE_METAL_ATTENTION=auto \
MARIAN_EDGE_METAL_FLASH_THRESHOLD=1 \
  target/release/marian-edge-server --backend metal --model-dir models/enzh
```

这台 Apple M1 / 16 GB 机器的吞吐甜点是
`--max-batch-size 16 --batch-window-us 750`，v0.6 资格测试覆盖的短请求并发约
32。历史 v0.4 mixed-f16 扫描显示并发 64 不再增加吞吐且中位延迟明显上升；这
只是容量预警，不是当前 v0.6 FP32 的 64 并发证明。M1 已验证的重复行宽度默认
为 9，可用
`MARIAN_EDGE_METAL_DUPLICATE_BATCH_WIDTH` 覆盖；它只在当前动态 batch 内补足
最多这么多 Metal 物理占位；逻辑重复项始终由 core 合并，也不会跨 batch 缓存
结果。当前 M1 profile 的重复行宽度、decode
row budget、单次提交步数和选词线程数分别是 9、54、6、256，自定义 FP32 GEMM
保持关闭；`/info` 会报告完整解析结果。M2、M3、M4 和 generic profile 目前只是
保守起点，必须在对应硬件复测后才能称为甜点。需要严格输出契约时用默认 FP32；
内存优先可启用
`MARIAN_EDGE_METAL_PRECISION=mixed-f16`，此前资格测试中 200 条语料有 2 条与 FP32
不同。

旧自动化仍可使用 `mlx` feature 或 `--backend mlx`，它们现在只是 direct Metal
实现的兼容 alias。MSL 源码内嵌在可执行文件里，并在进程启动时通过 Metal
framework 编译，因此不再需要 `libmlx.dylib`、外置 `.metallib`、MLX submodule
或 `scripts/build-mlx.sh`。Mac Release 只发布一个可执行文件；模型目录仍由用户
单独下载和准备。

模型准备脚本固定 Python 与转换依赖版本，并用 SHA-256 校验模型文件。模型、
转换后权重、缓存和构建产物不会进入 Git。

## 性能

最终 v0.6.0 release candidate 与现场重新构建的 v0.1.0 MLX 在同一台有桌面负载
的 M1 上做了三轮中位数 A/B。实测 candidate 因版本提交尚未完成而报告 0.5.0，
正式 v0.6.0 使用相同推理源码：1,000 个短请求从 486.64 提升到 546.19 item/s
（+12.2%），5 次 200 条不同
语料从 116.68 提升到 149.14 item/s（+27.8%）。最终二进制的 Flash q4 又分别比
classic attention 快 12.3% 和 4.9%，输出 hash 一致。Metal FP32 与 CPU FP32 在
200/200 条确定性语料上精确一致；300 请求 trace 中 40/40 个有标签 command
buffer 全部完成，GPU error 为 0。这些是 Apple M1 单机工程测量，不代表 M2-M4。
逐轮结果、历史安静机器峰值、延迟、内存、hash 和 trace 证据见
[BENCHMARKS](docs/BENCHMARKS.md)。

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
