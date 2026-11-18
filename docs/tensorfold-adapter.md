# TensorFold Native 双 Spark 工程适配

本 fork 在上游 SpecPilot `1058dc29481150e66ae11b8e8cf55ed60caf254a` 上增加独立适配层。目标接口为 TensorFold Native `f350cd3ea253a042e205746928271f9d8f71a82d`、Mia GLM-5.3-Flash 1104 recipe、EXL3、两台 DGX Spark GB10／SM121／TP2。保留上游的 SGLang 目录、setting、启动和论文复现脚本；不要求安装 SGLang、PyTorch 或本地 KSEO。

**当前交付范围是工程适配。开发具体自主优化任务之前，必须先与用户确认目标场景、性能和质量验收要求、可用时段与实验预算。** 本 fork 没有预设用户的优化任务、默认性能提升比例、自动搜索循环或模型修改。下面的部署和测量命令是供后续经确认的任务调用的接口。

## 1. 安装与入口

客户端使用 Python 3.10+ 标准库，可运行于 Windows 或 Linux。远端需要 Linux、Python 3、Bash、OpenSSH、Git、Docker、已准备好的 Mia recipe／模型及正常双机连接。head 必须可以通过密钥 SSH 登录 worker。

```bash
git clone https://github.com/wangtianwen07/SpecPilot.git
cd SpecPilot
python -m specpilot_tensorfold --help
# 等效入口，不需要 pip install：
python scripts/tensorfold.py --help
```

| 模块 | 工程职责 |
|---|---|
| `configuration.py` | 从可信部署读取配置、文件和镜像身份，生成外置 baseline 配置 |
| `deployment.py` | 双机检查、候选准备、切换、恢复、事务保留与操作记录 |
| `client.py` | 原生 tokenize、完整 SSE、真实 token IDs、客户端计时、单批并发测量 |
| `evaluation.py` | 显式合同下的离线 ABBA／cluster bootstrap／exact token 验收 |
| `__main__.py` | CLI 和 JSON 结果输出；默认不执行部署和生成 |

`.tensorfold-local/`、`tensorfold-runs/` 已忽略。地址、SSH 目标、原始请求、输出内容、部署配置及操作日志放在这些外置目录，不提交到公开 fork。API key 只通过环境变量传入；SSH 使用已配置的密钥和严格 host key 校验。

## 2. 先采集工程配置

以下地址和路径是占位符。`capture-config` 默认只打印计划；显式 `--execute` 才执行只读采集。它会 **source 你指定的可信 recipe 的 `scripts/config.sh`** 来解析配置；因此指定的 recipe 和其本地配置必须可信。不会调用 build、start、stop、模型生成或优化搜索。

```bash
python -m specpilot_tensorfold capture-config \
  --head spark-head --worker spark-worker \
  --recipe /home/operator/glm53-deployment \
  --container glm53-spark \
  --lock /home/operator/specpilot-control/dual-spark.lock \
  --transaction engineering-baseline \
  --output .tensorfold-local/deployment.json \
  --execute
```

head 的 SSH 别名由客户端解析，worker 的 SSH 别名由 head 解析。复杂 SSH 选项可通过 Python API 的 `head/worker.ssh_options` 提供，或放进各自的 SSH config。

采集后 `candidate` 为 `null`。配置包含：两节点的 `/etc/machine-id` 与 GPU UUID、冻结 recipe commit、部署文件库存和 SHA256、两节点 Docker image IDs、`tf.*` labels、实际运行设置、路径和事务信息。`runtime_sha256` 冻结实际命令、环境、挂载和相关 Docker 运行设置。不同 Docker image store 可能产生不同形式的 image ID；每节点分别冻结，并比较镜像配置及层的内容指纹。

采集同时将 **后续操作** 约束为 `MODEL_SOURCE=local`、`WORKER_WEIGHTS=check`、`CONTEXT_RETRY=0`、`MEMGUARD=1`。这些是防止下载／复制权重和静默缩小上下文的操作约束，不是对历史启动参数的声称。`auto` context／parallel 从现有容器实际命令冻结为具体值。

不要把示例路径或手填的假 SHA 当作有效配置。若部署与当前适配的原生 schema 不一致，应先修正适配或配置，不能跳过检查。

```bash
# 默认只打印检查计划，不连接 SSH：
python -m specpilot_tensorfold inspect --config .tensorfold-local/deployment.json
# 实际只读检查：
python -m specpilot_tensorfold inspect --config .tensorfold-local/deployment.json \
  --execute --output .tensorfold-local/inspection.json
# 只读取模型列表，不调用生成接口：
python -m specpilot_tensorfold inspect-api --base-url http://spark-head:PORT/v1 \
  --model GLM-5.3-Flash --output .tensorfold-local/api.json
```

`inspect-api` 成功仅证明模型列表可访问，**不证明** token、GPU 性能或独立 scorer 已验证。

## 3. 双机准备、切换和恢复接口

具体候选将在用户确认任务后开发。候选必须使用独立 recipe 目录、容器名、cache／state／log 路径，并保留冻结的模型、drafter、TensorFold pin 和核心 serving 参数。`files_sha256` 冻结完整 `scripts/`、`docker/`、`patches/engine/`、`templates/` 及相关根文件；`.env`／`scripts/local.sh` 不存在也要显式记录为 `null`。

配置校验的准确字段以 `DeploymentConfig.from_dict`、`REQUIRED_ENV` 和 `IMMUTABLE_ENV` 为准。候选准备阶段可以没有 image IDs；`prepare` 成功返回两节点 ID 后，先将其写入外置候选配置，才能调用 `activate`。改变配置后使用匹配的新事务身份，不能借此跳过未恢复的旧 reservation。

当前候选接口支持保持 serving 设置的代码补丁，以及 `TUNING_ENV` 中已有的 `TF_GLM_*` 参数。模型、精度、context、parallel、内存预留和 host cache 等设置保持冻结；要研究这些设置，需在后续任务确认时扩展合同与适配。仅改代码、保持镜像环境和启动方式时，候选可以复用基线的 `runtime_sha256`：指纹排除了容器名、镜像 ID 和补丁 label，并将候选目录及 cache／state 等路径归一化；白名单调参另按显式配置逐项核验。改变镜像环境、入口或其他运行语义的候选不在这个复用范围，不能填写假哈希绕过检查。

```bash
# 以下不加 --execute 均只输出可审查的命令和脚本。
python -m specpilot_tensorfold prepare --config .tensorfold-local/deployment.json
python -m specpilot_tensorfold activate --config .tensorfold-local/deployment.json
python -m specpilot_tensorfold restore --config .tensorfold-local/deployment.json
```

显式执行上述操作会影响服务：

- `prepare` 在事务中暂停基线、构建并传输候选，验证两节点镜像后恢复基线。GB10 使用统一内存，不能假定在服务旁编译不会干扰或耗尽内存。
- `activate` 先完成身份和占用检查，再停止基线、启动候选并验证；保持 reservation，直到 `restore` 完成。
- `restore` 清理受本事务识别的候选，恢复并核验冻结基线。候选损坏不能成为跳过基线恢复的理由。
- 失败时保留事件与恢复结果。SSH 断开／超时意味着远端状态可能未知，应先重新 `inspect`，不能自动重建或清除 owner。无法完成恢复时需人工处理，不能把操作报告为成功。

切换前检查 `/health.live.connections == 0`；固定 engine pin 的 [`live.zig`](https://github.com/ashhart/TensorFold/blob/f350cd3ea253a042e205746928271f9d8f71a82d/zig/src/server/live.zig) 将它定义为 engine 的 running + waiting，不包含健康检查 HTTP 连接。若启用 API 鉴权后健康接口隐藏这些字段，当前部署接口会拒绝切换；HTTP 测量客户端的 API key 支持不代表部署探针也支持该部署模式。

所有针对这对物理设备的调用者必须使用同一个 head、锁路径与冻结身份。`flock` 和 reservation 是合作式互斥，不能阻止其他账户、另一个锁路径或外部程序使用 GPU，也不是 OS 级隔离证明。对并发负载的可观测性不足要保留未知，不能由空闲内存反推“设备独占”。

GB10 的 NVML 进程列表可能为空，即使模型正在运行。适配层会结合 Docker 进程和 `/proc` 文件描述符观测；若权限不足或无法可靠观察设备占用，会拒绝服务切换。只读采集成功并不保证当前账户具备执行切换所需的观测权限。无法重建的外部 `TF_*`／`TENSORFOLD_*`／`NCCL_*` 环境覆盖也会阻止切换；日志只报告这类变量名，不保存其秘密值。

## 4. 测量协议

只有用户提供明确的 batch 文件并传入 `measure --execute` 才会发送生成请求。本轮不提供真实 batch 或优化目标。一个 batch 是同一 scenario／cluster／block／ABBA slot 的一波并发请求；请求数必须等于 concurrency，不是闭环服务负载发生器。

Batch 字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | `1` |
| `base_url`, `model` | 显式服务地址和 served model ID |
| `api_key_env` | 可选环境变量名，不是 key 本身 |
| `timeout_seconds`, `concurrency` | 请求超时、同批并发数 |
| `requests[].body` | 原生 chat 请求；必须显式 `temperature=0`、`max_tokens`、messages；当前仅支持 n=1 |
| `requests[].metadata` | `scenario_id, cluster_id, block_id, slot, arm, request_id, artifact_id` |

`slot=0,1,2,3` 分别声明 baseline、candidate、candidate、baseline。`artifact_id` 应绑定独立检查得到的源码／镜像／配置身份；模型 HTTP 响应不能自行证明这些身份。外层后续任务负责真实执行顺序、预热、缓存策略和实验排队，当前工具不会自行决定这些任务条件。

```bash
# 默认不调用 tokenize 或生成：
python -m specpilot_tensorfold measure --batch .tensorfold-local/batch.json \
  --output tensorfold-runs/batch-result.json
```

原生协议依据如下，全部来自固定 engine pin：

- [`tokens.zig`](https://github.com/ashhart/TensorFold/blob/f350cd3ea253a042e205746928271f9d8f71a82d/zig/src/server/tokens.zig)：`/v1/tokenize` 使用相同聊天模板，返回完整输入 IDs；原生 hash 为十进制 IDs 用逗号连接后的 SHA256。
- [`chat.zig`](https://github.com/ashhart/TensorFold/blob/f350cd3ea253a042e205746928271f9d8f71a82d/zig/src/server/chat.zig)：`return_token_ids:true` 返回完整生成 IDs，包含实际提交的 reasoning、结构标记和 EOS。
- [`openai.zig`](https://github.com/ashhart/TensorFold/blob/f350cd3ea253a042e205746928271f9d8f71a82d/zig/src/server/openai.zig)：终止 SSE 的 `tensorfold/speculative` 和后续 `usage` 是不同事件，必须累计；读到 `[DONE]` 前仍要检查错误。

计时与记录约定：

- tokenize 在正式计时前完成；E2E 从生成 HTTP 请求发起到收到 `[DONE]`，包含排队、网络和服务处理。
- `ttft_ns` 是客户端首次非空 content／reasoning／有效 tool payload 的延迟，排除 role 与 keepalive；不是服务器第一个模型 token 的时间。`ttft_kind=client_first_visible_delta`。
- `tpot` 是 `(E2E−上述客户端首输出延迟)/(完整输出 token 数−1)` 的代理指标；不等于逐 token 间隔分布。不能把服务器 TTFT 与客户端 E2E 混算。
- `tensorfold.seconds` 已是服务器 decode 时间，不能再减 prefill；保留原值供诊断，正式 E2E 使用客户端计时。
- 完整输出 IDs 必须存在，数量须与 usage 相同，原生完整 SHA256 必须匹配。文本重分词、短 hash、`null == null` 都不能作为一致性证据。
- 网络／推理失败保留为失败记录，不悄悄重试或取两组记录交集。tokenize 等前置失败会中止该批，不能制造成功测量。
- `batch_wall_tokens_per_second` 使用整个并发批次墙钟时间，不是相加每个请求的 tok/s。测量记录中的 `measured` 表示实际 HTTP 计时，GPU 与镜像身份仍需独立部署证据。

## 5. 离线验收接口

`validate-contract` 和 `evaluate` 必须读取显式合同，没有默认收益阈值。合同需要：

- `schema_version, evidence_kind, quality_mode, artifacts`；artifacts 分别标识 baseline／candidate。
- `scenarios[]` 的并发数、clusters、每个 cluster 的 request IDs 与 block IDs。
- 唯一 `target`：scenario、metric、`min_speedup` 与 `min_ci_speedup`；速度比方向为 baseline latency / candidate latency。
- 显式 `guards[]`：scenario、metric、`max_latency_ratio`；延迟比方向为 candidate / baseline。未列入的场景或指标不获得保护声明。
- 显式 `bootstrap`：resamples、confidence、seed、min_clusters。样本充分性和阈值由后续已确认任务确定。

所有 block 必须含完整 ABBA 四槽、全部请求及对应身份。重复实验先在 cluster 内聚合，再按 cluster bootstrap；不能把同一并发批次的请求伪装成独立样本，也不能复制相同 workload 改个 cluster 名增加样本量。

当前 `quality_mode=exact` 要求输入、请求参数和完整输出 token 序列符合一致性检查。baseline 本身不稳定返回 `INCONCLUSIVE`。`teacher_forced` 明确返回 `UNSUPPORTED`；现有 native prompt logprobs 不等同论文的独立 FP32-head scorer。

```bash
python -m specpilot_tensorfold validate-contract --contract .tensorfold-local/contract.json
python -m specpilot_tensorfold evaluate --contract .tensorfold-local/contract.json \
  --records tensorfold-runs/a0.json tensorfold-runs/b1.json tensorfold-runs/b2.json tensorfold-runs/a3.json \
  --output tensorfold-runs/comparison.json
```

`PASS` 只描述所提供证据在该合同上的结果。模拟数据只能得到 `SYNTHETIC_PASS`。所有结果均为 `promotion_allowed=false, sealed=false`。真实运行顺序、独立 scorer、候选不可访问的 holdout、OS 权限隔离和自动晋升不是这次工程适配已经实现的能力；不能把离线 PASS 当成论文 sealed final。

## 6. 验证与后续边界

```bash
python -m unittest discover -s tests/tensorfold -v
python -m specpilot_tensorfold --help
python scripts/tensorfold.py --help
git diff --check
```

测试使用 synthetic 合同、内存响应、本机假 HTTP 服务和模拟 SSH／Docker，不启动模型。CI 在 Windows／Linux 与 Python 3.10／3.12 上运行适配层 CPU 检查，不尝试安装或测试上游 GPU 引擎。

工程代码通过这些检查不等于在用户双 Spark 上取得性能收益。下一阶段先与用户确认真实优化目标，再开发相应任务、补齐所需质量验收并安排真实机器接入验证。W12、TLV、树验证和任何 TensorFold kernel 修改均不包含在本次适配交付中。
