# TensorFold 适配维护约定

- 先保留上游 SpecPilot 的 SGLang 复现路径；TensorFold 扩展位于 `specpilot_tensorfold/`。
- 扩展仅用 Python 标准库，兼容 Python 3.10+，不要求安装 SGLang 或 PyTorch。
- 本轮授权范围是工程适配。开发具体自主优化任务之前，先与用户确认目标场景、性能/质量门禁和预算；不要自行建立真实任务、启动搜索或部署候选。
- 部署、测量默认只输出计划。显式操作与目标合同分开，禁止以缺失身份、缺失 token IDs、缺失请求或 synthetic 证据认定真实验证成功。
- 双 Spark TP2 是一个实验资源组；候选与基线的 GPU 运行串行。双机镜像和固定 recipe 身份必须匹配。
- 真实环境地址、SSH 目标、API key、外置部署配置、原始用户请求与结果均不提交。key 只从环境读取。
- `exact` 比较使用原生输出 token IDs；不能对返回文本重新分词。不存在可用独立 scorer 时，不能把 teacher-forced 或 FP32-head 验收标记为通过。
- 代码 hash、离线统计 PASS 不代表 OS 隔离、sealed final 或生产晋升。当前扩展不声称提供论文完整自动优化控制器。
- 提交前运行 `python -m unittest discover -s tests/tensorfold -v`、CLI 帮助与 `git diff --check`。测试仅用 synthetic 数据或本机模拟服务，不调用真实 GPU。
