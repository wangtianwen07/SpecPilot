"""Engineering CLI. No optimization task, search loop, or default objective."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .client import TensorFoldClient, measure_batch
from .evaluation import evaluate, validate_contract


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def emit(value, output: str | None = None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        path = Path(output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    print(text, end="")


def parser():
    p = argparse.ArgumentParser(description="SpecPilot 的 TensorFold 双 Spark 工程适配；不创建或启动优化任务")
    sub = p.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-config", help="从可信 recipe 和现有双机服务采集外置部署配置；不推理/重启")
    capture.add_argument("--head", required=True, help="客户端到 head 的 SSH 目标或已配置别名")
    capture.add_argument("--worker", required=True, help="head 到 worker 的 SSH 目标或已配置别名")
    capture.add_argument("--recipe", required=True, help="head 上可信的绝对 recipe 路径")
    capture.add_argument("--container", required=True)
    capture.add_argument("--lock", required=True, help="head 上此物理双机组共用的绝对锁路径")
    capture.add_argument("--transaction", required=True)
    capture.add_argument("--execute", action="store_true", help="执行只读采集；会 source 指定可信 recipe 的 config.sh")
    capture.add_argument("--output", required=True, help="外置 JSON；采集冻结身份和后续 local/check/no-fallback/memguard 约束")
    for action in ("inspect", "prepare", "activate", "restore"):
        item = sub.add_parser(action, help={
            "inspect": "读取双机身份（默认仅输出计划）",
            "prepare": "构建/传输候选镜像（默认仅输出计划）",
            "activate": "切换至候选服务（默认仅输出计划）",
            "restore": "恢复冻结基线（默认仅输出计划）",
        }[action])
        item.add_argument("--config", required=True, help="外置部署配置，含两节点与冻结身份")
        item.add_argument("--execute", action="store_true", help="执行指定操作；prepare/activate/restore 会改变服务")
        item.add_argument("--journal", help="可选本地操作日志路径")
        item.add_argument("--output", help="可选 JSON 结果路径")
    api = sub.add_parser("inspect-api", help="仅 GET /v1/models，不发送生成请求")
    api.add_argument("--base-url", required=True)
    api.add_argument("--model", required=True)
    api.add_argument("--api-key-env", help="API key 的环境变量名；不通过命令行传 key")
    api.add_argument("--timeout", type=float, default=15)
    api.add_argument("--output")
    bench = sub.add_parser("measure", help="测量用户明确提供的一批请求；默认只检查输入，不推理")
    bench.add_argument("--batch", required=True, help="显式 batch JSON；任务目标须先由用户确认")
    bench.add_argument("--execute", action="store_true", help="发送生成请求，使用指定服务的 GPU")
    bench.add_argument("--output", required=True, help="保存原始测量结果")
    compare = sub.add_parser("evaluate", help="离线裁决已有记录；不部署、不晋升、不封存")
    compare.add_argument("--contract", required=True, help="用户确认后单独开发的显式合同")
    compare.add_argument("--records", nargs="+", required=True)
    compare.add_argument("--output")
    validate = sub.add_parser("validate-contract", help="检查显式合同完整性；不生成默认目标")
    validate.add_argument("--contract", required=True)
    validate.add_argument("--output")
    return p


def main(argv=None) -> int:
    # Windows redirected pipes otherwise default to the legacy ANSI code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.command == "capture-config":
            from .configuration import capture_baseline_config
            result = capture_baseline_config(
                {"ssh_target": args.head}, {"ssh_target": args.worker}, args.recipe,
                args.container, args.lock, args.transaction, execute=args.execute)
        elif args.command in ("inspect", "prepare", "activate", "restore"):
            from .deployment import DeploymentConfig, TensorFoldDeployment
            config = DeploymentConfig.from_json(args.config)
            deployment = TensorFoldDeployment(config, execute=args.execute, journal_path=args.journal)
            result = getattr(deployment, args.command)()
        elif args.command == "inspect-api":
            key = os.environ.get(args.api_key_env) if args.api_key_env else None
            if args.api_key_env and not key:
                raise ValueError("指定 API key 环境变量为空")
            result = TensorFoldClient(args.base_url, args.model, api_key=key,
                                     timeout=args.timeout).inspect_api()
        elif args.command == "measure":
            batch = load(args.batch)
            if not isinstance(batch, dict) or type(batch.get("schema_version")) is not int or batch["schema_version"] != 1:
                raise ValueError("batch 必须是 schema_version=1 对象")
            if not isinstance(batch.get("requests"), list) or not batch["requests"]:
                raise ValueError("batch.requests 必须显式提供")
            if type(batch.get("concurrency")) is not int or batch["concurrency"] != len(batch["requests"]):
                raise ValueError("单批请求数必须等于 concurrency")
            key_env = batch.get("api_key_env")
            key = os.environ.get(key_env) if key_env else None
            if args.execute and key_env and not key:
                raise ValueError("指定 API key 环境变量为空")
            client = TensorFoldClient(batch["base_url"], batch["model"], api_key=key,
                                     timeout=batch.get("timeout_seconds", 300))
            if not args.execute:
                result = {"status": "PLAN", "execute": False,
                          "requests": len(batch["requests"]), "concurrency": batch["concurrency"],
                          "inference_started": False, "objective_defined": False,
                          "note": "仅检查批次基本结构；未调用 tokenize 或生成接口，未验证服务能力"}
            else:
                result = measure_batch(client, batch["requests"], batch["concurrency"])
        elif args.command == "validate-contract":
            result = validate_contract(load(args.contract))
        else:
            records = []
            for path in args.records:
                blob = load(path)
                rows = blob.get("records") if isinstance(blob, dict) else blob
                if not isinstance(rows, list):
                    raise ValueError("records 文件必须为数组或含 records 数组的批次结果")
                records.extend(rows)
            result = evaluate(load(args.contract), records)
        emit(result, args.output)
        if result.get("status") in ("FAIL", "INCONCLUSIVE", "INVALID_CONTRACT", "INVALID_EVIDENCE", "UNSUPPORTED"):
            return 2
        if result.get("valid") is False or result.get("complete") is False:
            return 2
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # Avoid printing request headers, raw HTTP bodies, remote credentials or tracebacks.
        error = {"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}
        try:
            emit(error, getattr(args, "output", None))
        except OSError:
            emit(error)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
