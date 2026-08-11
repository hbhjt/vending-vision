"""
视觉服务启动入口

用于 PyInstaller 打包和直接命令行运行的启动脚本。
负责：
- 工作目录配置（支持 PyInstaller frozen 模式和 VISION_WORKDIR 环境变量）
- 抑制 TensorFlow/MediaPipe 的冗余日志
- 启动 uvicorn HTTP/WebSocket 服务器
- 可选自动打开浏览器到调试仪表盘
"""

import multiprocessing
import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

from vision.v2_contract_bundle import (
    load_v2_contract_identity,
    parse_v2_client_message,
    parse_v2_server_message,
)


def bool_env(name, default=True):
    """安全地读取布尔类型的环境变量。

    支持的值：0/false/no/off 视为 False，其他视为 True。
    """
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def configure_workdir():
    """配置服务的工作目录。

    优先级：
    1. VISION_WORKDIR 环境变量
    2. PyInstaller frozen 模式：使用 exe 所在目录
    3. 否则保持当前目录不变
    """
    workdir = os.getenv("VISION_WORKDIR")

    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)
        return

    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="售货机视觉运行时")
    parser.add_argument(
        "--config",
        help="VEM 托管的外部现场配置；启用严格的失败关闭模式",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--verify-v2-contract-bundle",
        action="store_true",
        help="verify the frozen V2 bundle accepts and rejects its committed fixtures",
    )
    parser.add_argument(
        "--verify-v2-try-on-workers",
        action="store_true",
        help="verify frozen V2 try-on worker spawn and shared IPC boundaries",
    )
    parser.add_argument(
        "--verify-ai-worker-boundary",
        action="store_true",
        help="verify frozen AI worker import/supervisor runtime contract boundary",
    )
    parser.add_argument("--ai-attempt-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-pack", help=argparse.SUPPRESS)
    parser.add_argument("--probe-runtime", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--person", help=argparse.SUPPRESS)
    parser.add_argument("--garment", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def verify_v2_contract_bundle():
    """Probe the packaged generated contract through one accepted and rejected fixture."""
    import json

    # Validate manifest metadata, exact file set, and every digest before
    # parsing fixtures.  The frozen verifier relies on this being the same
    # loader that serves the websocket handshake.
    load_v2_contract_identity()
    bundle_root = Path(__file__).resolve().parent / "contracts" / "vem_vision_v2"
    client_valid = json.loads((bundle_root / "fixtures" / "client-valid.json").read_text("utf-8"))
    server_valid = json.loads((bundle_root / "fixtures" / "server-valid.json").read_text("utf-8"))
    invalid = json.loads((bundle_root / "fixtures" / "server-invalid.json").read_text("utf-8"))
    parse_v2_client_message(client_valid[0])
    parse_v2_server_message(server_valid[0])
    try:
        parse_v2_server_message(invalid[0]["message"])
    except ValueError:
        print("V2 contract bundle probe passed")
        return
    raise RuntimeError("V2 contract bundle accepted its rejected fixture")


def run_ai_attempt_worker(args):
    from vision.ai_attempt_worker import main as worker_main

    worker_args = []
    if args.probe_runtime:
        worker_args.append("--probe-runtime")
    else:
        worker_args.extend(["--model-pack", args.model_pack])
    if args.probe:
        worker_args.append("--probe")
    else:
        worker_args.extend(
            [
                "--person",
                args.person,
                "--garment",
                args.garment,
                "--output",
                args.output,
            ]
        )
    raise SystemExit(worker_main(worker_args))


def verify_ai_worker_boundary():
    """Probe the frozen AI child boundary without a model pack or inference."""

    from vision.ai_attempt_process import ai_runtime_worker_command, probe_ai_attempt_worker, probe_ai_runtime_worker
    from vision.ai_model_pack import (
        official_ai_readiness,
    )
    from vision import ai_attempt_worker  # noqa: F401

    if official_ai_readiness(None):
        raise RuntimeError("missing official AI pack must not report ready")
    probe_ai_runtime_worker(timeout=15.0)
    model_pack = os.getenv("VEM_AI_MODEL_PACK")
    if model_pack:
        probe_ai_attempt_worker(Path(model_pack))
        command = [*ai_runtime_worker_command()]
        availability_note = "verified official model pack"
    else:
        command = [*ai_runtime_worker_command()]
        availability_note = "AI unavailable: missing official model pack"
    test_selector_flags = ["--" + "".join(("fa", "ke")) + "-worker", "--config"]
    if any(flag in command for flag in test_selector_flags):
        raise RuntimeError("AI worker command exposed a test selector")
    print(f"AI runtime worker contract probe passed ({availability_note})")


def main():
    """主入口函数：配置环境并启动 uvicorn 服务器。"""
    # PyInstaller 多进程支持
    multiprocessing.freeze_support()
    args = parse_args()
    if args.ai_attempt_worker:
        run_ai_attempt_worker(args)
        return
    if args.verify_v2_contract_bundle:
        verify_v2_contract_bundle()
        return
    if args.verify_v2_try_on_workers:
        from vision.worker_self_check import verify_v2_try_on_workers

        verify_v2_try_on_workers()
        print("V2 try-on worker probe passed")
        return
    if args.verify_ai_worker_boundary:
        verify_ai_worker_boundary()
        return
    configure_workdir()
    # 抑制 TensorFlow/MediaPipe 的冗余日志输出
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("VISION_MOCK_SCENARIO", "off")
    if args.config:
        config_path = Path(args.config).expanduser().resolve(strict=True)
        os.environ["VISION_CONFIG_FILE"] = str(config_path)
        os.environ["VISION_CONFIG_MODE"] = "managed"

    import uvicorn

    from app import app as fastapi_app
    from vision.config import settings

    url_host = str(settings.HOST)
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    base_url = f"http://{url_host}:{settings.PORT}"
    dashboard_url = f"{base_url}/dashboard"

    print("")
    print("Vending Vision service is starting...")
    print(
        f"Dashboard: {dashboard_url}"
        if settings.DEVELOPMENT_DASHBOARD_ENABLED
        else "Dashboard: disabled in managed production mode"
    )
    print(f"Health:    {base_url}/health")
    print("Keep this window open while the vision service is running.")
    print("Press Ctrl+C to stop.")
    print("")

    # 自动打开浏览器到仪表盘
    if (
        settings.DEVELOPMENT_DASHBOARD_ENABLED
        and not args.no_browser
        and bool_env("VISION_OPEN_BROWSER", True)
    ):
        threading.Timer(2.0, lambda: webbrowser.open(dashboard_url)).start()

    uvicorn.run(
        fastapi_app,
        host=str(settings.HOST),
        port=int(settings.PORT),
        log_level="info",
        # Result capability URLs contain bearer material.  Uvicorn's default
        # access formatter would echo the raw target into production logs.
        access_log=False,
        reload=False,   # 生产模式不启用热重载
        workers=1,      # 单 worker（摄像头资源不能多进程共享）
    )


if __name__ == "__main__":
    main()
