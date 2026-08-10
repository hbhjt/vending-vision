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


def main():
    """主入口函数：配置环境并启动 uvicorn 服务器。"""
    # PyInstaller 多进程支持
    multiprocessing.freeze_support()
    args = parse_args()
    if args.verify_v2_contract_bundle:
        verify_v2_contract_bundle()
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
