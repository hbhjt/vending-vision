"""共享录播夹具 spec：分辨率与帧率只在这里定义，避免散落漂移。

现场摄像头升级到 1080p 后（ADR：VM 先行），竖屏 front 夹具为 1080x1920，
横屏 top 夹具为 1920x1080。夹具源保持确定性并与 expected-results.json 摘要绑定。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PERSON_SOURCE = ROOT / "sources" / "person-man-front.png"
TOP_LEGACY_SOURCE = ROOT / "sources" / "top-legacy-320.mp4"

FRONT_FRAME_SIZE = (1080, 1920)
TOP_FRAME_SIZE = (1920, 1080)
FPS = 6
