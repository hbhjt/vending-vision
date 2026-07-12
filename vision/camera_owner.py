"""
前置摄像头所有权管理模块

中部（前置）摄像头是共享资源，可能被 vision 画像采集和 tryon 试衣前端同时竞争。
本模块实现基于优先级的准入控制：

优先级（从低到高）：
- idle (0): 无人使用
- vision (1): 视觉画像采集
- tryon_frontend (2): 试衣前端（最高优先级）

特性：
- 支持超时自动过期，防止死锁
- 线程安全操作
- 提供 I/O 锁用于需要独占前置摄像头的连续操作
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from vision.config import settings


# 合法的前置摄像头使用者
ALLOWED_FRONT_CAMERA_OWNERS = {"vision", "tryon_frontend"}
# 使用者优先级映射
FRONT_CAMERA_PRIORITY = {
    "idle": 0,
    "vision": 1,
    "tryon_frontend": 2,
}


def _now_iso():
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class FrontCameraOwner:
    """前置摄像头的所有权管理器。

    管理对前置摄像头的独占访问，支持：
    - 按优先级抢占（tryon_frontend > vision > idle）
    - 超时自动回收（防止持有者异常退出导致死锁）
    - 线程安全的所有权变更
    """

    def __init__(self):
        self.lock = threading.RLock()       # 可重入锁，保护所有状态变更
        self.owner = "idle"                 # 当前持有者
        self.reason = None                  # 持有/释放原因
        self.updated_at = _now_iso()        # 最近一次状态变更时间（ISO格式）
        self.updated_monotonic = time.time()  # 最近一次状态变更时间（单调时钟）

    def _expire_locked(self):
        """检查当前持有者是否超时，超时则强制回收（需在持有锁时调用）。

        超时时间由 FRONT_CAMERA_OWNER_TIMEOUT_MS 配置（默认 120 秒）。
        """
        if self.owner == "idle":
            return

        timeout_ms = max(int(settings.FRONT_CAMERA_OWNER_TIMEOUT_MS), 0)
        if timeout_ms <= 0:
            return

        elapsed_ms = int((time.time() - self.updated_monotonic) * 1000)
        if elapsed_ms < timeout_ms:
            return

        # 超时回收
        self.owner = "idle"
        self.reason = "owner_timeout"
        self.updated_at = _now_iso()
        self.updated_monotonic = time.time()

    def status(self):
        """获取当前所有权状态。"""
        with self.lock:
            self._expire_locked()
            return {
                "owner": self.owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
                "timeoutMs": settings.FRONT_CAMERA_OWNER_TIMEOUT_MS,
            }

    def acquire(self, owner: str, reason: str | None = None):
        """尝试获取前置摄像头的所有权。

        规则：
        - 只有 ALLOWED_FRONT_CAMERA_OWNERS 中的使用者可以获取
        - 如果当前持有者优先级更高，则拒绝请求
        - 同级别可以抢占（允许 vision 覆盖 vision）

        返回包含 ok 字段的结果字典。
        """
        if owner not in ALLOWED_FRONT_CAMERA_OWNERS:
            return {
                "ok": False,
                "owner": self.owner,
                "requestedOwner": owner,
                "error": "invalid_owner",
            }

        with self.lock:
            self._expire_locked()
            current_owner = self.owner
            current_priority = FRONT_CAMERA_PRIORITY.get(current_owner, 0)
            requested_priority = FRONT_CAMERA_PRIORITY[owner]

            # 优先级检查：不能抢占更高优先级的持有者
            if current_owner not in {"idle", owner} and requested_priority < current_priority:
                return {
                    "ok": False,
                    "owner": current_owner,
                    "requestedOwner": owner,
                    "reason": self.reason,
                    "error": "front_camera_busy",
                }

            self.owner = owner
            self.reason = reason
            self.updated_at = _now_iso()
            self.updated_monotonic = time.time()

            return {
                "ok": True,
                "owner": self.owner,
                "previousOwner": current_owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
            }

    def release(self, owner: str, reason: str | None = None):
        """释放前置摄像头的所有权。

        只有当前持有者本人才能释放。
        如果持有者不匹配，返回错误。
        """
        with self.lock:
            self._expire_locked()

            if owner not in ALLOWED_FRONT_CAMERA_OWNERS:
                return {
                    "ok": False,
                    "owner": self.owner,
                    "requestedOwner": owner,
                    "error": "invalid_owner",
                }

            if self.owner != owner:
                return {
                    "ok": False,
                    "owner": self.owner,
                    "requestedOwner": owner,
                    "error": "owner_mismatch",
                }

            previous_owner = self.owner
            self.owner = "idle"
            self.reason = reason
            self.updated_at = _now_iso()
            self.updated_monotonic = time.time()

            return {
                "ok": True,
                "owner": self.owner,
                "previousOwner": previous_owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
            }

    def renew(self, owner: str, reason: str | None = None):
        """Renew an active lease without changing ownership."""
        with self.lock:
            self._expire_locked()
            if self.owner != owner:
                return {
                    "ok": False,
                    "owner": self.owner,
                    "requestedOwner": owner,
                    "error": "owner_mismatch",
                }
            self.reason = reason or self.reason
            self.updated_at = _now_iso()
            self.updated_monotonic = time.time()
            return {
                "ok": True,
                "owner": self.owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
            }


# ---------------------------------------------------------------------------
# 全局单例和便捷函数
# ---------------------------------------------------------------------------

# 全局前置摄像头所有者实例
_front_camera_owner = FrontCameraOwner()
# 前置摄像头 I/O 锁：用于需要连续独占前置摄像头的操作（如画像采集）
_front_camera_io_lock = threading.RLock()


def get_front_camera_owner():
    """获取前置摄像头的当前所有权状态。"""
    return _front_camera_owner.status()


def acquire_front_camera(owner: str, reason: str | None = None):
    """尝试获取前置摄像头所有权。"""
    return _front_camera_owner.acquire(owner, reason=reason)


def release_front_camera(owner: str, reason: str | None = None):
    """释放前置摄像头所有权。"""
    return _front_camera_owner.release(owner, reason=reason)


def renew_front_camera(owner: str, reason: str | None = None):
    """Refresh an active camera-owner lease."""
    return _front_camera_owner.renew(owner, reason=reason)


@contextmanager
def front_camera_io_lock():
    """前置摄像头 I/O 上下文管理器。

    用于需要连续独占前置摄像头的操作（如画像采集、试衣流），
    防止操作中途被其他使用者打断。
    """
    with _front_camera_io_lock:
        yield
