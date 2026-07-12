"""
内存指标收集模块

提供轻量级的 Prometheus 风格指标注册表，支持：
- Counter: 累计计数器（如帧数、错误次数）
- Gauge: 瞬时值仪表盘（如当前连接数）
- Timer: 耗时统计（计数、总和、最小/最大/平均）

所有操作线程安全，通过 RLock 保护。
"""

from __future__ import annotations

import copy
import threading
import time


class MetricsRegistry:
    """内存中的指标注册表。

    支持带标签（label）的指标，标签用于区分不同维度的数据
    （例如：role=top vs role=front）。
    """

    def __init__(self):
        self.lock = threading.RLock()       # 线程锁
        self.started_at = time.time()       # 服务启动时间（用于计算 uptime）
        self.counters = {}                   # 计数器存储
        self.gauges = {}                     # 仪表盘存储
        self.timers = {}                     # 计时器存储

    def increment(self, name: str, value: int = 1, **labels):
        """增加计数器值。

        Args:
            name: 指标名称
            value: 增量值（默认 1）
            **labels: 标签键值对（如 role="top"）
        """
        key = self._key(name, labels)

        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name: str, value, **labels):
        """设置仪表盘值（覆盖旧值）。

        Args:
            name: 指标名称
            value: 当前值
            **labels: 标签键值对
        """
        key = self._key(name, labels)

        with self.lock:
            self.gauges[key] = value

    def observe_ms(self, name: str, value_ms: float, **labels):
        """记录一次耗时观测（毫秒）。

        自动更新该指标的：count、totalMs、minMs、maxMs、avgMs。

        Args:
            name: 指标名称
            value_ms: 本次耗时（毫秒）
            **labels: 标签键值对
        """
        key = self._key(name, labels)
        value_ms = round(float(value_ms), 3)

        with self.lock:
            timer = self.timers.setdefault(
                key,
                {
                    "count": 0,
                    "totalMs": 0.0,
                    "minMs": None,
                    "maxMs": None,
                },
            )
            timer["count"] += 1
            timer["totalMs"] = round(timer["totalMs"] + value_ms, 3)
            timer["minMs"] = (
                value_ms
                if timer["minMs"] is None
                else min(timer["minMs"], value_ms)
            )
            timer["maxMs"] = (
                value_ms
                if timer["maxMs"] is None
                else max(timer["maxMs"], value_ms)
            )
            timer["avgMs"] = round(timer["totalMs"] / timer["count"], 3)

    def snapshot(self):
        """获取所有指标的当前快照。

        Returns:
            包含 uptime、counters、gauges、timers 的字典。
        """
        with self.lock:
            return {
                "uptimeMs": int((time.time() - self.started_at) * 1000),
                "counters": self._expand(copy.deepcopy(self.counters)),
                "gauges": self._expand(copy.deepcopy(self.gauges)),
                "timers": self._expand(copy.deepcopy(self.timers)),
            }

    def _key(self, name: str, labels: dict):
        """根据名称和标签生成唯一键。

        格式：name{key1=val1,key2=val2}
        """
        if not labels:
            return name

        suffix = ",".join(
            f"{key}={labels[key]}"
            for key in sorted(labels)
        )
        return f"{name}{{{suffix}}}"

    def _expand(self, values: dict):
        """对指标字典按键排序后返回。"""
        return dict(sorted(values.items()))


# 全局指标注册表实例
metrics = MetricsRegistry()
