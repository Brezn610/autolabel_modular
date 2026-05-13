"""时序精炼（Temporal Refinement）子模块。

参考 MS3D (darrenjkt/MS3D, T-IV & ITSC 2024) 的 temporal_refinement 与 tracker/。
本项目把其中“跨帧关联 + 轨迹平滑 + 轨迹内去重”的核心思想做了轻量化实现，
以便在单序列（无 ego 全局位姿）场景下，仍能压制“同一目标跨相机/跨帧被重复检测”。
"""

from .temporal_refinement import refine_tracklets, draw_temporal_debug

__all__ = ["refine_tracklets", "draw_temporal_debug"]
