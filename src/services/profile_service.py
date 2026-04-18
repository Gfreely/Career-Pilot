from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.memory import MemoryManager


class ProfileService:
    """用户画像业务服务，提供查询、局部更新、全量替换与重载能力。"""

    def get_profile_bundle(self, conversation_manager: MemoryManager) -> Dict[str, Any]:
        """返回前端和 API 需要的画像完整视图。"""
        profile_manager = conversation_manager.profile_manager
        return {
            "profile": profile_manager.get_profile(),
            "profile_text": profile_manager.get_profile_text(),
            "filter_metadata": profile_manager.get_filter_metadata(),
        }

    def patch_profile(self, conversation_manager: MemoryManager, updates: Dict[str, Any]) -> Dict[str, Any]:
        """按现有合并规则增量更新画像。"""
        conversation_manager.profile_manager.update_profile(updates)
        return self.get_profile_bundle(conversation_manager)

    def replace_profile(self, conversation_manager: MemoryManager, profile: Dict[str, Any]) -> Dict[str, Any]:
        """全量替换画像，未提供的字段回落为默认空模板。"""
        profile_manager = conversation_manager.profile_manager
        profile_manager.reset()
        if profile:
            profile_manager.update_profile(profile)
        return self.get_profile_bundle(conversation_manager)

    def reload_profile(self, conversation_manager: MemoryManager) -> Dict[str, Any]:
        """从磁盘重新加载画像。"""
        conversation_manager.profile_manager.reload()
        return self.get_profile_bundle(conversation_manager)

    def get_profile_file(self, conversation_manager: MemoryManager) -> Dict[str, Any]:
        """读取画像原始文件内容，供前端直接编辑。"""
        profile_path = Path(conversation_manager.profile_manager.profile_path)
        if profile_path.exists():
            content = profile_path.read_text(encoding="utf-8")
        else:
            content = ""
        return {
            "path": str(profile_path.resolve()),
            "format": "yaml",
            "content": content,
        }

    def save_profile_file(self, conversation_manager: MemoryManager, content: str) -> Dict[str, Any]:
        """直接保存前端编辑后的画像原始文件，并重新加载到内存。"""
        parsed = yaml.safe_load(content) if content.strip() else {}
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("画像文件内容必须是 YAML 对象")

        profile_path = Path(conversation_manager.profile_manager.profile_path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(content, encoding="utf-8")
        conversation_manager.profile_manager.reload()

        result = self.get_profile_bundle(conversation_manager)
        result["file"] = {
            "path": str(profile_path.resolve()),
            "format": "yaml",
            "content": profile_path.read_text(encoding="utf-8"),
        }
        return result
