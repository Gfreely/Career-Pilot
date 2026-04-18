import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import JsonStorage, MemoryManager
from src.services import ProfileService


def build_test_memory_manager(temp_dir: str) -> MemoryManager:
    storage_dir = Path(temp_dir) / "conversations"
    profile_path = Path(temp_dir) / "user_profile.yaml"
    storage = JsonStorage(storage_dir=str(storage_dir))
    return MemoryManager(storage=storage, profile_path=str(profile_path))


def test_profile_service_patch_and_replace():
    """验证画像服务的增量更新与全量替换行为。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        service = ProfileService()

        patched = service.patch_profile(
            manager,
            {
                "major": "电子信息工程",
                "tech_stack": ["Python", "FPGA"],
                "project_experience": [
                    {
                        "name": "智能车控制系统",
                        "tech_stack": ["STM32"],
                        "description": "比赛项目",
                    }
                ],
            },
        )

        assert patched["profile"]["major"] == "电子信息工程"
        assert "Python" in patched["filter_metadata"]["tech_stack"]
        assert "STM32" in patched["filter_metadata"]["tech_stack"]

        replaced = service.replace_profile(
            manager,
            {
                "major": "通信工程",
                "target_cities": ["深圳"],
            },
        )

        assert replaced["profile"]["major"] == "通信工程"
        assert replaced["profile"]["tech_stack"] == []
        assert replaced["profile"]["target_cities"] == ["深圳"]


def test_profile_service_raw_file_roundtrip():
    """验证画像原始文件可读取并可直接保存回写。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        service = ProfileService()

        file_info = service.get_profile_file(manager)
        assert file_info["format"] == "yaml"
        assert "major:" in file_info["content"]

        updated = service.save_profile_file(
            manager,
            "major: 集成电路\ntech_stack:\n  - Verilog\n  - Python\n",
        )
        assert updated["profile"]["major"] == "集成电路"
        assert "Verilog" in updated["filter_metadata"]["tech_stack"]
        assert "major: 集成电路" in updated["file"]["content"]
