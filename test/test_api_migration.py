import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_chat_service, get_memory_manager, get_profile_service
from src.memory import JsonStorage, MemoryManager
from src.services import ProfileService


class FakeChatService:
    def generate_response(self, message, conversation_manager, stream_mode=True, model="", prompt_template="default"):
        conversation_id = conversation_manager.current_conversation_id
        conversation_manager.add_message(conversation_id, "user", message)
        yield "**判定意图：** DIRECT", ""
        yield "**判定意图：** DIRECT", "测试回复"
        conversation_manager.add_message(conversation_id, "assistant", "测试回复")


def build_test_memory_manager(temp_dir: str) -> MemoryManager:
    storage_dir = Path(temp_dir) / "conversations"
    profile_path = Path(temp_dir) / "user_profile.yaml"
    storage = JsonStorage(storage_dir=str(storage_dir))
    return MemoryManager(storage=storage, profile_path=str(profile_path))


def test_api_profile_and_chat_routes():
    """验证迁移出的 profile/chat API 基础路由可用。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        app = create_app()

        app.dependency_overrides[get_memory_manager] = lambda: manager
        app.dependency_overrides[get_profile_service] = lambda: ProfileService()
        app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

        client = TestClient(app)

        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        profile_resp = client.patch("/api/profile", json={"major": "微电子", "tech_stack": ["Verilog"]})
        assert profile_resp.status_code == 200
        assert profile_resp.json()["profile"]["major"] == "微电子"

        file_get_resp = client.get("/api/profile/file")
        assert file_get_resp.status_code == 200
        assert file_get_resp.json()["format"] == "yaml"

        file_put_resp = client.put(
            "/api/profile/file",
            json={"content": "major: 芯片设计\ntech_stack:\n  - SystemVerilog\n"},
        )
        assert file_put_resp.status_code == 200
        assert file_put_resp.json()["profile"]["major"] == "芯片设计"
        assert "SystemVerilog" in file_put_resp.json()["filter_metadata"]["tech_stack"]

        create_resp = client.post("/api/chat/conversations", json={"title": "迁移测试"})
        assert create_resp.status_code == 200
        conversation_id = create_resp.json()["id"]

        reply_resp = client.post(
            f"/api/chat/conversations/{conversation_id}/reply",
            json={"message": "你好", "model": "mock-model", "prompt_template": "default"},
        )
        assert reply_resp.status_code == 200
        assert reply_resp.json()["content"] == "测试回复"

        detail_resp = client.get(f"/api/chat/conversations/{conversation_id}")
        assert detail_resp.status_code == 200
        messages = detail_resp.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
