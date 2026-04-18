import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import JsonStorage, MemoryManager
from src.services.chat_service import ChatService


class FakeResponse:
    def __init__(self, content: str, reasoning: str = "") -> None:
        self.content = content
        self.reasoning = reasoning

    def model_dump(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content,
                        "reasoning_content": self.reasoning,
                    }
                }
            ]
        }


class FakeDelta:
    def __init__(self, reasoning_content=None, content=None) -> None:
        self.reasoning_content = reasoning_content
        self.content = content


class FakeChoice:
    def __init__(self, delta: FakeDelta) -> None:
        self.delta = delta


class FakeChunk:
    def __init__(self, reasoning_content=None, content=None) -> None:
        self.choices = [FakeChoice(FakeDelta(reasoning_content=reasoning_content, content=content))]


class FakeLLMClient:
    def __init__(self) -> None:
        self.large_calls = []

    def call_large_model(self, messages, model_name: str, stream: bool = True):
        self.large_calls.append(
            {
                "messages": messages,
                "model_name": model_name,
                "stream": stream,
            }
        )
        if stream:
            return [
                FakeChunk(reasoning_content="先分析背景"),
                FakeChunk(content="这是"),
                FakeChunk(content="流式回答"),
            ]
        return FakeResponse(content="这是非流式回答", reasoning="已完成推理")

    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        return '{"no_update": true}'


class FakeDispatcher:
    def dispatch(self, analysis, rag_graph, conversation_manager, emb_model):
        return {
            "merged_context": "这里是检索上下文",
            "route_results": {"RAG": {"status": "ok"}},
            "active_routes": ["RAG"],
            "display_info": "**判定意图：** RAG",
            "rag_final_state": {},
        }


def build_test_memory_manager(temp_dir: str) -> MemoryManager:
    storage_dir = Path(temp_dir) / "conversations"
    profile_path = Path(temp_dir) / "user_profile.yaml"
    storage = JsonStorage(storage_dir=str(storage_dir))
    return MemoryManager(storage=storage, profile_path=str(profile_path))


def build_test_service(fake_llm: FakeLLMClient) -> ChatService:
    return ChatService(
        llm_client=fake_llm,
        dispatcher=FakeDispatcher(),
        emb_model=object(),
        rag_graph_builder=lambda: object(),
        analyse_query_fn=lambda message, llm_client: {
            "intents": ["RAG"],
            "rewritten_query": message,
            "entities": {},
            "confidence": 1.0,
            "reasoning": "测试桩",
        },
    )


def test_chat_service_non_stream_single_write():
    """验证非流式调用只写入一轮 user/assistant 消息。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        conversation_id = manager.create_conversation("服务层拆解测试")
        manager.current_conversation_id = conversation_id
        manager.update_working_memory(conversation_id, "目标岗位", "嵌入式工程师")
        manager.profile_manager.update_profile({"major": "电子信息工程"})

        fake_llm = FakeLLMClient()
        service = build_test_service(fake_llm)

        updates = list(
            service.generate_response(
                message="帮我看看简历方向",
                conversation_manager=manager,
                stream_mode=False,
                model="mock-model",
            )
        )

        assert updates[-1][1] == "这是非流式回答"

        history = manager.get_conversation_history(conversation_id)
        assert len(history) == 2
        assert [item["role"] for item in history] == ["user", "assistant"]

        request_messages = fake_llm.large_calls[0]["messages"]
        assert request_messages[0]["role"] == "system"
        assert "电子信息工程" in request_messages[0]["content"]
        assert "目标岗位" in request_messages[0]["content"]


def test_chat_service_stream_response_persists_final_answer():
    """验证流式调用会持续产出内容，并在结束后完成持久化。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        conversation_id = manager.create_conversation("流式测试")
        manager.current_conversation_id = conversation_id

        fake_llm = FakeLLMClient()
        service = build_test_service(fake_llm)

        updates = list(
            service.generate_response(
                message="生成一份面试建议",
                conversation_manager=manager,
                stream_mode=True,
                model="mock-model",
            )
        )

        assert updates[-1][1] == "这是流式回答"
        assert any("思考过程" in thinking for thinking, _ in updates)

        history = manager.get_conversation_history(conversation_id)
        assert len(history) == 2
        assert history[-1]["content"] == "这是流式回答"


def test_chat_service_direct_route_does_not_load_rag_resources():
    """验证 DIRECT 路由不会提前初始化 RAG 图和向量模型。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        conversation_id = manager.create_conversation("DIRECT 路由测试")
        manager.current_conversation_id = conversation_id

        fake_llm = FakeLLMClient()
        counters = {"rag": 0, "emb": 0}

        service = ChatService(
            llm_client=fake_llm,
            dispatcher=FakeDispatcher(),
            rag_graph_builder=lambda: counters.__setitem__("rag", counters["rag"] + 1) or object(),
            analyse_query_fn=lambda message, llm_client: {
                "intents": ["DIRECT"],
                "rewritten_query": "",
                "entities": {},
                "confidence": 1.0,
                "reasoning": "测试 DIRECT",
            },
        )
        service.emb_model_builder = lambda: counters.__setitem__("emb", counters["emb"] + 1) or object()

        updates = list(
            service.generate_response(
                message="你好",
                conversation_manager=manager,
                stream_mode=False,
                model="mock-model",
            )
        )

        assert updates[-1][1] == "这是非流式回答"
        assert counters["rag"] == 0
        assert counters["emb"] == 0


if __name__ == "__main__":
    test_chat_service_non_stream_single_write()
    test_chat_service_stream_response_persists_final_answer()
    test_chat_service_direct_route_does_not_load_rag_resources()
