from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import src.core.template as template
from src.core.llm_client import UnifiedLLMClient
from src.memory import MemoryManager


DEFAULT_MODEL = "Pro/MiniMaxAI/MiniMax-M2.5"
DEFAULT_PROMPT_TEMPLATE = "default"


def build_default_rag_graph():
    """延迟导入 RAG 图构建器，降低服务层导入成本。"""
    from src.agents.rag_graph import build_rag_graph

    return build_rag_graph()


def build_default_emb_model():
    """延迟导入画像向量模型，便于测试环境绕开重依赖。"""
    from src.core.embedding_model import LocalBGEM3Embeddings

    return LocalBGEM3Embeddings()


def build_default_dispatcher():
    """延迟导入多路由分发器。"""
    from src.core.multi_router import MultiRouteDispatcher

    return MultiRouteDispatcher()


def default_retrieval_pipeline(**kwargs):
    """延迟导入检索流水线。"""
    from src.eval.Recall_test import execute_retrieval_pipeline

    return execute_retrieval_pipeline(**kwargs)


def default_analyse_query(query: str, llm_client: UnifiedLLMClient) -> Dict[str, Any]:
    """延迟导入意图分析函数。"""
    from src.core.multi_router import analyse_query

    return analyse_query(query, llm_client)


def load_prompts() -> Dict[str, Dict[str, str]]:
    """返回聊天服务使用的提示词模板定义。"""
    return {
        "default": {
            "system": template.RAG_TEMPLATE_XINGHUO,
            "description": "通用提示词",
        }
    }


def process_stream_response(stream) -> Generator[Tuple[str, str], None, Tuple[str, str]]:
    """解析流式大模型输出，持续返回推理内容与回答内容。"""
    full_response = ""
    reasoning = ""

    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            reasoning += reasoning_content
            yield reasoning, full_response

        content = getattr(delta, "content", None)
        if content:
            full_response += content
            yield reasoning, full_response

    return reasoning, full_response


def process_non_stream_response(response) -> Tuple[str, str]:
    """解析非流式大模型输出。"""
    response_data = response.model_dump()
    reasoning = ""
    content = ""

    if response_data.get("choices"):
        message = response_data["choices"][0].get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

    return reasoning, content


class ChatService:
    """聊天业务编排服务，负责意图分析、检索增强、Prompt 组装与消息落库。"""

    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        dispatcher: Optional[Any] = None,
        emb_model: Optional[Any] = None,
        rag_graph_builder: Optional[Callable[[], Any]] = None,
        retrieval_pipeline: Optional[Callable[..., Dict[str, Any]]] = None,
        analyse_query_fn: Optional[Callable[[str, UnifiedLLMClient], Dict[str, Any]]] = None,
    ) -> None:
        self.llm_client = llm_client or UnifiedLLMClient()
        self.dispatcher = dispatcher
        self.dispatcher_builder = build_default_dispatcher
        self.emb_model = emb_model
        self.emb_model_builder = build_default_emb_model
        self.rag_graph_builder = rag_graph_builder or build_default_rag_graph
        self.retrieval_pipeline = retrieval_pipeline or default_retrieval_pipeline
        self.analyse_query_fn = analyse_query_fn or default_analyse_query
        self._rag_graph = None

    def get_rag_graph(self):
        """按需构建并缓存 RAG 图，避免重复编译。"""
        if self._rag_graph is None:
            self._rag_graph = self.rag_graph_builder()
        return self._rag_graph

    def get_dispatcher(self):
        """按需获取多路由分发器，降低服务构造时的依赖成本。"""
        if self.dispatcher is None:
            self.dispatcher = self.dispatcher_builder()
        return self.dispatcher

    def get_emb_model(self):
        """按需获取画像向量模型，避免导入期加载重资源。"""
        if self.emb_model is None:
            self.emb_model = self.emb_model_builder()
        return self.emb_model

    def _needs_rag_resources(self, intents: List[str]) -> bool:
        """仅在当前请求确实需要 RAG 时才初始化重资源。"""
        return "RAG" in intents

    def generate_response(
        self,
        message: str,
        conversation_manager: MemoryManager,
        stream_mode: bool = True,
        model: str = DEFAULT_MODEL,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> Generator[Tuple[str, str], None, None]:
        """执行一次完整的聊天业务流程。"""
        conversation_id = self._ensure_conversation(conversation_manager)
        conversation_manager.add_message(conversation_id, "user", message)

        analysis = self.analyse_query_fn(message, self.llm_client)
        intents = analysis.get("intents", ["RAG"])
        intent_display = f"**判定意图：** 分析中... ({', '.join(intents)})"
        yield intent_display, ""

        context, intent_display = self._collect_context(
            message=message,
            analysis=analysis,
            intents=intents,
            conversation_manager=conversation_manager,
        )
        yield intent_display, ""

        request_messages = self._build_request_messages(
            conversation_manager=conversation_manager,
            conversation_id=conversation_id,
            prompt_template=prompt_template,
            context=context,
        )

        yield intent_display + "\n\n*正在载入大模型进行推理...*", ""

        response = self.llm_client.call_large_model(
            messages=request_messages,
            model_name=model,
            stream=stream_mode,
        )

        reasoning = ""
        content = ""

        if stream_mode:
            for reasoning, content in process_stream_response(response):
                yield self._format_thinking(intent_display, reasoning), content
        else:
            reasoning, content = process_non_stream_response(response)
            yield self._format_thinking(intent_display, reasoning), content

        conversation_manager.add_message(conversation_id, "assistant", content)
        self._try_update_profile(conversation_manager, conversation_id)

    def _ensure_conversation(self, conversation_manager: MemoryManager) -> str:
        """确保当前会话存在。"""
        if not conversation_manager.current_conversation_id:
            conversation_manager.create_conversation()
        return conversation_manager.current_conversation_id

    def _collect_context(
        self,
        message: str,
        analysis: Dict[str, Any],
        intents: List[str],
        conversation_manager: MemoryManager,
    ) -> Tuple[str, str]:
        """根据意图分发检索上下文，并提供失败回退。"""
        context = ""
        needs_rag_resources = self._needs_rag_resources(intents)

        try:
            dispatch_result = self.get_dispatcher().dispatch(
                analysis=analysis,
                rag_graph=self.get_rag_graph() if needs_rag_resources else None,
                conversation_manager=conversation_manager,
                emb_model=self.get_emb_model() if needs_rag_resources else None,
            )
            context = dispatch_result.get("merged_context", "")
            intent_display = dispatch_result.get("display_info", "**判定意图：** 未知")
            return context, intent_display
        except Exception as exc:
            print(f"[ChatService] dispatcher 异常: {exc}")
            if not needs_rag_resources:
                intent_display = f"**判定意图：** {', '.join(intents)}（分发异常，已跳过检索）"
                return "", intent_display
            intent_display = f"**判定意图：** {', '.join(intents)}（分发异常，回退直接检索）"

        try:
            retrieval_result = self.retrieval_pipeline(
                query=analysis.get("rewritten_query") or message,
                llm_client=self.llm_client,
                profile_text=conversation_manager.get_profile_text(),
                profile_vec=conversation_manager.get_profile_vector(self.get_emb_model()),
                profile_filter=conversation_manager.get_profile_filter(),
            )
            context = retrieval_result.get("context", "")
        except Exception as fallback_exc:
            print(f"[ChatService] fallback 检索失败: {fallback_exc}")
            context = ""

        return context, intent_display

    def _build_request_messages(
        self,
        conversation_manager: MemoryManager,
        conversation_id: str,
        prompt_template: str,
        context: str,
    ) -> List[Dict[str, str]]:
        """组装发送给大模型的消息列表，避免使用跨请求全局状态。"""
        prompts = load_prompts()
        base_system_prompt = prompts.get(prompt_template, prompts["default"])["system"]

        working_memory_text = conversation_manager.get_working_memory_text(conversation_id)
        if working_memory_text:
            base_system_prompt += f"\n\n[核心状态与系统记忆]\n{working_memory_text}"

        profile_text = conversation_manager.get_profile_text()
        if profile_text:
            base_system_prompt += f"\n\n[用户画像信息]\n{profile_text}"

        system_prompt = f"{base_system_prompt}\n\n检索到的参考内容：\n{context}"

        request_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        short_term_context = conversation_manager.get_short_term_context(conversation_id, max_turns=5)
        for message in short_term_context:
            request_messages.append({"role": message["role"], "content": message["content"]})

        return request_messages

    def _try_update_profile(self, conversation_manager: MemoryManager, conversation_id: str) -> None:
        """聊天结束后尝试抽取画像增量，不影响主流程。"""
        try:
            conversation_manager.extract_and_update_profile(conversation_id, self.llm_client)
        except Exception as exc:
            print(f"[ChatService] 画像提取失败（静默）: {exc}")

    @staticmethod
    def _format_thinking(intent_display: str, reasoning: str) -> str:
        """统一构造前端思考框文本。"""
        if reasoning:
            return f"{intent_display}\n\n**思考过程**\n{reasoning}"
        return intent_display
