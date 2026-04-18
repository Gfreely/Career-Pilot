import json
from typing import List, Dict, Optional, Any
from .storage import BaseStorage, JsonStorage
from .user_profile import UserProfileManager
from src.utils.json_utils import parse_json_object

class MemoryManager:
    """
    统一的三级内存管理接口：
    1. 短期记忆 (Short-Term Memory): 最近的 N 轮对话滑窗。
    2. 工作记忆 (Working Memory): 核心事实、画像状态摘要。
    3. 结构化用户画像 (Structured User Profile): 跨会话持久化的用户背景信息。
       替代了原有的"长期记忆"原始对话向量存储，改为提取并维护结构化字段。
       原始对话落盘仍然保留作为日志级存档。
    """
    def __init__(self, storage: Optional[BaseStorage] = None,
                 profile_path: str = "user_profile.yaml"):
        self.storage = storage or JsonStorage()
        self.current_conversation_id = None
        self.profile_manager = UserProfileManager(profile_path=profile_path)
        
    def create_conversation(self, title: Optional[str] = None) -> str:
        self.current_conversation_id = self.storage.create_conversation(title)
        return self.current_conversation_id

    def get_conversation(self, conversation_id: str) -> Dict:
        return self.storage.get_conversation(conversation_id)

    def get_all_conversations(self) -> List[Dict]:
        return self.storage.get_all_conversations()

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.storage.delete_conversation(conversation_id)

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        return self.storage.update_title(conversation_id, title)

    # --- 长期记忆 (全量记录与获取，保留作为日志级存档) ---
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """记录长久明细（日志级归档）"""
        return self.storage.add_message(conversation_id, role, content)
        
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取完整版会话历史 (日志级存档读取)"""
        conversation = self.storage.get_conversation(conversation_id)
        if not conversation:
            return []
            
        messages = conversation.get("messages", [])
        history = []
        for msg in messages:
            history.append({"role": msg["role"], "content": msg["content"]})
        return history

    # --- 短期记忆 (最近上下文滑动窗口) ---
    def get_short_term_context(self, conversation_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        获取短期上下文（滑动窗口机制）。
        max_turns 指保留的回合数，通常一回合等于 user + assistant 两条记录。
        """
        history = self.get_conversation_history(conversation_id)
        if not history:
            return []
            
        max_messages = max_turns * 2
        
        # 截取最后的 max_messages
        if len(history) > max_messages:
            return history[-max_messages:]
        return history

    # --- 工作记忆/实体记忆 (高价值画像或状态) ---
    def get_working_memory_text(self, conversation_id: str) -> str:
        """从 Storage 中读取提炼过的高优摘要或状态，转为文本供 prompt 使用"""
        conv_data = self.storage.get_conversation(conversation_id)
        if not conv_data:
            return ""
        working_memory_dict = conv_data.get("working_memory", {})
        if not working_memory_dict:
            return ""
            
        lines = []
        for k, v in working_memory_dict.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
        
    def update_working_memory(self, conversation_id: str, key: str, value: str) -> bool:
        """写入/更新实体或状态到工作记忆的字典中"""
        return self.storage.update_working_memory(conversation_id, key, value)

    # --- 结构化用户画像 (Structured User Profile) ---
    def extract_and_update_profile(self, conversation_id: str, llm_client: Any) -> bool:
        """
        从最近一轮 user + assistant 消息中提取用户画像增量，并合并到全局画像。
        
        Args:
            conversation_id: 当前会话 ID
            llm_client: 统一 LLM 客户端实例（用于调用小模型）
        
        Returns:
            True 表示画像有更新，False 表示无更新
        """
        import src.core.template as template

        # 取最近一轮对话
        history = self.get_conversation_history(conversation_id)
        if len(history) < 2:
            return False

        # 找最后一对 user + assistant
        last_user_msg = None
        last_assistant_msg = None
        for msg in reversed(history):
            if msg["role"] == "assistant" and last_assistant_msg is None:
                last_assistant_msg = msg["content"]
            elif msg["role"] == "user" and last_user_msg is None:
                last_user_msg = msg["content"]
            if last_user_msg and last_assistant_msg:
                break

        if not last_user_msg or not last_assistant_msg:
            return False

        # 构建提取 prompt
        current_profile_json = json.dumps(self.profile_manager.get_profile(), ensure_ascii=False, indent=2)
        prompt = template.PROFILE_EXTRACTION_TEMPLATE.replace(
            "{current_profile}", current_profile_json
        ).replace(
            "{user_message}", last_user_msg
        ).replace(
            "{assistant_message}", last_assistant_msg
        )

        try:
            # 调用小模型提取画像增量
            result = llm_client.call_small_model(system_prompt=prompt)

            # 小模型异常时 UnifiedLLMClient 会返回空串，此处直接跳过，避免 json.loads("")
            result = (result or "").strip()
            if not result:
                print("⚠ 画像提取结果为空，已跳过本轮画像更新")
                return False

            updates = parse_json_object(result)
            
            # 合并到画像
            has_update = self.profile_manager.update_profile(updates)
            if has_update:
                print(f"✅ 用户画像已更新: {list(updates.keys())}")
            return has_update

        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            preview = result[:200].replace("\n", "\\n") if "result" in locals() else ""
            print(f"⚠ 画像提取 JSON 解析失败: {e}; 原始输出片段: {preview}")
            return False
        except Exception as e:
            print(f"⚠ 画像提取流程异常: {e}")
            return False

    def get_profile_text(self) -> str:
        """获取用户画像的文本格式（供 prompt 注入）"""
        return self.profile_manager.get_profile_text()

    def get_profile_filter(self) -> Dict[str, Any]:
        """获取用于检索增强的画像关键字段"""
        return self.profile_manager.get_filter_metadata()

    def get_profile_vector(self, emb_model) -> Optional[list]:
        """获取画像 Embedding 向量（委托给 UserProfileManager 的写穿缓存）"""
        return self.profile_manager.get_profile_vector(emb_model)
