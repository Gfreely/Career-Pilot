import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class BaseStorage(ABC):
    @abstractmethod
    def create_conversation(self, title: Optional[str] = None) -> str:
        """创建新会话并返回 ID"""
        pass
        
    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Dict:
        """获取目标 ID 会话的所有明细"""
        pass
        
    @abstractmethod
    def get_all_conversations(self) -> List[Dict]:
        """获取所有对话的轻量级列表"""
        pass
        
    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """添加对话消息"""
        pass
        
    @abstractmethod
    def update_title(self, conversation_id: str, title: str) -> bool:
        """更新对话标题"""
        pass
        
    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """删除指定会话数据"""
        pass
        
    @abstractmethod
    def update_working_memory(self, conversation_id: str, key: str, value: str) -> bool:
        """更新工作实体记忆键值对"""
        pass


class JsonStorage(BaseStorage):
    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        # 索引统一在内存中维护，index.json 仅存标题与时间结构
        self.index: Dict[str, Dict] = self._load_index()
        
    def _load_index(self) -> Dict[str, Dict]:
        index_file = self.storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
        
    def _save_index(self):
        with open(self.storage_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
            
    def _get_conversation_file(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{conversation_id}.json"

    def create_conversation(self, title: Optional[str] = None) -> str:
        conversation_id = str(int(time.time()))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not title:
            title = f"对话 {timestamp}"
            
        # 存储在简略 index 中
        self.index[conversation_id] = {
            "id": conversation_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp
        }
        self._save_index()
        
        # 存储具体明细在新文件中
        conv_data = {
            "id": conversation_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
            "working_memory": {} # 用于提取工作记忆的关键画像特征
        }
        with open(self._get_conversation_file(conversation_id), "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
            
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Dict:
        file_path = self._get_conversation_file(conversation_id)
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_all_conversations(self) -> List[Dict]:
        """返回轻量的索引列表，按时间倒序"""
        conversations_list = list(self.index.values())
        return sorted(conversations_list, key=lambda x: x.get("updated_at", ""), reverse=True)

    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        if conversation_id not in self.index:
            return False
            
        conv_data = self.get_conversation(conversation_id)
        if not conv_data:
            return False
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "messages" not in conv_data:
            conv_data["messages"] = []
            
        conv_data["messages"].append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        conv_data["updated_at"] = timestamp
        
        # 保存明细
        with open(self._get_conversation_file(conversation_id), "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
            
        # 更新并保存索引
        self.index[conversation_id]["updated_at"] = timestamp
        self._save_index()
        
        return True

    def update_title(self, conversation_id: str, title: str) -> bool:
        if conversation_id not in self.index:
            return False
            
        self.index[conversation_id]["title"] = title
        self._save_index()
        
        conv_data = self.get_conversation(conversation_id)
        if conv_data:
            conv_data["title"] = title
            with open(self._get_conversation_file(conversation_id), "w", encoding="utf-8") as f:
                json.dump(conv_data, f, ensure_ascii=False, indent=2)
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.index:
            del self.index[conversation_id]
            self._save_index()
            
        file_path = self._get_conversation_file(conversation_id)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        return True

    def update_working_memory(self, conversation_id: str, key: str, value: str) -> bool:
        file_path = self._get_conversation_file(conversation_id)
        if not file_path.exists():
            return False
            
        conv_data = self.get_conversation(conversation_id)
        if not conv_data:
            return False
            
        if "working_memory" not in conv_data:
            conv_data["working_memory"] = {}
            
        conv_data["working_memory"][key] = value
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)
            
        return True
