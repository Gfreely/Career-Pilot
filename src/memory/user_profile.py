"""
结构化用户画像管理器 (Structured User Profile Manager)

全局跨会话持久化的用户画像存储，使用 YAML 格式便于 IDE 手动编辑。
负责画像的读写、智能合并（list 并集去重、str 覆盖更新）和格式化输出。
支持热加载：每次读取时从磁盘重新加载，确保外部手动编辑立即生效。
"""

import yaml
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


# 画像字段的默认空模板（包含扩展字段）
_DEFAULT_PROFILE = {
    "major": None,               # 专业
    "degree": None,               # 学历：本科/硕士/博士
    "graduation_year": None,      # 毕业年份
    "target_cities": [],          # 意向城市列表
    "tech_stack": [],             # 技术栈列表
    "job_preferences": [],        # 岗位偏好列表
    "offer_status": {             # Offer 状态
        "received": [],
        "pending": [],
        "rejected": []
    },
    "experience_level": None,     # 经验水平
    "concerns": [],               # 关注点/顾虑列表

    # --- 扩展字段 ---
    "education_background": [],   # 学历背景列表，每项为 {"school": "XX大学", "level": "985/211/双非", "degree": "本科/硕士", "major": "..."}
    "project_experience": [],     # 项目经历列表，每项为 {"name": "项目名", "tech_stack": ["技术1"], "description": "简述"}
    "internship_experience": [],  # 实习经历列表，每项为 {"company": "公司名", "position": "岗位", "duration": "时长"}

    "last_updated": None          # 最后更新时间
}

# 列表类型的顶层字段（合并时做并集去重，仅适用于简单 str 列表）
_SIMPLE_LIST_FIELDS = {"target_cities", "tech_stack", "job_preferences", "concerns"}

# 复合对象列表字段（需要按名称/公司等主键去重合并）
_OBJECT_LIST_FIELDS = {"education_background", "project_experience", "internship_experience"}

# offer_status 的子字段（也是列表类型）
_OFFER_SUB_FIELDS = {"received", "pending", "rejected"}

# YAML 文件头部注释模板
_YAML_HEADER = """# ===== 用户画像 (User Profile) =====
# 可在 IDE 中直接手动编辑此文件，系统也会在对话中自动提取更新。
# 列表项使用 YAML 列表语法（每行一个 - 开头的条目）。
# 修改保存后系统会自动热加载，无需重启。
#
# 字段说明:
#   major            - 专业
#   degree           - 最高学历 (本科/硕士/博士)
#   graduation_year  - 毕业年份
#   experience_level - 经验水平 (应届生/1-3年/3-5年/5年以上)
#   target_cities    - 意向城市列表
#   tech_stack       - 技术栈列表
#   job_preferences  - 岗位偏好列表
#   offer_status     - Offer 状态 (received/pending/rejected)
#   concerns         - 关注点/顾虑列表
#   education_background - 学历背景 (school, level[985/211/双非], degree, major)
#   project_experience   - 项目经历 (name, tech_stack, description)
#   internship_experience - 实习经历 (company, position, duration)
# ======================================

"""


class UserProfileManager:
    """用户画像管理器：YAML 读写、智能合并、热加载、格式化输出"""

    def __init__(self, profile_path: str = "user_profile.yaml"):
        self.profile_path = Path(profile_path)
        self.profile: Dict[str, Any] = self._load_profile()
        # 画像 Embedding 写穿缓存（仅画像文本变化时重新计算向量）
        self._cached_profile_text: Optional[str] = None
        self._cached_profile_vec: Optional[List[float]] = None

    def _load_profile(self) -> Dict[str, Any]:
        """从磁盘加载 YAML 画像，若不存在则初始化空画像并写入带注释的模板"""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data is None:
                    data = {}
                # 确保所有默认字段都存在（兼容旧版本）
                for key, default_val in _DEFAULT_PROFILE.items():
                    if key not in data:
                        data[key] = copy.deepcopy(default_val)
                # 确保 list 字段不为 None（用户可能误改为 null）
                for key in _SIMPLE_LIST_FIELDS | _OBJECT_LIST_FIELDS:
                    if data.get(key) is None:
                        data[key] = []
                if data.get("offer_status") is None:
                    data["offer_status"] = {"received": [], "pending": [], "rejected": []}
                return data
            except Exception as e:
                print(f"⚠ 画像文件读取失败，将初始化空画像: {e}")
                return copy.deepcopy(_DEFAULT_PROFILE)
        # 文件不存在 — 创建带注释的模板
        profile = copy.deepcopy(_DEFAULT_PROFILE)
        self._save_with_header(profile)
        return profile

    def reload(self):
        """从磁盘热加载画像（支持外部编辑后同步）"""
        self.profile = self._load_profile()

    def _save_with_header(self, data: Dict[str, Any]):
        """持久化画像到磁盘（带注释头部）"""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            f.write(_YAML_HEADER)
            f.write(yaml_content)

    def save(self):
        """持久化当前画像到磁盘"""
        self._save_with_header(self.profile)

    def get_profile(self) -> Dict[str, Any]:
        """返回当前画像（先热加载磁盘版本，确保手动编辑同步）"""
        self.reload()
        return copy.deepcopy(self.profile)

    def update_profile(self, updates: Dict[str, Any]) -> bool:
        """
        智能合并画像更新。
        - 简单 list 类型字段：做并集去重
        - str/scalar 类型字段：直接覆盖
        - offer_status 子字段：list 并集去重
        - 复合对象 list 字段：按主键去重追加
        
        返回 True 表示有实际更新，False 表示无更新。
        """
        # 先热加载，确保基于最新版本合并
        self.reload()

        if not updates or updates.get("no_update"):
            return False

        has_change = False

        for key, value in updates.items():
            if key in ("no_update", "last_updated"):
                continue

            if key in _SIMPLE_LIST_FIELDS:
                # 简单列表字段：并集去重
                if isinstance(value, list) and value:
                    existing = self.profile.get(key, [])
                    if existing is None:
                        existing = []
                    merged = list(dict.fromkeys(existing + value))
                    if merged != existing:
                        self.profile[key] = merged
                        has_change = True
                elif isinstance(value, str) and value:
                    existing = self.profile.get(key, [])
                    if existing is None:
                        existing = []
                    if value not in existing:
                        existing.append(value)
                        self.profile[key] = existing
                        has_change = True

            elif key in _OBJECT_LIST_FIELDS:
                # 复合对象列表：按主键去重追加
                if isinstance(value, list) and value:
                    existing = self.profile.get(key, [])
                    if existing is None:
                        existing = []
                    # 根据字段类型确定主键
                    pk_map = {
                        "education_background": "school",
                        "project_experience": "name",
                        "internship_experience": "company"
                    }
                    pk = pk_map.get(key, "name")
                    existing_pks = {item.get(pk) for item in existing if isinstance(item, dict)}
                    for item in value:
                        if isinstance(item, dict) and item.get(pk) not in existing_pks:
                            existing.append(item)
                            existing_pks.add(item.get(pk))
                            has_change = True
                    self.profile[key] = existing

            elif key == "offer_status":
                if isinstance(value, dict):
                    existing_offer = self.profile.get("offer_status", {
                        "received": [], "pending": [], "rejected": []
                    })
                    for sub_key in _OFFER_SUB_FIELDS:
                        if sub_key in value and isinstance(value[sub_key], list):
                            old_list = existing_offer.get(sub_key, [])
                            if old_list is None:
                                old_list = []
                            merged = list(dict.fromkeys(old_list + value[sub_key]))
                            if merged != old_list:
                                existing_offer[sub_key] = merged
                                has_change = True
                    self.profile["offer_status"] = existing_offer

            elif key in _DEFAULT_PROFILE:
                if value is not None and value != self.profile.get(key):
                    self.profile[key] = value
                    has_change = True

        if has_change:
            self.profile["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            self.save()
            # 画像内容变更，清除向量缓存（下次调用 get_profile_vector 时惰性重算）
            self._cached_profile_vec = None
            self._cached_profile_text = None

        return has_change

    def get_profile_text(self) -> str:
        """将画像格式化为可注入 prompt 的文本（先热加载）"""
        self.reload()
        lines = []
        field_labels = {
            "major": "专业",
            "degree": "学历",
            "graduation_year": "毕业年份",
            "target_cities": "意向城市",
            "tech_stack": "技术栈",
            "job_preferences": "岗位偏好",
            "experience_level": "经验水平",
            "concerns": "关注点",
        }

        for key, label in field_labels.items():
            val = self.profile.get(key)
            if val:
                if isinstance(val, list) and val:
                    lines.append(f"- {label}: {', '.join(str(v) for v in val)}")
                elif isinstance(val, str):
                    lines.append(f"- {label}: {val}")

        # 学历背景
        edu = self.profile.get("education_background", [])
        if edu:
            edu_parts = []
            for e in edu:
                if isinstance(e, dict):
                    parts = []
                    if e.get("school"):
                        parts.append(e["school"])
                    if e.get("level"):
                        parts.append(f"({e['level']})")
                    if e.get("degree"):
                        parts.append(e["degree"])
                    if e.get("major"):
                        parts.append(e["major"])
                    edu_parts.append(" ".join(parts))
            if edu_parts:
                lines.append(f"- 学历背景: {'; '.join(edu_parts)}")

        # 项目经历
        proj = self.profile.get("project_experience", [])
        if proj:
            proj_parts = []
            for p in proj:
                if isinstance(p, dict) and p.get("name"):
                    tech = ", ".join(p.get("tech_stack", [])) if p.get("tech_stack") else ""
                    desc = f"[{tech}]" if tech else ""
                    proj_parts.append(f"{p['name']}{desc}")
            if proj_parts:
                lines.append(f"- 项目经历: {'; '.join(proj_parts)}")

        # 实习经历
        intern = self.profile.get("internship_experience", [])
        if intern:
            intern_parts = []
            for i in intern:
                if isinstance(i, dict):
                    company = i.get("company", "")
                    position = i.get("position", "")
                    intern_parts.append(f"{company}-{position}" if position else company)
            if intern_parts:
                lines.append(f"- 实习经历: {'; '.join(intern_parts)}")

        # Offer 状态
        offer = self.profile.get("offer_status", {})
        offer_parts = []
        if offer and isinstance(offer, dict):
            if offer.get("received"):
                offer_parts.append(f"已获得: {', '.join(offer['received'])}")
            if offer.get("pending"):
                offer_parts.append(f"等待中: {', '.join(offer['pending'])}")
            if offer.get("rejected"):
                offer_parts.append(f"已拒绝: {', '.join(offer['rejected'])}")
        if offer_parts:
            lines.append(f"- Offer状态: {'; '.join(offer_parts)}")

        return "\n".join(lines)

    def get_filter_metadata(self) -> Dict[str, Any]:
        """
        提取可用于检索增强的关键字段子集（先热加载）。
        返回非空字段，供搜索关键词扩充使用。
        """
        self.reload()
        result = {}
        for key in ["target_cities", "tech_stack", "job_preferences", "major"]:
            val = self.profile.get(key)
            if val:
                if isinstance(val, list) and val:
                    result[key] = val
                elif isinstance(val, str):
                    result[key] = val

        # 从项目经历中提取技术栈补充
        proj_techs = []
        for p in self.profile.get("project_experience", []):
            if isinstance(p, dict) and p.get("tech_stack"):
                proj_techs.extend(p["tech_stack"])
        if proj_techs:
            existing_tech = result.get("tech_stack", [])
            if isinstance(existing_tech, str):
                existing_tech = [existing_tech]
            merged = list(dict.fromkeys(existing_tech + proj_techs))
            result["tech_stack"] = merged

        return result

    def get_profile_vector(self, emb_model) -> Optional[List[float]]:
        """
        获取画像 Embedding 向量（带写穿缓存）。

        仅当画像文本内容实际发生变化时才重新调用 Embedding API，
        否则直接返回缓存的向量。实现画像更新逻辑与 Embedding 计算的融合：
        - update_profile() 有变更时主动清除缓存
        - reload() 读取外部编辑后文本变化也会通过比对自动检测
        """
        current_text = self.get_profile_text()
        if not current_text or not current_text.strip():
            return None
        # 文本未变 → 缓存命中
        if current_text == self._cached_profile_text and self._cached_profile_vec is not None:
            return self._cached_profile_vec
        # 文本变化或首次调用 → 重新计算
        print("[Profile Cache] 画像内容变化，重新计算 Embedding 向量")
        self._cached_profile_vec = emb_model.embed_query(current_text)
        self._cached_profile_text = current_text
        return self._cached_profile_vec

    def reset(self):
        """重置画像为空模板（调试/测试用）"""
        self.profile = copy.deepcopy(_DEFAULT_PROFILE)
        self._cached_profile_vec = None
        self._cached_profile_text = None
        self.save()
