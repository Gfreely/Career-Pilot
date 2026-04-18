import os
import sys
import tempfile

# 将当前脚本的父目录（即项目根目录）加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.memory import MemoryManager
from src.memory.storage import JsonStorage
from src.memory.user_profile import UserProfileManager


def test_memory():
    """原有三级内存测试"""
    manager = MemoryManager()
    
    # 1. 创建对话
    cid = manager.create_conversation("测试会话")
    print(f"创建对话: {cid}")
    
    # 2. 加入消息记录 (10 轮交互 = 20 条消息)
    for i in range(1, 11):
        manager.add_message(cid, "user", f"问题 {i}")
        manager.add_message(cid, "assistant", f"回答 {i}")
        
    # 3. 获取长期记忆
    full_history = manager.get_conversation_history(cid)
    print(f"\n长期全量记忆获取记录条数: {len(full_history)}")
    assert len(full_history) == 20
    
    # 4. 获取短期记忆
    short_term_context = manager.get_short_term_context(cid, max_turns=5)
    print(f"短期截断窗口记忆记录条数 (max_turns=5): {len(short_term_context)}")
    assert len(short_term_context) == 10
    
    # 5. 测试工作记忆
    manager.update_working_memory(cid, "language", "python")
    manager.update_working_memory(cid, "preferences", "代码优先")
    wm_text = manager.get_working_memory_text(cid)
    assert "language" in wm_text
    
    # 6. 删除对话
    manager.delete_conversation(cid)
    assert not manager.get_conversation(cid)
    print("✅ 基础三级内存测例全数通过")


def test_user_profile_yaml():
    """测试 YAML 格式的 UserProfileManager"""
    test_path = "conversations/_test_profile.yaml"
    
    try:
        pm = UserProfileManager(profile_path=test_path)
        
        # 1. 空画像初始化
        profile = pm.get_profile()
        assert profile.get("major") is None
        assert profile.get("target_cities") == []
        assert profile.get("education_background") == []
        assert profile.get("project_experience") == []
        assert profile.get("internship_experience") == []
        print("\n--- YAML 用户画像测试 ---")
        print("✅ 空画像初始化通过")
        
        # 2. 标量字段更新
        pm.update_profile({
            "major": "电子信息工程",
            "degree": "硕士",
            "graduation_year": "2026",
            "experience_level": "应届生"
        })
        assert pm.profile["major"] == "电子信息工程"
        assert pm.profile["degree"] == "硕士"
        print("✅ 标量字段更新通过")
        
        # 3. 列表字段并集去重
        pm.update_profile({"target_cities": ["深圳", "上海"]})
        pm.update_profile({"target_cities": ["上海", "杭州"]})
        assert pm.profile["target_cities"] == ["深圳", "上海", "杭州"]
        print(f"✅ 列表去重通过: {pm.profile['target_cities']}")
        
        # 4. 技术栈测试
        pm.update_profile({"tech_stack": ["FPGA", "Verilog"]})
        pm.update_profile({"tech_stack": ["Python", "FPGA"]})
        assert pm.profile["tech_stack"].count("FPGA") == 1
        print(f"✅ 技术栈去重通过: {pm.profile['tech_stack']}")
        
        # 5. Offer 状态合并
        pm.update_profile({"offer_status": {"received": ["华为-硬件工程师"], "pending": ["大疆-算法工程师"]}})
        pm.update_profile({"offer_status": {"received": ["中兴-嵌入式开发", "华为-硬件工程师"]}})
        assert pm.profile["offer_status"]["received"].count("华为-硬件工程师") == 1
        assert "中兴-嵌入式开发" in pm.profile["offer_status"]["received"]
        print(f"✅ Offer状态合并通过: {pm.profile['offer_status']}")
        
        # 6. 学历背景（复合对象列表）
        pm.update_profile({"education_background": [
            {"school": "电子科技大学", "level": "985", "degree": "硕士", "major": "电子信息工程"}
        ]})
        pm.update_profile({"education_background": [
            {"school": "XX大学", "level": "双非", "degree": "本科", "major": "通信工程"},
            {"school": "电子科技大学", "level": "985", "degree": "硕士", "major": "电子信息工程"}  # 重复，应去重
        ]})
        assert len(pm.profile["education_background"]) == 2
        schools = [e["school"] for e in pm.profile["education_background"]]
        assert "电子科技大学" in schools
        assert "XX大学" in schools
        print(f"✅ 学历背景合并通过: {schools}")
        
        # 7. 项目经历
        pm.update_profile({"project_experience": [
            {"name": "智能车控制系统", "tech_stack": ["STM32", "PID"], "description": "基于STM32的智能循迹小车"}
        ]})
        pm.update_profile({"project_experience": [
            {"name": "FPGA图像处理", "tech_stack": ["Verilog", "Quartus"], "description": "基于FPGA的实时图像滤波"},
            {"name": "智能车控制系统", "tech_stack": ["STM32"], "description": "重复项"}  # 按 name 去重
        ]})
        assert len(pm.profile["project_experience"]) == 2
        proj_names = [p["name"] for p in pm.profile["project_experience"]]
        assert "FPGA图像处理" in proj_names
        print(f"✅ 项目经历合并通过: {proj_names}")
        
        # 8. 实习经历
        pm.update_profile({"internship_experience": [
            {"company": "华为", "position": "嵌入式开发实习生", "duration": "3个月"}
        ]})
        pm.update_profile({"internship_experience": [
            {"company": "大疆", "position": "算法实习生", "duration": "2个月"},
            {"company": "华为", "position": "嵌入式开发实习生", "duration": "3个月"}  # 按 company 去重
        ]})
        assert len(pm.profile["internship_experience"]) == 2
        companies = [i["company"] for i in pm.profile["internship_experience"]]
        assert "大疆" in companies
        print(f"✅ 实习经历合并通过: {companies}")
        
        # 9. no_update 测试
        result = pm.update_profile({"no_update": True})
        assert result == False
        print("✅ no_update 跳过通过")
        
        # 10. get_profile_text 输出
        text = pm.get_profile_text()
        print(f"\n画像文本输出:\n{text}")
        assert "电子信息工程" in text
        assert "深圳" in text
        assert "华为" in text
        assert "电子科技大学" in text
        assert "智能车控制系统" in text
        print("✅ get_profile_text 通过（含扩展字段）")
        
        # 11. get_filter_metadata 输出（含项目技术栈补充）
        filters = pm.get_filter_metadata()
        print(f"\n画像过滤字段: {filters}")
        assert "tech_stack" in filters
        # 项目的技术栈应该也被合并进来
        assert "STM32" in filters["tech_stack"]
        assert "Verilog" in filters["tech_stack"]
        print("✅ get_filter_metadata 通过（含项目技术栈补充）")
        
        # 12. YAML 持久化 + 热加载
        pm2 = UserProfileManager(profile_path=test_path)
        assert pm2.profile["major"] == "电子信息工程"
        assert len(pm2.profile["education_background"]) == 2
        assert len(pm2.profile["project_experience"]) == 2
        print("✅ YAML 持久化重载通过")
        
        # 13. 手动编辑模拟（直接修改文件后 reload）
        import yaml
        with open(test_path, "r", encoding="utf-8") as f:
            content = f.read()
        data = yaml.safe_load(content)
        data["major"] = "通信工程_手动修改"
        with open(test_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        
        # 热加载测试
        hot_profile = pm.get_profile()
        assert hot_profile["major"] == "通信工程_手动修改"
        print("✅ 手动编辑后热加载通过")
        
        # 14. reset 测试
        pm.reset()
        assert pm.profile["major"] is None
        assert pm.profile["education_background"] == []
        print("✅ reset 通过")
        
        # 15. 验证 YAML 文件带注释头
        with open(test_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        assert "用户画像" in first_line
        print("✅ YAML 文件注释头通过")
        
        print("\n✅ YAML 用户画像所有测例全数通过")
        
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"🧹 已清理测试文件: {test_path}")


def test_memory_manager_profile_integration():
    """测试 MemoryManager 中画像代理方法（YAML 版本）"""
    test_path = "conversations/_test_profile_int.yaml"
    
    try:
        manager = MemoryManager(profile_path=test_path)
        
        # 手动更新画像
        manager.profile_manager.update_profile({
            "major": "通信工程",
            "tech_stack": ["5G NR", "MATLAB"],
            "education_background": [
                {"school": "北京邮电大学", "level": "211", "degree": "硕士", "major": "通信工程"}
            ],
            "project_experience": [
                {"name": "5G基站信号处理", "tech_stack": ["MATLAB", "Simulink"], "description": "5G NR 信号仿真"}
            ]
        })
        
        # 通过 MemoryManager 代理方法读取
        text = manager.get_profile_text()
        assert "通信工程" in text
        assert "北京邮电大学" in text
        assert "5G基站信号处理" in text
        print(f"\nMemoryManager 画像代理文本:\n{text}")
        
        filters = manager.get_profile_filter()
        assert filters["major"] == "通信工程"
        assert "5G NR" in filters["tech_stack"]
        # 项目技术栈也应包含
        assert "MATLAB" in filters["tech_stack"]
        assert "Simulink" in filters["tech_stack"]
        print(f"MemoryManager 画像代理过滤: {filters}")
        
        print("\n✅ MemoryManager 画像集成测例全数通过")
        
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


class FakeEmptyLLMClient:
    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        return ""


class FakeCodeFenceLLMClient:
    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        return """```json
{"tech_stack": ["Verilog"], "target_cities": ["深圳"]}
```"""


def test_extract_and_update_profile_skips_empty_result():
    """验证画像提取在小模型返回空串时静默跳过，不再触发 json.loads 空串报错。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_dir = os.path.join(temp_dir, "conversations")
        profile_path = os.path.join(temp_dir, "user_profile.yaml")
        manager = MemoryManager(storage=JsonStorage(storage_dir=storage_dir), profile_path=profile_path)

        cid = manager.create_conversation("画像空结果测试")
        manager.add_message(cid, "user", "你好")
        manager.add_message(cid, "assistant", "你好，我可以帮你分析求职问题。")

        updated = manager.extract_and_update_profile(cid, FakeEmptyLLMClient())

        assert updated is False
        assert manager.profile_manager.get_profile()["tech_stack"] == []


def test_extract_and_update_profile_accepts_code_fence_json():
    """验证画像提取可以解析被 Markdown 代码块包裹的 JSON。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_dir = os.path.join(temp_dir, "conversations")
        profile_path = os.path.join(temp_dir, "user_profile.yaml")
        manager = MemoryManager(storage=JsonStorage(storage_dir=storage_dir), profile_path=profile_path)

        cid = manager.create_conversation("画像代码块测试")
        manager.add_message(cid, "user", "我会 Verilog，想去深圳找工作")
        manager.add_message(cid, "assistant", "已记录你的技术栈和目标城市。")

        updated = manager.extract_and_update_profile(cid, FakeCodeFenceLLMClient())

        assert updated is True
        profile = manager.profile_manager.get_profile()
        assert "Verilog" in profile["tech_stack"]
        assert "深圳" in profile["target_cities"]


if __name__ == "__main__":
    test_memory()
    test_user_profile_yaml()
    test_memory_manager_profile_integration()
    test_extract_and_update_profile_skips_empty_result()
    test_extract_and_update_profile_accepts_code_fence_json()
