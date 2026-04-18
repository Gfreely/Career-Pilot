import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import (
    get_interview_question_service,
    get_memory_manager,
    get_profile_analysis_service,
)
from src.memory import JsonStorage, MemoryManager
from src.services import InterviewQuestionService, ProfileAnalysisService


class FakeAnalysisLLMClient:
    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        return """
        {
          "summary": "适合冲刺嵌入式方向，但项目深度仍需加强",
          "match_score": 78,
          "strengths": ["专业方向匹配", "有项目经历"],
          "gaps": ["缺少量化成果"],
          "risks": ["目标岗位竞争激烈"],
          "action_plan": ["补齐项目指标", "完善简历"],
          "suggested_roles": ["嵌入式工程师", "硬件测试工程师"],
          "interview_focus": ["项目细节追问", "基础知识复盘"]
        }
        """


class FakeInterviewLLMClient:
    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        return """
        {
          "target_position": "嵌入式工程师",
          "difficulty": "中等",
          "question_count": 2,
          "questions": [
            {
              "question": "请介绍你在智能车项目中如何做控制闭环设计？",
              "question_type": "项目深挖",
              "focus": "控制系统设计",
              "reference_answer": "说明采样、控制算法和调参过程",
              "follow_up": "如果出现震荡你会如何定位？",
              "reason": "用户有相关项目经历"
            },
            {
              "question": "SPI 和 I2C 的典型差异是什么？",
              "question_type": "八股",
              "focus": "通信总线基础",
              "reference_answer": "从速率、线数、主从机制等角度回答",
              "follow_up": "你在项目里更常用哪一种？",
              "reason": "目标岗位需要硬件接口基础"
            }
          ]
        }
        """


class FakeProfileAnalysisService:
    def analyze(self, **kwargs):
        return {
            "summary": "分析结果",
            "match_score": 81,
            "strengths": ["优势A"],
            "gaps": ["短板B"],
            "risks": ["风险C"],
            "action_plan": ["动作D"],
            "suggested_roles": ["岗位E"],
            "interview_focus": ["准备点F"],
        }


class FakeInterviewQuestionService:
    def generate_questions(self, **kwargs):
        return {
            "target_position": "嵌入式工程师",
            "difficulty": "中等",
            "question_count": 1,
            "questions": [
                {
                    "question": "请介绍一个你做过的嵌入式项目",
                    "question_type": "项目深挖",
                    "focus": "项目表达",
                    "reference_answer": "从背景、方案、结果展开",
                    "follow_up": "项目里最难的问题是什么？",
                    "reason": "画像里有项目经验",
                }
            ],
        }


def build_test_memory_manager(temp_dir: str) -> MemoryManager:
    storage_dir = Path(temp_dir) / "conversations"
    profile_path = Path(temp_dir) / "user_profile.yaml"
    storage = JsonStorage(storage_dir=str(storage_dir))
    return MemoryManager(storage=storage, profile_path=str(profile_path))


def test_profile_analysis_service_and_interview_service():
    """验证分析服务与面试题服务的结构化输出。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        conversation_id = manager.create_conversation("分析测试")
        manager.current_conversation_id = conversation_id
        manager.profile_manager.update_profile(
            {
                "major": "电子信息工程",
                "project_experience": [
                    {
                        "name": "智能车控制系统",
                        "tech_stack": ["STM32"],
                        "description": "比赛项目",
                    }
                ],
            }
        )
        manager.add_message(conversation_id, "user", "我想找嵌入式岗位")
        manager.add_message(conversation_id, "assistant", "可以重点准备项目表达")

        analysis_service = ProfileAnalysisService(llm_client=FakeAnalysisLLMClient())
        analysis = analysis_service.analyze(
            conversation_manager=manager,
            target_position="嵌入式工程师",
            target_city="深圳",
        )
        assert analysis["match_score"] == 78
        assert "专业方向匹配" in analysis["strengths"]

        interview_service = InterviewQuestionService(llm_client=FakeInterviewLLMClient())
        questions = interview_service.generate_questions(
            conversation_manager=manager,
            target_position="嵌入式工程师",
            question_count=2,
            question_types=["项目深挖", "八股"],
        )
        assert questions["question_count"] == 2
        assert questions["questions"][0]["question_type"] == "项目深挖"


def test_analysis_and_interview_api_routes():
    """验证分析与面试题 API 路由可用。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = build_test_memory_manager(temp_dir)
        app = create_app()

        app.dependency_overrides[get_memory_manager] = lambda: manager
        app.dependency_overrides[get_profile_analysis_service] = lambda: FakeProfileAnalysisService()
        app.dependency_overrides[get_interview_question_service] = lambda: FakeInterviewQuestionService()

        client = TestClient(app)

        analysis_resp = client.post(
            "/api/profile/analyze",
            json={
                "target_position": "嵌入式工程师",
                "target_city": "深圳",
                "target_direction": "嵌入式",
                "notes": "优先考虑校招",
            },
        )
        assert analysis_resp.status_code == 200
        assert analysis_resp.json()["match_score"] == 81

        interview_resp = client.post(
            "/api/interview/questions/generate",
            json={
                "target_position": "嵌入式工程师",
                "difficulty": "中等",
                "question_count": 1,
                "question_types": ["项目深挖"],
            },
        )
        assert interview_resp.status_code == 200
        assert interview_resp.json()["questions"][0]["question_type"] == "项目深挖"
