"""
mcp_stub.py — MCP 工具调用占位骨架

当前状态：STUB（占位模式）
所有函数已定义标准接口，但实际数据由提示文字代替。
接入真实 MCP 服务时，只需在各函数的 `# TODO` 处替换实现。

接口规范：
  每个执行函数返回 dict:
    {
        "context": str,        # 可直接注入 system prompt 的文本
        "status":  str,        # "ok" | "stub" | "error"
        "source":  str,        # 来源标识
        "entities": dict,      # 透传的实体信息（方便调试）
    }
"""

from __future__ import annotations
from typing import Optional


# ============================================================
# 内部常量
# ============================================================
_STATUS_STUB = "stub"
_STATUS_OK   = "ok"
_STATUS_ERR  = "error"

_STUB_NOTICE = (
    "⚠️ [占位模式] 当前岗位/公司实时数据尚未接入真实 MCP 服务，"
    "以下信息基于常识生成，仅供参考，不代表真实情况。"
)


# ============================================================
# 岗位实时查询 (MCP_JOB)
# ============================================================

def execute_mcp_job_retrieval(
    entities: dict,
    rewritten_query: str,
    timeout: float = 5.0,        # 超时秒数（真实接入时启用）
) -> dict:
    """
    查询特定公司/岗位的实时招聘信息（HC 状态、投递方式、补招等）。

    Parameters
    ----------
    entities : dict
        由 analyse_query 提取的实体信息，包含 company/position/location 等字段。
    rewritten_query : str
        改写后的统一检索短语，可直接用于 MCP 搜索参数。
    timeout : float
        超时阈值（秒）。超时后降级为 STUB 返回，不阻塞主流程。

    Returns
    -------
    dict
        包含 context、status、source、entities 的标准化结果字典。
    """
    company  = entities.get("company") or "目标公司"
    position = entities.get("position") or "目标岗位"
    location = entities.get("location") or "不限城市"

    # ------------------------------------------------------------------
    # TODO: 接入真实 MCP SDK
    # 示例（伪代码）：
    #   from mcp_sdk import JobSearchClient
    #   client = JobSearchClient(api_key=os.getenv("MCP_API_KEY"))
    #   result = client.search(
    #       company=company, position=position, location=location,
    #       timeout=timeout
    #   )
    #   context = _format_job_result(result)
    #   return {"context": context, "status": _STATUS_OK, "source": "MCP_JOB", "entities": entities}
    # ------------------------------------------------------------------

    stub_context = (
        f"{_STUB_NOTICE}\n\n"
        f"【岗位查询占位信息】\n"
        f"查询关键词：{rewritten_query}\n"
        f"目标公司：{company} ｜ 岗位：{position} ｜ 城市：{location}\n\n"
        f"当前尚未接入实时招聘数据源。请建议用户前往官方招聘网站或 Boss 直聘、"
        f"拉勾、牛客网等渠道查询最新 HC 状态和投递入口。"
    )

    return {
        "context":  stub_context,
        "status":   _STATUS_STUB,
        "source":   "MCP_JOB",
        "entities": entities,
    }


# ============================================================
# 公司评价与薪资查询 (MCP_COMPANY)
# ============================================================

def execute_mcp_company_insight(
    entities: dict,
    rewritten_query: str,
    timeout: float = 5.0,
) -> dict:
    """
    查询特定公司的口碑、薪资、加班情况、技术氛围、面试难度等信息。

    Parameters
    ----------
    entities : dict
        由 analyse_query 提取的实体信息。
    rewritten_query : str
        改写后的检索短语。
    timeout : float
        超时阈值（秒）。

    Returns
    -------
    dict
        包含 context、status、source、entities 的标准化结果字典。
    """
    company  = entities.get("company") or "目标公司"
    position = entities.get("position") or "相关岗位"

    # ------------------------------------------------------------------
    # TODO: 接入真实 MCP SDK（如 Maimai/看准/脉脉 数据源）
    # 示例（伪代码）：
    #   from mcp_sdk import CompanyInsightClient
    #   client = CompanyInsightClient(api_key=os.getenv("MCP_API_KEY"))
    #   result = client.get_insight(company=company, position=position, timeout=timeout)
    #   context = _format_company_result(result)
    #   return {"context": context, "status": _STATUS_OK, "source": "MCP_COMPANY", "entities": entities}
    # ------------------------------------------------------------------

    stub_context = (
        f"{_STUB_NOTICE}\n\n"
        f"【公司评价占位信息】\n"
        f"查询关键词：{rewritten_query}\n"
        f"目标公司：{company} ｜ 相关岗位：{position}\n\n"
        f"当前尚未接入实时公司评价数据源。请建议用户参考以下渠道获取真实口碑：\n"
        f"- 看准网 (kanzhun.com)：薪资数据、员工评价\n"
        f"- 脉脉 (maimai.cn)：内部动态、职级体系\n"
        f"- 牛客网 (nowcoder.com)：面试经验、OC 情况\n"
        f"- 知乎同名话题：综合评价与技术氛围"
    )

    return {
        "context":  stub_context,
        "status":   _STATUS_STUB,
        "source":   "MCP_COMPANY",
        "entities": entities,
    }
