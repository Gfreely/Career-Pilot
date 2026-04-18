from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict


def strip_code_fence(text: str) -> str:
    """移除模型常见的 Markdown 代码块包裹。"""
    cleaned = (text or "").strip()
    if "```json" in cleaned:
        return cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in cleaned:
        return cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned


def extract_first_json_object(text: str) -> str:
    """从文本中提取首个花括号包裹的 JSON 对象，忽略字符串内部括号。"""
    start = text.find("{")
    if start < 0:
        raise ValueError("未找到 JSON 对象起始位置")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("JSON 对象未正常闭合")


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _insert_missing_commas_between_fields(text: str) -> str:
    """修复常见的字段间漏逗号场景，例如 `"a": 1 "b": 2`。"""
    chars = []
    in_string = False
    escape = False
    prev_significant = ""

    for index, char in enumerate(text):
        if in_string:
            chars.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            lookahead = text[index:]
            starts_next_key = re.match(r'"[^"]+"\s*:', lookahead) is not None
            if starts_next_key and prev_significant in {'"', "]", "}", "e", "l"}:
                chars.append(",")
            in_string = True
            chars.append(char)
            prev_significant = char
            continue

        chars.append(char)
        if not char.isspace():
            prev_significant = char

    return "".join(chars)


def _normalize_json_candidate(text: str) -> str:
    normalized = (
        text.replace("\ufeff", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    normalized = _remove_trailing_commas(normalized)
    normalized = _insert_missing_commas_between_fields(normalized)
    return normalized


def _try_literal_eval(text: str) -> Dict[str, Any]:
    python_like = (
        text.replace(": null", ": None")
        .replace(": true", ": True")
        .replace(": false", ": False")
    )
    result = ast.literal_eval(python_like)
    if not isinstance(result, dict):
        raise ValueError("解析结果不是对象")
    return result


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    """尽量从模型输出中恢复出合法 JSON 对象。"""
    if not raw_text or not raw_text.strip():
        raise ValueError("模型未返回可解析内容")

    cleaned = strip_code_fence(raw_text)
    candidate = extract_first_json_object(cleaned)

    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    repaired = _normalize_json_candidate(candidate)
    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return _try_literal_eval(repaired)
