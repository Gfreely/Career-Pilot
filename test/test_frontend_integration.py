import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_frontend_static_mount():
    """验证独立前端页面与静态资源已接入 FastAPI。"""
    client = TestClient(create_app())

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "XinghuoLLM 工作台" in index_response.text
    assert "chat-grid" in index_response.text
    assert "scrollToLatestBtn" in index_response.text

    js_response = client.get("/assets/app.js")
    assert js_response.status_code == 200
    assert "bootstrap" in js_response.text
    assert "renderMarkdown" in js_response.text
    assert "settlePendingAssistantMessage" in js_response.text
    assert "discardPendingAssistantMessage" in js_response.text
    assert "syncConversationListInBackground" in js_response.text
    assert 'await loadConversation(state.currentConversationId)' not in js_response.text

    css_response = client.get("/assets/styles.css")
    assert css_response.status_code == 200
    assert "--accent" in css_response.text
    assert ".chat-grid" in css_response.text
    assert ".markdown-body" in css_response.text
    assert "position: sticky" in css_response.text
    assert "overscroll-behavior: contain" in css_response.text
