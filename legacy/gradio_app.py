from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

from src.memory import MemoryManager
from src.services import ChatService, load_prompts


chat_service = ChatService()


custom_css = """
#chatbot .user {
    font-family: 'Arial', 'STXihei', sans-serif !important;
    border-radius: 10px;
}

#chatbot .bot {
    font-family: 'Arial', 'STXihei', sans-serif !important;
    border-radius: 10px;
}

.avatar-container {
    width: 50px !important;
    height: 50px !important;
}

#thinking_box {
    font-family: 'Verdana', sans-serif !important;
    font-size: 24px;
    border-radius: 10px;
}
"""


def convert_messages_to_chat_history(messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """将持久化消息转换为 Gradio Chatbot 需要的二元组格式。"""
    chat_history: List[Tuple[str, str]] = []
    index = 0

    while index < len(messages):
        current = messages[index]
        if (
            current["role"] == "user"
            and index + 1 < len(messages)
            and messages[index + 1]["role"] == "assistant"
        ):
            chat_history.append((current["content"], messages[index + 1]["content"]))
            index += 2
            continue
        index += 1

    return chat_history


def create_interface():
    """创建旧版 Gradio 界面，业务逻辑仍通过服务层调用。"""
    conversation_manager = MemoryManager()
    if not conversation_manager.get_all_conversations():
        conversation_manager.create_conversation()

    prompts = load_prompts()

    with gr.Blocks(
        title="“就”决定是你了",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        gr.Markdown("# 电子信息学习就业助手")

        current_conversation_id = gr.State(conversation_manager.current_conversation_id)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=600,
                    avatar_images=("data/avatar/user_avatar.jpg", "data/avatar/bot_avatar.jpg"),
                    show_copy_button=True,
                    elem_id="chatbot",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="请输入您的问题...",
                        container=False,
                        scale=8,
                    )
                    submit_btn = gr.Button("发送", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=[
                            "Pro/MiniMaxAI/MiniMax-M2.5",
                            "deepseek-ai/DeepSeek-R1",
                            "Qwen/Qwen3.5-397B-A17B",
                        ],
                        value="Pro/MiniMaxAI/MiniMax-M2.5",
                        label="模型选择",
                        interactive=True,
                    )

                    prompt_template_selector = gr.Dropdown(
                        choices=[(value["description"], key) for key, value in prompts.items()],
                        value="default",
                        label="提示词模板",
                        interactive=True,
                    )

                    stream_checkbox = gr.Checkbox(
                        value=True,
                        label="流式输出",
                        info="是否使用流式响应模式",
                        interactive=True,
                    )

                thinking_box = gr.Textbox(
                    label="思考过程",
                    placeholder="这里将显示模型的思考过程（如果有）",
                    lines=5,
                    interactive=False,
                    visible=True,
                    elem_id="thinking_box",
                )

            with gr.Column(scale=1):
                gr.Markdown("## 对话管理")

                with gr.Row():
                    with gr.Column():
                        conversation_dropdown = gr.Dropdown(
                            choices=[
                                (conv["title"], conv["id"])
                                for conv in conversation_manager.get_all_conversations()
                            ],
                            value=conversation_manager.current_conversation_id,
                            label="选择对话",
                            interactive=True,
                        )

                with gr.Row():
                    new_conversation_btn = gr.Button("新建对话")
                    delete_conversation_btn = gr.Button("删除对话")

                    title_input = gr.Textbox(
                        label="对话标题",
                        placeholder="输入新的对话标题...",
                        interactive=True,
                    )
                    rename_btn = gr.Button("重命名")

        def respond(message, chat_history, conversation_id, stream_mode, model, prompt_template):
            if not message:
                return "", chat_history, conversation_id, ""

            if not conversation_id:
                conversation_id = conversation_manager.create_conversation()

            conversation_manager.current_conversation_id = conversation_id
            chat_history = chat_history + [(message, None)]

            response_generator = chat_service.generate_response(
                message=message,
                conversation_manager=conversation_manager,
                stream_mode=stream_mode,
                model=model,
                prompt_template=prompt_template,
            )

            for thinking, content in response_generator:
                if content is not None:
                    chat_history[-1] = (message, content)
                yield "", chat_history, conversation_id, thinking

        def create_new_conversation():
            new_id = conversation_manager.create_conversation()
            conversation_manager.current_conversation_id = new_id
            return (
                [],
                "",
                "",
                new_id,
                gr.Dropdown(
                    choices=[
                        (conv["title"], conv["id"])
                        for conv in conversation_manager.get_all_conversations()
                    ],
                    value=new_id,
                ),
            )

        def clear_current_chat():
            return [], "", ""

        def load_conversation(conversation_id):
            if not conversation_id:
                return [], "", ""

            conversation_manager.current_conversation_id = conversation_id
            conversation = conversation_manager.get_conversation(conversation_id)
            title = conversation.get("title", "")
            chat_history = convert_messages_to_chat_history(conversation.get("messages", []))
            return chat_history, title, conversation_id

        def delete_conversation(conversation_id):
            if not conversation_id:
                return gr.Dropdown(), "", "", []

            conversation_manager.delete_conversation(conversation_id)
            all_conversations = conversation_manager.get_all_conversations()

            if all_conversations:
                new_id = all_conversations[0]["id"]
                title = all_conversations[0]["title"]
            else:
                new_id = conversation_manager.create_conversation()
                title = ""
                all_conversations = conversation_manager.get_all_conversations()

            conversation_manager.current_conversation_id = new_id
            return (
                gr.Dropdown(
                    choices=[(conv["title"], conv["id"]) for conv in all_conversations],
                    value=new_id,
                ),
                new_id,
                title,
                [],
            )

        def rename_conversation(conversation_id, new_title):
            if not conversation_id or not new_title:
                return gr.Dropdown(), ""

            conversation_manager.update_conversation_title(conversation_id, new_title)
            return (
                gr.Dropdown(
                    choices=[
                        (conv["title"], conv["id"])
                        for conv in conversation_manager.get_all_conversations()
                    ],
                    value=conversation_id,
                ),
                "",
            )

        msg.submit(
            fn=respond,
            inputs=[
                msg,
                chatbot,
                current_conversation_id,
                stream_checkbox,
                model_selector,
                prompt_template_selector,
            ],
            outputs=[msg, chatbot, current_conversation_id, thinking_box],
        )
        submit_btn.click(
            fn=respond,
            inputs=[
                msg,
                chatbot,
                current_conversation_id,
                stream_checkbox,
                model_selector,
                prompt_template_selector,
            ],
            outputs=[msg, chatbot, current_conversation_id, thinking_box],
        )
        conversation_dropdown.change(
            fn=load_conversation,
            inputs=conversation_dropdown,
            outputs=[chatbot, title_input, current_conversation_id],
        )
        new_conversation_btn.click(
            fn=create_new_conversation,
            inputs=None,
            outputs=[chatbot, thinking_box, msg, current_conversation_id, conversation_dropdown],
        )
        delete_conversation_btn.click(
            fn=delete_conversation,
            inputs=current_conversation_id,
            outputs=[conversation_dropdown, current_conversation_id, title_input, chatbot],
        )
        clear_btn.click(
            fn=clear_current_chat,
            inputs=None,
            outputs=[chatbot, msg, thinking_box],
        )
        rename_btn.click(
            fn=rename_conversation,
            inputs=[current_conversation_id, title_input],
            outputs=[conversation_dropdown, title_input],
        )

    return demo


def main() -> None:
    """启动旧版 Gradio 界面。"""
    Path("conversations").mkdir(exist_ok=True)
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1")


if __name__ == "__main__":
    main()
