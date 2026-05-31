"""Gradio demo for ModelScope Spaces — K12 数学教研 RAG 问答."""

from __future__ import annotations

import os

import gradio as gr

from rag_query import query

# ModelScope 创空间 Secrets 使用 DASHSCOPE_API_KEY
os.environ.setdefault(
    "OPENAI_API_KEY",
    os.environ.get("DASHSCOPE_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
)

# 拉宽整页布局（创空间默认较窄）
CUSTOM_CSS = """
.gradio-container {
    max-width: min(1200px, 96vw) !important;
    width: 96% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.contain {
    max-width: min(1200px, 96vw) !important;
}
"""


def ask(question: str) -> tuple[str, str]:
    if not (question or "").strip():
        return "请输入问题", ""

    try:
        result = query(question)
    except Exception as exc:
        return f"查询失败：{exc}", "（无引用来源）"

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    if sources:
        lines = []
        for i, chunk in enumerate(sources[:5],1):
            src = chunk.get("source", "未知来源")
            # 提取文件名并简化
            import os
            filename = os.path.basename(src)
            filename = filename.replace("【人教版】", "人教版·").replace("数学电子课本.pdf", "").replace("(1).pdf", "").replace(".pdf", "").strip()
            src = filename
            content = (chunk.get("content", "") or "")[:80]
            lines.append(f"{i}. 📖 {src}\n   {content}...")
        sources_text = "\n\n".join(lines)
    else:
        sources_text = "（无引用来源）"

    return answer, sources_text


EXAMPLES = [
    "一元一次方程怎么解？",
    "什么是等腰三角形？",
    "勾股定理的内容是什么？",
    "导数是什么？",
    "平行线有哪些性质？",
]

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(
        label="输入问题",
        placeholder="例：勾股定理的内容是什么？",
        lines=4,
    ),
    outputs=[
        gr.Textbox(label="回答", lines=12),
        gr.Textbox(label="引用来源", lines=8),
    ],
    title="K12 数学教研 RAG 问答",
    description=(
        "面向初中数学教研场景的模块化 RAG 问答，支持混合检索与引用级回答。"
        "超纲问题（如导数）会明确提示。"
    ),
    examples=EXAMPLES,
    examples_per_page=len(EXAMPLES),
    cache_examples=False,
    flagging_mode="never",
    fill_width=True,
    css=CUSTOM_CSS,
)

if __name__ == "__main__":
    demo.launch()
