"""
Grounding system text + structured citations from selected chunks.

Pure string I/O — no network, no FAISS.
"""

from __future__ import annotations

from typing import Any

from rag.chunk_meta import ensure_chunk_metadata
from rag.prompts.text import safe_message_text


class RAGPromptBuilder:
    """Turn ranked chunk dicts into a grounding block and API citation list."""

    def __init__(self, *, rag_block_limit: int = 2600, citation_text_limit: int = 480) -> None:
        self.rag_block_limit = rag_block_limit
        self.citation_text_limit = citation_text_limit

    def build(self, selected: list[dict[str, Any]], *, conversational: bool = False) -> tuple[str, list[dict[str, Any]]]:
        if not selected:
            return "", []

        header_lines = [
            "【知识库摘录】每段含编号 [S1]、[S2]…，以及元数据 chunk_id 与 source_name（出处标签）。你是「仅依据摘录作答」模式，必须遵守：",
            "- 回答必须严格基于下列摘录中的内容；不得引入摘录未出现的书名细节、人物、情节、原话或常识补充（如同人设定）。",
            "- 若摘录不足以完整回答、或问题超出摘录范围：必须先在回复中明确写出「我不知道」或「当前摘录不足以回答这个问题」（二选一或等价清晰表述），简要说明缺什么；之后最多再给一两句不依赖虚构情节的温和陪伴，且不得假装摘录里有。",
            "- 引用规范（最终回复须同时满足）：(1) 凡引用摘录内容，须在对应句末或句中标出所用 chunk_id（示例：（依据 kb_002））；(2) 还须标明对应的 source_name 出处标签（示例：〔出处：little_prince_kb〕或与 chunk_id 成对写出）。",
            "- 结尾建议单独一行作引用汇总，格式示例：「引用：kb_001 (little_prince_kb)、kb_003 (little_prince_kb)」——涵盖本次用到的全部 chunk_id 与 source_name；若仅用一条，二者仍都要出现。",
            "- 可同时保留 [S1] 等与摘录编号对照；chunk_id 与 source_name 为强制引用信息，不可省略。",
            "",
        ]
        if conversational:
            header_lines = [
                "【聊天参考】这些资料帮助你理解原著，不改变前面的朋友聊天语气。先回答用户当前问题，通常 1-4 句。",
                "- 具体人物、情节、数字和原话要有资料依据；内容梗概和释义不是逐字原文，不能加引号冒充原话。",
                "- 对含义和为什么，可以结合资料与上下文作合理解释。用‘我觉得’或‘可以这样理解’自然区分解读与原文，不因资料没有逐字写出解释就拒答。",
                "- 转述不能改变事实、因果和时间关系：不要把静态特征说成发生过的变化，不把假设举例说成实际经历，不为解释补造细节。",
                "- 只有具体事实确实无法确定时，简短说明哪一点不知道，再回答有依据的部分；不要把‘当前摘录不足’当成每次开头。用户提出资料里没有的前提，也不要当作原著事实。",
                "- 正文不写资料编号、内部字段、数据库名、引用汇总或参考标签。引用信息由接口单独提供；用户问出处时，可以说自然的章节名称。",
                "- 短追问接着上一轮解释，不重复定义；换回日常话题就跟着聊，不强行继续讲故事。",
                "",
            ]
        body_lines: list[str] = []
        citations: list[dict[str, Any]] = []
        for i, c in enumerate(selected, start=1):
            label = f"S{i}"
            meta = ensure_chunk_metadata(c)
            cid = meta["chunk_id"]
            src_name = meta["source_name"]
            text = str(c.get("text", ""))
            body_lines.append(f"参考内容：{c.get('chapter') or '相关资料'}" if conversational else f"[{label}] chunk_id={cid} source_name={src_name}")
            body_lines.append(text)
            body_lines.append("")
            citations.append(
                {
                    "label": label,
                    "chunk_id": cid,
                    "source_name": src_name,
                    "text": safe_message_text(text, limit=self.citation_text_limit),
                }
            )

            for key in ("chapter", "source_url", "chunk_type"):
                if c.get(key):
                    citations[-1][key] = c[key]

        rag_block = safe_message_text("\n".join(header_lines + body_lines), limit=self.rag_block_limit)
        return rag_block, citations

