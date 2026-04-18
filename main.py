from __future__ import annotations

from pathlib import Path

import uvicorn


def main() -> None:
    """默认启动 FastAPI 服务，完成前端与业务层的最终解耦。"""
    Path("conversations").mkdir(exist_ok=True)
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    main()
