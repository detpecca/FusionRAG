"""`python -m fusionrag` 启动 API 服务。"""

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "fusionrag.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
