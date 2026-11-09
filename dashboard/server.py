"""窗口的后端：薄薄一层 FastAPI，把 URL 与请求体里的名字校验后交给 data.py（读日志）和 worker（生成）。

只读 runs/，不写任何训练状态。处理函数都是普通 def，FastAPI 会放进线程池，
一次慢的生成不会堵住页面每 5 秒一次的轮询。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard import data
from dashboard.worker import WorkerClient, WorkerError, WorkerTimeout
from minids.logging_io import EVAL_FILE, EVENTS_FILE, METRICS_FILE, SAMPLES_FILE, read_jsonl

STATIC_DIR = Path(__file__).resolve().parent / "static"


class GenerateRequest(BaseModel):
    run: str
    checkpoint: str
    prefix: str = Field(min_length=1, max_length=20000)
    suffix: str | None = Field(default=None, max_length=20000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    device: Literal["cpu", "cuda"] = "cpu"


def create_app(runs_dir: Path, worker=None, repo_root: Path | None = None) -> FastAPI:
    runs_dir = Path(runs_dir)
    gen_worker = worker if worker is not None else WorkerClient(repo_root=repo_root)  # 进程在第一次生成时才启动

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        gen_worker.close()

    app = FastAPI(title="黑米 Heimi 训练窗口", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(data.RunNotFound)
    async def _not_found(request: Request, exc: data.RunNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return PlainTextResponse("前端尚未构建：缺少 dashboard/static/index.html")
        return FileResponse(page, media_type="text/html")

    @app.get("/api/runs")
    def api_runs() -> list[dict]:
        return data.list_runs(runs_dir)

    @app.get("/api/runs/{run}/metrics")
    def api_metrics(run: str, since: int | None = None) -> list[dict]:
        return read_jsonl(data.safe_run_dir(runs_dir, run) / METRICS_FILE, since)

    @app.get("/api/runs/{run}/eval")
    def api_eval(run: str) -> list[dict]:
        return read_jsonl(data.safe_run_dir(runs_dir, run) / EVAL_FILE)

    @app.get("/api/runs/{run}/events")
    def api_events(run: str) -> list[dict]:
        return read_jsonl(data.safe_run_dir(runs_dir, run) / EVENTS_FILE)

    @app.get("/api/runs/{run}/samples")
    def api_samples(run: str) -> list[dict]:
        return read_jsonl(data.safe_run_dir(runs_dir, run) / SAMPLES_FILE)

    @app.get("/api/runs/{run}/overview")
    def api_overview(run: str) -> dict:
        return data.overview(data.safe_run_dir(runs_dir, run))

    @app.get("/api/status")
    def api_status(run: str | None = None) -> dict:
        return data.status(runs_dir, run=run)  # 页面带上正在看的 run；不带则报最近更新的那个

    @app.post("/api/generate")
    def api_generate(body: GenerateRequest) -> dict:
        run_dir = data.safe_run_dir(runs_dir, body.run)
        ckpt = data.safe_checkpoint(run_dir, body.checkpoint)  # 只有列得出来的检查点名才会走到生成进程
        req = {
            "ckpt_path": str(ckpt.absolute()),
            "prefix": body.prefix,
            "suffix": body.suffix,
            "max_new_tokens": body.max_new_tokens,
            "temperature": body.temperature,
            "device": body.device,
        }
        try:
            result = gen_worker.generate(req)
        except WorkerTimeout as e:
            raise HTTPException(status_code=504, detail=str(e)) from e
        except WorkerError as e:
            raise HTTPException(status_code=500, detail=f"生成失败：{e}") from e
        return {
            "text": result.get("text", ""),
            "seconds": result.get("seconds"),
            "tokens": result.get("tokens"),
            "checkpoint": body.checkpoint,
            "device": body.device,
        }

    # check_dir=False：静态目录不存在（前端还没建）时应用照样能启动，取不到的文件就是 404
    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    return app
