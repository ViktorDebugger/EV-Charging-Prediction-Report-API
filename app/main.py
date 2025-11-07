try:
    from typing import List
    _closure = getattr(List.__getitem__, "__closure__", None)
    if _closure:
        _cell = _closure[0].cell_contents
        if not hasattr(_cell, "cache_clear"):
            setattr(_cell, "cache_clear", lambda: None)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import training, inference, monitor

app = FastAPI(
    title="ML Model API",
    description="API для тренування моделі та передбачення",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router, prefix="", tags=["training"])
app.include_router(inference.router, prefix="", tags=["inference"])
app.include_router(monitor.router, prefix="/monitor", tags=["monitor"])


'''
python -m uvicorn app.main:app --reload
'''