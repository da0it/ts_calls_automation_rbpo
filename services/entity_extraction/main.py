# services/entity-extraction/main.py
import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from extractor.entity_extractor import EntityExtractor
from extractor.models import Segment, Entities

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# State для хранения экстрактора
class AppState:
    extractor: Optional[EntityExtractor] = None
    startup_error: str = ""

state = AppState()

# Создание приложения
app = FastAPI(
    title="Entity Extraction Service",
    description="NER сервис для извлечения сущностей из диалогов",
    version="1.0.0"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _service_status() -> dict:
    if state.extractor is None:
        return {
            "status": "starting",
            "service": "entity-extraction",
            "mode": "starting",
            "ner_loaded": False,
            "ready": False,
            "startup_error": state.startup_error or "",
        }

    degraded = state.extractor.ner_model is None
    return {
        "status": "degraded" if degraded else "healthy",
        "service": "entity-extraction",
        "mode": state.extractor.mode,
        "ner_loaded": not degraded,
        "ready": True,
        "startup_error": state.extractor.startup_error or state.startup_error or "",
    }


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Entity Extraction Service...")
    use_ner = _env_bool("ENTITY_USE_NER", True)
    allow_download = _env_bool("ENTITY_NER_DOWNLOAD_ON_STARTUP", False)
    allow_install = _env_bool("ENTITY_NER_INSTALL_ON_STARTUP", False)

    try:
        logger.info(
            "Initializing EntityExtractor: use_ner=%s download_on_startup=%s install_on_startup=%s",
            use_ner,
            allow_download,
            allow_install,
        )
        state.extractor = EntityExtractor(
            use_ner=use_ner,
            allow_download=allow_download,
            allow_install=allow_install,
        )
        state.startup_error = state.extractor.startup_error
        if state.extractor.ner_model is not None:
            logger.info("Entity Extraction Service is ready with DeepPavlov NER")
        else:
            logger.warning(
                "Entity Extraction Service is ready in regex-only mode. startup_error=%s",
                state.startup_error or "none",
            )
    except Exception as e:
        state.startup_error = str(e)
        logger.error(
            "Failed to initialize EntityExtractor with NER, falling back to regex-only mode: %s",
            e,
            exc_info=True,
        )
        state.extractor = EntityExtractor(use_ner=False)
        logger.warning("Entity Extraction Service started in forced regex-only mode")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Entity Extraction Service...")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response модели
class ExtractRequest(BaseModel):
    segments: List[dict]  # List of {start, end, speaker, role, text}

    class Config:
        schema_extra = {
            "example": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "SPEAKER_00",
                        "role": "звонящий",
                        "text": "Здравствуйте, меня зовут Иван Петров"
                    }
                ]
            }
        }


class ExtractResponse(BaseModel):
    entities: Entities


# Эндпоинты
@app.post("/api/extract-entities", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    """
    Извлекает сущности из сегментов диалога
    """
    if state.extractor is None:
        raise HTTPException(
            status_code=503,
            detail="EntityExtractor not initialized. Service is starting up."
        )
    
    try:
        logger.info(f"Extracting entities from {len(request.segments)} segments")
        
        # Конвертируем в модели
        segments = [Segment(**seg) for seg in request.segments]
        
        # Извлекаем сущности
        entities = state.extractor.extract(segments)
        
        logger.info(f"Extracted: {len(entities.persons)} persons, "
                   f"{len(entities.phones)} phones, {len(entities.emails)} emails")
        
        return ExtractResponse(entities=entities)
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return _service_status()


@app.get("/")
async def root():
    """Root endpoint"""
    status = _service_status()
    return {
        "service": "Entity Extraction Service",
        "version": "1.0.0",
        "description": "NER сервис для извлечения сущностей из транскрибированных диалогов",
        "endpoints": {
            "extract": "POST /api/extract-entities",
            "health": "GET /health",
            "docs": "GET /docs",
            "redoc": "GET /redoc"
        },
        "status": status["status"],
        "mode": status["mode"],
        "ner_loaded": status["ner_loaded"],
        "ready": status["ready"],
        "startup_error": status["startup_error"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001,
        log_level="info"
    )
