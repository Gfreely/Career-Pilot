from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import os
import shutil

from src.api.dependencies import (
    get_memory_manager,
    get_profile_analysis_service,
    get_profile_service,
)
from src.api.schemas.profile import (
    ProfileAnalysisRequest,
    ProfileAnalysisResponse,
    ProfileBundleResponse,
    ProfileFileResponse,
    ProfileFileUpdateRequest,
    ProfileFileUpdateResponse,
    ResumeUploadResponse,
    UserProfilePayload,
)
from src.memory import MemoryManager
from src.services import ProfileAnalysisService, ProfileService
from src.utils.pdf_parser import PdfToMarkdownParser


router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("", response_model=ProfileBundleResponse)
def get_profile(
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """获取当前用户画像。"""
    return profile_service.get_profile_bundle(conversation_manager)


@router.put("", response_model=ProfileBundleResponse)
def replace_profile(
    payload: UserProfilePayload,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """全量替换用户画像。"""
    return profile_service.replace_profile(
        conversation_manager,
        payload.model_dump(exclude_none=True),
    )


@router.patch("", response_model=ProfileBundleResponse)
def patch_profile(
    payload: UserProfilePayload,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """增量更新用户画像。"""
    return profile_service.patch_profile(
        conversation_manager,
        payload.model_dump(exclude_unset=True, exclude_none=True),
    )


@router.post("/reload", response_model=ProfileBundleResponse)
def reload_profile(
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """从磁盘重新加载用户画像。"""
    return profile_service.reload_profile(conversation_manager)


@router.get("/file", response_model=ProfileFileResponse)
def get_profile_file(
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """读取画像原始 YAML 文件，供前端直接编辑。"""
    return profile_service.get_profile_file(conversation_manager)


@router.put("/file", response_model=ProfileFileUpdateResponse)
def save_profile_file(
    payload: ProfileFileUpdateRequest,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_service: ProfileService = Depends(get_profile_service),
):
    """保存前端编辑后的画像原始 YAML 文件。"""
    try:
        return profile_service.save_profile_file(conversation_manager, payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload_resume", response_model=ResumeUploadResponse)
def upload_resume(file: UploadFile = File(...)):
    """上传 PDF 简历并解析为 Markdown。"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只允许上传 PDF 文件")

    # 确保存储目录存在
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "resumes")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")
    finally:
        file.file.close()

    # 解析 PDF
    md_content = PdfToMarkdownParser.parse_pdf_with_mineru(file_path, save_dir)
    if not md_content:
        raise HTTPException(status_code=500, detail="PDF 解析失败，未能生成 Markdown")

    md_content = PdfToMarkdownParser.clean_text(md_content)
    return ResumeUploadResponse(markdown_content=md_content, file_path=file_path)


@router.post("/analyze", response_model=ProfileAnalysisResponse)
def analyze_profile(
    payload: ProfileAnalysisRequest,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    profile_analysis_service: ProfileAnalysisService = Depends(get_profile_analysis_service),
):
    """基于用户画像与上下文生成求职分析报告。"""
    return profile_analysis_service.analyze(
        conversation_manager=conversation_manager,
        model_name=payload.model_name,
        target_position=payload.target_position,
        target_city=payload.target_city,
        target_direction=payload.target_direction,
        notes=payload.notes,
        conversation_id=payload.conversation_id,
    )
