from fastapi import APIRouter, HTTPException, File, UploadFile
from app.schemas import IngestRequest, IngestResponse
from app.services.ingest import IngestService

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_papers(request: IngestRequest):
    """Ingest papers from ArXiv metadata JSON"""
    try:
        ingest_service = IngestService()
        
        # Try different file paths
        possible_paths = [
            "arxiv-metadata-oai-snapshot.json",
            "../data/arxiv-metadata-oai-snapshot.json",
            "data/arxiv-metadata-oai-snapshot.json",
            "../frontend/public/data/arxiv-metadata-oai-snapshot.json"
        ]
        
        file_path = None
        for path in possible_paths:
            try:
                response = await ingest_service.ingest_from_file(
                    path, 
                    request.download_pdfs, 
                    request.batch_size
                )
                return response
            except FileNotFoundError:
                continue
        
        raise HTTPException(status_code=404, detail="ArXiv metadata file not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_from_upload(
    file: UploadFile = File(...),
    download_pdfs: bool = False,
    batch_size: int = 1000
):
    """Ingest papers from uploaded JSON file"""
    try:
        ingest_service = IngestService()
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            response = await ingest_service.ingest_from_file(tmp_path, download_pdfs, batch_size)
            return response
        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload ingest failed: {str(e)}")
