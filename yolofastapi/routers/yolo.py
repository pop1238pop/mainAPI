# For API operations and standards
from fastapi import APIRouter, UploadFile, Response, status, HTTPException
# Our detector objects
from yolofastapi.detectors import yolov8
# For encoding images
import cv2
# For response schemas
from yolofastapi.schemas.yolo import ImageAnalysisResponse

router = APIRouter(tags=["Image Upload and analysis"], prefix="/yolo")
images = []

@router.post("/",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successfully Analyzed Image."}
    },
    response_model=ImageAnalysisResponse,
)
async def yolo_image_upload(file: UploadFile) -> ImageAnalysisResponse:

    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels = await dt()
    success, encoded_image = cv2.imencode(".png", frame)
    images.append(encoded_image)
    return ImageAnalysisResponse(id=len(images), labels=labels)


@router.get(
    "/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image ID Not Found."}
    },
    response_class=Response,
)
async def yolo_image_download(image_id: int) -> Response:

    try:
        return Response(content=images[image_id - 1].tobytes(), media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail="Image not found") 