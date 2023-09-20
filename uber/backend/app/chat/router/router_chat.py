from typing import List, Optional

from fastapi import Depends, Response

from app.utils import AppModel

from ..service import Service, get_service
from . import router


class ChatResponse(AppModel):
    answer: str = "Hi there"


@router.post("/predict", response_model=ChatResponse)
def create_post(
    identity: str,
    svc: Service = Depends(get_service),
):
    svc.ml_service.tst()
    final_results = svc.ml_service.predict(
        identity=identity
    )

    return {"answer": final_results}
