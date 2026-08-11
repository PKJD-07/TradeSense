from fastapi import APIRouter

from backend.app.api.schemas.historical import HistoricalDataset

router = APIRouter()


@router.post("/market/history", response_model=HistoricalDataset)
def validate_historical_data(dataset: HistoricalDataset):
    return dataset