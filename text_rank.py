from typing import List
from base_model import BaseModel


class TextRank(BaseModel):
    def __init__(self):
        super().__init__()
        pass

    def train(self, train_data: List[str], references: List[str]) -> None:
        pass

    def predict(self, data: List[str]) -> List[str]:
        pass
