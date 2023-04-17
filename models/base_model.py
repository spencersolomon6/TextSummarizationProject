from typing import List


class BaseModel:
    """
    Base Model implementation
    """

    def __init__(self):
        pass

    def train_model(self, train_data: List[str], references: List[str]) -> None:
        """
        Trains the model on the given training data

        :param train_data: A list of strings containing each article in the training data
        :param references: A list of strings containing the reference summary for each article
        """
        raise NotImplementedError()

    def predict(self, data: List[str]) -> List[str]:
        """
        Generate a list of predictions for the data

        :param data: A list of strings containing articles to summarize
        :return: A generate a prediction for each item in the input
        """
        raise NotImplementedError()