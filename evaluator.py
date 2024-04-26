from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class Evaluator(ABC):
    model_name: Optional[str] = None

    @abstractmethod
    def process(self, conversations: List[Dict]):
        pass

    @abstractmethod
    def evaluate(self, conversation: Dict) -> bool:
        pass

    @abstractmethod
    def evaluate_batch(self, batch: List[Dict]) -> List[Dict]:
        pass


class DummyEvaluator(Evaluator):
    model_name = 'dummy model'

    def process(self, conversations: List[Dict]):
        if len(conversations) > 0:
            return self.evaluate_batch(conversations)
        else:
            return self.evaluate(conversations[0])

    def evaluate(self, conversation: Dict) -> bool:
        return True

    def evaluate_batch(self, batch: List[Dict]) -> List[bool]:
        response: List[bool] = [True]*len(batch)
        return response
