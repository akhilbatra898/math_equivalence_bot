import pandas as pd
import json
import json_repair
from typing import Type
from evaluator import Evaluator


class DatasetProcessor:
    @staticmethod
    def json_parse_conversation_column(conv: str):
        try:
            return json.loads(conv)
        except json.JSONDecodeError:
            return json_repair.loads(conv) or ""

    @classmethod
    def process_file(cls, file_name: str, evaluator_cls: Type[Evaluator]):
        evaluator = evaluator_cls()
        dataset: pd.DataFrame = pd.read_csv(file_name)
        dataset['Conversation History'] = dataset['Conversation History'].apply(cls.json_parse_conversation_column)
        dataset = dataset.iloc[:10]
        responses = evaluator.process(
            dataset[['Conversation History', 'User Response', 'DialogID Hash']].to_dict(orient='records'))
        print(responses)
        final_data = dataset.merge(pd.DataFrame(responses)[['DialogID Hash', 'equivalence_prediction']],
                                   on='DialogID Hash', how='left')
        final_data.to_csv('final_data.csv')
