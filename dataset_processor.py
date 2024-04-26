import json_repair
import pandas as pd
import json
import ast
from typing import Type
from evaluator import Evaluator


class DatasetProcessor:
    @staticmethod
    def json_parse_conversation_column(conv: str):
        try:
            return json.loads(conv)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(conv) or ""
            except SyntaxError:
                return json_repair.loads(conv) or ""

    @classmethod
    def process_file(cls, file_name: str, evaluator_cls: Type[Evaluator]):
        evaluator = evaluator_cls()
        dataset: pd.DataFrame = pd.read_csv(file_name)
        dataset['Conversation History'] = dataset['Conversation History'].apply(cls.json_parse_conversation_column)
        dataset = dataset.drop(columns=['LLM Equivalence Evaluation (Response)', 'Time taken to complete the request'])
        dataset = dataset
        responses = evaluator.process(
            dataset[['Conversation History', 'User Response', 'DialogID Hash', 'Index']].to_dict(orient='records'))
        final_data = dataset.merge(pd.DataFrame(responses)[['Index', 'LLM Equivalence Evaluation (Response)',
                                                            'Time taken to complete the request']],
                                   on='Index', how='left')
        final_data.to_csv('final_data.csv')
