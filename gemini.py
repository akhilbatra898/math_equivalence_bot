import json
from typing import List, Dict
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from evaluator import Evaluator


class GeminiEvaluator(Evaluator):
    model_name = 'gemini model'

    def __init__(self):
        self.token_length = 32760
        self.model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.05)

    def process(self, conversations: List[Dict]):
        if len(conversations) > 1:
            return self.evaluate_batch(conversations)
        else:
            return self.evaluate(conversations[0])

    @staticmethod
    def single_query_template(conversation: Dict) -> str:
        for msg in reversed(conversation['Conversation History']):
            if 'bot' in msg:
                break
        return f"""Given question, validate if the answer is right. respond in true or false only
        question - {msg['bot']}
        answer - {conversation['User Response']}"""

    @staticmethod
    def batch_query_template(conversations: List[Dict]):
        data = []
        token_counts = 0
        for conv in conversations:
            for msg in reversed(conv['Conversation History']):
                if 'bot' in msg:
                    data.append({'question': msg['bot'], 'answer': conv['User Response']})
                    token_counts += (len(msg['bot'].split())) + len(conv['User Response'].split())
                    break
            if token_counts > 500:
                yield f"""Given pairs of questions and answers in the data provided, validate if answer are right for corresponding question. return true or false as an array of boolean values (true/false) 
                data - {json.dumps(data)}"""
                data = []
                token_counts = 0
        if len(data) > 0:
            yield f"""Given pairs of questions and answers in the data provided, validate if answer are right for corresponding question. return true or false as an array of boolean values (true/false)
                data - {json.dumps(data)}"""

    def evaluate(self, conversation: Dict) -> str:
        query: str = self.single_query_template(conversation)
        result = self.model.invoke(query)
        if "true" in result.content.lower():
            return 'EQUIVALENT'
        else:
            return 'NOT_EQUIVALENT'

    @staticmethod
    def get_result_from_resp(resp) -> List[bool]:
        return json.loads(resp)

    def evaluate_batch(self, batch: List[Dict]) -> List[Dict]:
        results = []
        for query in self.batch_query_template(conversations=batch):
            result = self.model.invoke(query)
            res = self.get_result_from_resp(result.content)
            results.append(res)

        print(results)
        for i in range(len(batch)):
            if results[i]:
                batch[i]['equivalence_prediction'] = 'EQUIVALENT'
            else:
                batch[i]['equivalence_prediction'] = 'NOT_EQUIVALENT'
        return batch
