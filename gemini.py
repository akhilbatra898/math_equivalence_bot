import json
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from evaluator import Evaluator


class GeminiEvaluator(Evaluator):
    model_name = 'gemini model'

    def __init__(self):
        self.token_length = 32760
        self.model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.1)

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
        for conv in conversations:
            for msg in reversed(conv['Conversation History']):
                if 'bot' in msg:
                    data.append({'question': msg['bot'], 'answer': conv['User Response']})
                    break
        return f"""Given pairs of questions and answers in the json provided, validate if answer are right for corresponding question. return true or false as a list
                json - {json.dumps(data)}"""
    def evaluate(self, conversation: Dict) -> bool:
        query: str = self.single_query_template(conversation)
        result = self.model.invoke(query)
        print(query)
        print(result)
        if "true" in result.content.lower():
            return True
        else:
            return False

    def evaluate_batch(self, batch: List[Dict]) -> List[bool]:
        query: str = self.batch_query_template(conversations=batch)
        result = self.model.invoke(query)
        print(result)
        return [True]*5
