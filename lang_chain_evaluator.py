import json
from typing import List, Dict
import time
import re

from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

from langchain_core.prompts.chat import ChatPromptValue
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from evaluator import Evaluator


class LangChainEvaluator(Evaluator):

    def __init__(self, model=None,multi_modal_model=None, model_name='lang chain model', **kwargs):
        self.model = model
        self.model_name = model_name
        self.multi_modal_model = multi_modal_model
        self.params = kwargs

    def process(self, conversations: List[Dict]):
        if len(conversations) > 1:
            return self.evaluate_batch(conversations)
        else:
            return self.evaluate__using_chaining(conversations[0])

    @staticmethod
    def single_query_template(conversation: Dict) -> str:
        for i in range(len(conversation['Conversation History'])):
            if 'date' in conversation['Conversation History'][i]:
                conversation['Conversation History'][i].pop('date')
        msg = conversation['Conversation History'][-1]
        return f"""Given a QA Thread, first check if  the last  message in thread is sufficient enoigh to validate whether the response provided is right or wrong  for the last message? 
        If yes, then validate the response in true or false, otherwise take the whole thread as the context to validate the response in true/false only 
        Note - Need only 1 word output in true or false only.        
        QA Thread - {msg}
        Response - {conversation['User Response']}"""

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
                yield f"""Given pairs of questions and answers in the data provided, validate if answer are right for corresponding question and return validations as boolean array (true/false) 
                data - {json.dumps(data)}"""
                data = []
                token_counts = 0
        if len(data) > 0:
            yield f"""Given pairs of questions and answers in the data provided, validate if answer are right for corresponding question and return validations as boolean array (true/false)
                data - {json.dumps(data)}"""

    def evaluate(self, conversation: Dict) -> str:
        query: str = self.single_query_template(conversation)
        result = self.model.invoke(query)
        print(query, result)
        if "true" in result.content.lower():
            return 'EQUIVALENT'
        else:
            return 'NOT_EQUIVALENT'

    @staticmethod
    def get_result_from_resp(resp) -> List[bool]:
        return json.loads(resp)

    def parse_context(self, context):
        pass

    def handle_context_and_image(self, context):
        if 'image_url' in str(context):
            img_url = re.findall(r"https:.*jpg", str(context))[0]
            try:
                image_data = self.multi_modal_model.invoke([HumanMessage(
                    [{"type": "text", "text": "Extract the problem text from the image"},
                     {"type": "image_url", "image_url": img_url}])])
            except Exception as _:
                image_data = ""
            if image_data != "":
                return PromptTemplate.from_template("""Provided history of the chat between the user and the bot as Chat 
                History below, can you check whats the problem that user need to solve and then check if the Answer 
                provided at the end is right/wrong. Important Note: reply with a one word answer true/false
        
                                    Chat History - {context}
                                    Question - {img_question}
                                    Answer - {answer}
                                    """, partial_variables={
                    'img_question': image_data.content}) | self.model | StrOutputParser()

        return (PromptTemplate.from_template("""Provided history of the chat between the user and the bot as Chat 
        History below, can you check whats the problem that user need to solve and then check if the Answer 
        provided at the end is right/wrong. Important Note: reply with a one word answer true/false

                                        Chat History - {context}
                                        Answer - {answer}
                                        """) | self.model | StrOutputParser())

    def route_logic(self, resp):
        print(resp)
        valid_context_followup_chain = (PromptTemplate.from_template("""For the given question and answer provided by the user. Validate if the answer provided is right or wrong.
         Question - {question}
         Answer - {answer}
         
        Note - Respond only 1 word - true/false""") | self.model | StrOutputParser())
        if 'true' in resp['resp'].lower():
            return valid_context_followup_chain

        return self.handle_context_and_image(context=resp['context'])

    def evaluate__using_chaining(self, conversation: Dict):
        context_validity_chain = (PromptTemplate.from_template("""Given Question below, is it possible for user to provide an answer?
        Provide only 1 word output true/false 
        Question - {question}""")
                                  | self.model | StrOutputParser())
        full_chain = {"resp": context_validity_chain, "answer": lambda x: x['answer'],
                      "question": lambda x: x['question'],
                      "context": lambda x: x['context']} | RunnableLambda(self.route_logic)

        result = full_chain.invoke({"answer": conversation['User Response'],
                                    "question": conversation['Conversation History'][-1],
                                    "context": conversation['Conversation History']})
        print(result)
        if 'true' in result.lower():
            return "EQUIVALENT"
        else:
            return "NOT_EQUIVALENT"

    def evaluate_batch(self, batch: List[Dict]) -> List[Dict]:
        for i in range(len(batch)):
            start_time = time.time()
            batch[i]['LLM Equivalence Evaluation (Response)'] = self.evaluate__using_chaining(batch[i])
            batch[i]['Time taken to complete the request'] = (time.time() - start_time)
        return batch
