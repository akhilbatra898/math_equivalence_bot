from langchain_google_genai import ChatGoogleGenerativeAI
from lang_chain_evaluator import LangChainEvaluator


class GeminiModel(LangChainEvaluator):
    def __init__(self, model=None, **kwargs):
        super().__init__(model, **kwargs)
        self.model_name = 'gemini pro'
        self.model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.0)
        self.multi_modal_model = ChatGoogleGenerativeAI(model='gemini-pro-vision')
