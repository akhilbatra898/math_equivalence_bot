import os
from dataset_processor import DatasetProcessor
from gemini import GeminiEvaluator

import os


print(DatasetProcessor.process_file(file_name=os.path.abspath('./Accuracy Calculation Dataset.csv'),
                                    evaluator_cls=GeminiEvaluator))
