import os
from dataset_processor import DatasetProcessor
from gemini import GeminiModel
import sys

import os


if __name__ == '__main__':

    DatasetProcessor.process_file(file_name=os.path.abspath(sys.argv[1]),
                                  evaluator_cls=GeminiModel)
