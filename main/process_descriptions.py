import html2text
import re
import sys
# import numpy as np
import pandas as pd

from training_description.database_queries import DatabaseQueries
from training_description.cleaning import TrainingDescriptionCleaner

if __name__ == '__main__':
    USER_NAME = 'eduardo oliveira'
    queries = DatabaseQueries()
    raw_descriptions, planner = queries.describe_descriptions_by_name(USER_NAME)
    training_description_cleaner = TrainingDescriptionCleaner()
    cleaned_description = training_description_cleaner.clean_training_descriptions(raw_descriptions, planner)
    # print('cleaned_description: ', cleaned_description)
