"""Athletes Data Loader

This script allows the user to load the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following class:

    * DataLoader - supports loading all data files for the system.
"""
# Packages
import os
import re
import pandas as pd
import numpy as np
import functools
import operator
import html2text
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# Self-defined modules
import utility
from configparser import ConfigParser
from training_description.database_queries import DatabaseQueries

# Settings of Packages
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 


class DataLoader():
    """
    A class used to load data

    ...

    Attributes
    ----------
    file_or_dir_name : str
        The name of the file that is about to load (default '{}/data')

    Methods
    -------
    load_athlete_dataframe()
        Load the data frame
    """

    def __init__(self, data_type='spreadsheet'):
        """The constructor include the parameter data_type which indicates spreadsheet/additional data
        so that it can avoid data structure problems given that the two functions in the class
        have return in different data types.

        """
        self.data_path = '{}/data'.format(os.path.pardir)
        self.data_type = data_type
        data_names = '{}/main/config/data_file_names.cfg'.format(os.path.pardir)
        self.config = ConfigParser()
        self.config.read(data_names)


    def load_spreadsheet_data(self, file_name=None, athletes_name=None):
        """Load the spreadsheet data for an athlete

        Returns
        -------
            Pandas data frame
        """
        if file_name == None and athletes_name == None:
            raise Exception("No inputs in load_spreadsheet_data. Input a file name or an athlete's name.")
        if self.data_type == 'spreadsheet':
            file_path = '{}/{}'.format(self.data_path, file_name)
            if os.path.isfile(file_path):
                return pd.read_csv(file_path, sep=',')
            try:
                athletes_file_name = self.config.get('SPREADSHEET-DATA-SETS', athletes_name.lower())
                file_path = '{}/{}'.format(self.data_path, athletes_file_name)
                return pd.read_csv(file_path, sep=',')
            except:
                return None
        if self.data_type == 'additional':
            print('Invalid function call. Given type \'additional\'.')
            return None


    def load_additional_data(self, athletes_name, activity_type='', split_type=''):
        """Load the additional data for an athlete

        Parameters
        -------
        activity_type : str
            The activity type. Options are '', 'cycling', 'running', 'swimming' and 'training'
        split_type : str
            The split and laps types. Options are '', 'real-time', 'laps', 'starts'
        athletes_name : str
            The name of the athlete from whom the data converted from fit files is about to get

        Returns
        -------
            List of file names in strings converted from fit files
        """
        if self.data_type == 'spreadsheet':
            print('Invalid function call. Given type \'spreadsheet\'.')
            return None
        if self.data_type == 'additional':
            if athletes_name.startswith('csv_'):
                dir_path = '{}/{}/fit_csv'.format(self.data_path, athletes_name)
            else:
                dir_path = '{}/csv_{}/fit_csv'.format(self.data_path, '_'.join(athletes_name.lower().split()))
            if os.path.isdir(dir_path):
                return ['{}/{}'.format(dir_path, file_name) for file_name in os.listdir(dir_path)
                        if file_name.startswith(activity_type) and file_name.endswith('{}.csv'.format(split_type))]
            else:
                return None


    def load_cleaned_spreadsheet_data(self, dir_name=None, athletes_name=None):
        """Load the cleaned spreadsheet data for an athlete

        Returns
        -------
            Pandas data frame
        """
        if self.data_type == 'spreadsheet':
            try:
                file_path = '{}/{}'.format(self.data_path, dir_name)
                if os.path.isfile(file_path):
                    return pd.read_csv(file_path, sep=',')

                file_name = self.config.get('CLEANED-SPREADSHEET-DATA-SETS', athletes_name.lower())
                file_path = '{}/{}'.format(self.data_path, file_name)
                return pd.read_csv(file_path, sep=',')
            except Exception as e:
                print('Exception: Cannot load the data.', e)

        if self.data_type == 'additional':
            print('Invalid function call. Given type \'additional\'.')
            return None


    def load_cleaned_additional_data(self, athletes_name, activity_type='', split_type=''):
        if self.data_type == 'spreadsheet':
            print('Invalid function call. Given type \'spreadsheet\'.')
            return None
        if self.data_type == 'additional':
            dir_path = '{}/cleaned_additional/{}'.format(self.data_path, '_'.join(athletes_name.lower().split()))
            if os.path.isdir(dir_path):
                return ['{}/{}'.format(dir_path, file_name) for file_name in os.listdir(dir_path)
                        if file_name.startswith(activity_type) and file_name.endswith('{}.csv'.format(split_type))]
            else:
                return None


    def load_merged_data(self, athletes_name):
        file_path = '{}/merged_dataframes/merged_{}.csv'.format(self.data_path, '_'.join(athletes_name.lower().split()))
        return pd.read_csv(file_path, sep=',')

    def load_training_description_with_watch_data(self, athletes_name):
        # load watch data
        watch_data_dataframe = self.load_merged_data(athletes_name=athletes_name)

        # load training descriptions for [athletes_name]
        queries = DatabaseQueries()
        sessions, planner = queries.describe_descriptions_by_name(athletes_name)
        training_description_cleaner = TrainingDescriptionsDataCleaner()
        cleaned_description = training_description_cleaner.clean_training_descriptions(sessions, planner)

        # one hot encoding
        tokens_set = set(functools.reduce(operator.iconcat, list(map(lambda item: item['tokens'] ,cleaned_description)), []))
        dim = len(tokens_set)
        all_data_td_map = {}
        idx = 0
        for token in tokens_set:
            all_data_td_map[token] = idx
            idx += 1
        for i, item in enumerate(cleaned_description):
            one_hot_td = []
            for token in item['tokens']:
                onehot = np.zeros(dim)
                onehot[all_data_td_map[token]] = 1
                one_hot_td.append(onehot)
            cleaned_description[i]['onehot'] = one_hot_td
        training_description_dataframe = pd.DataFrame(cleaned_description)
        training_description_dataframe = utility.keep_keys_in_dataframe(['date', 'onehot'], training_description_dataframe)

        # process date
        for index, record in watch_data_dataframe.iterrows():
            date = str(record['Date']).split(' ')[0]
            watch_data_dataframe.at[index, 'Date'] = date

        # merge training descriptions with watch data
        final_df = pd.merge(watch_data_dataframe, training_description_dataframe, left_on='Date', right_on='date', how='left')

        return final_df.dropna(subset=['onehot'])


class TrainingDescriptionsDataCleaner():
    def grep(self, array, string):
        return [i for i, item in enumerate(array) if re.search(string, item)]

    def prefix_dict(self, di_, prefix_s=''):
        """
        Add prefix_s to every key in dict
        :param di_:
        :param prefix_s:
        :return:
        """
        return {prefix_s + k: v for k, v in di_.items()}

    def spear_dict(self, di_, con_s='_', with_k=True):
        """
        :param di_: input dict
        :param con_s: connection character
        :return: dict with depth 1
        """
        ret_di = {}
        for k, v in di_.items():
            if type(v) is dict:
                v = self.spear_dict(v)
                ret_di.update(self.prefix_dict(v, prefix_s=k + con_s if with_k else ''))
            else:
                ret_di.update({k: v})
        return ret_di
    
    def _keep_useful_keys(self, item):
        preserved_keys = ['description', 'associate_no_of_days']
        new_obj = {}
        for key in preserved_keys:
            value = item[key]
            new_obj[key] = value
        return new_obj

    def _is_unrelated_activities(self, item):
        title = item['title'].lower()
        return bool(re.search('run | ride | swim', title))
    
    def _handle_date_time(self, item, start_date):
        copy = item.copy()
        associate_no_of_days = copy['associate_no_of_days']
        date = str(start_date + timedelta(days=associate_no_of_days))
        copy['date'] = date
        return copy

    def _remove_note_and_link_in_description(self, description_splitted_list):
        min_i = len(description_splitted_list)
        for i, token in enumerate(description_splitted_list):
            if i < min_i:
                if token.startswith('[ ') or bool(re.search('[Dd]ownload', token)):
                    min_i = i
        return description_splitted_list[:min_i]

    def _parse_html_to_text(self, item):
        html_parsed = html2text.html2text(item['description']).replace('*', '')
        description_splitted_list = html_parsed.split('\n')
        description_splitted_list = list(filter(None, map(lambda token: None if not token.strip() else token.strip(), description_splitted_list)))
        description_splitted_list = self._remove_note_and_link_in_description(description_splitted_list)
        activity_name = description_splitted_list[0] if len(description_splitted_list) else 'Unknown Activity Name'
        # warm_up_params = self._get_warm_up_params(description_splitted_list)
        processed_descriptions = {
            # 'activity_name': activity_name,
            # 'warmup': warm_up_params
        }
        del item['description']
        item['description'] = description_splitted_list
        description_str = ' '.join(description_splitted_list)
        item['description_str'] = description_str
        item['processed_descriptions'] = self.spear_dict(processed_descriptions)
        return self.spear_dict(item, '', False)

    def _get_warm_up_params(self, description_splitted_list):
        REGEX_WARM_UP = r'[Ww]arm[\ ]?[Uu]p'
        REGEX_MAIN_SET = r'[Mm]ain[\ ]?[Ss]et'
        REGEX_MINUTES = r'(\d+)(\ )*[Mm][Ii][Nn](ute)?(s)?'
        REGEX_T_ZONE = r'T(\d)'
        REGEX_RPM = r'(\d+)[\ ]*([Rr][Pp][Mm]([Ss])?)'

        # some of the training descriptions don't have the `Warm up` keyword.
        try:
            warm_up_ind_list = self.grep(description_splitted_list, REGEX_WARM_UP)
            main_set_ind_list = self.grep(description_splitted_list, REGEX_MAIN_SET)

            warm_up_index = warm_up_ind_list[0] if len(
                warm_up_ind_list) > 0 else -1
            main_set_index = main_set_ind_list[0] if len(
                main_set_ind_list) > 0 else -1

            warm_up_description = description_splitted_list[1:main_set_index] if warm_up_index == - \
                1 else description_splitted_list[warm_up_index+1:main_set_index]
            print('-'*30)
            print('warm_up_description: ', warm_up_description)

            warm_up_params = {}
            for i, item in enumerate(warm_up_description):
                session_number = i + 1
                minutes_match = re.search(REGEX_MINUTES, item)
                t_zone_match = re.findall(REGEX_T_ZONE, item)
                print('item: ', item)
                rpms_match = re.search(REGEX_RPM, item)

                if minutes_match:
                    warm_up_params['activity_{}_time_minutes'.format(
                        session_number)] = int(minutes_match.group(1))
                if len(t_zone_match):
                    print('t_zone_match: ', t_zone_match)
                    for zone in t_zone_match:
                        warm_up_params['activity_{}_t_zone_{}'.format(session_number, zone)] = 1
                if rpms_match:
                    warm_up_params['activity_{}_rpms'.format(
                        session_number)] = int(rpms_match.group(1))
                # TODO: activity regex

            return warm_up_params
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        return []

    def clean_training_descriptions(self, descriptions, planner):
        start_date = planner['starting_date']
        end_interval = planner['end_interval']
        descriptions_with_date = []
        cleaned_descriptions = []

        # process descriptions
        for i, item in enumerate(descriptions):
            item = self._keep_useful_keys(item)
            item = self._parse_html_to_text(item)
            descriptions[i] = item

        # repeat intervals and process date
        for i in range(end_interval):
            for item in descriptions:
                item = self._handle_date_time(item, start_date)
                descriptions_with_date.append(item)
            start_date = start_date + relativedelta(months=1)

        # merge descriptions with same date
        tmp = {}
        for i, item in enumerate(descriptions_with_date):
            description, description_str, date = item['description'], item['description_str'], item['date']
            prev_item = descriptions_with_date[i-1]
            if i > 0:
                if date == prev_item['date']:
                    tmp['description'] = prev_item['description'] + item['description']
                    tmp['description_str'] = prev_item['description_str'] + item['description_str']
                    tmp['date'] = date
                else:
                    cleaned_descriptions.append(tmp)
                    tmp = {}

        # nlp part
        for item in cleaned_descriptions:
            # remove stopwords and tokenize
            item['tokens'] = [w for w in word_tokenize(item['description_str']) if not w in stop_words] 

        return cleaned_descriptions


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira', 'xu chen', 'carly hart']
    # ====== Get all file names of the spreadsheet data ======
    data_loader_spreadsheet = DataLoader('spreadsheet')
    spreadsheet_data_names = utility.get_all_spreadsheet_data_file_names()
    print('Spreadsheet data file names: ', spreadsheet_data_names)
    # Load spreadsheet data in two ways
    spreadsheet_df_example1 = data_loader_spreadsheet.load_spreadsheet_data(file_name=spreadsheet_data_names[1])   # Load with file name
    spreadsheet_df_example2 = data_loader_spreadsheet.load_spreadsheet_data(athletes_name=athletes_names[1])   # Load with athlete's name
    print(spreadsheet_df_example1.head())
    print(spreadsheet_df_example2.head())
    # Load cleaned spreadsheet data in two ways
    cleaned_spreadsheet_df_example1 = data_loader_spreadsheet.load_cleaned_spreadsheet_data(dir_name='cleaned_spreadsheet/Eduardo Oliveira (Intermediate).csv')
    cleaned_spreadsheet_df_example2 = data_loader_spreadsheet.load_cleaned_spreadsheet_data(athletes_name=athletes_names[1])
    print(cleaned_spreadsheet_df_example1.head())
    print(cleaned_spreadsheet_df_example2.head())


    # ====== Get all folder names of the additional data ======
    data_loader_additional = DataLoader('additional')
    add_data_folder_names = utility.get_all_additional_data_folder_names()
    print('Additional data folder names: ', add_data_folder_names)
    # Load additional data in two ways
    additional_df_example1 = data_loader_additional.load_additional_data(add_data_folder_names[1])    # Load with folder name
    additional_df_example2 = data_loader_additional.load_additional_data(athletes_name=athletes_names[1],
                                                                         activity_type='swimming',
                                                                         split_type='real-time')  # Load with athlete's name
    print(additional_df_example1)
    print(additional_df_example2)
    # Load cleaned additional data
    cleaned_additional_df_example = data_loader_additional.load_cleaned_additional_data(athletes_name=athletes_names[1])
    print(cleaned_additional_df_example)


    # ====== Load merged data ======
    merged_data_example = DataLoader().load_merged_data(athletes_names[0])
    print(merged_data_example.head())


