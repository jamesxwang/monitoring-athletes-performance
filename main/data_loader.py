"""Athletes Data Loader

This script allows the user to load the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following class:

    * DataLoader - supports loading all data files for the system.
"""

import os
import pandas as pd
import utility
from configparser import ConfigParser


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


