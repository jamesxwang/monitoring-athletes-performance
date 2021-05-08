import sys
import re
import html2text
from datetime import timedelta
from dateutil.relativedelta import relativedelta

def grep(array, string):
    return [i for i, item in enumerate(array) if re.search(string, item)]

def prefix_dict(di_, prefix_s=''):
    """
    Add prefix_s to every key in dict
    :param di_:
    :param prefix_s:
    :return:
    """
    return {prefix_s + k: v for k, v in di_.items()}

def spear_dict(di_, con_s='_', with_k=True):
    """
    :param di_: input dict
    :param con_s: connection character
    :return: dict with depth 1
    """
    ret_di = {}
    for k, v in di_.items():
        if type(v) is dict:
            v = spear_dict(v)
            ret_di.update(prefix_dict(v, prefix_s=k + con_s if with_k else ''))
        else:
            ret_di.update({k: v})
    return ret_di

class TrainingDescriptionCleaner():
    def _keep_useful_keys(self, item):
        preserved_keys = ['id', 'title', 'unit', 'distance', 'description', 'level', 'rpe_load', 'associate_no_of_days', 'program.title']
        new_obj = {}
        for key in preserved_keys:
            value = item[key]
            new_obj[key] = value
        return new_obj

    def _is_unrelated_activities(self, item):
        title = item['title'].lower()
        return bool(re.search('run | ride | swim', title))
    
    def _handle_date_time(self, item, start_date):
        associate_no_of_days = item['associate_no_of_days']
        item['date'] = start_date + timedelta(days=associate_no_of_days)
        # del item['associate_no_of_days']
        return item

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
        # warm_up_params = self.get_warm_up_params(description_splitted_list)
        processed_descriptions = {
            # 'activity_name': activity_name,
            # 'warmup': warm_up_params
        }
        del item['description']
        item['description'] = description_splitted_list
        item['processed_descriptions'] = spear_dict(processed_descriptions)
        return spear_dict(item, '', False)

    def get_warm_up_params(self, description_splitted_list):
        REGEX_WARM_UP = r'[Ww]arm[\ ]?[Uu]p'
        REGEX_MAIN_SET = r'[Mm]ain[\ ]?[Ss]et'
        REGEX_MINUTES = r'(\d+)(\ )*[Mm][Ii][Nn](ute)?(s)?'
        REGEX_T_ZONE = r'T(\d)'
        REGEX_RPM = r'(\d+)[\ ]*([Rr][Pp][Mm]([Ss])?)'

        # some of the training descriptions don't have the `Warm up` keyword.
        try:
            warm_up_ind_list = grep(description_splitted_list, REGEX_WARM_UP)
            main_set_ind_list = grep(description_splitted_list, REGEX_MAIN_SET)

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
        cleaned_descriptions = []
        tmp = []
        for item in descriptions:
            item = self._keep_useful_keys(item)
            item = self._parse_html_to_text(item)
            tmp.append(item)

        for i in range(end_interval):
            start_date = start_date + relativedelta(months=1)
            for item in tmp:
                item = self._handle_date_time(item, start_date)
                print('-' * 50)
                print('cleaned_descriptions item: ', item)
                cleaned_descriptions.append(item)

        return cleaned_descriptions

