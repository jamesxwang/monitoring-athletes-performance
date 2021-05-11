import sys
import pymysql.cursors
from training_description.database_utils import queryone, queryall

class DatabaseQueries():

    def describe_user_id(self, user_name):
        user_id_map = {
            'eduardo oliveira': 56
        }
        return user_id_map[user_name.lower()]
    
    def describe_planner_id_by_user_id(self, user_id):
        sql = """
            SELECT `planner_id` FROM `wp_cm_athlete_planner`
            WHERE `user_id` = {}
        """.format(user_id)
        return queryone(sql)

    def describe_associate_program_start_date_by_planner_id(self, planner_id):
        sql = """
            SELECT * FROM `wp_cm_planner`
            WHERE `id` = {}
        """.format(planner_id)
        return queryone(sql)

    def describe_program_id_by_planner(self, planner):
        sql = """
            SELECT `program_id` FROM `wp_cm_planner_program`
            WHERE `planner_id` = {} AND `associate_program_start_date` = '{}'
        """.format(planner['id'], planner['starting_date'])
        return queryone(sql)

    def describe_sessions_by_program_id(self, program_id):
        sql = """
            SELECT *
            FROM `wp_cm_session` AS session
            JOIN `wp_cm_session_program` AS session_program ON session.id = session_program.session_id
            JOIN `wp_cm_program` AS program ON session_program.program_id = program.id
            JOIN `wp_cm_session_activity_type` AS session_activity_type ON session_activity_type.id = session.activity_type_id
            WHERE `program_id` = {}
            ORDER BY session_program.associate_no_of_days, session_program.session_order
        """.format(program_id)
        return queryall(sql)

    def describe_descriptions_by_name(self, name):
        user_id = self.describe_user_id(name)
        planner_id = self.describe_planner_id_by_user_id(user_id)
        planner = self.describe_associate_program_start_date_by_planner_id(planner_id)
        program_id = self.describe_program_id_by_planner(planner)
        sessions = self.describe_sessions_by_program_id(program_id)
        return sessions, planner


if __name__ == '__main__':
    USER_NAME = 'eduardo oliveira'
    queries = DatabaseQueries()

    sessions = queries.describe_descriptions_by_name(USER_NAME)
    print('\nsessions: ', sessions)
