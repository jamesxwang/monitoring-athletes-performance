import os
import json
import traceback
import pymysql.cursors

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def connect_mysql():
    try:
        config = find("db_config.json", os.path.abspath("./config"))
        with open(config, "r") as file:
            load_dict = json.load(file)
        return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **load_dict)
    except Exception as e:
        print(traceback.format_exc())
        print("cannot create mysql connect")

def queryone(sql, param=None):
    """
    :param sql: sql query
    :param param: string|tuple|list
    :return: {}
    """
    con = connect_mysql()
    cur = con.cursor()
    row = None
    try:
        cur.execute(sql, param)
        row = cur.fetchone()
    except Exception as e:
        con.rollback()
        print(traceback.format_exc())
        print(":{} [param]:{}".format(sql, param))
    cur.close()
    con.close()
    return simple_value(row)

def queryall(sql, param=None):
    """
    :param sql: sql query
    :param param: tuple|list
    :return: [{},{},{}...]
    """
    con = connect_mysql()
    cur = con.cursor()
    rows = None
    try:
        cur.execute(sql, param)
        rows = cur.fetchall()
    except Exception as e:
        con.rollback()
        print(traceback.format_exc())
        print(":{} [param]:{}".format(sql, param))
    cur.close()
    con.close()
    return rows

def simple_list(rows):
    """
    :param rows: [{'id': 1}, {'id': 2}, {'id': 3}]
    :return: [1, 2, 3]
    """
    if not rows:
        return rows
    if len(rows[0].keys()) == 1:
        simple_list = []
        # print(rows[0].keys())
        key = list(rows[0].keys())[0]
        for row in rows:
            simple_list.append(row[key])
        return simple_list
    return rows

def simple_value(row):
    """
    :param row: {'count(*)': 3}
    :return: 3
    """
    if not row:
        return None
    if len(row.keys()) == 1:
        # print(row.keys())
        key = list(row.keys())[0]
        return row[key]
    return row

