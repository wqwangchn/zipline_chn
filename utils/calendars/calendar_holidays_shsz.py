#coding: utf-8
import datetime
import os
import zipline.utils.paths as pth

def int_to_date(d):
    d = str(d)
    return datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))

def str_to_int(s):
    return int(s)

def get_from_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        return set([int_to_date(str_to_int(i.rstrip('\n'))) for i in data])
    return set([])

def get_holidays(data_file_name='holidays_shsz.txt'):
    datafilepath = os.path.join(pth.data_root(), data_file_name)
    return get_from_file(datafilepath)

def check_expired(get_holidays):
    """
    check if local or cached data need update
    :return: true/false
    """
    last_date = max(get_holidays())
    now = datetime.datetime.now().date()
    if now > last_date:
            return True
    return False

def is_trading_day(get_holidays):
    def _is_trading_day(dt):
        if type(dt) is datetime.datetime:
            dt = dt.date()
        if dt.weekday() >= 5:
            return False
        holidays = get_holidays()
        if dt in holidays:
            return False
        return True
    return _is_trading_day

def previous_trading_day(is_trading_day):
    def _previous_trading_day(dt):
        if type(dt) is datetime.datetime:
            dt = dt.date()
        while True:
            dt = dt - datetime.timedelta(days=1)
            if is_trading_day(dt):
                return dt
    return _previous_trading_day

def next_trading_day(is_trading_day):
    def _next_trading_day(dt):
        if type(dt) is datetime.datetime:
            dt = dt.date()

        while True:
            dt = dt + datetime.timedelta(days=1)
            if is_trading_day(dt):
                return dt
    return _next_trading_day

def trading_days_between(get_holidays):
    def _trading_days_between(start, end):
        if type(start) is datetime.datetime:
            start = start.date()

        if type(end) is datetime.datetime:
            end = end.date()

        dataset = get_holidays()
        if start > end:
            return
        curdate = start
        while curdate <= end:
            if curdate.weekday() < 5 and not(curdate in dataset):
                yield curdate
            curdate = curdate + datetime.timedelta(days=1)
    return _trading_days_betwee



def get_remote_and_cache(data_file_name='holidays_shsz.txt'):
    """
    get newest data file from network and cache on local machine
    :return: a list contains all holiday data, element with datatime.date format
    """
    response = requests.get('https://raw.githubusercontent.com/rainx/cn_stock_holidays/master/cn_stock_holidays/data.txt')
    datafilepath = os.path.join(pth.data_root(), data_file_name)
    with open(datafilepath, 'wb') as f:
        f.write(response.content)

def sync_data(check_expired, get_remote_and_cache):
    logging.basicConfig(level=logging.INFO)
    if check_expired():
        logging.info("trying to fetch data...")
        get_remote_and_cache()
        logging.info("done")
    else:
        logging.info("local data is not exipired, do not fetch new data")

'''
    import requests
    import logging
    from cn_stock_holidays.common import function_cache, int_to_date, print_result, _get_from_file

    def meta_get_remote_and_cache(get_cached, get_cache_path):
        """
        get newest data file from network and cache on local machine
        :return: a list contains all holiday data, element with datatime.date format
        """
        response = requests.get('https://raw.githubusercontent.com/rainx/cn_stock_holidays/master/cn_stock_holidays/data.txt')
        cache_path = get_cache_path()

        with open(cache_path, 'wb') as f:
            f.write(response.content)

        get_cached.cache_clear()

        return get_cached()
    def meta_sync_data(check_expired, get_remote_and_cache):
        logging.basicConfig(level=logging.INFO)
        if check_expired():
            logging.info("trying to fetch data...")
            get_remote_and_cache()
            logging.info("done")
        else:
            logging.info("local data is not exipired, do not fetch new data")
    def meta_get_cache_path(data_file_name='data.txt'):
        def get_cache_path():
            usr_home = os.path.expanduser('~')
            cache_dir = os.path.join(usr_home, '.cn_stock_holidays')
            if not(os.path.isdir(cache_dir)):
                os.mkdir(cache_dir)
            return os.path.join(cache_dir, data_file_name)
        return get_cache_path
'''