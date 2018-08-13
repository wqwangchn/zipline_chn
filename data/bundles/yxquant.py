"""
Module for building a complete daily dataset Internal dataset.
"""
import numpy as np
import pandas as pd
from six import iteritems
from logbook import Logger
from . import core as bundles
from sqlalchemy import create_engine
from zipline.utils.calendars import register_calendar_alias

log = Logger(__name__)
def gen_asset_metadata(data, show_progress):
    if show_progress:
        log.info('Generating asset metadata.')

    data = data.groupby(
        by='symbol'
    ).agg(
        {'date': [np.min, np.max]}
    )
    data.reset_index(inplace=True)
    data['start_date'] = data.date.amin
    data['end_date'] = data.date.amax
    del data['date']
    data.columns = data.columns.get_level_values(0)

    data['exchange'] = 'YXQUANT'
    data['auto_close_date'] = data['end_date'].values + pd.Timedelta(days=1)
    return data

def parse_pricing_and_vol(data,
                          sessions,
                          symbol_map):
    for asset_id, symbol in iteritems(symbol_map):
        asset_data = data.xs(
            symbol,
            level=1
        ).reindex(
            sessions.tz_localize(None),method='ffill'
        ).fillna(0.0)
        yield asset_id, asset_data

def parse_splits(data, show_progress):
    if show_progress:
        log.info('Parsing split data.')

    data['split_ratio'] = 1.0 / data.split_ratio
    data.rename(
        columns={
            'split_ratio': 'ratio',
            'date': 'effective_date',
        },
        inplace=True,
        copy=False,
    )
    return data

def parse_dividends(data, show_progress):
    if show_progress:
        log.info('Parsing dividend data.')

    data['record_date'] = data['declared_date'] = data['pay_date'] = pd.NaT
    data.rename(
        columns={
            'ex_dividend': 'amount',
            'date': 'ex_date',
        },
        inplace=True,
        copy=False,
    )
    return data


"""Parameters
    ----------
    name : str
        The name of the bundle.
    f : callable(yxdm_bundle)
        The ingest function. This function will be passed:
        environ : mapping
          The environment this is being run with.
        asset_db_writer : AssetDBWriter
          The asset db writer to write into.
        minute_bar_writer : BcolzMinuteBarWriter
          The minute bar writer to write into.
        daily_bar_writer : BcolzDailyBarWriter
          The daily bar writer to write into.
        adjustment_writer : SQLiteAdjustmentWriter
          The adjustment db writer to write into.
        calendar : zipline.utils.calendars.TradingCalendar
          The trading calendar to ingest for.
        start_session : pd.Timestamp
          The first session of data to ingest.
        end_session : pd.Timestamp
          The last session of data to ingest.
        cache : DataFrameCache
          A mapping object to temporarily store dataframes.
          This should be used to cache intermediates in case the load
          fails. This will be automatically cleaned up after a
          successful load.
        show_progress : bool
          Show the progress for the current load where possible.
    calendar_name : str, optional
        The name of a calendar used to align bundle data.
        Default is 'NYSE'.
    start_session : pd.Timestamp, optional
        The first session for which we want data. If not provided,
        or if the date lies outside the range supported by the
        calendar, the first_session of the calendar is used.
    end_session : pd.Timestamp, optional
        The last session for which we want data. If not provided,
        or if the date lies outside the range supported by the
        calendar, the last_session of the calendar is used.
    minutes_per_day : int, optional
        The number of minutes in each normal trading day.
    create_writers : bool, optional
        Should the ingest machinery create the writers for the ingest
        function. This can be disabled as an optimization for cases where
        they are not needed, like the ``quantopian-quandl`` bundle.
"""
@bundles.register(
    'yxquant',
    calendar_name='SHSZ',#'SHSZ', # US equities
    start_session=None,
    end_session=None,
    minutes_per_day=240,
    create_writers=True
)

def yxdm_bundle(environ,
                 asset_db_writer,
                 minute_bar_writer,
                 daily_bar_writer,
                 adjustment_writer,
                 calendar,
                 start_session,
                 end_session,
                 cache,
                 show_progress,
                 output_dir,
                 tframe='daily'):
    """
    Build a zipline data bundle from the local files.
    """
    ## 1.链接数据库
    hive_username = environ.get('HIVE_USERNAME','dev')
    hive_password = environ.get('HIVE_PASSWORD','222222')
    print(hive_username,hive_password)

    try:
        #session_bars = create_engine('hive://localhost:8080/hive/default')
        session_bars = create_engine('sqlite:////Users/administrator/.zipline/data/tdx/2018-07-10T10;45;45.711151/session-bars.sqlite')
    except Exception as e:
        raise ValueError(
            "The hive database link failed，please make sure your username and password is correct and retry."
        )
    
    ## 2.抽取元数据
    raw_data = pd.read_sql("select  CAST(id AS int) as symbol, day as date, open, high, low, close, volume from bars;",
                        session_bars,
                        parse_dates=['date']
                        )
    raw_data['symbol']=raw_data['symbol'].apply(lambda x:format(x,'06'))

    #"symbol day open    high    low close   volume"
    asset_metadata = gen_asset_metadata(
        raw_data[['symbol', 'date']],
        show_progress
    )

    asset_db_writer.write(equities=asset_metadata)

    ## 3.daily_bar_writer
    symbol_map = asset_metadata.symbol
    end_session = min(end_session, pd.Timestamp.utcnow()-pd.Timedelta("1 days"))
    sessions = calendar.sessions_in_range(start_session, end_session)
    raw_data.set_index(['date', 'symbol'], inplace=True)

    if tframe == 'minute':
        writer = minute_bar_writer
    else:
        writer = daily_bar_writer
    writer.write(parse_pricing_and_vol(raw_data, sessions, symbol_map), show_progress=show_progress)

    ## 4.adjustment_writer
    raw_data.reset_index(inplace=True)
    raw_data['symbol'] = raw_data['symbol'].astype('category')
    raw_data['sid'] = raw_data.symbol.cat.codes
    if 'split_ratio' not in raw_data.columns:
        raw_data['split_ratio'] = 1
    if 'ex_dividend' not in raw_data.columns:
        raw_data['ex_dividend'] = 0
    adjustment_writer.write(
        splits=parse_splits(
            raw_data[['sid', 'date', 'split_ratio',]].loc[raw_data.split_ratio != 1],
            show_progress=show_progress
        ),
        dividends=parse_dividends(
            raw_data[['sid', 'date', 'ex_dividend',]].loc[raw_data.ex_dividend != 0],
            show_progress=show_progress
        )
    )

register_calendar_alias("YXQUANT", "SHSZ")


###### del
'''
    from io import BytesIO
    import tarfile
    from zipfile import ZipFile

    from click import progressbar
    from logbook import Logger
    import pandas as pd
    import requests
    from six.moves.urllib.parse import urlencode
    from six import iteritems

    from zipline.utils.calendars import register_calendar_alias
    from zipline.utils.deprecate import deprecated
    from . import core as bundles
    import numpy as np

    from numpy import empty
    from pandas import DataFrame, read_csv, Index, Timedelta, NaT
    from zipline.utils.cli import maybe_show_progress
    import os
    import sys
    from click import progressbar
    def yxdm_bundle_csv(environ,
                     asset_db_writer,
                     minute_bar_writer,
                     daily_bar_writer,
                     adjustment_writer,
                     calendar,
                     start_session,
                     end_session,
                     cache,
                     show_progress,
                     output_dir,
                     tframe='daily'):
        """
        Build a zipline data bundle from the local files.
        """
        print("1234567890")
        hive_username = environ.get('HIVE_USERNAME','dev')
        hive_password = environ.get('HIVE_PASSWORD','222222')
        print(hive_username,hive_password)
     
        symbols = ['000001', '000002']

        
        divs_splits = {'divs': DataFrame(columns=['sid', 'amount',
                                                  'ex_date', 'record_date',
                                                  'declared_date', 'pay_date']),
                       'splits': DataFrame(columns=['sid', 'ratio',
                                                    'effective_date'])}

        dtype = [('start_date', 'datetime64[ns]'),
                 ('end_date', 'datetime64[ns]'),
                 ('auto_close_date', 'datetime64[ns]'),
                 ('symbol', 'object')]
        metadata = DataFrame(empty(len(symbols), dtype=dtype))
        print(metadata)
        

        if tframe == 'minute':
            writer = minute_bar_writer
        else:
            writer = daily_bar_writer

        writer.write(_pricing_iter(symbols, metadata, divs_splits,show_progress),
                     show_progress=show_progress)

        
        # Hardcode the exchange to "CSVDIR" for all assets and (elsewhere)
        # register "CSVDIR" to resolve to the NYSE calendar, because these
        # are all equities and thus can use the NYSE calendar.
        metadata['exchange'] = "YXDM"

        asset_db_writer.write(equities=metadata)

        divs_splits['divs']['sid'] = divs_splits['divs']['sid'].astype(int)
        divs_splits['splits']['sid'] = divs_splits['splits']['sid'].astype(int)
        adjustment_writer.write(splits=divs_splits['splits'],
                                dividends=divs_splits['divs'])


    def _pricing_iter( symbols, metadata, divs_splits, show_progress):
        csvdir = '/Users/administrator/Application/boundless/zipline/tests/resources/csvdir_samples/csvdir/daily_t'
        symbols = ['000001', '000002']
        with maybe_show_progress(symbols, show_progress,
                                 label='Loading custom pricing data: ') as it:
            files = os.listdir(csvdir)
            for sid, symbol in enumerate(it):
                logger.debug('%s: sid %s' % (symbol, sid))

                try:
                    fname = [fname for fname in files
                             if '%s.csv' % symbol in fname][0]
                except IndexError:
                    raise ValueError("%s.csv file is not in %s" % (symbol, csvdir))

                dfr = read_csv(os.path.join(csvdir, fname),
                               parse_dates=[0],
                               infer_datetime_format=True,
                               index_col=0).sort_index()

                start_date = dfr.index[0]
                end_date = dfr.index[-1]

                # The auto_close date is the day after the last trade.
                ac_date = end_date + Timedelta(days=1)
                metadata.iloc[sid] = start_date, end_date, ac_date, symbol

                if 'split' in dfr.columns:
                    tmp = 1. / dfr[dfr['split'] != 1.0]['split']
                    split = DataFrame(data=tmp.index.tolist(),
                                      columns=['effective_date'])
                    split['ratio'] = tmp.tolist()
                    split['sid'] = sid

                    splits = divs_splits['splits']
                    index = Index(range(splits.shape[0],
                                        splits.shape[0] + split.shape[0]))
                    split.set_index(index, inplace=True)
                    divs_splits['splits'] = splits.append(split)

                if 'dividend' in dfr.columns:
                    # ex_date   amount  sid record_date declared_date pay_date
                    tmp = dfr[dfr['dividend'] != 0.0]['dividend']
                    div = DataFrame(data=tmp.index.tolist(), columns=['ex_date'])
                    div['record_date'] = NaT
                    div['declared_date'] = NaT
                    div['pay_date'] = NaT
                    div['amount'] = tmp.tolist()
                    div['sid'] = sid

                    divs = divs_splits['divs']
                    ind = Index(range(divs.shape[0], divs.shape[0] + div.shape[0]))
                    div.set_index(ind, inplace=True)
                    divs_splits['divs'] = divs.append(div)
                yield sid, dfr
'''