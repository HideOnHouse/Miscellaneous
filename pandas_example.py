import numpy as np
import pandas as pd

"""
PCAP 그 파일의 경로를 엘라스틱 서치에 올리면 될 것 같다
Pandas - CSV to Elasticsearch


How to read csv with python
    1. numpy
    2. pandas
    3. python read(much slower)

pandas merge acts like RDBMS(All data is in table which is very formulaic), fast
Elastic search has withdraw like join (application of data is pretty hard compare to RDBMS -> pandas)

Graph database(NEO)
    1. data are already joined, kakaotalk using graph database which makes it fast
    2. timeline database
    
"""


def make_dataframe(log_type):
    """

    :param log_type:
    :return:
    """
    df = pd.read_csv()
    return df


def do():
    df = make_dataframe('conn')
    conn_df = make_dataframe('conn')
    file_df = make_dataframe('files')
    http_df = make_dataframe('http')
    http_df['fuid'] = http_df.apply(
        lambda x: x['sfuids'] if x['dfuids'] == '-' else x['sfuids'] == '-' if x['dfuids'] == '-' else + ',' + x[
            'dfuids'], axis=1)

    file_df = pd.merge(file_df, http_df, how='inner', on=['fuid'])
    print(df[0])

    print(df[0].apply(lambda x: pd.datetime.fromtimestamp(x).strftime("%Y-%M-%D")))


def main():
    [print(i, end='') for i in open(__file__, encoding='utf-8')]


if __name__ == '__main__':
    main()
