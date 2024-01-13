import numpy as np
import polars as pl
import pandas as pd


def filter_pd(df:pd.DataFrame, filter_cols:list, group_by:str|list='ALL', n:float = 6, IQR:bool = True):
    if('ALL' not in df.columns and group_by == 'ALL'):
        df['ALL'] = 'all'
    
    inp_sp = df.shape[0]
    ret_cols = df.columns
    cols_df = [group_by]
    cols_df.extend(filter_cols)
    df = pd.DataFrame(df).copy()
    
    if(IQR):
        dfu = df[cols_df].groupby(group_by).quantile(.75).sort_index()
        dfl = df[cols_df].groupby(group_by).quantile(.25).sort_index()
        QR =  (dfu-dfl)*n
        dfu += QR
        dfl -= QR
        
    else:
        
        dfm = df[cols_df].groupby(group_by).mean().sort_index()
        dfs = df[cols_df].groupby(group_by).std().sort_index()
        dfl =  dfm-dfs*n
        dfu =  dfm+dfs*n
        
    df = pd.merge(df, dfl, on = group_by, suffixes=['', '_L'])    
    df = pd.merge(df, dfu, on = group_by, suffixes=['','_U'])
    for f in filter_cols:
        df = df[(df[f]>df[f'{f}_L']) & (df[f]<df[f'{f}_U'])]
    return df[ret_cols], inp_sp - df.values.shape[0]


def filter_pl(df:pl.DataFrame, filter_cols:list, group_by:str|list='ALL', n:float = 8, IQR = True):
    def sub_pl(x, filter_cols, n = 1):
        m = [x[0]]
        for i in range(len(filter_cols)):
            m.append((x[i+1] - x[i+len(filter_cols)+1])*n)
        return tuple(m)
    def add_pl(x, filter_cols, n = 1):
        m = [x[0]]
        for i in range(len(filter_cols)):
            m.append((x[i+1] + x[i+len(filter_cols)+1])*n)
        return tuple(m)
    inp_sp = df.shape[0]
    ret_cols = df.columns
    if('ALL' not in df.columns and group_by == 'ALL'):
        df = df.with_columns(pl.lit('all').alias('ALL'))
    
    cols_df = [group_by]
    cols_df.extend(filter_cols)
    if(IQR):

        dfu = df.select(pl.col(cols_df)).groupby(group_by).quantile(.75)
        dfl = df.select(pl.col(cols_df)).groupby(group_by).quantile(.25)
        QR = dfu.join(dfl, on = group_by).apply(lambda x: sub_pl(x, filter_cols, n = n)).rename({f"column_{i}":cols_df[i] for i in range(len(cols_df))})
        dfu = dfu.join(QR, on=group_by).apply(lambda x: add_pl(x, filter_cols=filter_cols, n = 1)).rename({f"column_{i}":f"{cols_df[i]}_U" for i in range(len(cols_df))})
        dfl = dfl.join(QR, on=group_by).apply(lambda x: sub_pl(x, filter_cols=filter_cols, n = 1)).rename({f"column_{i}":f"{cols_df[i]}_L" for i in range(len(cols_df))})
        
    else:
        
        dfstd = df.select(pl.col(cols_df)).groupby(group_by).std()
        dfmn = df.select(pl.col(cols_df)).groupby(group_by).mean()
        dfu = dfmn.join(dfstd, on=group_by).apply(lambda x: add_pl(x, filter_cols=filter_cols, n = n)).rename({f"column_{i}":f"{cols_df[i]}_U" for i in range(len(cols_df))})
        dfl = dfmn.join(dfstd, on=group_by).apply(lambda x: sub_pl(x, filter_cols=filter_cols, n = n)).rename({f"column_{i}":f"{cols_df[i]}_L" for i in range(len(cols_df))})
    df = df.join(dfl, left_on=group_by, right_on=f"{group_by}_L")
    df = df.join(dfu,left_on=group_by, right_on=f"{group_by}_U")    
    for f in filter_cols:
        df = df.filter(
            (pl.col(f)>pl.col(f"{f}_L"))
            & (pl.col(f)<pl.col(f"{f}_U"))
            )
    
    return df.select(pl.col(ret_cols)), inp_sp - df.shape[0]
    
       
