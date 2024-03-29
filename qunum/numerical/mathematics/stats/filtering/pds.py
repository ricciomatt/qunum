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