from scipy.stats import normaltest

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import ipywidgets as widgets
from datetime import datetime
from sklearn.metrics import r2_score


pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

"""
Funções deste arquivo:
    eda_date_convert - criar coluna de data ou transformar uma coluna já formatada
    eda_categorical_transform - transformar colunas 'object' em 'categorical'
    eda_read_df - criar análise do dataframe
    eda_overview - imprimir análise [depende do eda_read_df]
    eda_freq_tables - cria a tabela de frequência para uma coluna categórica
    eda_cat_charts - imprime barplot e tabela de freq para colunas categóricas
    eda_num_charts - imprime histograma e boxplot para colunas numéricas
"""


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_date_convert(df, make=(), transform=[], make_format='%Y-%m-%d', transform_format='%Y-%m-%d', drop_make=False):
    """Criar uma data a partir de três colunas com ano, mês e dia.
    Ou transformar em datime64 colunas que já estejam no formato de data.

    Args:
        df (dataframe): dataframe que contém as colunas
        make (tuple, optional): tupla com as três colunas nesta sequência: Y, M, D.
        transform (list, optional): lista com as colunas já no formato de data apenas para transformação.
        make_format (str, optional): Formato que estão os dados. Defaults to '%Y-%m-%d'.
        transform_format (str, optional): Formato que estão os dados. Defaults to '%Y-%m-%d'.
        drop_make (bool, optional): Excluir colunas que compõem o male. Defaults to False.
    """

    if len(make) == 3:
        y, m, d = make
        df['full_date'] = df[y].astype(str) +"-"+ df[m].astype(str) +"-"+ df[d].astype(str)
        df['full_date'] = pd.to_datetime(df['full_date'], format=make_format)
        if drop_make:
            df.drop(columns=[y,m,d], inplace=True)

    if len(transform) > 0:
        for t in transform:
            df[t] = pd.to_datetime(df[t], format=transform_format, errors='coerce')

    if (len(make)!=3) and (len(transform)==0):
        print("Erro nos parâmetros")


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_categorical_transform(df, cat_key_val={}):
    """Transformar colunas 'object' em ' categorical', permitindo
    ordenação das classes das colunas categóricas.
    Caso alguma coluna ou valor passada no dicionário não exista no
    dataframe, retornará erro.

    Args:
        df (dataframe): dataframe que contém as colunas
        cat_key_val (dict, optional): key- colunas do df; cal- classes ordenadas
    """

    keys_check = all([k in df.columns for k in cat_key_val.keys()])
    values_check = all([v in df[k].to_list() for k in cat_key_val.keys() for v in cat_key_val[k] ])
    dict_check = all([len(cat_key_val[k])>0 for k in cat_key_val.keys() ])
    
    if keys_check and values_check and dict_check:
        for k,v in cat_key_val.items():
            df[k] = pd.Categorical(df[k], categories=v, ordered=True)
    else:
        print(f'Keys check: {keys_check} \tValues check: {values_check} \t Dict check: {dict_check}')


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_read_df(df):
    """Analisar o dataframe. Sugestão de uso:
    df_eda, num_cols, cat_cols, dat_cols = eda_read_df(df)

    Args:
        df (dataframe): dataframe que será analisado

    Returns:
        df_eda: dataframe da análise
        num_cols: lista com colunas numéricas
        cat_cols: lista com colunas categóricas
        dat_cols: lista com colunas de datas

    Sugestão de uso:
        df_eda, num_cols, cat_cols, dat_cols = eda_read_df(df)
        eda_overview(df_eda=df_eda, df=df)
    """

    temp = {
        'col':[],
        'dtype':[],
        'nan_#':[],
        'nan_%':[],
        'special_char':[],
        ## numerical ------------
        'outliers':[],
        'zero':[],
        'negative':[],
        'positive':[],
        'stdev':[],
        'mean':[],        
        'median':[],
        'mode':[],        
        'min':[],
        'perc_25':[],
        'perc_75':[],
        'max':[],
        'range':[],
        'p_normal':[],
        ## categorical ----------
        'classes':[],
        'top3_classes':[]
    }

    num_cols = []
    cat_cols = []
    dat_cols = []

    colunas = df.columns
    rows = df.shape[0]

    for col in colunas:
        temp['col'].append(                 col)
        temp['dtype'].append(               df[col].dtype)
        temp['nan_#'].append(               df[col].isna().sum())
        temp['nan_%'].append(               round(df[col].isna().sum() / rows *100,2))

        if df[col].dtype in ['O', 'category']:
            top3 = df[col].value_counts()[:3].values.sum() / rows
            cat_cols.append(col)
            temp['mode'].append(            df[col].mode()[0])
            temp['special_char'].append(    df[col].str.contains(r'[-@&/#,+()`|$~^%.:*?!<>{}]').sum())
            temp['classes'].append(         len(df[col].unique()))
            temp['top3_classes'].append(    f'{top3:.2%} > ' + " | ".join( [ f'{v[0]} {v[1]/rows:.2%}' for v in df[col].value_counts()[:3].items() ] ))
            
            temp['outliers'].append(0)
            temp['zero'].append(0)
            temp['negative'].append(0)
            temp['positive'].append(0)
            temp['mean'].append(0)
            temp['stdev'].append(0)
            temp['median'].append(0)
            temp['max'].append(0)
            temp['min'].append(0)
            temp['range'].append(0)
            temp['perc_25'].append(0)
            temp['perc_75'].append(0)
            temp['p_normal'].append(0)
        
        elif df[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            num_cols.append(col)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            _, p = stats.normaltest(df[col])

            temp['mode'].append(            df[col].mode()[0])
            temp['outliers'].append(        ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum())
            temp['zero'].append(            np.sum((df[col] == 0).values.ravel()))
            temp['negative'].append(        np.sum((df[col] < 0).values.ravel()))
            temp['positive'].append(        np.sum((df[col] > 0).values.ravel()))
            temp['mean'].append(            round(df[col].mean(), 2))
            temp['stdev'].append(           round(df[col].std(), 2))
            temp['median'].append(          df[col].median())
            temp['max'].append(             df[col].max())
            temp['min'].append(             df[col].min())
            temp['range'].append(           df[col].max() - df[col].min())
            temp['perc_25'].append(         q1)
            temp['perc_75'].append(         q3)
            temp['p_normal'].append(        p)
            
            temp['special_char'].append(0)
            temp['classes'].append(0)
            temp['top3_classes'].append('')

        elif df[col].dtype == 'datetime64[ns]':
            dat_cols.append(col)
            temp['mode'].append(            df[col].mode()[0].date())
            temp['median'].append(          df[col].median().date())
            temp['max'].append(             df[col].max().date())
            temp['min'].append(             df[col].min().date())
            temp['range'].append(           df[col].max() - df[col].min())
            temp['classes'].append(         len(df[col].unique()))
            temp['top3_classes'].append(    " | ".join( [ f'{v[0].date()} {v[1]/rows:.2%}' for v in df[col].value_counts()[:3].items() ] ))

            temp['outliers'].append(0)
            temp['zero'].append(0)
            temp['negative'].append(0)
            temp['positive'].append(0)
            temp['mean'].append(0)
            temp['stdev'].append(0)
            temp['special_char'].append(0)
            temp['perc_25'].append(0)
            temp['perc_75'].append(0)
            temp['p_normal'].append(0)


    df_eda = pd.DataFrame.from_dict(data= temp).sort_values(by=['col'])
    return df_eda, num_cols, cat_cols, dat_cols


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_overview(df_eda, df):
    """Imprimir informações extraídas da função eda_read_df.

    Args:
        df_eda (_type_): dataframe com metadados
        df (_type_): dataframe com dados
    
    Sugestão de uso:
        df_eda, num_cols, cat_cols, dat_cols = eda_read_df(df)
        eda_overview(df_eda=df_eda, df=df)
    """
    dupli = df.duplicated().sum()
    rows = df.shape[0]
    
    # OVERVIEW -------------------------------------------------------------------------------------
    t1 = np.sum(df.isna().sum(axis=1) > 0)
    t2 = df_eda[df_eda['nan_#'] == df_eda['nan_#'].max()]['col'].to_list()[0]
    t3 = df_eda[df_eda['col']==t2]['nan_#'].to_list()[0]
    nan_up5 = df_eda[(df_eda['nan_%'] < 5) & (df_eda['nan_%'] > 0)]['col'].to_list()
    nan_bi30 = df_eda[(df_eda['nan_%'] > 30)]['col'].to_list()
    t4 = np.sum(df[nan_up5].isna().sum(axis=1) > 0)
    print('='*100, '\nOverview', '\n'+'='*100)
    print(f'Rows: {df.shape[0]:,} \t\tw/ some nan: {t1:,} | {t1/rows:.2%}')
    print(f'\t\t\tMax nan column: {t2} | {t3:,} | {t3/rows:.2%}')
    print(f'\t\t\tCols more than 30% nan: {len(nan_bi30)}  |  {nan_bi30}')
    print(f'\t\t\tCols up to 5% nan: {len(nan_up5)}  |  {nan_up5}')
    print(f'\t\t\tIt represents: {t4:,} | {t4/rows:.2%}')
    print(f'\t\t\tDuplicates: {dupli:,} | {dupli/rows:.2%}\n')

    t1 = np.sum([ c in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'] for c in df_eda['dtype'].astype(str).to_list() ])
    t2 = np.sum([ c in ['object', 'category'] for c in df_eda['dtype'].astype(str).to_list() ])
    t3 = np.sum([ c in ['datetime64[ns]'] for c in df_eda['dtype'].astype(str).to_list() ])
    print(f'Features: {df.shape[1]:,} \t\tNum_cols: {t1} \t\tCat_cols: {t2} \t\tDat_cols: {t3}')

    t1 = np.sum((df_eda['nan_#'] > 0).values.ravel())
    t2 = np.sum((df_eda['special_char'] > 0).values.ravel())
    print(f'\t\t\tw/ Nan: {t1} \t\tw/ Special char: {t2}')

    t1 = np.sum((df_eda['outliers'] > 0).values.ravel())
    t2 = df_eda['classes'].max()
    print(f'\t\t\tw/ Outlier: {t1} \t\tw/ Max classes: {t2}')

    t1 = np.sum((df_eda['negative'] > 0).values.ravel())
    t2 = df_eda[df_eda['classes']>0]['classes'].min()
    print(f'\t\t\tw/ Negative: {t1} \t\tw/ Min classes: {t2}')

    t1 = np.sum((df_eda['p_normal'] > 0.0500).values.ravel())
    print(f'\t\t\tNormaltest: {t1} (scipy)')

    cols = df_eda[ (df_eda['min']==0) & (df_eda['max']==1) & (df_eda['dtype']!='O') ]['col'].to_list()
    print(f'\nPossible bool columns defined as numerical:\n\t\t\t{cols}')

    
    # NUM -------------------------------------------------------------------------------------
    print('\n'+'='*100, '\nNumerical Features', '\n'+'='*100)
    display(df_eda[ (df_eda['dtype']!='O') & 
                    (df_eda['dtype']!='datetime64[ns]') & 
                    (df_eda['dtype']!='category')].drop(columns=['special_char', 'classes', 'top3_classes']))
    
    # CAT -------------------------------------------------------------------------------------
    print('\n'+'='*100, '\nCategorical Features', '\n'+'='*100)
    display( df_eda[ (df_eda['dtype']=='O') |
                     (df_eda['dtype']=='category')][['col', 'dtype', 'nan_#', 'nan_%', 'special_char', 'classes', 'top3_classes']])
    
    # DAT -------------------------------------------------------------------------------------
    print('\n'+'='*100, '\nDate Features', '\n'+'='*100)
    display( df_eda[df_eda['dtype']=='datetime64[ns]'].drop(columns=['special_char','outliers', 'zero','negative','positive','mean','stdev','perc_25','perc_75', 'p_normal']))


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_freq_tables(df, col, n, hue=None):
    """Retorna a tabela de frequências absoluta, relativa
    e relativa acumulada de uma determinada coluna.
    Também agrupa as categorias que ficarem de fora do limite
    escolhido (top) numa única categoria chamada '_Others'.
    Caso categorias sejam somadas em '_Others', é adicionada uma
    coluna 'cat_#' indicando quantas categorias foram agrupadas.

    Args:
        df (dataframe): dataframe
        col (str): nome da coluna categórica
        n (int): quantidade de categorias
        hue (str): nome da coluna categórica
    """
    col = [col]
    if hue:
        col.append(hue)
        top = df.groupby(col, dropna=False)[col[0]].count().unstack().fillna(0)
    else:
        top = df.groupby(col, dropna=False)[col[0]].count().to_frame(name='abs')
    x = int(n)

    top['abs'] = top.sum(axis=1)
    top.sort_values('abs', ascending=False, inplace=True)
    top_n = top[:x].copy()
    if df[col[0]].nunique() > x:
        top_n.loc['_Others', :] = top[x:].sum(axis=0)
        top_n['cat_#'] = '_'
        top_n.loc['_Others','cat_#'] = str(len(top[x:])) + f'/{len(top)}'

    if hue:
        top_hue = top_n.reset_index().melt(id_vars=col[0], value_vars=top_n.columns.drop(['abs', 'cat_#'], errors='ignore'),var_name =col[1], value_name='count')
        top_hue = pd.merge(top_hue, top_n['abs'], on=col[0]).sort_values(by=['abs', 'count'], ascending=[0,0]).reset_index(drop=True)
        top_hue['abs'] = top_hue['count']
        top_hue.drop(columns='count', inplace=True)
        if df[col[0]].nunique() > x:
            idx = top_hue[top_hue[col[0]] == '_Others'].index
            before = top_hue.iloc[[i for i in top_hue.index if i not in idx], :]
            after = top_hue.iloc[idx, :]
            top_hue = pd.concat([before, after])
            del(before)
            del(after)
        top_hue['rel_%'] = round(top_hue['abs']/ top_hue['abs'].sum(axis=0) *100,2)
        top_hue['rel_%_acc'] = round(top_hue['rel_%'].cumsum(), 2)

    top_n['rel_%'] = round(top_n['abs']/ top_n['abs'].sum(axis=0) *100,2)
    top_n['rel_%_acc'] = round(top_n['rel_%'].cumsum(), 2)
    
    if df[col[0]].nunique() > x:
        top_n = top_n[['abs', 'rel_%', 'rel_%_acc', 'cat_#']]
    else:
        top_n = top_n[['abs', 'rel_%', 'rel_%_acc']]

    del(top)
    if hue:
        return top_n, top_hue
    else:
        return top_n


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_cat_charts(df, cat_cols, n=99, hue=None):
    """Análise das variáveis categóricas.

    Args:
        df (dataframe): dataframe
        cat_cols (list): lista de colunas categóricas
        n (int, optional): quantidade de categorias. Ordenadas pela contagem.
                           Caso a coluna tenha mais categorias do que valor informado, as
                           demais categorias serão agrupadas em '_Others'.
        hue (str, optional): nome da coluna categórica que será usado como parâmetro 'hue' nos gráficos
    
    Sugestão de uso:
        df_eda, num_cols, cat_cols, dat_cols = eda_read_df(train_transf)
        eda_cat_charts(df=df, cat_cols=cat_cols, n=5, hue='TARGET')
    """
    y=0
    bbox=[0, 0, 1, 1]

    if hue:
        c = 3
        gridspec_kw={"width_ratios": (.50, .25, .30)}
        width=17
    else:
        c = 2
        gridspec_kw={"width_ratios": (.60, .40)}
        width=13
        
    lin = -(- len(cat_cols) //c)
    f, axes = plt.subplots(lin*2, c, figsize=(width, 4*lin+n), gridspec_kw=gridspec_kw) 

    for col in cat_cols:
        axes[y, 0].set_title(">> " + col, loc='left')
        if hue:
            info, top_hue = eda_freq_tables(df=df, col=col, n=n, hue=hue)
            leg = list(top_hue[hue].unique())
            sns.barplot(data = top_hue, x='abs', y=col,  ax=axes[y, 0], palette="CMRmap_r", hue=hue, edgecolor='white', dodge=False)
            axes[y, 0].legend(ncol=len(leg), bbox_to_anchor=(1, 1.23), edgecolor='white', loc=1, framealpha=0)
            axes[y, 0].set(ylabel=None)
            info2 = top_hue.drop(columns=['rel_%', 'rel_%_acc']).pivot_table(values='abs', index=col, columns=hue, sort=False)
            axes[y, 2].table(cellText = info2.values, bbox=bbox, colLabels=info2.columns).auto_set_font_size(False)
            axes[y, 2].axis('off')
            axes[y, 2].set_title(f"Freq Table by {hue}", loc='left')
        else:
            info = eda_freq_tables(df=df, col=col, n=n)
            sns.barplot(x=info['abs'], y=list(info.index),  ax=axes[y, 0], color='b')
        
        axes[y, 1].set_title("Freq Table", loc='left')
        axes[y, 1].table(cellText = info.values, rowLabels = info.index, bbox=bbox, colLabels=info.columns).auto_set_font_size(False)
        axes[y, 1].axis('off')

        y0 = 1/(lin*2)*y 
        axes[y, 0].plot([0, 1], [y0,y0], color='gray', lw=1, transform=plt.gcf().transFigure, clip_on=False)
        y+=1

    plt.tight_layout()
    print('='*100, '\nCategorical Features Charts', '\n'+'='*100)
    plt.show()


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def eda_num_charts(df_eda, df, num_cols, hue=None):
    """Análise das variáveis numéricas.

    Args:
        df_eda (dataframe): dataframe com metadados
        df (dataframe): dataframe com dados
        num_cols (list): lista de colunas numéricas
        hue(str): nome da coluna categórica que será usada como parâmetro 'hue' nos gráficos
    
    Sugestão de uso:
        df_eda, num_cols, cat_cols, dat_cols = eda_read_df(train_transf)
        eda_num_charts(df_eda=df_eda, df=df, num_cols=num_cols, hue='TARGET')
    """
    if (hue) and (hue not in df.columns.to_list()):
        print('Parâmetro "hue" não contém no dataframe.')
        return False

    c = 2
    lin = -(- len(num_cols) //c)
    if hue:
        leg = list(df[hue].unique())

    f, axes = plt.subplots(lin*2, c, figsize=(17, 4*lin), gridspec_kw={"height_ratios": (.85, .15)*lin}) 
    y=0

    for name in num_cols:
        i, j = divmod(y, c)
        i*=2
        # -----------------------------------------------------------------------------------------------
        sns.histplot(data=df, x=name, kde=False, hue=hue, ax=axes[i, j]) 
        if hue:
            axes[i, j].legend(leg, ncol=len(leg), bbox_to_anchor=(1, 1.13), edgecolor='white', loc=1, framealpha=0)
        axes[i, j].set_title(">> " + name, loc='left')
        axes[i, j].set_xlabel("")

        axes[i, j].axvline(df_eda[df_eda['col']==name]['mean'].to_list()[0], color="gray", linestyle='-')
        axes[i, j].axvline(df_eda[df_eda['col']==name]['median'].to_list()[0], color="black", linestyle=':')
        axes[i, j].axvline(df_eda[df_eda['col']==name]['mode'].to_list()[0], color="yellow", linestyle='--')

        info = str(df_eda[df_eda['col']==name].drop(columns=['col', 'special_char', 'classes', 'top3_classes', 'perc_25', 'perc_75']).T)
        info = info[info.find('dtype'):]
        axes[i, j].text(1.01, 1., str(info), ha='left', va='top', transform=axes[i, j].transAxes)
        
        # -----------------------------------------------------------------------------------------------
        sns.boxplot(x=df[name], ax=axes[i+1, j])
        axes[i+1, j].axis('off')

        info = str(df_eda[df_eda['col']==name][['perc_25', 'perc_75']].T)
        info = info[info.find('perc_'):]
        axes[i+1, j].text(1.01, 1.05, str(info), ha='left', va='top', transform=axes[i+1, j].transAxes)
        
        # -----------------------------------------------------------------------------------------------
        axes[i+1, j].plot([0.5, 0.5], [0, 1], color='gray', lw=1, transform=plt.gcf().transFigure, clip_on=False)
        if j==0:
            y0 = 1/(lin*2)*y 
            axes[i+1, j].plot([0, 1], [y0, y0], color='gray', lw=1, transform=plt.gcf().transFigure, clip_on=False)
        
        y+=1

    if len(num_cols) < lin*c:
        i, j = divmod(y, c)
        i*=2
        axes[i, j].axis('off')
        axes[i+1, j].axis('off')

    plt.tight_layout()
    print('='*100, '\nNumerical Features Charts', '\n'+'='*100)
    print('Lines: \nMean = gray - \tMedian = black : \tMode = Yellow --')
    plt.show()


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def heatmap_ticks(heat):
    min = heat.min().min() if heat.ndim > 1 else heat.min()
    max = heat.max().max() if heat.ndim > 1 else heat.max()
    med = (max-min)/2
    ticks = [min, min+med/2, min+med, max-med/2,max]
    return ticks


def heatmap_subplot(heat, mycolors, annot, fmt, square, ax, bar_anchor, bar_shrink, bar_orientation, bar_location=None):
    tick = heatmap_ticks(heat)
    sns.heatmap(heat, cmap=mycolors, square=square, annot=annot, fmt=fmt,
                linewidth=0.01, linecolor="#444", vmin=heat.min().min(), vmax=heat.max().max(), cbar=True,
                cbar_kws = {'use_gridspec':False, 'location':bar_location, 'orientation':bar_orientation, 
                            'anchor':bar_anchor, 'shrink':bar_shrink, 'ticks':tick},
                ax=ax)
    

def heatmap_plot(df, row, col, value, aggfunc='mean', annot=False, color_bin=False):
    heat = df.pivot_table(index=row, columns=col, values=value, aggfunc=aggfunc)
    
    if color_bin:
        mycolors=[(0.7,0.7,0.7),(0.4,0.5,0.4),(0.2,0.3,0.3),(0.1,0.7,0)]
    else:
        mycolors="BuPu"
    
    fmt = ',.1f' if aggfunc=='mean' else ',.0f'
    
    fig, axd = plt.subplot_mosaic([['UL', 'UR'],
                                   ['LL', 'LR']],
                                  figsize=(17,9), constrained_layout=True,
                                  gridspec_kw={'width_ratios': (.9,.1), 'height_ratios': (.85,.15)})
    
    heatmap_subplot(heat=heat, mycolors=mycolors, annot=annot,
                    fmt=fmt, square=False, ax=axd['UL'], 
                    bar_anchor='NE', bar_shrink=0.3, bar_orientation='horizontal', bar_location='top')
    
    heatmap_subplot(heat=heat.sum(axis=0)[np.newaxis, :], mycolors=mycolors, annot=annot,
                    fmt=fmt, square=False, ax=axd['LL'], 
                    bar_anchor='SE', bar_shrink=0.3, bar_orientation='horizontal')
    
    heatmap_subplot(heat=heat.sum(axis=1)[:, np.newaxis], mycolors=mycolors, annot=annot,
                    fmt=fmt, square=False, ax=axd['UR'], 
                    bar_anchor='NE', bar_shrink=0.4, bar_orientation='vertical')
        
    axd['UL'].set_title(f'{aggfunc.upper()}  of  {value.upper()}   by:   {row.upper()}   and   {col.upper()}', loc='left')
    axd['UR'].axis('off')
    axd['LR'].axis('off')
    axd['LL'].axis('off')
    plt.xlabel('')
    plt.ylabel('')
    plt.show();
    

def eda_heatmap_ux(df, max_cat=99):
    heat_cols = [ k for k,v in df.select_dtypes(include=[int, object, 'category']).nunique().to_dict().items() if v<=max_cat]
    heat_vals = df.select_dtypes(include=[int, float]).columns

    items0 = [
        widgets.Dropdown(options=heat_cols, value=heat_cols[0], description='row feature:', disabled=False),
        widgets.Dropdown(options=heat_cols, value=heat_cols[1], description='col feature:', disabled=False),
        widgets.Dropdown(options=heat_vals, value=heat_vals[2], description='metric:', disabled=False),
    ]
    items1 = [
        widgets.Dropdown(options=['mean', 'sum', 'count', 'median', 'max', 'min'], value='mean', description='aggfunc:', disabled=False),
        widgets.Checkbox(value=False, description='annotations'),
        widgets.Checkbox(value=False, description='color bins')
    ]

    box_layout = widgets.Layout(display='flex', flex_flow='row', justify_content='space-between', align_items='center')
    box0 = widgets.Box(children=items0, layout=box_layout)
    box1 = widgets.Box(children=items1, layout=box_layout)
    box = widgets.VBox([box0, box1])
    
    out = widgets.interactive_output(heatmap_plot, {
                                    'df':widgets.fixed(df), 
                                    'row':items0[0], 
                                    'col':items0[1], 
                                    'value':items0[2], 
                                    'aggfunc':items1[0],
                                    'annot':items1[1],
                                    'color_bin':items1[2]}
                                    )
    
    display(box, out)


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
def run_models_get_results(models, X_train, y_train):
    """Rodando o dicionário de modelos construído.

    Args:
        models (dict): dicionário com as estruturas dos modelos
        X_train (dataframe): dataframe com as features de treino
        y_train (array): vetor com os targets
    
    Sugestão de uso:
        results, best_model_fitted = run_models_get_results(models=models, X_train=X_train, y_train=y_train)
        compare_model_score(results=results, score=score, gscv=gscv, n_models=len(models.keys()), task='classification')
        model_ = 'logistic'
        estimator = best_model_fitted[model_]
        y_pred = estimator.predict(X_test)
        print(classification_report(y_test, y_pred))
    """
    results = {}
    best_model_fitted = {}
    t = datetime.now()

    for alg, clf in models.items():
        t0 = datetime.now()
        print(f"Treinando: {alg}")
        clf.fit(X_train, y_train) 
        t1 = datetime.now()
        best_model_fitted[alg] = clf.best_estimator_
        results[alg] = clf.cv_results_
        print(f"\t{alg} treinado: \t Duração: {t1-t0}")

    print(f"\nTempo total: {t1-t}")
    results = pd.DataFrame.from_dict(results)

    return results, best_model_fitted


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_best_param_each_score(results):
    """Retorna os melhores parâmetros de cada score.

    Args:
        results (dict): resultados do crossvalidations da função run_models_get_results
    """
    best_param_each_score = {}
    for indice in [x for x in results.index if 'rank_test_' in x]:
        best_each = [x[0]-1 for x in results.loc[indice]]
        best_param_each_score[indice]= [ x[best_each[i]] for i,x in enumerate(results.loc['params'])]
    text=''
    for key, value in best_param_each_score.items():
        text+= ' > ' + str(key) + '\n   ' + '\n   '.join([str(x) for x in value]) + '\n\n'
    return text


def get_mean_test_each_score(results, score):
    """Retorna o resultado médio de cada score

    Args:
        results (dict): resultados do crossvalidations da função run_models_get_results
        score (list): lista dos scores selecionados para avaliação
    """
    mean_test_each_score = {}
    for scr in score:
        for indice in [x for x in results.index if scr in x and "mean_test_" in x]:
            best_each = [x[0]-1 for x in results.loc['rank_test_'+scr]]
            mean_test_each_score[indice]= [ x[best_each[i]] for i,x in enumerate(results.loc[indice])]
    return mean_test_each_score


def get_best_score_each_model(results, score):
    """Retorna o o melhor score de cada modelo

    Args:
        results (dict): resultados do crossvalidations da função run_models_get_results
        score (list): lista dos scores selecionados para avaliação
    """
    best_score_each_model={}
    for scr in score:
        for indice in [x for x in results.index if scr in x and "split" in x]:
            best_each = [x[0]-1 for x in results.loc['rank_test_'+scr]]
            best_score_each_model[indice]= [ x[best_each[i]] for i,x in enumerate(results.loc[indice])]
    df_best_score_each_model = pd.DataFrame(best_score_each_model).transpose()
    df_best_score_each_model.columns = results.columns
    return df_best_score_each_model


def make_compare_charts(best_score_each_model, best_param_each_score, mean_test_each_score, score, gscv, n_models, task):
    """Constrói boxplot comparando cada score de cada modelo.

    Args:
        best_score_each_model: resultado da função get_best_score_each_model
        best_param_each_score: resultado da função get_best_param_each_score
        mean_test_each_score: resultado da função get_mean_test_each_score
        score (list): lista dos scores selecionados para avaliação
        gscv: objeto com o crossvalidation
        n_models (int): quantidade de modelos no dicionário
        task (str): classificação ou regressão
    """
    share_y = True if task=='classification' else False
    f, axes = plt.subplots(1, len(score), figsize=(15,5), sharey=share_y)  

    print('>> Best Params', '='*100)
    print(best_param_each_score)
    for idx, scr in enumerate(score):
        linhas = [x for x in best_score_each_model.index if scr in x]
        box_df = best_score_each_model.loc[linhas]
        axes[idx].boxplot([ scores for alg, scores in box_df.iteritems() ])
        axes[idx].set_xticklabels(box_df.columns)
        [axes[idx].axhline( x, color="gray", linestyle=':') for x in mean_test_each_score['mean_test_'+scr]]
        if task=='classification':
            [axes[idx].text(0.53, x, box_df.columns[i], color='gray') for i,x in enumerate(mean_test_each_score['mean_test_'+scr])]
            [axes[idx].text(n_models+0.5, x, f'{x:.2f}', color="gray") for x in mean_test_each_score['mean_test_'+scr]]
        else:
            axes[idx].set_yscale("log")
        #axes[idx].text(0.5, 1, "mean", color='gray')
        axes[idx].set_title(scr)
        #label = best_param_each_score['rank_test_'+scr]
        #print('>>', scr)

    print('>> Charts', '='*100)
    print(f'   Results from Cross Validation with {gscv.get_n_splits()} folds.')
    plt.tight_layout()
    plt.show()


def compare_model_score(results, score, gscv, n_models, task='classification'):
    """Única função que precisa ser chamada para execução de todos os passos acima.

    Args:
        results (dict): resultados do crossvalidations da função run_models_get_results
        score (list): lista dos scores selecionados para avaliação
        gscv: objeto com o crossvalidation
        n_models (int): quantidade de modelos no dicionário
    
    Sugestão de uso:
        gscv = StratifiedKFold(n_splits=10, shuffle=True)
        score = ['roc_auc', 'f1', 'accuracy']
        rft = 'f1'

        models = {
            'logistic':  GridSearchCV(
                Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('log', LogisticRegression())]), 
                param_grid={
                    'log__penalty': ['l1', 'l2'],
                    'log__solver': ['liblinear'],
                },
                scoring=score,
                #error_score='raise',
                refit=rft,
                cv=gscv),

            'randomforest':  GridSearchCV(
                Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('rf', RandomForestClassifier())]), 
                param_grid={
                    'rf__max_depth': [5, 20],
                    'rf__criterion': ['entropy', 'gini'],
                },
                scoring=score,
                refit=rft,
                cv=gscv),
                
            'svmrbf': GridSearchCV(
                Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('svm', LinearSVC())]), 
                param_grid={
                    'svm__C': [1.0, 5.0],
                    'svm__penalty': ['l2'],
                },
                scoring=score,
                refit=rft,
                cv=gscv),
        }

        results = run_models_get_results(models=models, X_train=X_train, y_train=y_train)
        compare_model_score(results=results, score=score, gscv=gscv, n_models=len(models.keys()))

    """
    if task not in ['classification', 'regression']:
        print("Task must be: classification or regression")
        return False
    best_param_each_score = get_best_param_each_score(results)
    mean_test_each_score = get_mean_test_each_score(results, score)
    best_score_each_model = get_best_score_each_model(results, score)
    make_compare_charts(score=score, gscv=gscv, n_models=n_models,
                        best_param_each_score=best_param_each_score,
                        best_score_each_model=best_score_each_model,
                        mean_test_each_score=mean_test_each_score,
                        task=task)


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pred_and_plot_regression(X_test, y_test, models, best_model_fitted):
    """Fazer predição e plotar o scatterplot do Real vs Previsto.

    Args:
        X_test (dataframe): features
        y_test (series): target
        models (dict): dicionário com as estruturas dos modelos
        best_model_fitted (dict): resultado da função 'run_models_get_results'
    
    Sugestão de uso:
        results, best_model_fitted = run_models_get_results(models=models, 
                                                                          X_train=X_train,
                                                                          y_train=y_train)
        pred_and_plot_regression(X_test = X_test, 
                                y_test = y_test, 
                                models = models, 
                                best_model_fitted = best_model_fitted)
    """
    model_ = models.keys()
    c = 2
    lin = -(- len(model_) //c)

    f, axes = plt.subplots(lin, c, figsize=(10,10), sharey=False)  
    axy=0
    for model in model_:
        i, j = divmod(axy, c)

        estimator = best_model_fitted[model]
        y_pred = estimator.predict(X_test)
        r2= r2_score(y_test, y_pred)
       
        x = np.linspace(0, y_test.max())
        y = x

        axes[i, j].set_title(f"{model} | Actual x Pred | r2: {r2:.4f}")
        axes[i, j].plot(x, y, color="red", ls=":")
        axes[i, j].scatter(x=y_test, y=y_pred)
        axes[i, j].set_xlabel("Actual")
        axes[i, j].set_ylabel("Pred")
        axy+=1

    if len(model_) < lin*c:
        i, j = divmod(axy, c)
        axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


# ************************************************************************************************************
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++