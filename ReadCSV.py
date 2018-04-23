import pandas as pd


def read_base(csv_file, base):
    df = pd.read_csv(csv_file, sep=";")
    if base == 'completa':
        df = remover_atributor(df)
        return df.iloc[:,1:].values
    if base == 'shape':
        df = remover_atributor(df)
        return df.iloc[:,1:7].values
    if base == 'rgb':
        return df.iloc[:, 10:].values


def remover_atributor(df):
    df = df.drop(['REGION-PIXEL-COUNT'], axis=1)
    df = df.drop(['SHORT-LINE-DENSITY-5'], axis=1)
    df = df.drop(['SHORT-LINE-DENSITY-2'], axis=1)

    return df


def calc_unique(csv_file):
    df2 = pd.read_csv(csv_file, sep=";")
    df2 = df2.iloc[:,0].values
    unique = []
    for i in range(len(df2)):
        if df2[i] not in unique:
            unique.append(df2[i])

    return unique


def subst_classes(unique, csv_file):
    df3 = pd.read_csv(csv_file, sep=";")
    df3 = df3.iloc[:, 0].values
    classes = []
    for i in range(len(df3)):
        for k in range(len(unique)):
            if df3[i] == unique[k]:
                classes.append(k)

    return classes