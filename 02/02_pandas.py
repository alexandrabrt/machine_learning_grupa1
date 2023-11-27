import pandas as pd

# data = {'Nume': ['Ana', 'Bogdan', 'Cristina'],
#         'Varsta': [25, 30, 22],
#         'Salariu': [50000, 60000, 45000]}
#
# df = pd.DataFrame(data)
# print(df)
# # nume = df['Nume']
# # print(nume)
# salariu_bogdan = df.at[1, 'Salariu']
# print(salariu_bogdan)
# df['Experienta'] = [2, 8, 1]
# print(df)
# print('+++++++++++++++++')
# df.set_index('Nume', inplace=True)
# print(df.loc['Bogdan'])
# print(df.iloc[1])
# print(df.loc[['Ana', 'Cristina'], ['Varsta', 'Salariu']])
# df_filtrat = df[df['Varsta'] > 25]
# print(df_filtrat)
# df_filtrat_complex = df[(df['Varsta'] >= 22) & (df['Experienta'] >= 2)]
# print(df_filtrat_complex)

# df_sortat = df.sort_values(by='Varsta', ascending=True)
# print(df_sortat)

# data = {'Nume': ['Ana', 'Bogdan', 'Ana', 'Cristina', 'David', 'Ana'],
#         'Varsta': [25, 30, 25, 22, 35, 25],
#         'Salariu': [50000, 60000, 50000, 45000, 70000, 50000]}
#
# df = pd.DataFrame(data)
# print(df)
# print('===============')
# df_fara_duplicate = df.drop_duplicates()
# print(df_fara_duplicate)

# data = {'Nume': ['Ana', 'Bogdan', None, 'Cristina', 'David'],
#         'Varsta': [25, 30, None, 22, 35],
#         'Salariu': [50000, None, 45000, 70000, 60000]}
# df = pd.DataFrame(data)
# print(df)
# print('===========')
# df_fara_none = df.dropna()
# print(df_fara_none)
# df_cu_zero = df.fillna(0)
# print(df_cu_zero)

# a = 2
# b = 3
# c = 4
# d = 0
# print((a + b + c + d) / 4)
# df['Experienta'] = [2, 5, 1, 3, 4]
# print(df)
# df_redenumire = df.rename(columns={'Nume': 'Numele', 'Varsta': 'Varsta', 'Salariu': 'Salariul'})
# df.rename(columns={'Nume': 'Numele', 'Varsta': 'Varsta', 'Salariu': 'Salariul'}, inplace=True)
# print(df)

# data = {'Nume': ['Ana', 'Bogdan', 'Cristina', 'David', 'Elena', 'Florin'],
#         'Varsta': [25, 30, 22, 35, 28, 40],
#         'Salariu': [50000, 60000, 45000, 70000, 55000, 80000],
#         'Departamente': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR']}

# df = pd.DataFrame(data)
# print(df)
# print('=============')
# grupuri_departamente = df.groupby('Departamente')
# for nume_departament, grup in grupuri_departamente:
#     print(grup)
# medie_salarii = grupuri_departamente['Salariu'].mean()
# print(medie_salarii)
# rezultate_agregare = grupuri_departamente.agg({'Varsta': 'mean', 'Salariu': ['sum', 'median'], 'Nume': 'count'})
# print(rezultate_agregare)

# data = {'Data': ['2022-01-01', '2022-02-01', '2022-03-01'],
#         'Vanzari': [100, 150, 200]}

# df = pd.DataFrame(data)
# print(df)
# df['Data'] = pd.to_datetime(df['Data'])
# print(df.dtypes)
# df['Zi'] = df['Data'].dt.day
# df['Luna'] = df['Data'].dt.month
# df['An'] = df['Data'].dt.year
# print(df)
# df['Diferenta_zi'] = (df['Data'] - pd.to_datetime('2022-01-01')).dt.days
# print(df)

# df1 = pd.DataFrame({'Nume': ['Ana', 'Bogdan', 'Silviu'], 'Varsta': [25, 30, 34]}, index=[1, 2, 3])
# df2 = pd.DataFrame({'ID': [2, 3, 4], 'Nume': ['Cristina', 'David', 'Maria'], 'Varsta': [22, 35, 21]}, index=[2, 3, 4])
# df2 = pd.DataFrame({'Experienta': [2, 3, 4], 'Departament': ['HR', 'IT', 'Sales']}, index=[2, 3, 4])

# df_concat_randuri = pd.concat([df1, df2], ignore_index=True)
# df_concat_randuri = pd.concat([df1, df2])
# print(df_concat_randuri)

# df_concat_coloane = pd.concat([df1, df2], axis=1)
# print(df_concat_coloane)
# df_merge = pd.merge(df1, df2, on='ID', how='outer')
# df_merge = pd.merge(df1, df2, on='ID', how='outer')
# print(df_merge)
# df_join = df1.join(df2, how='inner')
# print(df_join)
data = {'Nume': ['Ana', 'Bogdan', 'Cristina', 'David', 'Elena', 'Florin'],
        'Varsta': [25, 30, 22, 35, 28, 40],
        'Salariu': [50000, 60000, 45000, 70000, 55000, 80000],
        'Departamente': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR']}
# df = pd.DataFrame(data)
# print(df)
# df.to_csv('date.csv', index=False)
# df.to_csv('date_cu_index.csv')

df = pd.read_csv('date.csv')
print(df)
