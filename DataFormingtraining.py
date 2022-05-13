import pandas as pd
import xlsxwriter
import numpy as np
#from openpyxl import Workbook
#wb = Workbook()
import xlwt
from xlwt import Workbook
workbook = xlsxwriter.Workbook('train.xlsx')
worksheet = workbook.add_worksheet()
# Workbook is created
#wb = Workbook()
#ws = wb.active
# add_sheet is used to create sheet.
#sheet1 = wb.add_sheet('Sheet 1')


df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\RawTrainDataPart2.xlsx","Sheet1")
print(len(df))
#ws[1][0]=42

for i in range(0,len(df)+1):
     print(df.iloc[i,1])
     x=int(df.iloc[i,0])
     y=int(df.iloc[i,1])

     worksheet.write(x-1, y-1, 1)






df1=pd.read_excel("train.xlsx","Sheet1")
print(df1.shape)
df2 = df1.replace(np.nan, 0, regex=True)
print(df2.shape)


with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP

     df2.to_excel(writer, sheet_name='Sheet_name_1')


"""print(df1.shape)
for i in range (0,1060):
     for j in range (0,3558):
          if(df1.iloc[i,j]!="1"):
               worksheet.write(i, j, 0)"""

#workbook.close()