import pandas as pd
import numpy as np

s1 = pd.Series([10, 20 , 30])
# print("-----s1 ------")
# print(s1)

s2 = pd.Series([10, 20 , 30], index = ['a', 'b', 'c'])
# print("-----s2 ------")
# print(s2)

dic1 = { 'a': 50, 'b': 60, 'c': 70}
s3 = pd.Series(dic1)
# print("-----s3 ------")
# print(s3)

report = pd.DataFrame({ "Name": ["Same", "Alice", "Mark"], "marks": [78, 92, 49]  })
print(report)
print(report.shape)
