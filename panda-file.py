import pandas as pd

# netflix = pd.read_csv("~/Downloads/netflix_titles.csv")
netflix = pd.read_csv("./csv_files/netflix_titles.csv")
# d = netflix.iloc[0:20, [2,5]]
# print(d)

# d = netflix.iloc[0:20, [2, 5]]
# print(d)


# // for named index we have to use loc
d = netflix.loc[0:20, ["show_id", "title"]]
print(d)

