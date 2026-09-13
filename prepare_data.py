import pandas as pd
import re

def clean_html(text):
    return re.sub(r"<.*?>", "", str(text)).strip()

books = pd.read_csv("book1-100k.csv")
descs = pd.read_csv("desc1-100k.csv")

books["RatingDistTotal"] = books["RatingDistTotal"].str.replace("total:", "").astype(int)
books["RatingDist1"] = books["RatingDist1"].str.replace("1:", "").astype(int)
books["RatingDist2"] = books["RatingDist2"].str.replace("2:", "").astype(int)
books["RatingDist3"] = books["RatingDist3"].str.replace("3:", "").astype(int)
books["RatingDist4"] = books["RatingDist4"].str.replace("4:", "").astype(int)
books["RatingDist5"] = books["RatingDist5"].str.replace("5:", "").astype(int)

english_variants = ["en-US", "en-GB", "en-CA", "en"]
books["Language"] = (
    books["Language"].replace(english_variants, "eng").fillna("unknown")
)

descs["Description"] = descs["Description"].apply(clean_html)
books = books.merge(
    descs[["Id", "Description"]].drop_duplicates(subset=["Id"]),
    on="Id",
    how="left",
)

books = books.dropna(subset=["Description", "Name"])
books = books[books["Description"].str.len() > 100]
books = books.sort_values(by="RatingDistTotal", ascending=False)
books = books.drop_duplicates(subset=["Name", "Authors"], keep="first")
books = books.iloc[:25000]



books.to_csv("clean_books.csv", index=False)