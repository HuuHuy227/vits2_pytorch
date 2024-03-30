from text import _symbol_to_id, cleaned_text_to_sequence
# print(_symbol_to_id)
from text.vietnamese import g2p
from viphoneme import vi2IPA_split
from text.cleaners import clean_text
# print(vi2IPA_split("Sáng hôm sau, thi thể Sherri được tìm thấy trong chiếc minivan của gia đình ở một bãi đậu xe bên ngoài thành phố Pensacola, Florida", delimit="/"))
p, t = clean_text('Chung Mong-Gyu ')
print(p)
print(t)
print(len(p), len(t))
# print(cleaned_text_to_sequence(p, t))


# import pandas as pd
# df = pd.read_csv("D:\demo\dataset\metadata1.csv")
# print(df["Text"][0])