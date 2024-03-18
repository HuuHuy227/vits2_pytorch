pad = "_"
punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
# Vietnamese
vi_symbols = [
           'ɯəj', 
           'ɤ̆j', 
           'ʷiə', 
           'ɤ̆w', 
           'ɯəw', 
           'ʷet', 
           'iəw', 
           'uəj', 
           'ʷen', 
           'tʰw', 
           'ʷɤ̆', 
           'ʷiu', 
           'kwi', 
           'ŋ͡m', 
           'k͡p', 
           'cw', 
           'jw', 
           'uə', 
           'eə', 
           'bw', 
           'oj', 
           'ʷi', 
           'vw', 
           'ăw', 
           'ʈw', 
           'ʂw', 
           'aʊ', 
           'fw', 
           'ɛu', 
           'tʰ', 
           'tʃ', 
           'ɔɪ', 
           'xw', 
           'ʷɤ', 
           'ɤ̆', 
           'ŋw', 
           'ʊə', 
           'zi', 
           'ʷă', 
           'dw', 
           'eɪ', 
           'aɪ', 
           'ew', 
           'iə', 
           'ɣw', 
           'zw', 
           'ɯj', 
           'ʷɛ', 
           'ɯw', 
           'ɤj', 
           'ɔ:', 
           'əʊ', 
           'ʷa', 
           'mw', 
           'ɑ:', 
           'hw', 
           'ɔj', 
           'uj', 
           'lw', 
           'ɪə', 
           'ăj', 
           'u:', 
           'aw', 
           'ɛj', 
           'iw', 
           'aj', 
           'ɜ:', 
           'kw', 
           'nw', 
           't∫', 
           'ɲw', 
           'eo', 
           'sw', 
           'tw', 
           'ʐw', 
           'iɛ', 
           'ʷe', 
           'i:', 
           'ɯə', 
           'dʒ', 
           'ɲ', 
           'θ', 
           'ʌ', 
           'l', 
           'w', 
           'ɪ', 
           'ɯ', 
           'd', 
           '∫', 
           'p', 
           'ə', 
           'u', 
           'o', 
           'ɣ', 
           '!', 
           'ð', 
           'ʧ', 
           'ʒ', 
           'ʐ', 
           'z', 
           'v', 
           'g', 
           'ă', 
           'æ', 
           'ɤ', 
           'ʤ', 
           'i', 
           '.', 
           'ɒ', 
           'b', 
           'h', 
           'n', 
           'ʂ', 
           'ɔ', 
           'ɛ', 
           'k', 
           'm', 
           ' ', 
           'c', 
           'j', 
           'x', 
           'ʈ', 
           ',', 
           'ʊ', 
           's', 
           'ŋ', 
           'a', 
           'ʃ', 
           '?', 
           'r', 
           ':', 
           'η', 
           'f', 
           ';', 
           'e', 
           't', 
           "'"]

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]

num_tones = 10

# Export all symbols:
symbols = [pad] + sorted(set(vi_symbols + en_symbols + pu_symbols))

# Special symbol ids
SPACE_ID = symbols.index(" ")
