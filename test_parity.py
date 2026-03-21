
def test_split(s, maxsplit):
    res = s.split(None, maxsplit)
    print(f"split('{s}', {maxsplit}) -> {res}")

def test_rsplit(s, maxsplit):
    res = s.rsplit(None, maxsplit)
    print(f"rsplit('{s}', {maxsplit}) -> {res}")

cases = [
    ("  a  b  ", 0),
    ("  a  b  ", 1),
    ("  a  b  ", 2),
    ("  a  b  c  ", 1),
    ("  a  b  c  ", 2),
    ("a b  ", 1),
    ("  a b", 1),
    ("   ", 0),
    ("   ", 1),
    ("", 0),
    ("a    b", 1),
    ("\u00A0a\u00A0\u00A0b\u00A0", 1),
]

for s, m in cases:
    test_split(s, m)
    test_rsplit(s, m)
