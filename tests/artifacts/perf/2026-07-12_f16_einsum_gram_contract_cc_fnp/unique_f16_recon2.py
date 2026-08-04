import numpy as np

def hexs(arr):
    return [hex(x) for x in np.asarray(arr).view(np.uint16)]

# nan selection: -nan (0xfe00) FIRST vs LAST; multiple payloads
for name, vals in (("neg-nan first", [np.uint16(0xfe00), np.uint16(0x7e00), np.uint16(0x7e01)]),
                   ("pos-nan first", [np.uint16(0x7e00), np.uint16(0xfe00)]),
                   ("payload order", [np.uint16(0x7e05), np.uint16(0x7e01)])):
    a = np.array(vals, dtype=np.uint16).view(np.float16)
    print(f"{name}: {hexs(np.unique(a))}")
# zeros deep check: first-occurrence rule at larger scale + mixed with values
a = np.array([1.0, -0.0, 2.0, 0.0, -1.0], dtype=np.float16)
print("mixed neg0-first:", hexs(np.unique(a)))
a = np.array([1.0, 0.0, 2.0, -0.0, -1.0], dtype=np.float16)
print("mixed pos0-first:", hexs(np.unique(a)))
# nan + zeros + full combo
a = np.array([0.0, np.nan, -0.0, -np.nan, 5.0], dtype=np.float16)
print("combo:", hexs(np.unique(a)))
print(np.__version__)
