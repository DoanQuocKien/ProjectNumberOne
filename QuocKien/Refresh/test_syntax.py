def assemble_dataset(candidates, target_tx, tables, flags, is_inference=False):
    pass

try:
    assemble_dataset(1, 2, 3, is_inference=False, flags=4)
    print("Valid!")
except Exception as e:
    print(repr(e))
