import torch
test_data = torch.load("Dataset/Test/test_clean.pt", map_location="cpu", weights_only=False)
print("type(test_data):", type(test_data))
print("len:", len(test_data) if hasattr(test_data, '__len__') else "n/a")
item = test_data[0]
print("type(item):", type(item))
print("item:", item)
if isinstance(item, tuple):
    print("tuple len:", len(item))
    for i, x in enumerate(item):
        print(f"  [{i}] type={type(x)}, val={x}")