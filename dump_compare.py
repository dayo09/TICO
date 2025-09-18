import pickle
import numpy as np

with open('0821_seedr.pkl', 'rb') as f:
    a = pickle.load(f)
with open('0909_seedr.pkl', 'rb') as f:
    b = pickle.load(f)
# with open('circle_result_260_0909.pkl', 'rb') as f:
#     c = pickle.load(f)

# for key in a:
#     if key in b:
#         diff = np.abs(a[key] - b[key])
#         print(f"{key}: max diff={diff.max()}, mean diff={diff.mean()}, shape={a[key].shape}")

import torch
try:
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
except Exception as e:
    print("-------------------")
    print(f"0821 and 0909")
    print(e)

# try:
#     torch.testing.assert_close(a, c, rtol=1e-3, atol=1e-3)
# except Exception as e:
#     print("-------------------")
#     print(f"circle_result_260_upgrade and circle_result_260_0909")
#     print(e)

# try:
#     torch.testing.assert_close(b, c, rtol=1e-3, atol=1e-3)
# except Exception as e:
#     print("-------------------")
#     print(f"circle_result_260 and circle_result_260_0909")
#     print(e)
