import scipy.io as sio

mat = sio.loadmat(
    r"D:\EEG\data_preprocessed_matlab\s01.mat"
)

print(mat.keys())
print(mat["data"].shape)
print(mat["labels"].shape)
