import numpy as np

# load one processed frame
train_seq = np.load('dataset/processed/TOLONG/TOLONG_001.npy')
print("Train Frame 0 shoulder center:")
print((train_seq[0, 11] + train_seq[0, 12]) / 2)
print("Hand tip (idx 0 left, 0 right):")
print("Left:", train_seq[0, 33])  # left wrist
print("Right:", train_seq[0, 54])
