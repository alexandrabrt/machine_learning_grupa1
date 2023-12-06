import tensorflow as tf

tensor_0d = tf.constant(42)
print(f"Tensor 0D (scalar): {tensor_0d.numpy()}")

tensor_1d = tf.constant([1, 2, 3, 4])
print(f"Tensor 1D (vector): {tensor_1d.numpy()}")

tensor_2d = tf.constant([[1, 2, 3], [4, 5, 6]])
print(f"Tensor 2d (Matrice): {tensor_2d.numpy()}")

tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7,8]]])
print(f"Tensor 3D: {tensor_3d.numpy()}")
