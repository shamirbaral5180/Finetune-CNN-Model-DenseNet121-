C:\Users\SRB\AppData\Local\Programs\Python\Python310\python.exe "C:\Users\SRB\Desktop\PytTorch And AI\train2.py"
Found 1354 files belonging to 9 classes.
Using 1219 files for training.
2025-09-06 14:24:40.098584: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Found 1354 files belonging to 9 classes.
Using 135 files for validation.
Classes: ['Anthracnose', 'Bacterial Blight', 'Citrus Canker', 'Curl Virus', 'Deficiency Leaf', 'Dry Leaf', 'Healthy Leaf', 'Sooty Mould', 'Spider Mites']
Epoch 1/5
2025-09-06 14:24:45.264677: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 134217728 exceeds 10% of free system memory.
2025-09-06 14:24:45.341120: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 138444800 exceeds 10% of free system memory.
2025-09-06 14:24:45.433378: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 67108864 exceeds 10% of free system memory.
2025-09-06 14:24:45.490895: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 67108864 exceeds 10% of free system memory.
2025-09-06 14:24:45.535591: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 67108864 exceeds 10% of free system memory.
39/39 [==============================] - ETA: 0s - loss: 2.3668 - accuracy: 0.2395
C:\Users\SRB\AppData\Local\Programs\Python\Python310\lib\site-packages\keras\src\engine\training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
Epoch 1: val_accuracy improved from -inf to 0.22963, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 49s 1s/step - loss: 2.3668 - accuracy: 0.2395 - val_loss: 2.0630 - val_accuracy: 0.2296 - lr: 4.0000e-04
Epoch 2/5
39/39 [==============================] - ETA: 0s - loss: 1.6395 - accuracy: 0.5029
Epoch 2: val_accuracy improved from 0.22963 to 0.42963, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 45s 1s/step - loss: 1.6395 - accuracy: 0.5029 - val_loss: 1.5796 - val_accuracy: 0.4296 - lr: 4.0000e-04
Epoch 3/5
39/39 [==============================] - ETA: 0s - loss: 1.2735 - accuracy: 0.6579
Epoch 3: val_accuracy improved from 0.42963 to 0.65185, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 47s 1s/step - loss: 1.2735 - accuracy: 0.6579 - val_loss: 1.2836 - val_accuracy: 0.6519 - lr: 4.0000e-04
Epoch 4/5
39/39 [==============================] - ETA: 0s - loss: 1.1389 - accuracy: 0.7235
Epoch 4: val_accuracy improved from 0.65185 to 0.79259, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 45s 1s/step - loss: 1.1389 - accuracy: 0.7235 - val_loss: 1.0752 - val_accuracy: 0.7926 - lr: 4.0000e-04
Epoch 5/5
39/39 [==============================] - ETA: 0s - loss: 1.0368 - accuracy: 0.7744
Epoch 5: val_accuracy improved from 0.79259 to 0.88148, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 45s 1s/step - loss: 1.0368 - accuracy: 0.7744 - val_loss: 0.9615 - val_accuracy: 0.8815 - lr: 4.0000e-04
Epoch 1/25
39/39 [==============================] - ETA: 0s - loss: 0.9817 - accuracy: 0.8113
Epoch 1: val_accuracy did not improve from 0.88148
39/39 [==============================] - 47s 1s/step - loss: 0.9817 - accuracy: 0.8113 - val_loss: 0.9194 - val_accuracy: 0.8815 - lr: 1.0000e-05
Epoch 2/25
39/39 [==============================] - ETA: 0s - loss: 0.9998 - accuracy: 0.7925
Epoch 2: val_accuracy improved from 0.88148 to 0.88889, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 45s 1s/step - loss: 0.9998 - accuracy: 0.7925 - val_loss: 0.8925 - val_accuracy: 0.8889 - lr: 1.0000e-05
Epoch 3/25
39/39 [==============================] - ETA: 0s - loss: 0.9903 - accuracy: 0.8105
Epoch 3: val_accuracy did not improve from 0.88889
39/39 [==============================] - 44s 1s/step - loss: 0.9903 - accuracy: 0.8105 - val_loss: 0.8711 - val_accuracy: 0.8889 - lr: 1.0000e-05
Epoch 4/25
39/39 [==============================] - ETA: 0s - loss: 1.0101 - accuracy: 0.7916
Epoch 4: val_accuracy did not improve from 0.88889
39/39 [==============================] - 44s 1s/step - loss: 1.0101 - accuracy: 0.7916 - val_loss: 0.8568 - val_accuracy: 0.8815 - lr: 1.0000e-05
Epoch 5/25
39/39 [==============================] - ETA: 0s - loss: 0.9811 - accuracy: 0.8154
Epoch 5: val_accuracy did not improve from 0.88889
39/39 [==============================] - 44s 1s/step - loss: 0.9811 - accuracy: 0.8154 - val_loss: 0.8453 - val_accuracy: 0.8889 - lr: 1.0000e-05
Epoch 6/25
39/39 [==============================] - ETA: 0s - loss: 0.9613 - accuracy: 0.8236
Epoch 6: val_accuracy did not improve from 0.88889
39/39 [==============================] - 44s 1s/step - loss: 0.9613 - accuracy: 0.8236 - val_loss: 0.8351 - val_accuracy: 0.8889 - lr: 1.0000e-05
Epoch 7/25
39/39 [==============================] - ETA: 0s - loss: 0.9915 - accuracy: 0.8039
Epoch 7: val_accuracy did not improve from 0.88889
39/39 [==============================] - 42s 1s/step - loss: 0.9915 - accuracy: 0.8039 - val_loss: 0.8257 - val_accuracy: 0.8889 - lr: 1.0000e-05
Epoch 8/25
39/39 [==============================] - ETA: 0s - loss: 0.9864 - accuracy: 0.8138
Epoch 8: val_accuracy improved from 0.88889 to 0.89630, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 43s 1s/step - loss: 0.9864 - accuracy: 0.8138 - val_loss: 0.8178 - val_accuracy: 0.8963 - lr: 1.0000e-05
Epoch 9/25
39/39 [==============================] - ETA: 0s - loss: 0.9532 - accuracy: 0.8269
Epoch 9: val_accuracy did not improve from 0.89630
39/39 [==============================] - 42s 1s/step - loss: 0.9532 - accuracy: 0.8269 - val_loss: 0.8117 - val_accuracy: 0.8963 - lr: 1.0000e-05
Epoch 10/25
39/39 [==============================] - ETA: 0s - loss: 0.9639 - accuracy: 0.8097
Epoch 10: val_accuracy improved from 0.89630 to 0.90370, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 42s 1s/step - loss: 0.9639 - accuracy: 0.8097 - val_loss: 0.8059 - val_accuracy: 0.9037 - lr: 1.0000e-05
Epoch 11/25
39/39 [==============================] - ETA: 0s - loss: 0.9786 - accuracy: 0.8187
Epoch 11: val_accuracy did not improve from 0.90370
39/39 [==============================] - 42s 1s/step - loss: 0.9786 - accuracy: 0.8187 - val_loss: 0.8001 - val_accuracy: 0.9037 - lr: 1.0000e-05
Epoch 12/25
39/39 [==============================] - ETA: 0s - loss: 0.9786 - accuracy: 0.8187
Epoch 12: val_accuracy did not improve from 0.90370
39/39 [==============================] - 43s 1s/step - loss: 0.9786 - accuracy: 0.8187 - val_loss: 0.7963 - val_accuracy: 0.9037 - lr: 1.0000e-05
Epoch 13/25
39/39 [==============================] - ETA: 0s - loss: 0.9596 - accuracy: 0.8146
Epoch 13: val_accuracy did not improve from 0.90370
39/39 [==============================] - 42s 1s/step - loss: 0.9596 - accuracy: 0.8146 - val_loss: 0.7928 - val_accuracy: 0.9037 - lr: 1.0000e-05
Epoch 14/25
39/39 [==============================] - ETA: 0s - loss: 0.9466 - accuracy: 0.8261
Epoch 14: val_accuracy did not improve from 0.90370
39/39 [==============================] - 42s 1s/step - loss: 0.9466 - accuracy: 0.8261 - val_loss: 0.7902 - val_accuracy: 0.9037 - lr: 1.0000e-05
Epoch 15/25
39/39 [==============================] - ETA: 0s - loss: 0.9563 - accuracy: 0.8294
Epoch 15: val_accuracy improved from 0.90370 to 0.91111, saving model to outputs_20250906_142440\model_best.h5
39/39 [==============================] - 43s 1s/step - loss: 0.9563 - accuracy: 0.8294 - val_loss: 0.7883 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 16/25
39/39 [==============================] - ETA: 0s - loss: 0.9774 - accuracy: 0.8171
Epoch 16: val_accuracy did not improve from 0.91111
39/39 [==============================] - 43s 1s/step - loss: 0.9774 - accuracy: 0.8171 - val_loss: 0.7878 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 17/25
39/39 [==============================] - ETA: 0s - loss: 0.9533 - accuracy: 0.8277
Epoch 17: val_accuracy did not improve from 0.91111
39/39 [==============================] - 41s 1s/step - loss: 0.9533 - accuracy: 0.8277 - val_loss: 0.7864 - val_accuracy: 0.9037 - lr: 1.0000e-05
Epoch 18/25
39/39 [==============================] - ETA: 0s - loss: 0.9506 - accuracy: 0.8187
Epoch 18: val_accuracy did not improve from 0.91111
39/39 [==============================] - 41s 1s/step - loss: 0.9506 - accuracy: 0.8187 - val_loss: 0.7833 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 19/25
39/39 [==============================] - ETA: 0s - loss: 0.9729 - accuracy: 0.8146
Epoch 19: val_accuracy did not improve from 0.91111
39/39 [==============================] - 41s 1s/step - loss: 0.9729 - accuracy: 0.8146 - val_loss: 0.7818 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 20/25
39/39 [==============================] - ETA: 0s - loss: 0.9747 - accuracy: 0.8187
Epoch 20: val_accuracy did not improve from 0.91111
39/39 [==============================] - 41s 1s/step - loss: 0.9747 - accuracy: 0.8187 - val_loss: 0.7811 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 21/25
39/39 [==============================] - ETA: 0s - loss: 0.9427 - accuracy: 0.8187
Epoch 21: val_accuracy did not improve from 0.91111
39/39 [==============================] - 42s 1s/step - loss: 0.9427 - accuracy: 0.8187 - val_loss: 0.7805 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 22/25
39/39 [==============================] - ETA: 0s - loss: 0.9672 - accuracy: 0.8220
Epoch 22: val_accuracy did not improve from 0.91111
39/39 [==============================] - 42s 1s/step - loss: 0.9672 - accuracy: 0.8220 - val_loss: 0.7804 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 23/25
39/39 [==============================] - ETA: 0s - loss: 0.9452 - accuracy: 0.8384
Epoch 23: val_accuracy did not improve from 0.91111
39/39 [==============================] - 42s 1s/step - loss: 0.9452 - accuracy: 0.8384 - val_loss: 0.7787 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 24/25
39/39 [==============================] - ETA: 0s - loss: 0.9544 - accuracy: 0.8269
Epoch 24: val_accuracy did not improve from 0.91111
39/39 [==============================] - 42s 1s/step - loss: 0.9544 - accuracy: 0.8269 - val_loss: 0.7772 - val_accuracy: 0.9111 - lr: 1.0000e-05
Epoch 25/25
39/39 [==============================] - ETA: 0s - loss: 0.9363 - accuracy: 0.8236
Epoch 25: val_accuracy did not improve from 0.91111
39/39 [==============================] - 42s 1s/step - loss: 0.9363 - accuracy: 0.8236 - val_loss: 0.7767 - val_accuracy: 0.9111 - lr: 1.0000e-05
plot_model failed (install graphviz & pydot to enable): You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model to work.

All artifacts saved in: outputs_20250906_142440
Best weights: outputs_20250906_142440\model_best.h5
Final model : outputs_20250906_142440\model_final.h5

Process finished with exit code 0