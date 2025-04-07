# Ensemble CNN Model for Image Classification

This project implements an ensemble of Convolutional Neural Networks (CNNs) using popular architectures like VGG16, ResNet50, and EfficientNetB0 for image classification tasks. The models are fine-tuned on a custom dataset and their predictions are combined to improve overall accuracy.

## Features

- Utilizes three powerful CNN architectures: VGG16, ResNet50, and EfficientNetB0
- Implements model freezing for efficient transfer learning
- Trains each model separately and combines their predictions
- Includes data visualization tools
- Saves the best performing models during training

## Model Architecture

The ensemble approach combines predictions from three base models:

1. VGG16
2. ResNet50
3. EfficientNetB0

Each model is pre-trained on ImageNet and fine-tuned on the custom dataset. Their predictions are averaged to produce the final output.

## Training Process

1. **Model Creation**: Base models are created with pre-trained weights and modified for the specific number of classes.
2. **Freezing Layers**: Base model layers are frozen to prevent overfitting.
3. **Training**: Each model is trained separately with early stopping and model checkpointing.
4. **Ensemble Prediction**: Predictions from all three models are averaged to produce the final result.

## Usage

### Data Preparation

Split your dataset into training, validation, and test sets:
```python
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
```

### Model Training

Train each model individually:
```python
model_resnet, history_resnet = train_model(model_resnet, 'resnet50')
model_efficientnet, history_efficientnet = train_model(model_efficientnet, 'efficientnetb0')
model_vgg, history_vgg = train_model(model_vgg, 'vgg16')
```

### Ensemble Prediction

Combine predictions from all models:
```python
y_pred_ensemble = (y_pred_vgg + y_pred_resnet + y_pred_efficientnet) / 3
y_pred_ensemble_classes = np.argmax(y_pred_ensemble, axis=1)
```

### Visualization

Visualize sample images with their predicted labels:
```python
plot_sample_images(X_test, y_pred_ensemble_classes)
```

## Requirements

- Python 3.10+
- TensorFlow/Keras
- NumPy
- Matplotlib

## Results

The ensemble model typically outperforms individual models by combining their strengths. During training, the best models are saved based on validation accuracy.

Example of model performance during training:
```
Epoch 1/30
115/115 [==============================] - 34s 278ms/step - accuracy: 0.2733 - loss: 1.3848 - val_accuracy: 0.2790 - val_loss: 1.3834 - learning_rate: 0.0010
...
Epoch 7: val_accuracy improved from 0.69694 to 0.71882, saving model to best_resnet50.keras
```

## Model Inspection

You can inspect model details using:
```python
inspect_model(model)
```

This will print information about the model architecture, input/output shapes, and layer details.

## Conclusion

This ensemble approach leverages the strengths of multiple CNN architectures to achieve better generalization and accuracy in image classification tasks. It demonstrates the power of combining different models to create a more robust solution.

## Future Improvements

- Experiment with different weighting schemes for ensemble predictions
- Add more base models to the ensemble
- Implement data augmentation techniques
- Fine-tune hyperparameters for each base model

Feel free to contribute or modify the code to suit your specific needs!
