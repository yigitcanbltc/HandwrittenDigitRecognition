# Handwritten Digit Recognition Application

A real-time handwritten digit recognition application powered by artificial intelligence. The model is trained on the MNIST dataset using TensorFlow and Keras, implementing a Convolutional Neural Network (CNN) architecture.

## Features

- User-friendly graphical interface
- Real-time digit recognition
- High accuracy predictions
- Confidence score display
- Convenient keyboard shortcuts

## Requirements
numpy
tensorflow
pillow
tkinter

## Installation

1. Clone the repository:
git clone https://github.com/username/HandwrittenDigitRecognition.git

2. Install required packages:
```
pip install -r requirements.txt
```

## Usage

1. Launch the application:
```
python main.py
```

2. Draw a digit (0-9) in the drawing area
3. Click "Predict" or press Enter
4. View the prediction and confidence score
5. Use "Clear" button or press Ctrl+Z for a new prediction

## Keyboard Shortcuts

- `Enter`: Make prediction
- `Ctrl+Z`: Clear drawing area

## Project Structure

```
HandwrittenDigitRecognition/
│
├── main.py                    # Main application file
├── train_model.py            # Model training script
├── handwritten_digit_model.keras  # Trained model
│
└── Datasets/
    └── train.csv             # Training dataset
```

## About the Model

- Model architecture: Convolutional Neural Network (CNN)
- Training dataset: MNIST
- Accuracy: ~98%
- Input size: 28x28 pixels

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback, please open an [issue](https://github.com/username/HandwrittenDigitRecognition/issues).w