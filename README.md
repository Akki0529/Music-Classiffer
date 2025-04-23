Project Overview
This project aimed to accurately classify audio tracks into their respective genres using advanced machine learning techniques. Specifically, a Convolutional Neural Network (CNN) was employed to achieve robust and high-accuracy predictions across ten different music genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. The dataset comprised approximately 100 audio tracks per genre, each 30 seconds long, provided as .wav files.

Data Preprocessing
Audio data were initially raw waveforms, unsuitable directly for CNN input. To transform these signals into a more CNN-friendly format, Mel-Frequency Cepstral Coefficients (MFCCs) were extracted using librosa. MFCCs are advantageous for audio processing as they effectively represent the frequency spectrum of sound in a compact form, capturing human auditory perception characteristics.

A dedicated preprocessing function was designed:

Input: Audio data (.wav files)

Processing:

Sampling rate standardized at 22050 Hz

FFT (Fast Fourier Transform) with a window length (n_fft) of 2048

Hop length set to 512 samples

13 MFCC coefficients per audio segment extracted

Each track split into 10 segments (each 3 seconds long)

Output: MFCC features saved in JSON format for easy data handling and future loading.

Final Data Shape: (samples, 132, 13)

After feature extraction, the dataset was split into training (80%), validation (10%), and test (10%) sets, maintaining stratification to ensure balanced class distributions.

Initial Model and Errors Encountered
Initial CNN implementations produced accuracy around 79%. Several challenges were encountered:

Incorrect Data Reshaping: CNN models require four-dimensional inputs (samples, height, width, channels). Initially, data were incorrectly flattened or reshaped to three dimensions, leading to input dimensionality errors during convolution operations.

Overfitting Issues: Without sufficient regularization, early models quickly overfitted the training data, reducing test accuracy.

Model Optimization
To resolve these issues and enhance model performance toward approximately 90%, key improvements were implemented:

Correct Input Shaping: Data correctly reshaped as (samples, 132, 13, 1), resolving dimensionality errors and allowing convolutional layers to capture spatial relationships effectively.

Architecture Enhancement: The CNN was expanded to four convolutional blocks with increased filters (64, 128, and 256) and appropriate kernel sizes. This deeper architecture allowed the model to learn more complex features.

Regularization Techniques: Implemented Batch Normalization after each convolution, Dropout layers at strategic points, and Global Average Pooling layers to significantly reduce overfitting.

Activation and Optimization: Used ReLU activations to address vanishing gradients and ensure efficient training. The Adam optimizer with a reduced learning rate (1e-4) was selected for smooth convergence.

Training Callbacks: Introduced Early Stopping to halt training when validation performance plateaued, and ReduceLROnPlateau to dynamically lower the learning rate for enhanced convergence.

Baseline Comparison
To validate the effectiveness of CNN, traditional machine learning classifiers (Logistic Regression, Random Forest, and Support Vector Machines) were trained as baseline models using flattened MFCC features. These models typically achieved lower accuracy (~60–75%) compared to the optimized CNN (~85–90%), underscoring the CNN's superior capability in feature extraction and pattern recognition.

Results and Evaluation
The optimized CNN achieved significantly higher accuracy, closer to the targeted 90%, highlighting its robustness and effectiveness in classifying music genres from audio data. A confusion matrix was generated to visually inspect model performance, confirming that the CNN accurately distinguished between various genres.

Conclusion and Future Work
This project demonstrates CNN's effectiveness for audio classification tasks, particularly when paired with MFCC preprocessing and robust optimization techniques. Future improvements could include ensemble methods, incorporating more extensive datasets, and experimenting with hybrid CNN-RNN models to further boost performance.

