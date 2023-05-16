#include "NeuralNetwork.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

NeuralNetwork::NeuralNetwork(int inputCount, int hiddenCount, int outputCount) {
    this->inputCount = inputCount;
    this->hiddenCount = hiddenCount;
    this->outputCount = outputCount;

    srand(time(0));

    // Initialize weights and biases
    for (int i = 0; i < hiddenCount; i++) {
        std::vector<double> weight;
        for (int j = 0; j < inputCount; j++) {
            weight.push_back((double)rand() / RAND_MAX);
        }
        weightsInputHidden.push_back(weight);
        biasInputHidden.push_back((double)rand() / RAND_MAX);
    }

    for (int i = 0; i < outputCount; i++) {
        std::vector<double> weight;
        for (int j = 0; j < hiddenCount; j++) {
            weight.push_back((double)rand() / RAND_MAX);
        }
        weightsHiddenOutput.push_back(weight);
        biasHiddenOutput.push_back((double)rand() / RAND_MAX);
    }
}

std::vector<double> NeuralNetwork::feedforward(std::vector<double> input) {
    hiddenLayerOutput.clear();
    outputLayerOutput.clear();

    for (int i = 0; i < hiddenCount; i++) {
        double activation = biasInputHidden[i];
        for (int j = 0; j < inputCount; j++) {
            activation += weightsInputHidden[i][j] * input[j];
        }
        hiddenLayerOutput.push_back(sigmoid(activation));
    }

    for (int i = 0; i < outputCount; i++) {
        double activation = biasHiddenOutput[i];
        for (int j = 0; j < hiddenCount; j++) {
            activation += weightsHiddenOutput[i][j] * hiddenLayerOutput[j];
        }
        outputLayerOutput.push_back(sigmoid(activation));
    }

    return outputLayerOutput;
}

void NeuralNetwork::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets, double lr, int epochs) {
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < inputs.size(); i++) {
            std::vector<double> input = inputs[i];
            std::vector<double> target = targets[i];

            feedforward(input);

            // Calculate output layer errors
            std::vector<double> outputErrors;
            for (int i = 0; i < outputCount; i++) {
                double error = target[i] - outputLayerOutput[i];
                outputErrors.push_back(error);
            }

            // Calculate hidden layer errors
            std::vector<double> hiddenErrors(hiddenCount, 0.0);
            for (int i = 0; i < outputCount; i++) {
                for (int j = 0; j < hiddenCount; j++) {
                    hiddenErrors[j] += outputErrors[i] * weightsHiddenOutput[i][j];
                }
            }

            // Update weights and biases between hidden and output layers
            for (int i = 0; i < outputCount; i++) {
                for (int j = 0; j < hiddenCount; j++) {
                    double gradient = outputErrors[i] * sigmoidDerivative(outputLayerOutput[i]);
                    weightsHiddenOutput[i][j] += lr * gradient * hiddenLayerOutput[j];
                }
                biasHiddenOutput[i] += lr * outputErrors[i] * sigmoidDerivative(outputLayerOutput[i]);
            }

            // Update weights and biases between input and hidden layers
            for (int i = 0; i < hiddenCount; i++) {
                for (int j = 0; j < inputCount; j++) {
                    double gradient = hiddenErrors[i] * sigmoidDerivative(hiddenLayerOutput[i]);
                    weightsInputHidden[i][j] += lr * gradient * input[j];
                }
                biasInputHidden[i] += lr * hiddenErrors[i] * sigmoidDerivative(hiddenLayerOutput[i]);
            }
        }
    }
}

