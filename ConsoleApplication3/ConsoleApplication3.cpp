#include "NeuralNetwork.h"
#include <iomanip>  
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>

std::vector<std::vector<double>> trainingInputs;
std::vector<std::vector<double>> trainingTargets;
std::vector<std::vector<double>> testingInputs;
std::vector<std::vector<double>> testingTargets;

void loadIrisDataset(std::string filename, double splitRatio) {
    std::ifstream file(filename);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> input(4);
        std::vector<double> target(3, 0);
        std::string value;

        // Read the input values
        for (int i = 0; i < 4; i++) {
            if (!std::getline(ss, value, ',')) {
                std::cerr << "Error: Could not read input values from line " << line << std::endl;
                continue;
            }
            input[i] = std::stod(value);
        }

        // Read the class label
        if (!std::getline(ss, value, ',')) {
            std::cerr << "Error: Could not read class label from line " << line << std::endl;
            continue;
        }
        std::string classLabel = value;

        if (classLabel == "Iris-setosa") {
            target[0] = 1;
        }
        else if (classLabel == "Iris-versicolor") {
            target[1] = 1;
        }
        else if (classLabel == "Iris-virginica") {
            target[2] = 1;
        }
        else {
            std::cerr << "Error: Unknown class label " << classLabel << " in line " << line << std::endl;
            continue;
        }

        if ((double)rand() / RAND_MAX < splitRatio) {
            trainingInputs.push_back(input);
            trainingTargets.push_back(target);
        }
        else {
            testingInputs.push_back(input);
            testingTargets.push_back(target);
        }
    }

    // Check if the dataset was split correctly
    if (trainingInputs.empty() || testingInputs.empty()) {
        std::cerr << "Error: The dataset was not split correctly" << std::endl;
    }
}



void test(NeuralNetwork& nn) {
    int correctCount = 0;
    for (int i = 0; i < testingInputs.size(); i++) {
        std::vector<double> output = nn.feedforward(testingInputs[i]);
        int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        int actual = std::distance(testingTargets[i].begin(), std::max_element(testingTargets[i].begin(), testingTargets[i].end()));
        if (predicted == actual) {
            correctCount++;
        }
    }
    double accuracy = (double)correctCount / testingInputs.size() * 100;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
}

int main() {
    srand(time(0));

    // Load the Iris dataset and split it into a training set and a testing set
    loadIrisDataset("iris.data", 0.7);

    // Initialize a neural network with 4 inputs, 5 hidden neurons, and 3 outputs
    NeuralNetwork nn(4, 5, 3);

    // Train the neural network with the training set
    nn.train(trainingInputs, trainingTargets, 0.1, 10000);

    // Test the neural network with the testing set
    test(nn);

    return 0;
}