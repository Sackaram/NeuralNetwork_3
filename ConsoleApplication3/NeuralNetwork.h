#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int inputCount, int hiddenCount, int outputCount);
    std::vector<double> feedforward(std::vector<double> input);
    void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets, double lr, int epochs);

private:
    int inputCount;
    int hiddenCount;
    int outputCount;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<double> biasInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasHiddenOutput;

    std::vector<double> hiddenLayerOutput;
    std::vector<double> outputLayerOutput;
};


#endif // NEURALNETWORK_H
