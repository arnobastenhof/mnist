#include <cassert>
#include <iostream>
#include <string>

#include "img_parser.hpp"
#include "lab_parser.hpp"

#include "neural.hpp"

using std::cin;
using std::cout;
using std::string;

using arma::Mat;

using mnist::ImageParser;
using mnist::LabelParser;
using mnist::NeuralNet;

// Constants regarding the MNIST data files
enum {
  kTrainingSetSz = 60000,
  kTestSetSz = 10000,
  kImageSz = 784
};

// MNIST image- and label files
static const string kTrainingSetImageFile = "train-images.idx3-ubyte";
static const string kTrainingSetLabelFile = "train-labels.idx1-ubyte";
static const string kTestSetImageFile     = "t10k-images.idx3-ubyte";
static const string kTestSetLabelFile     = "t10k-labels.idx1-ubyte";

// Read a value from standard input or use a default otherwise
template <typename T>
static auto ReadValue(const string prompt, T default_value)
{
  T val;
  cout << prompt;
  // Use default value if the user entered a newline
  if (cin.peek() == '\n') {
    val = default_value;
    cin.get();
  } else {
    // Resort to default value if the user entered invalid input
    if (!(cin >> val)) {
      cout << "Invalid input. Using " << default_value << ".\n";
      val = default_value;
      cin.clear();
    }
    // Skip other values entered until newline
    while (cin.get() != '\n')
      ;
  }
  return val;
}

int main(int argc, char *argv[])
{
  // Set path to MNIST data files
  string path;
  if (argc == 2) {
    path = argv[1];
  } else {
    cout << "Absolute path to MNIST data files: ";
    cin >> path;
    cin.get();
  }

  // Set learning rate
  const double rate = ReadValue("Learning rate (default 0.015): ", 0.015);

  // Set regularization parameter
  const double reg = ReadValue("Regularization param (default 0.095): ", 0.095);

  // Set no. of epochs
  const int epochs = ReadValue("No. of epochs (default 20): ", 20);

  // Training set
  Mat<uint8_t> training_set(kTrainingSetSz, kImageSz + 1);

  // Parse images from the training set
  ImageParser img_train_parser{(path + kTrainingSetImageFile).c_str()};
  training_set.head_cols(kImageSz) = img_train_parser.Parse();

  // Parse labels from the training set
  LabelParser lab_train_parser{(path + kTrainingSetLabelFile).c_str()};
  training_set.tail_cols(1) = lab_train_parser.Parse();

  // Test set
  Mat<uint8_t> test_set(kTestSetSz, kImageSz + 1);

  // Parse images from the test set
  ImageParser img_test_parser{(path + kTestSetImageFile).c_str()};
  test_set.head_cols(kImageSz) = img_test_parser.Parse();

  // Parse labels from the test set
  LabelParser lab_test_parser{(path + kTestSetLabelFile).c_str()};
  test_set.tail_cols(1) = lab_test_parser.Parse();

  // Create neural network
  NeuralNet nn{};

  // Learn weights from training set and output cost
  nn.LearnWeights(training_set, rate, reg, epochs);

  // Evaluate test set and output cost
  cout << nn.Evaluate(test_set) << '\n';

  return 0;
}
