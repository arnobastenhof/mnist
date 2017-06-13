/**
 * @file
 * @brief Implementation of a three-layer feedforward neural network.
 * @author Arno Bastenhof
 */

#include "neural.hpp"

#include <cassert>
#include <cmath>

// Read/write access to the tail columns of a matrix (i.e., minus the first)
#define TAIL_COLS(m) ((m).tail_cols((m).n_cols - 1))

using std::begin;
using std::end;
using std::runtime_error;
using std::transform;

using arma::Col;
using arma::Mat;
using arma::accu;
using arma::conv_to;
using arma::join_vert;
using arma::mat;
using arma::rowvec;
using arma::shuffle;
using arma::vec;
using arma::vectorise;

static inline arma::mat Sigmoid(const arma::mat m)
{
  return 1.0 / (1.0 + arma::exp(-m));
}

static inline arma::mat SigmoidGrad(const arma::mat m)
{
  // m assumed to be of the form Sigmoid(n)
  return m % (1 - m);
}

// See footnote 2, p.7 of exercise set 4 (Ng's Machine Learning @ Coursera)
static inline double Eps(const double in_sz, const double out_sz)
{
  return sqrt(6) / sqrt(in_sz + out_sz);
}

namespace mnist {

/**
 * @brief Applies mini-batch gradient descent to learn the network parameters.
 * @param[in] data The input data, incl. labels (placed in the last column).
 * @param[in] rate The learning rate.
 * @param[in] reg The regularization parameter.
 * @param[in] epochs The number of full iterations to run over the input data.
 */
void NeuralNet::LearnWeights(const Mat<uint8_t>& data, const double rate,
    const double reg, int epochs)
{
  ValidateSize(data);

  // Randomly initialize weights
  InitWeights();

  while (epochs-- != 0) {
    // Shuffle training examples at the start of each epoch
    shuffle(data);

    // Separate the images from the labels
    const auto img = data.head_cols(kInputLayerSz);
    const auto lab = data.tail_cols(1);

    // Run forward- and backward propagation on each single training example
    // and update the weights accordingly
    for (unsigned int i = 0; i != img.n_rows; i += kBatchSz) {
      ForwardProp(img.rows(i, i + kBatchSz - 1));
      const auto grads = BackProp(lab.rows(i, i + kBatchSz - 1), reg);
      weights_ -= rate * grads;
    }
  }
}

/**
 * @brief Evaluates the learned network parameters on a test set.
 * @param[in] data The input data, incl. labels (placed in the last column).
 * @return The percentage of examples from the input data that were classified
 *         correctly.
 */
double NeuralNet::Evaluate(const Mat<uint8_t>& data) const
{
  ValidateSize(data);
  const auto img = data.head_cols(kInputLayerSz); // images
  const auto lab = data.tail_cols(1);             // labels
  double cnt = 0;
  
  for (unsigned int i = 0; i != data.n_rows; i += kBatchSz) {
    // Run forward propagation on a single image
    ForwardProp(img.rows(i, i + kBatchSz - 1));

    // Base predictions on the output nodes with the highest probabilities
    Col<uint8_t> predictions(kBatchSz);
    for (unsigned int j = 0; j != activ_l3_.n_rows; ++j) {
      predictions.row(j) = activ_l3_.row(j).index_max();
    }

    // Compare predictions to the actual labels
    predictions -= lab.rows(i, i + kBatchSz - 1);
    predictions.transform([](const uint8_t val){ return val == 0; });

    // Update count with the number of correctly classified images
    cnt += accu(predictions);
  }
  return (cnt / data.n_rows) * 100;
}

void NeuralNet::InitWeights()
{
  static const double eps_hidden = Eps(kInputLayerSz, kHiddenLayerSz);
  static const double eps_out = Eps(kHiddenLayerSz, kOutputLayerSz);

  weights_.randu();
  weights_.head_rows(kWeightsHeadSz) *= 2 * eps_hidden;
  weights_.head_rows(kWeightsHeadSz) -= eps_hidden;
  weights_.tail_rows(kWeightsTailSz) *= 2 * eps_out;
  weights_.tail_rows(kWeightsTailSz) -= eps_out;
}

void NeuralNet::ForwardProp(const Mat<uint8_t>& img) const
{
  assert(img.n_rows == kBatchSz && img.n_cols == kInputLayerSz);

  // reshape weights
  const mat weights_l12 = reshape(weights_.head_rows(kWeightsHeadSz),
    kHiddenLayerSz, kInputLayerSz + 1);
  const mat weights_l23 = reshape(weights_.tail_rows(kWeightsTailSz),
    kOutputLayerSz, kHiddenLayerSz + 1);

  // first activation layer (equals input)
  TAIL_COLS(activ_l1_) = conv_to<mat>::from(img);

  // second activation layer
  const mat prod_l2 = weights_l12 * activ_l1_.t();
  TAIL_COLS(activ_l2_) = Sigmoid(prod_l2.t());

  // third activation layer
  const mat prod_l3 = weights_l23 * activ_l2_.t();
  activ_l3_ = Sigmoid(prod_l3.t());
}

vec NeuralNet::BackProp(const Col<uint8_t>& lab, const double reg) const
{
  assert(lab.n_rows == kBatchSz);

  // reshape weights
  const mat weights_l12 = reshape(weights_.head_rows(kWeightsHeadSz),
    kHiddenLayerSz, kInputLayerSz + 1);
  const mat weights_l23 = reshape(weights_.tail_rows(kWeightsTailSz),
    kOutputLayerSz, kHiddenLayerSz + 1);

  // output layer
  mat yk(kOutputLayerSz, kBatchSz);
  for (int k = 0; k != kOutputLayerSz; ++k) {
    auto dest = yk.begin_row(k);
    transform(begin(lab), end(lab), dest, [k](uint8_t y){ return y==k; });
  }
  mat err_l3 = activ_l3_.t() - yk;
  err_l3 %= SigmoidGrad(activ_l3_.t());

  // hidden layer
  mat err_l2 = TAIL_COLS(weights_l23).t() * err_l3;
  err_l2 %= SigmoidGrad(TAIL_COLS(activ_l2_).t());

  // gradients
  mat grad_l23 = (err_l3 * activ_l2_) / kBatchSz;
  mat grad_l12 = (err_l2 * activ_l1_) / kBatchSz;

  // regularization
  TAIL_COLS(grad_l23) += (reg * TAIL_COLS(weights_l23)) / kBatchSz;
  TAIL_COLS(grad_l12) += (reg * TAIL_COLS(weights_l12)) / kBatchSz;

  // unroll gradients
  return join_vert(vectorise(grad_l12), vectorise(grad_l23));
}

} // namespace mnist
