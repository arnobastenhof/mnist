/**
 * @file
 * @brief Interface for MNIST label file parsers.
 * @author Arno Bastenhof
 */

#ifndef LAB_PARSER_HPP_
#define LAB_PARSER_HPP_

#include <armadillo>

#include "mnist_parser.hpp"

namespace mnist {

class LabelParser : public MnistParser<arma::Col<uint8_t>> {
public:
  explicit              LabelParser(const char *);
                        LabelParser(const LabelParser&) = delete;
                        LabelParser(LabelParser&&) = delete;
  LabelParser&          operator=(const LabelParser&) = delete;
  arma::Col<uint8_t>    Parse() override;
private:
  enum {
    kBufferSize = 256
  };
  uint8_t               buffer_[kBufferSize];
};

} // namespace mnist

#endif // LAB_PARSER_HPP_
