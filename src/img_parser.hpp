/**
 * @file
 * @brief Interface for MNIST image file parsers.
 * @author Arno Bastenhof
 */

#ifndef IMG_PARSER_H_
#define IMG_PARSER_H_

#include <armadillo>

#include "mnist_parser.hpp"

namespace mnist {

class ImageParser : public MnistParser<arma::Mat<uint8_t>> {
public:
  explicit              ImageParser(const char *);
                        ImageParser(const ImageParser&) = delete;
                        ImageParser(ImageParser&&) = delete;
  ImageParser&          operator=(const ImageParser&) = delete;
  arma::Mat<uint8_t>    Parse() override;
private:
  int                   num_row_;
  int                   num_col_;
  uint8_t               buffer_[kImageSize];
};

} // namespace mnist

#endif // IMG_PARSER_H_
