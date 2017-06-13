/**
 * @file
 * @brief Implementation of a parser for MNIST image files.
 * @author Arno Bastenhof
 */

#include "img_parser.hpp"

#include <algorithm>

using std::min;
using std::runtime_error;

using arma::Mat;
using arma::Row;

namespace mnist {

/**
 * @brief Constructor that opens an image file and parses its header.
 * @param[in] filename The full path of the image file.
 */
ImageParser::ImageParser(const char * filename)
  : MnistParser<Mat<uint8_t>>{filename}
{
  // Read image file header into buffer
  if (!file_.read(reinterpret_cast<char *>(buffer_), kHeaderSizeImageFile)) {
    throw runtime_error{"Could not read image file header."};
  }

  // Validate magic number
  if (ReadBigEndianInt32(buffer_) != kMagicNumberImageFile) {
    throw runtime_error{"Unexpected magic number for image file."};
  }

  // Read numbers of images and of rows and columns per image
  num_items_ = ReadBigEndianInt32(buffer_ + 4);
  num_row_   = ReadBigEndianInt32(buffer_ + 8);
  num_col_   = ReadBigEndianInt32(buffer_ + 12);
}

/**
 * @brief Parses the image file and returns its contents as a matrix.
 *
 * The body of the input file contains grayscale images of 28 x 28 pixels each,
 * constituting the rows of the returned matrix after their reshaping
 * (row-wise) into arrays of 784 bytes.
 *
 * @return An n x 784 byte matrix, where n is the number of images contained in
 *         the input file.
 */
Mat<uint8_t> ImageParser::Parse()
{
  if (IsDone()) {
    throw runtime_error{"Image file already parsed."};
  }
  Mat<uint8_t> mat(num_items_, kImageSize);
  for (int i = 0; --num_items_ != 0; ++i) {
    // read a single image into an input buffer ...
    if (!file_.read(reinterpret_cast<char *>(buffer_), kImageSize)) {
      throw runtime_error{"Could not read image."};
    }
    // ... and store it as a row in the returned matrix
    mat.row(i) = Row<uint8_t>(buffer_, kImageSize);
  }
  return mat;
}

} // namespace mnist
