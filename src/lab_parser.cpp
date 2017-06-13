/**
 * @file
 * @brief Implementation of a parser for MNIST label files.
 * @author Arno Bastenhof
 */

#include "lab_parser.hpp"

#include <algorithm>
#include <cassert>

using std::begin;
using std::copy;
using std::end;
using std::min;
using std::runtime_error;

using arma::Col;
using arma::join_cols;

namespace mnist {

/**
 * @brief Constructor that opens a label file and parses its header.
 * @param[in] filename The full path of the label file.
 */
LabelParser::LabelParser(const char *filename)
  : MnistParser<Col<uint8_t>>{filename}
{
  // Read label file header into buffer
  if (!file_.read(reinterpret_cast<char *>(buffer_), kHeaderSizeLabelFile)) {
    throw runtime_error{"Could not read label file header."};
  }

  // Validate magic number
  if (ReadBigEndianInt32(buffer_) != kMagicNumberLabelFile) {
    throw runtime_error{"Unexpected magic number for label file."};
  }

  // Read number of labels
  num_items_ = ReadBigEndianInt32(buffer_ + 4);
}

/**
 * @brief Parses the label file and returns its contents as a vector.
 * @return A column vector with each entry containing a label for a
 *         corresponding image file.
 */
Col<uint8_t> LabelParser::Parse()
{
  if (IsDone()) {
    throw runtime_error{"Label file already parsed."};
  }
  Col<uint8_t> col(num_items_);
  auto it = begin(col);
  do {
    // number of bytes to read is the minimum of the buffer size and the number
    // of labels left to parse
    const int cnt = min((int)kBufferSize, num_items_);

    // read cnt bytes from the label file into an input buffer
    if (!file_.read(reinterpret_cast<char *>(buffer_), cnt)) {
      throw runtime_error{"Could not read labels."};
    }
    num_items_ -= cnt;

    // copy the buffer's contents to the output vector and continue
    copy(buffer_, buffer_ + cnt, it);
    it += cnt;
  } while (num_items_ != 0);
  assert(it == end(col));
  return col;
}

} // namespace mnist
