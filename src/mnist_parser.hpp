/**
 * @file
 * @brief Interface for MNIST data file parsers.
 * @author Arno Bastenhof
 */

#ifndef MNIST_PARSER_HPP_
#define MNIST_PARSER_HPP_

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mnist {

template <typename T>
class MnistParser {
public:
                 MnistParser(const MnistParser&) = delete;
                 MnistParser(MnistParser&&) = delete;
  virtual        ~MnistParser() = default;
  MnistParser&   operator=(const MnistParser&) = delete;
  bool           IsDone() const;
  virtual T      Parse() = 0;      /**< @brief Parses the input file. */
protected:
  /** @brief Constants used in the definitions of the MNIST file formats.*/
  enum {
    kMagicNumberLabelFile = 0x801, /**< @brief Label file magic number. */
    kMagicNumberImageFile = 0x803, /**< @brief Image file magic number. */
    kHeaderSizeLabelFile = 8,      /**< @brief Label file header size. */
    kHeaderSizeImageFile = 16,     /**< @brief Image file header size. */
    kImageSize = 784               /**< @brief Image size (24 x 24 pixels) */
  };
  explicit       MnistParser(const char *);
  static int     ReadBigEndianInt32(const uint8_t *);
  std::ifstream  file_;            /**< @brief The input file stream. */
  int            num_items_;       /**< @brief No. of labels or images. */
};

/**
 * @brief Constructor that opens a given file for reading.
 * @param[in] filename The full path of the input file.
 */
template <typename T>
MnistParser<T>::MnistParser(const char * filename)
  : file_{filename, std::ios::in | std::ios::binary}, num_items_(0)
{
  if (!file_) {
    throw std::runtime_error{std::string("File not found: ") + filename};
  }
}

/**
 * @brief Returns whether the input file has been parsed.
 */
template <typename T>
inline bool MnistParser<T>::IsDone() const
{
  return num_items_ == 0;
}

/**
 * @brief Convenience method for reading integers from the input file.
 *
 * Reads and returns a 32-bit integer from its representation by a byte array
 * in Big Endian order.
 */
template <typename T>
inline int MnistParser<T>::ReadBigEndianInt32(const uint8_t * bytes)
{
  return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

} // namespace mnist

#endif // MNIST_PARSER_HPP_
