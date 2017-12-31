/**
 * This file can be compiled into a stand-alone application for transforming
 * the MNIST data into CSV format. It should in particular be run from the same
 * directory containing the original data files.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
  kImgHeaderSz  = 16,
  kLabHeaderSz  = 8,
  kImgMagicNo   = 0x803,
  kLabMagicNo   = 0x801,
  kImgSz        = 784
};

static void         Process(const char * const, const char * const,
                            const char * const);
static inline int   ReadBigEndianInt32(const uint8_t *);

static const char * const kImgTr = "train-images.idx3-ubyte";
static const char * const kLabTr = "train-labels.idx1-ubyte";
static const char * const kTrOut = "train.csv";
static const char * const kImgTe = "t10k-images.idx3-ubyte";
static const char * const kLabTe = "t10k-labels.idx1-ubyte";
static const char * const kTeOut = "test.csv";

int
main()
{
  Process(kImgTr, kLabTr, kTrOut);
  Process(kImgTe, kLabTe, kTeOut);

  return 0;
}

static void
Process(const char * const img, const char * const lab,
            const char * const csv)
{
  FILE *    f_img;
  FILE *    f_lab;
  FILE *    f_csv;
  uint8_t   b_img[kImgSz];
  uint8_t   b_lab[kLabHeaderSz];
  int       n_item;
  int       i_line;
  int       i_col;
  
  /* Open files. */
  f_img = fopen(img, "r");
  f_lab = fopen(lab, "r");
  f_csv = fopen(csv, "w");

  printf("Opened %s and %s for reading and %s for writing.\n", img, lab, csv);

  /* Write output file header. */
  fputs("id", f_csv);
  for (i_col = 0; i_col != kImgSz; ++i_col) {
    fprintf(f_csv, ",X%d", i_col);
  }
  fputs(",Y\n", f_csv);

  /* Read image file header. */
  if (fread(&b_img, 1, kImgHeaderSz, f_img) != kImgHeaderSz) {
    fprintf(stderr, "Could not read image file header.\n");
    goto exit;
  }

  /* Read label file header. */
  if (fread(&b_lab, 1, kLabHeaderSz, f_lab) != kLabHeaderSz) {
    fprintf(stderr, "Could not read label file header.\n");
    goto exit;
  }

  /* Validate image file magic number. */
  if (ReadBigEndianInt32(b_img) != kImgMagicNo) {
    fprintf(stderr, "Unexpected magic number of image file.\n");
    goto exit;
  }

  /* Validate label file magic number. */
  if (ReadBigEndianInt32(b_lab) != kLabMagicNo) {
    fprintf(stderr, "Unexpected magic number of label file.\n");
    goto exit;
  }

  /* Read number of images. */
  n_item = ReadBigEndianInt32(b_img + 4);
  if (n_item != ReadBigEndianInt32(b_lab + 4)) {
    fprintf(stderr, "Numbers of images and labels don't match.\n");
    goto exit;
  }
  printf("Processing %d items...\n", n_item);

  /* Read number of rows and columns per image. */
  ReadBigEndianInt32(b_img + 8);   /* no. of rows */
  ReadBigEndianInt32(b_img + 12);  /* no. of columns */

  /* Process file contents. */
  for (i_line = 0; n_item--; ++i_line) {
    /* Read a single image. */
    if (fread(&b_img, 1, kImgSz, f_img) != kImgSz) {
      fprintf(stderr, "Could not read image.");
      goto exit;
    }

    /* Read a single label. */
    if (fread(&b_lab, 1, 1, f_lab) != 1) {
      fprintf(stderr, "Could not read label.");
      goto exit;
    }

    /* Write a row to the output CSV. */
    fprintf(f_csv, "%d", i_line);
    for (i_col = 0; i_col != kImgSz; ++i_col) {
      fprintf(f_csv, ",%d", b_img[i_col]);
    }
    fprintf(f_csv, ",%d\n", b_lab[0]);
  }

exit:
  /* Close files. */
  fclose(f_img);
  fclose(f_lab);
  fclose(f_csv);
  printf("Closed %s, %s and %s.\n", img, lab, csv);
}

static inline int
ReadBigEndianInt32(const uint8_t *bytes)
{
  return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
