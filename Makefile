# reset suffix list
.SUFFIXES:
.SUFFIXES: .cpp .hpp .o

# use spaces instead of tabs
.RECIPEPREFIX != ps

# paths
PATHS = src/
PATHB = build/
PATHO = build/obj/
PATHH = build/html/

# keep object files
.PRECIOUS: $(PATHO)%.o

# commands and flags
CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -Werror -g3
ALL_CFLAGS = -O3 -std=c++14 -I$(PATHS) $(CFLAGS)

# file lists

BUILD_PATHS = $(PATHB) $(PATHO) $(PATHH)

OBJECTS = $(PATHO)main.o $(PATHO)img_parser.o $(PATHO)lab_parser.o \
          $(PATHO)neural.o

.PHONY: all html clean

all : $(PATHB)main html

html : $(PATHH)
  doxygen Doxyfile

clean:
  -rm -rf $(BUILD_PATHS)

# build directories

$(PATHB):
  -mkdir $(PATHB)

$(PATHO): $(PATHB)
  -mkdir $(PATHO)

$(PATHH): $(PATHB)
  -mkdir $(PATHH)

# object files

-include $(OBJECTS:.o=.d)

$(PATHO)%.o : $(PATHS)%.cpp $(PATHO)
  $(CC) $(ALL_CFLAGS) -c $< -o $@
  $(CC) $(ALL_CFLAGS) $< -MM -MF $(basename $@).d

# executable

$(PATHB)main: $(OBJECTS)
  $(CC) -o $@ $^ -larmadillo
