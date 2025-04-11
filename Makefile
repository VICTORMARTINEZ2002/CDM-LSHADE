HEADER = de.h
TARGET = solver

OBJS := $(patsubst %.cpp,%.o,$(shell find src -name '*.cpp'))
OPTION = -std=c++14 -O3

# Link the static library directly
LDFLAGS = ./lib/libpyclustering.a -lm

# Add -I to specify the header file directory
CFLAGS = -I./include -I./include/pyclustering

$(TARGET): $(OBJS)
	mpicxx -o $(TARGET) $(OBJS) $(OPTION) $(LDFLAGS)

%.o: %.cpp
	mpicxx $(CFLAGS) -c $< -o $@

run:
	mpirun -np $(n) ./$(TARGET)

cls:
	rm -rf src/*.o $(TARGET)
