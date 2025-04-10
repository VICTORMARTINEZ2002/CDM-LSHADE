HEADER = de.h
TARGET = solver

OBJS := $(patsubst %.cpp,%.o,$(shell find src -name '*.cpp'))
OPTION = -std=c++14 -O3

# Link the static library directly
LDFLAGS = ./lib/libpyclustering.a -lm

# Add -I to specify the header file directory
CFLAGS = -I./include -I./include/pyclustering

$(TARGET): $(OBJS)
	g++ -o $(TARGET) $(OBJS) $(OPTION) $(LDFLAGS)

%.o: %.cpp
	g++ $(CFLAGS) -c $< -o $@

cls:
	rm -rf src/*.o $(TARGET)
