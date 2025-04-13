HEADER = de.h
TARGET = solver

OBJS := $(patsubst %.cpp,%.o,$(shell find src -name '*.cpp'))
OPTION = -std=c++14 -O3

# Link the static library directly
LDFLAGS = ./lib/libpyclustering.a -lm

# Add -I to specify the header file directory
CFLAGS = -I./include -I./include/pyclustering

# Parametros da Linha de Comando
n ?= 1
FUNCAO  ?= 9 # Ras
MAXVAR  ?= 10
MAXFAVL ?= 10000 # Raphael 10000
MAXFPOP ?= 18    # Raphael    18

$(TARGET): $(OBJS)
	mpicxx -o $(TARGET) $(OBJS) $(OPTION) $(LDFLAGS)

%.o: %.cpp
	mpicxx $(CFLAGS) -c $< -o $@

run:
	rm -rf src/*.o $(TARGET)
	$(MAKE)
	mpirun -np $(n) ./$(TARGET) $(FUNCAO) $(MAXVAR) $(MAXFPOP) $(MAXFAVL)



cls:
	rm -rf src/*.o $(TARGET)
