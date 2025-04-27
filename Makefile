HEADER = de.h
OUTPUT = solver

OBJS := $(patsubst %.cpp,%.o,$(shell find src -name '*.cpp'))
OPTION = -std=c++17 -O3

# Link the static library directly
LDFLAGS = ./lib/libpyclustering.a -lm

# Add -I to specify the header file directory
CFLAGS = -I./include -I./include/pyclustering

# Parametros da Linha de Comando
n ?= 1
FUNCAO  ?= 9 # Ras
MAXVAR  ?= 10    # Must be 10, 30, 50, 100
MAXFAVL ?= 10000 # Raphael 10000
MAXFPOP ?= 18    # Raphael    18
MAXFSLV ?= 24    # % Elite Enviada [main -> min=3]
DIVERSD ?= 1

SCRIPTF ?= 0     # No Print Results
TARGET  ?= -1    # Time to Target (Assume Min Fitness como Melhor)

$(OUTPUT): $(OBJS)
	mpicxx -o $(OUTPUT) $(OBJS) $(OPTION) $(LDFLAGS)

%.o: %.cpp
	mpicxx $(CFLAGS) -c $< -o $@

build:
	$(MAKE)

run: #Exec Silenciosa
	@mpirun -np $(n) ./$(OUTPUT) $(FUNCAO) $(MAXVAR) $(MAXFPOP) $(MAXFSLV) $(MAXFAVL) $(DIVERSD) $(TARGET) $(SCRIPTF)		

all:
	rm -rf src/*.o $(OUTPUT)
	$(MAKE)
	clear
	mpirun -np $(n) ./$(OUTPUT) $(FUNCAO) $(MAXVAR) $(MAXFPOP) $(MAXFSLV) $(MAXFAVL) $(DIVERSD) $(TARGET) $(SCRIPTF) 	



cls:
	rm -rf src/*.o $(OUTPUT)
