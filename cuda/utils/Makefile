###########################################################

## USER SPECIFIC DIRECTORIES ##

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc

NVCC_FLAGS=-arch sm_50

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = ../src

# Object file directory:
OBJ_DIR = ../bin

# Include header file diretory:
INC_DIR = ../include

##########################################################

## Make variables ##

# Target executable name:
EXE = DataGen

# Object files:
OBJS = $(OBJ_DIR)/DataGen.o $(OBJ_DIR)/DataInit.o $(OBJ_DIR)/sort.o $(OBJ_DIR)/save.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ -L/usr/local/lib -lcnpy -lz

# Compile main.cu file to object files:
$(OBJ_DIR)/DataGen.o : DataGen.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -I ./$(INC_DIR) -I ~/YHs_Sample/cnpy

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean objects in object directory.
clean:
	$(RM) ../bin/* *.o $(EXE)



