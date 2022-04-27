## Compilers
CC = g++
CU = nvcc
## Flags
FLAGS = -std=c++11 -O3 -w
CU_FLAGS = #-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
##-Wno-deprecated-gpu-targets
INC = -I./
## Boost header files and library
BOOST_INCLUDE = /usr/include
BOOST_LIB = /usr/lib
dir_guard = @mkdir -p $(@D)

##
DIR_BIN = bin
DIR_OBJ = build
DIR_SRC = lib
CU_SRC = $(wildcard $(DIR_SRC)/details/*.cu) \
				 $(wildcard $(DIR_SRC)/ntt_gpu/*.cu) \
				 $(wildcard $(DIR_SRC)/*.cu)
CC_SRC = $(wildcard $(DIR_SRC)/details/*.cc) \
				 $(wildcard $(DIR_SRC)/*.cc)
CU_OBJ = $(patsubst $(DIR_SRC)/%,$(DIR_OBJ)/%,$(CU_SRC:.cu=.o))
CU_DEP = $(CU_OBJ:.o=.d)
CC_OBJ = $(patsubst $(DIR_SRC)/%,$(DIR_OBJ)/%,$(CC_SRC:.cc=.o))
CC_DEP = $(CC_OBJ:.o=.d)

##
targets: $(DIR_BIN)/libredcufhe.so \
	$(DIR_BIN)/multigpu_gates_example \
	$(DIR_BIN)/multigpu_arithmetic_example

##
all: $(targets)
gpu: $(DIR_BIN)/libredcufhe.so $(DIR_BIN)/multigpu_gates_example $(DIR_BIN)/multigpu_arithmetic_example
clean:
	rm -r $(DIR_BIN) $(DIR_OBJ)

$(DIR_BIN)/multigpu_gates_example: $(DIR_OBJ)/examples/multigpu_gates_example.o
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) -o $@ $(DIR_OBJ)/examples/multigpu_gates_example.o -L$(DIR_BIN) -lredcufhe -Xcompiler -fopenmp

$(DIR_OBJ)/examples/multigpu_gates_example.o: examples/multigpu_gates_example.cu
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) $(INC) -M -o $(@:%.o=%.d) $< -Xcompiler -fopenmp
	$(CU) $(FLAGS) $(CU_FLAGS) $(INC) -c -o $@ $< -Xcompiler -fopenmp

$(DIR_BIN)/multigpu_arithmetic_example: $(DIR_OBJ)/examples/multigpu_arithmetic_example.o
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) -o $@ $(DIR_OBJ)/examples/multigpu_arithmetic_example.o -L$(DIR_BIN) -lredcufhe -Xcompiler -fopenmp

$(DIR_OBJ)/examples/multigpu_arithmetic_example.o: examples/multigpu_arithmetic_example.cu
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) $(INC) -M -o $(@:%.o=%.d) $< -Xcompiler -fopenmp
	$(CU) $(FLAGS) $(CU_FLAGS) $(INC) -c -o $@ $< -Xcompiler -fopenmp

$(DIR_BIN)/libredcufhe.so: $(CU_OBJ) $(DIR_OBJ)/cufhe.o $(DIR_OBJ)/cufhe_io.o
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) -shared -o $@ $(CU_OBJ) $(DIR_OBJ)/cufhe.o $(DIR_OBJ)/cufhe_io.o

$(CC_OBJ): $(CC_SRC)
	$(dir_guard)
	$(CC) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cc))
	$(CC) $(FLAGS) $(INC) -fPIC -c -o $@ \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cc))

$(CU_OBJ): $(CU_SRC)
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) $(INC) -M -o $(@:%.o=%.d) \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cu))
	$(CU) $(FLAGS) $(CU_FLAGS) $(INC) -c -o $@ \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cu)) -Xcompiler '-fPIC'