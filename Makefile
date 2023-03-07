CC = g++
CU = nvcc
FLAGS = -std=c++11 -O3 -w
CU_FLAGS = #-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
##-Wno-deprecated-gpu-targets
INC = -I./
BOOST_INCLUDE = /usr/include
BOOST_LIB = /usr/lib
dir_guard = @mkdir -p $(@D)
PREFIX = /usr/local
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

build: $(DIR_BIN)/libredcufhe.so $(DIR_BIN)/multigpu_gates_example $(DIR_BIN)/multigpu_arithmetic_example ## Build lib and examples

clean: ## Remove executables and object files
	@rm -r $(DIR_BIN) $(DIR_OBJ) 2>/dev/null || true

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

$(DIR_BIN)/libredcufhe.so: $(CU_OBJ) $(DIR_OBJ)/redcufhe.o $(DIR_OBJ)/redcufhe_io.o
	$(dir_guard)
	$(CU) $(FLAGS) $(CU_FLAGS) -shared -o $@ $(CU_OBJ) $(DIR_OBJ)/redcufhe.o $(DIR_OBJ)/redcufhe_io.o

install: ## Install lib to usr/local by default 
	install -d $(PREFIX)/lib
	install -m 644 bin/libredcufhe.so $(PREFIX)/lib
	install -d $(PREFIX)/include/REDcuFHE/
	install -m 644 include/redcufhe.h $(PREFIX)/include/REDcuFHE/
	install -m 644 include/cufhe_core.h $(PREFIX)/include/REDcuFHE/
	install -m 644 include/redcufhe_bootstrap_gpu.cuh $(PREFIX)/include/REDcuFHE/
	install -m 644 include/redcufhe_gpu.cuh $(PREFIX)/include/REDcuFHE/
	install -d $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt_1024_device.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt_1024_twiddle.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt_conv_kind.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt_ffp.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt_shifting.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -m 644 include/ntt_gpu/ntt_single_thread.cuh $(PREFIX)/include/REDcuFHE/ntt_gpu/
	install -d $(PREFIX)/include/REDcuFHE/details/
	install -m 644 include/details/allocator.h $(PREFIX)/include/REDcuFHE/details/
	install -m 644 include/details/allocator_gpu.cuh $(PREFIX)/include/REDcuFHE/details/
	install -m 644 include/details/assert.h $(PREFIX)/include/REDcuFHE/details/
	install -m 644 include/details/error_gpu.cuh $(PREFIX)/include/REDcuFHE/details/
	install -m 644 include/details/math.h $(PREFIX)/include/REDcuFHE/details/
	install -m 644 include/details/utils_gpu.cuh $(PREFIX)/include/REDcuFHE/details/

uninstall: ## Completely uninstall library
	rm -rf $(PREFIX)/include/REDcuFHE
	rm $(PREFIX)/lib/libredcufhe.so

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

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
