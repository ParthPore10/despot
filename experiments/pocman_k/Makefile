CXX ?= c++
CXXFLAGS ?= -O3 -Wall -Wextra -Wno-sign-compare
DESPOTHOME := ../..
POCMAN := $(DESPOTHOME)/examples/cpp_models/pocman/src
BIN := pocman_k

.DELETE_ON_ERROR:
.PHONY: all core clean

all: $(BIN)

core:
	$(MAKE) -C $(DESPOTHOME) core

$(BIN): main.cpp $(POCMAN)/pocman.cpp $(POCMAN)/pocman.h core
	$(CXX) $(CXXFLAGS) -std=c++11 -I$(DESPOTHOME)/include -I$(POCMAN) main.cpp $(POCMAN)/pocman.cpp $(DESPOTHOME)/build/*.o -o $@

clean:
	rm -f $(BIN)
