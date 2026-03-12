include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++17
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ifdef PROFILE
PROF_FLAGS = -pg -g -fno-omit-frame-pointer
CXXFLAGS += $(PROF_FLAGS)
CXXFLAGS2 += $(PROF_FLAGS)
LDFLAGS += -pg
endif

ALL= ant_simu.exe 

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ gmon.out profile.txt

.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $^ -o $@	

ant_simu.exe : ant.o fractal_land.o renderer.o window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB) $(LDFLAGS)	

profile: clean all
	@rm -f gmon.out profile.txt
	@echo "Running profiling session..."
	@./ant_simu.exe
	@gprof ./ant_simu.exe --benchmark gmon.out > profile_vectorized.txt
	@gprof ./ant_simu.exe --benchmark--oop gmon.out >> profile_oop.txt
	@echo "gprof report generated: profile.txt"

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "    profile        : run gprof profiling and write profile.txt"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Add PROFILE=yes to compile with gprof instrumentation"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
