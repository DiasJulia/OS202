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

ALL= ant_simu.exe ant_simu_v2.exe ant_simu_v3.exe ant_simu_v4.exe
V4_NP ?= 2
V4_OMP_THREADS ?= 2

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ gmon.out profile.txt profile_original.txt profile_v2.txt profile_v3.txt profile_v4_t2.txt profile_v4_t4.txt profile_v4_t8.txt

.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $^ -o $@	

ant_simu.exe : ant.o fractal_land.o renderer.o window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB) $(LDFLAGS)	

ant_simu_v2.exe : ant.o fractal_land.o renderer.o window.o ant_simu_v2.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB) $(LDFLAGS)

ant_simu_v3.exe : ant.o fractal_land.o renderer.o window.o ant_simu_v3.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB) $(LDFLAGS)

ant_simu_v4.o : ant_simu_v4.cpp
	$(MPICXX) $(CXXFLAGS2) -c $< -o $@

ant_simu_v4.exe : ant.o fractal_land.o renderer.o window.o ant_simu_v4.o
	$(MPICXX) $(CXXFLAGS2) $^ -o $@ $(LIB) $(LDFLAGS)

run_v4: ant_simu_v4.exe
	OMP_NUM_THREADS=$(V4_OMP_THREADS) mpirun -np $(V4_NP) ./ant_simu_v4.exe

profile: clean
	$(MAKE) PROFILE=yes all
	@rm -f gmon.out profile_original.txt profile_v2.txt profile_v3.txt profile_v4_t2.txt profile_v4_t4.txt profile_v4_t8.txt profile.txt
	@echo "Profiling original mode..."
	@./ant_simu.exe
	@gprof ./ant_simu.exe gmon.out > profile_original.txt
	@echo "Profiling vectorized mode..."
	@rm -f gmon.out
	@./ant_simu_v2.exe
	@gprof ./ant_simu_v2.exe gmon.out > profile_v2.txt
	@echo "Profiling parallel mode..."
	@rm -f gmon.out
	@./ant_simu_v3.exe
	@gprof ./ant_simu_v3.exe gmon.out > profile_v3.txt
	@echo "Profiling first parallelization strategy (OMP_NUM_THREADS=2)..."
	@rm -f gmon.out
	@OMP_NUM_THREADS=2 mpirun -np $(V4_NP) ./ant_simu_v4.exe
	@gprof ./ant_simu_v4.exe gmon.out > profile_v4_t2.txt
	@echo "Profiling first parallelization strategy (OMP_NUM_THREADS=4)..."
	@rm -f gmon.out
	@OMP_NUM_THREADS=4 mpirun -np $(V4_NP) ./ant_simu_v4.exe
	@gprof ./ant_simu_v4.exe gmon.out > profile_v4_t4.txt
	@echo "Profiling first parallelization strategy (OMP_NUM_THREADS=8)..."
	@rm -f gmon.out
	@OMP_NUM_THREADS=8 mpirun -np $(V4_NP) ./ant_simu_v4.exe
	@gprof ./ant_simu_v4.exe gmon.out > profile_v4_t8.txt
	@cat profile_original.txt profile_v2.txt profile_v3.txt profile_v4_t2.txt profile_v4_t4.txt profile_v4_t8.txt > profile.txt

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "    profile        : run gprof profiling and write profile.txt"
	@echo "    run_v4         : execute ant_simu_v4.exe with MPI/OMP"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Add PROFILE=yes to compile with gprof instrumentation"
	@echo "Add V4_NP=<n> and V4_OMP_THREADS=<n> for run_v4"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
