ECHO = @echo
OUTNAME = ai
CC := g++
CUCC := nvcc -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_61,code=sm_61
SRCDIR := src
OBJDIR := objs
BINDIR := workspace

CFLAGS := -std=c++11 -g -O3 -fopenmp -w
CUFLAGS := -std=c++11 -g -O3 -w
INC_OPENCV := /usr/include/opencv/ /usr/include/opencv2
INC_LOCAL := ./src ./src/builder ./src/common ./src/infer ./src/plugin ./src/plugin/plugins
INC_SYS := /usr/local/include
INC_CUDA := /usr/local/cuda/include /usr/local/TensorRT-6.0.1.5/include
INCS := $(INC_OPENCV) $(INC_LOCAL) $(INC_SYS) $(INC_CUDA)
INCS := $(foreach inc, $(INCS), -I$(inc))

LIB_CUDA := /usr/local/cuda/lib64 /usr/local/TensorRT-6.0.1.5/lib
LIB_SYS := /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
LIBS := $(LIB_SYS) $(LIB_CUDA)
LIBS := $(foreach lib, $(LIBS),-L$(lib))

RPATH := $(LIB_SYS) $(LIB_CUDA)
RPATH := $(foreach lib, $(RPATH),-Wl,-rpath $(lib))

LD_OPENCV := opencv_core opencv_highgui opencv_imgproc opencv_ml opencv_video opencv_photo opencv_flann opencv_stitching opencv_videostab
LD_NVINFER := nvinfer nvinfer_plugin nvparsers nvonnxparser
LD_CUDA := cuda curand cublas cudart cudnn
LD_SYS := dl pthread stdc++
LDS := $(LD_OPENCV) $(LD_NVINFER) $(LD_CUDA) $(LD_SYS)
LDS := $(foreach lib, $(LDS), -l$(lib))

SRCS := $(shell cd $(SRCDIR) && find -name "*.cpp")
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
OBJS := $(foreach item,$(OBJS),$(OBJDIR)/$(item))
CUS := $(shell cd $(SRCDIR) && find -name "*.cu")
CUOBJS := $(patsubst %.cu,%.o,$(CUS))
CUOBJS := $(foreach item,$(CUOBJS),$(OBJDIR)/$(item))
OBJS := $(subst /./,/,$(OBJS))
CUOBJS := $(subst /./,/,$(CUOBJS))

all: $(BINDIR)/$(OUTNAME)
	$(ECHO) Done, now you can run this program with \"make run\" command.

run:
	@cd $(BINDIR) && ./$(OUTNAME)

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	@$(CC) $(CFLAGS) $(LIBS) -o $@ $^ $(LDS) $(RPATH)

$(CUOBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CUCC) $(CUFLAGS) $(INCS) -c -o $@ $<

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@if [ ! -d $@ ]; then mkdir -p $(dir $@); fi
	$(ECHO) Compiling: $<
	@$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)/$(OUTNAME)