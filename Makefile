ECHO = @echo
OUTNAME = trtrun
CC := g++
CUCC := nvcc
SRCDIR := src
OBJDIR := objs
LEAN := /datav/newbb/lean
#BINDIR := $(LEAN)/tensorRTIntegrate
BINDIR := workspace

CFLAGS := -std=c++11 -fPIC -m64 -g -O3 -fopenmp -w -DONNX_ML -DNDEBUG 
CUFLAGS := -std=c++11 -m64 -Xcompiler -fPIC -g -O3 -w -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_61,code=sm_61
INC_OPENCV := $(LEAN)/opencv4.2.0/include/opencv4
INC_LOCAL := ./src ./src/builder ./src/common ./src/infer ./src/plugin ./src/plugin/plugins
INC_SYS := /usr/local/protobuf/include
INC_CUDA := $(LEAN)/cuda10.2/include $(LEAN)/TensorRT-7.0.0.11/include $(LEAN)/cudnn7.6.5.32-cuda10.2
INCS := $(INC_SYS) $(INC_OPENCV) $(INC_LOCAL) $(INC_CUDA)
INCS := $(foreach inc, $(INCS), -I$(inc))

LIB_CUDA := $(LEAN)/cuda10.2/lib $(LEAN)/TensorRT-7.0.0.11/lib $(LEAN)/cudnn7.6.5.32-cuda10.2
LIB_SYS := /usr/local/protobuf/lib
LIB_OPENCV := $(LEAN)/opencv4.2.0/lib $(LEAN)/jasper
LIBS := $(LIB_SYS) $(LIB_CUDA) $(LIB_OPENCV)
LIBS := $(foreach lib, $(LIBS),-L$(lib))

RPATH := $(LIB_SYS) $(LIB_CUDA) $(LIB_OPENCV)
RPATH := $(foreach lib, $(RPATH),-Wl,-rpath=$(lib))

LD_OPENCV := opencv_core opencv_highgui opencv_imgproc opencv_video opencv_videoio opencv_imgcodecs
LD_NVINFER := nvinfer nvinfer_plugin nvparsers
LD_CUDA := cuda curand cublas cudart cudnn
LD_SYS := stdc++
LDS := $(LD_SYS) $(LD_OPENCV) $(LD_NVINFER) $(LD_CUDA)
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

run: all
	@cd $(BINDIR) && ./$(OUTNAME)

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	@g++  $(LIBS) -o $@ $^ $(LDS) $(RPATH) -pthread -lprotobuf 

#-pthread -lprotobuf -lonnx_proto
#@if [ ! -d $(BINDIR) ]; then mkdir -p $(BINDIR); fi
#@$(CC) $(CFLAGS) $(LIBS) -o $@ $^ -lprotobuf -lonnx_proto $(LDS) $(RPATH)
#@$(ECHO) copy hpp to $(BINDIR)
#@cp src/builder/*.hpp $(BINDIR)/
#@cp src/common/*.hpp $(BINDIR)/
#@cp src/infer/*.hpp $(BINDIR)/
#@cp src/plugin/*.hpp $(BINDIR)/

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
