## Notes on HIP for NVIDIA
As of 17/09/2024, it doesn't work right out of the box and requires some tinkerings to get it work.

If you want to compile a HIP program for NVIDIA GPUs, it can be quirky when using CMake. I've experienced a few problems before getting it to work. If you experience similar issues, look at the HIP relatedfiles inside `/usr/share/cmake-[version]/Modules/` to see what's going on.

Here's the steps that worked for me
1. I used Ubuntu 22.04.5 (on WSL) with CUDA 12.3
2. Install CMake 3.30.3 from the [Kitware repository](https://apt.kitware.com/)
3. Add the [ROCm repository](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/native-install/ubuntu.html), and **don't install rocm**.
4. `sudo apt install hip-runtime-nvidia hip-dev` 
5. Get [hipBLAS](https://github.com/ROCm/hipBLAS/releases/tag/rocm-6.2.0) and install it via running `./install.sh -i`. It will install hipblas inside `/opt/rocm/hipblas`
6. Set below values for `~/.bashrc` or your shell configuration file
```bash
export PATH=$PATH:/opt/rocm/bin
export PATH=$PATH:/usr/local/cuda/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

export HIP_PLATFORM=nvidia
export HIP_COMPILER=nvcc
export HIP_RUNTIME=cuda
```

### Errors realted to CMake / Environment Variables.
Below are the error messages I've faced, in case someone's experiencing same problems. They were mostly related to the old CMake version included in the default system packages or setting the environment variables. **I also experienced the issue of hipcc treating `--cuda-gpu-arch` like `--offload-arch` as mentioned on the [the issue report](https://github.com/ROCm/HIP/issues/3479#issuecomment-2305038649)**. I don't know which caused which as I didn't throughly tested each setting.

```
nvcc fatal   : Unknown option '-print-libgcc-file-name'
CMake Error at /opt/rocm-6.2.0/lib/cmake/hip-lang/hip-lang-config.cmake:122 (message):
  hip-lang Error:1 - clangrt builtins lib could not be found.
```
```
CMake Error at /usr/share/cmake-3.22/Modules/CMakeDetermineCompilerABI.cmake:49 (try_compile):
  Failed to configure test project build system.
Call Stack (most recent call first):
  /usr/share/cmake-3.22/Modules/CMakeTestHIPCompiler.cmake:29 (CMAKE_DETERMINE_COMPILER_ABI)
  CMakeLists.txt:2 (project)
```

```
CMake Error at /usr/share/cmake-3.30/Modules/CMakeTestHIPCompiler.cmake:73 (message):
  The HIP compiler

    "/usr/local/cuda/bin/nvcc"

  is not able to compile a simple test program.

  It fails with the following output:

    Change Dir: '/mnt/c/GitHub/SGEMM_HIP/build/CMakeFiles/CMakeScratch/TryCompile-LK3fRO'

    Run Build Command(s): /usr/bin/cmake -E env VERBOSE=1 /usr/bin/gmake -f Makefile cmTC_393f2/fast
    /usr/bin/gmake  -f CMakeFiles/cmTC_393f2.dir/build.make CMakeFiles/cmTC_393f2.dir/build
    gmake[1]: Entering directory '/mnt/c/GitHub/SGEMM_HIP/build/CMakeFiles/CMakeScratch/TryCompile-LK3fRO'
    Building HIP object CMakeFiles/cmTC_393f2.dir/testHIPCompiler.hip.o
    /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler  --options-file CMakeFiles/cmTC_393f2.dir/includes_HIP.rsp --offload-arch=75 -MD -MT CMakeFiles/cmTC_393f2.dir/testHIPCompiler.hip.o -MF CMakeFiles/cmTC_393f2.dir/testHIPCompiler.hip.o.d -o CMakeFiles/cmTC_393f2.dir/testHIPCompiler.hip.o -x cu -c /mnt/c/GitHub/SGEMM_HIP/build/CMakeFiles/CMakeScratch/TryCompile-LK3fRO/testHIPCompiler.hip
    gcc: error: unrecognized command-line option ‘--offload-arch=75’; did you mean ‘--offload-abi=’?
    gmake[1]: *** [CMakeFiles/cmTC_393f2.dir/build.make:80: CMakeFiles/cmTC_393f2.dir/testHIPCompiler.hip.o] Error 1
    gmake[1]: Leaving directory '/mnt/c/GitHub/SGEMM_HIP/build/CMakeFiles/CMakeScratch/TryCompile-LK3fRO'
    gmake: *** [Makefile:127: cmTC_393f2/fast] Error 2
  CMake will not be able to correctly generate this project.
```

### Errors related to hipBLAS Library
It also yielded some errors related to not being able to find hipblas when it is properly installed and registered to ldconfig. Or, when I manually linked the libhipblas.so file, it was compiled but also gave the error that the hipblas functions were undefined. The related errors was gone when I installed hipblas manually.

```
/usr/bin/ld: cannot find -lhipblas: No such file or directory
collect2: error: ld returned 1 exit status
failed to execute:/usr/local/cuda/bin/nvcc  -Wno-deprecated-gpu-targets -lcuda -lcudart -L/usr/local/cuda/lib64  -I /usr/local/cuda/include CMakeFiles/sgemm.dir/sgemm.cpp.o CMakeFiles/sgemm.dir/src/runner.cpp.o -o "sgemm" -lhipblas -lcudadevrt -lcudart_static -lrt
gmake[2]: *** [CMakeFiles/sgemm.dir/build.make:113: sgemm] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:85: CMakeFiles/sgemm.dir/all] Error 2
gmake: *** [Makefile:91: all] Error 2
```