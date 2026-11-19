# PyTorch nâng cao: quá trình build (setup.py)

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *pytorch编译流程解析* trong *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/2-deep_learning_basic/pytorch_code/pytorch编译流程解析.md), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)), phân tích tệp `setup.py` của PyTorch khoảng bản 2.7 (đầu 2025). Mã Python giữ nguyên, chú thích dịch sang tiếng Việt; sơ đồ mermaid của bản gốc (site không vẽ được mermaid) chuyển thành danh sách các bước. `setup.py` của PyTorch thay đổi theo từng phiên bản, nên chi tiết có thể khác với mã nguồn hiện tại; cấu trúc tổng thể thì vẫn vậy. Biên dịch PyTorch mất cả giờ nên trang này không chạy build; khối code cuối chỉ đọc các thông tin build của bản đã cài.

Cách gọi build và cài đặt từ mã nguồn ở mục 1.3 bài [Tổng quan kiến trúc](tools/pt_arch_vi).

## 1 Luồng xử lý của hàm main trong setup.py

Trong `setup.py`, việc chính là gọi `setuptools.setup()`, hàm cốt lõi của setuptools, **dùng để khai báo siêu dữ liệu, phụ thuộc và quy trình build của gói Python**. `pytorch/setup.py` gọi `setup()` trong hàm `main()`.

Tham số của `setup` chia thành các nhóm:

1. **Tham số cơ bản**: siêu dữ liệu của gói: tên `name`, phiên bản `version`, mô tả ngắn `description`, tác giả `author`, trang dự án `url`, các gói con `packages`, phụ thuộc `install_requires`, mô tả dài `long_description`...
2. **Mô-đun mở rộng** `ext_modules`: biên dịch mã `C++/CUDA`.
3. **Quy trình build tuỳ biến** `cmdclass`: thay các lệnh mặc định.

Có chú thích:

```python
setup(
    # 1. Thông tin cơ bản
    name=package_name,                  # tên gói trên PyPI và khi cài, ví dụ "torch"
    version=version,                    # số phiên bản, thường theo semantic versioning (1.0.0)
    description=(                       # mô tả ngắn, hiện ở tiêu đề trang PyPI
        "Tensors and Dynamic neural networks in Python with strong GPU acceleration"
    ),
    long_description=long_description,  # mô tả chi tiết, thường đọc từ README.md
    long_description_content_type="text/markdown",

    # 2. Biên dịch và build
    ext_modules=extensions,   # danh sách mô-đun mở rộng C/C++ cần biên dịch
    cmdclass=cmdclass,        # ánh xạ lệnh build/cài tuỳ biến, thay build_ext mặc định...

    # 3. Gói Python và điểm vào
    packages=packages,        # các gói Python cần đóng gói, thường lấy bằng find_packages()
    entry_points=entry_points,# lệnh dòng lệnh đăng ký khi cài (console_scripts)

    # 4. Phụ thuộc bắt buộc và tuỳ chọn
    install_requires=install_requires,  # gói bắt buộc phải cài kèm
    extras_require=extras_require,      # phụ thuộc tuỳ chọn: pip install package[extra]

    # 5. Tệp, tài nguyên kèm theo
    package_data=package_data,          # tệp không phải .py cần đóng gói (header, .so/.dll, .cmake...)

    # 6. Trang chủ và siêu dữ liệu khác
    url="https://pytorch.org/",
    download_url="https://github.com/pytorch/pytorch/tags",
    author="PyTorch Team",
    author_email="packages@pytorch.org",

    # 7. Phiên bản Python và phân loại
    python_requires=f">={python_min_version_str}",   # phiên bản Python tối thiểu
    classifiers=[                       # bộ phân loại trên PyPI: trạng thái, đối tượng, giấy phép...
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        # ... (lược bớt)
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}"     # thêm "Python :: 3.X" cho từng phiên bản hỗ trợ
       for i in range(python_min_version[1], version_range_max)],

    # 8. Giấy phép và từ khoá
    license="BSD-3-Clause",
    keywords="pytorch, machine learning",
)
```

Các bước của `main()` (bản gốc vẽ bằng sơ đồ mermaid):

1. **Kiểm tra xung đột chế độ build**: `BUILD_PYTHON_ONLY` và `BUILD_LIBTORCH_WHL` chỉ được bật một.
2. **Khai báo danh sách phụ thuộc Python cơ bản** (`install_requires`).
3. **Tối ưu script liên kết cho ARM + CUDA** (nếu bật `USE_PRIORITIZED_TEXT_FOR_LD`).
4. **Phân tích tham số dòng lệnh** (tham số thứ hai thường là `build`, `install`, `develop`...) và kiểm tra hợp lệ.
5. **Tiền xử lý**: `mirror_files_into_torchgen()` chép các tệp cần thiết vào `torchgen`; `build_deps()` chạy CMake để build toàn bộ phần C++ (bước tốn thời gian nhất).
6. **Cấu hình mô-đun mở rộng C/C++**: gọi `configure_extension_build()` (mục 2), lấy danh sách extension và phụ thuộc thêm.
7. **Đọc `README.md`** làm `long_description`.
8. **Dựng danh sách `package_data`**, tuỳ các tuỳ chọn build mà chọn thư viện động và header nào đóng gói:
   - nếu không build wheel libtorch: đóng gói cả `libtorch_python.*`;
   - nếu không phải build chỉ-Python: đóng gói các `lib/*.so`, `.dll`, `.lib` khác;
   - nếu bật `USE_TENSORPIPE`: thêm header của tensorpipe;
   - nếu bật `USE_KINETO`: thêm header của kineto (profiler);
   - nếu không build wheel libtorch: đóng gói cả `torchgen`; ngược lại bỏ các extension.
9. **Gọi `setuptools.setup()`** để đóng gói/cài gói Python.
10. **In cảnh báo, gợi ý build** nếu cần.

> **Biên tập:** Bản gốc viết `package_data` "có hai trường hợp" rồi liệt kê năm; đã ghi lại thành danh sách điều kiện ở bước 8.

Hàm `main()` (tác giả rút gọn) có chú thích:

```python
def main():
    # 1. Kiểm tra xung đột chế độ build
    # chỉ build phần Python (BUILD_PYTHON_ONLY) và build wheel libtorch (BUILD_LIBTORCH_WHL) loại trừ nhau
    if BUILD_LIBTORCH_WHL and BUILD_PYTHON_ONLY:
        raise RuntimeError(
            "Conflict: 'BUILD_LIBTORCH_WHL' and 'BUILD_PYTHON_ONLY' can't both be 1. Set one to 0 and rerun."
        )

    # 2. Phụ thuộc Python cơ bản khi cài PyTorch
    install_requires = [
        "filelock",
        "typing-extensions>=4.10.0",
        'setuptools ; python_version >= "3.12"',
        "sympy>=1.13.3",
        "networkx",
        "jinja2",
        "fsspec",
    ]
    # chỉ build phần Python thì cần gói libtorch tương ứng
    if BUILD_PYTHON_ONLY:
        install_requires.append(f"{LIBTORCH_PKG_NAME}=={get_torch_version()}")

    # 3. Tối ưu script liên kết cho ARM + CUDA
    use_prioritized_text = str(os.getenv("USE_PRIORITIZED_TEXT_FOR_LD", ""))
    # trên Linux + AArch64 mà chưa đặt biến này thì nhắc người dùng
    if use_prioritized_text == "" and platform.system() == "Linux" and platform.processor() == "aarch64":
        print_box("""
            WARNING: we strongly recommend enabling linker script optimization for ARM + CUDA.
            To do so please export USE_PRIORITIZED_TEXT_FOR_LD=1
            """)
    # nếu bật: sinh script liên kết và thêm vào cờ biên dịch, liên kết
    if use_prioritized_text == "1" or use_prioritized_text == "True":
        gen_linker_script(filein="cmake/prioritized_text.txt", fout="cmake/linker_script.ld")
        linker_script_path = os.path.abspath("cmake/linker_script.ld")
        os.environ["LDFLAGS"] = os.getenv("LDFLAGS", "") + f" -T{linker_script_path}"
        os.environ["CFLAGS"] = os.getenv("CFLAGS", "") + " -ffunction-sections -fdata-sections"
        os.environ["CXXFLAGS"] = os.getenv("CXXFLAGS", "") + " -ffunction-sections -fdata-sections"

    # 4. Phân tích tham số dòng lệnh (build, install, sdist, bdist_wheel...)
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.distutils.errors.DistutilsArgError as e:
        print(e)
        sys.exit(1)

    # 5. Tiền xử lý: chép tệp cần thiết vào torchgen, build các phụ thuộc (CMake)
    mirror_files_into_torchgen()
    if RUN_BUILD_DEPS:
        build_deps()

    # 6. Cấu hình extension C/C++, lấy danh sách extension và phụ thuộc thêm
    extensions, cmdclass, packages, entry_points, extra_install_requires = configure_extension_build()
    install_requires += extra_install_requires

    # phụ thuộc tuỳ chọn, ví dụ pip install torch[optree]
    extras_require = {
        "optree": ["optree>=0.13.0"],
        "opt-einsum": ["opt-einsum>=3.3"],
    }

    # (đoạn đọc README.md làm long_description được lược bỏ)

    # 7. Dựng danh sách tệp đóng gói vào gói Python (package_data)
    version_range_max = max(sys.version_info[1], 13) + 1   # để sinh bộ phân loại phiên bản Python
    # các tệp đưa vào gói 'torch' (thư viện, header, script cmake...)
    torch_package_data = [
        "py.typed", "bin/*", "test/*", "*.pyi", "**/*.pyi",
        "lib/*.pdb", "lib/**/*.pdb", "lib/*shm*", "lib/torch_shm_manager",
        "lib/*.h", "lib/**/*.h",
        "include/*.h", "include/**/*.h", "include/*.hpp", "include/**/*.hpp",
        "include/*.cuh", "include/**/*.cuh",
        "_inductor/codegen/*.h", "_inductor/codegen/aoti_runtime/*.cpp", "_inductor/script.ld",
        "_export/serde/*.yaml", "_export/serde/*.thrift",
        "share/cmake/ATen/*.cmake", "share/cmake/Caffe2/*.cmake",
        # ... (thêm các thư mục cmake của Caffe2, Gloo, Tensorpipe, Torch)
        "utils/benchmark/utils/*.cpp",
        "utils/model_dump/skeleton.html", "utils/model_dump/code.js", "utils/model_dump/*.mjs",
    ]
    # ... (lược bớt)

    # không build wheel libtorch thì đóng gói cả torchgen; ngược lại không cần extension Python
    if not BUILD_LIBTORCH_WHL:          # BUILD_LIBTORCH_WHL mặc định là 0
        package_data["torchgen"] = torchgen_package_data
    else:
        extensions = []

    # 8. Gọi setuptools.setup() để đóng gói/cài
    setup(
        name=package_name,
        version=version,
        description="Tensors and Dynamic neural networks in Python with strong GPU acceleration",
        ext_modules=extensions,       # extension C/C++
        cmdclass=cmdclass,            # lệnh build tuỳ biến
        packages=packages,            # các gói Python
        entry_points=entry_points,    # lệnh dòng lệnh
        install_requires=install_requires,
        extras_require=extras_require,
        package_data=package_data,
        url="https://pytorch.org/",
        download_url="https://github.com/pytorch/pytorch/tags",
        author="PyTorch Team",
        author_email="packages@pytorch.org",
        python_requires=f">={python_min_version_str}",
        license="BSD-3-Clause",
        keywords="pytorch, machine learning",
    )

if __name__ == "__main__":
    main()
```

## 2 Luồng xử lý của configure_extension_build

`configure_extension_build` cấu hình và sinh các tham số then chốt để build và cài các mô-đun mở rộng C/C++ của PyTorch:

- `ext_modules`: danh sách mô-đun mở rộng C/C++ để setuptools biên dịch khi cài;
- `cmdclass`: dict các lệnh tuỳ biến, để làm thêm việc khi cài (biên dịch tài nguyên khác, chạy script...);
- `packages`: danh sách gói Python đưa vào gói cài đặt;
- `entry_points`: dict khai báo lệnh dòng lệnh hay điểm vào plugin;
- `extra_install_requires`: **danh sách** phụ thuộc thêm (lấy từ biến môi trường `PYTORCH_EXTRA_INSTALL_REQUIREMENTS`).

> **Biên tập:** Bản gốc ghi `extra_install_requires` là dict dùng cho "phụ thuộc tuỳ chọn"; trong mã, nó là **list** các phụ thuộc bắt buộc thêm (được cộng vào `install_requires`), còn phụ thuộc tuỳ chọn là `extras_require`.

Các bước của `configure_extension_build()`:

```text
configure_extension_build()
├─ đọc biến cache của CMake (cmake_cache_vars)
├─ cấu hình tuỳ chọn biên dịch
│  ├─ lấy cấu hình từ CMakeCache.txt
│  ├─ đặt cờ biên dịch / liên kết theo hệ điều hành
│  ├─ xử lý cờ chế độ debug / release
│  ├─ xử lý biên dịch chéo trên macOS
│  └─ hàm phụ sinh tham số -rpath cho trình liên kết
├─ khai báo mô-đun mở rộng và gói
│  ├─ loại các thư mục không đóng gói (caffe2, tools)
│  ├─ tạo mô-đun mở rộng chính (torch._C)
│  └─ thêm functorch._C nếu cần
├─ định nghĩa lệnh build tuỳ biến (cmdclass)
├─ cấu hình điểm vào (console_scripts)
└─ trả về các tham số
```

`rpath` (runtime search path) là cơ chế chỉ định đường dẫn tìm thư viện động lúc chạy, để tệp thực thi hay thư viện động biết tìm các thư viện phụ thuộc ở đâu. Mặc định hệ thống tìm trong các đường dẫn chuẩn (`/usr/lib`, `/usr/local/lib`) hoặc trong biến môi trường `LD_LIBRARY_PATH` (macOS: `DYLD_LIBRARY_PATH`). Đặt rpath tương đối `$ORIGIN/lib` cho `_C...so` nghĩa là "tìm thư viện trong thư mục `lib` cạnh chính tệp này", nhờ vậy gói torch cài ở đâu cũng chạy được mà không cần đặt `LD_LIBRARY_PATH`.

Mã `configure_extension_build` có chú thích:

```python
def configure_extension_build():
    r"""
    Cấu hình build extension C/C++ của PyTorch và trả về một phần tham số cho setuptools.setup(...).
    Trả về tuple 5 phần tử: (extensions, cmdclass, packages, entry_points, extra_install_requires)
    """
    cmake_cache_vars = get_cmake_cache_vars()   # đọc cấu hình từ CMakeCache.txt

    # 1. Cờ biên dịch và liên kết cơ bản
    library_dirs = []
    extra_install_requires = []
    if IS_WINDOWS:
        # /NODEFAULTLIB:LIBCMT.LIB: chỉ liên kết bản DLL của runtime
        extra_link_args = ["/NODEFAULTLIB:LIBCMT.LIB"]
        # /MD: runtime động; /FS: ghi PDB song song; /EHsc: xử lý ngoại lệ C++ chuẩn
        extra_compile_args = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = []
        extra_compile_args = [
            "-Wall", "-Wextra", "-Wno-strict-overflow", "-Wno-unused-parameter",
            "-Wno-missing-field-initializers", "-Wno-unknown-pragmas",
            # -fno-strict-aliasing: tương thích Python cũ (2.6), tránh tối ưu nhạy cảm với alias
            "-fno-strict-aliasing",
        ]

    library_dirs.append(lib_path)               # thư mục thư viện đã build, để liên kết

    main_compile_args = []
    main_libraries = ["torch_python"]           # mặc định liên kết tới libtorch_python
    main_link_args = []
    main_sources = ["torch/csrc/stub.c"]        # tệp nguồn duy nhất của torch._C
    # build wheel libtorch: không cần torch_python, liên kết "torch", không dùng stub.c
    if BUILD_LIBTORCH_WHL:
        main_libraries = ["torch"]
        main_sources = []

    # cờ cho chế độ Debug / RelWithDebInfo
    if build_type.is_debug():
        if IS_WINDOWS:
            extra_compile_args.append("/Z7")          # thông tin gỡ lỗi đầy đủ
            extra_link_args.append("/DEBUG:FULL")
        else:
            extra_compile_args += ["-O0", "-g"]       # không tối ưu, có thông tin gỡ lỗi
            extra_link_args += ["-O0", "-g"]
    if build_type.is_rel_with_deb_info():
        if IS_WINDOWS:
            extra_compile_args.append("/Z7")
            extra_link_args.append("/DEBUG:FULL")
        else:
            extra_compile_args += ["-g"]
            extra_link_args += ["-g"]

    # phụ thuộc thêm từ biến môi trường, phân tách bằng "|"
    pytorch_extra_install_requirements = os.getenv("PYTORCH_EXTRA_INSTALL_REQUIREMENTS", "")
    if pytorch_extra_install_requirements:
        report(f"pytorch_extra_install_requirements: {pytorch_extra_install_requirements}")
        extra_install_requires += pytorch_extra_install_requirements.split("|")

    # biên dịch chéo trên macOS: CMAKE_OSX_ARCHITECTURES là arm64 hay x86_64
    if IS_DARWIN:
        macos_target_arch = os.getenv("CMAKE_OSX_ARCHITECTURES", "")
        if macos_target_arch in ["arm64", "x86_64"]:
            macos_sysroot_path = os.getenv("CMAKE_OSX_SYSROOT")
            if macos_sysroot_path is None:
                # lấy sysroot của macOS bằng xcrun --show-sdk-path
                macos_sysroot_path = (subprocess.check_output(["xcrun", "--show-sdk-path", "--sdk", "macosx"])
                                      .decode("utf-8").strip())
            # thêm -arch và -isysroot để khớp kiến trúc đích
            extra_compile_args += ["-arch", macos_target_arch, "-isysroot", macos_sysroot_path]
            extra_link_args += ["-arch", macos_target_arch]

    # hàm phụ: sinh tham số rpath tương đối
    def make_relative_rpath_args(path):
        if IS_DARWIN:
            return [f"-Wl,-rpath,@loader_path/{path}"]   # macOS: @loader_path
        elif IS_WINDOWS:
            return []                                    # Windows không dùng rpath
        else:
            return [f"-Wl,-rpath,$ORIGIN/{path}"]        # Linux: $ORIGIN

    # 2. Khai báo mô-đun mở rộng và gói Python
    extensions = []
    excludes = ["tools", "tools.*", "caffe2", "caffe2.*"]  # không đóng gói làm gói Python
    if not cmake_cache_vars["BUILD_FUNCTORCH"]:
        excludes.extend(["functorch", "functorch.*"])
    packages = find_packages(exclude=excludes)

    # mô-đun mở rộng chính: torch._C
    C = Extension(
        "torch._C",                             # tên đầy đủ: cài xong import được torch._C
        libraries=main_libraries,               # thư viện cần liên kết
        sources=main_sources,                   # tệp nguồn biên dịch thành mô-đun
        language="c",                           # ngôn ngữ, quyết định trình biên dịch
        extra_compile_args=main_compile_args + extra_compile_args,
        include_dirs=[],
        library_dirs=library_dirs,              # nơi tìm thư viện khi liên kết
        extra_link_args=(extra_link_args + main_link_args
                         + make_relative_rpath_args("lib")),   # rpath: tìm thư viện ở ./lib
    )
    extensions.append(C)

    # build functorch thì thêm một extension giữ chỗ functorch._C (CMake build và chép vào)
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        extensions.append(Extension(name="functorch._C", sources=[]))

    # lệnh tuỳ biến, thay build_ext, bdist_wheel... mặc định của setuptools
    cmdclass = {
        "bdist_wheel": wheel_concatenate,
        "build_ext": build_ext,
        "clean": clean,
        "install": install,
        "sdist": sdist,
    }

    # lệnh dòng lệnh khi cài
    entry_points = {
        "console_scripts": ["torchrun = torch.distributed.run:main"],     # lệnh torchrun
        "torchrun.logs_specs": ["default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs"],
    }
    # build có phân tán thì thêm lệnh torchfrtrace
    if cmake_cache_vars["USE_DISTRIBUTED"]:
        entry_points["console_scripts"].append("torchfrtrace = tools.flight_recorder.fr_trace:main")

    # (tác giả thêm) in kết quả trước khi trả về
    logger.info("=== configure_extension_build Results ===")
    logger.info("extensions: %s", extensions)
    logger.info("cmdclass: %s", cmdclass)
    logger.info("packages: %s", packages)
    logger.info("entry_points: %s", entry_points)
    logger.info("extra_install_requires: %s", extra_install_requires)
    return extensions, cmdclass, packages, entry_points, extra_install_requires
```

Đoạn trên trông dài, nhưng cốt lõi nhất là phần khai báo mô-đun mở rộng chính `torch._C`: một extension chỉ có **một tệp nguồn** `torch/csrc/stub.c`, liên kết với `libtorch_python` (thư viện chứa toàn bộ phần ràng buộc Python đã được CMake build ở bước `build_deps()`), và có rpath trỏ vào `lib`. Bài [Tổng quan kiến trúc](tools/pt_arch_vi) cho thấy kết quả: tệp `_C...so` chỉ khoảng 10 KB.

`library_dirs` chứa thư mục thư viện đã build:

```python
lib_path = os.path.join(cwd, "torch", "lib")
library_dirs.append(lib_path)
```

Thư mục `pytorch/torch/lib` chứa **các thư viện động sinh ra khi build, tạo thành backend C++ lõi của PyTorch**: mã nhị phân cài đặt phép toán tensor, vi phân tự động, quản lý thiết bị, huấn luyện phân tán... Chúng nối với tầng Python qua cơ chế mở rộng của Python (mô-đun `_C`).

Kết quả in của `configure_extension_build` trên máy tác giả (danh sách `packages` đã được tác giả rút gọn: `torch.fx.*`, `torch.nn.quantized.*`, `torch.nn.qat.*`, `torch.utils.*` đại diện cho cả nhóm mô-đun con):

```text
INFO:__main__:=== configure_extension_build Results ===
INFO:__main__:extensions: [<setuptools.extension.Extension('torch._C') at 0x100db26c0>, <setuptools.extension.Extension('functorch._C') at 0x100b2b0e0>]
INFO:__main__:cmdclass: {'bdist_wheel': <class '__main__.wheel_concatenate'>, 'build_ext': <class '__main__.build_ext'>, 'clean': <class '__main__.clean'>, 'install': <class '__main__.install'>, 'sdist': <class '__main__.sdist'>}
INFO:__main__:packages: ['torch', 'torchgen', 'functorch', 'torch.monitor', 'torch._dispatch', 'torch.nn', 'torch.onnx', 'torch.distributed', 'torch.autograd', 'torch.profiler', 'torch.export', 'torch.compiler', 'torch.func', 'torch.nn.attention', 'torch.nn.modules', 'torch.utils.tensorboard', 'torch.nn.qat.*', 'torch.nn.quantized.*', 'torch.fx.*', 'torch.utils.*', ...]
INFO:__main__:entry_points: {'console_scripts': ['torchrun = torch.distributed.run:main', 'torchfrtrace = tools.flight_recorder.fr_trace:main'], 'torchrun.logs_specs': ['default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs']}
INFO:__main__:extra_install_requires: []
```

## 3 Bổ sung: xem thông tin build của bản đã cài

Mục này do người biên soạn viết thêm. Không cần build lại, ta vẫn đọc được kết quả của các bước trên từ bản PyTorch đã cài: cờ build (đọc từ CMake), các lệnh `console_scripts` đã đăng ký, phụ thuộc bắt buộc, và rpath của `_C...so`:

<!-- run -->
```python
import os, subprocess, importlib.metadata as md, torch
cfg = torch.__config__.show()
print("\n".join(l for l in cfg.splitlines() if l.strip().startswith(("- GCC", "- C++ Version", "- CPU capability"))))
settings = cfg.split("Build settings:")[1]
for key in ["BUILD_TYPE", "USE_CUDA", "USE_MKLDNN", "USE_OPENMP", "TORCH_VERSION"]:
    val = [p.split("=")[1] for p in settings.replace(",", " ").split() if p.startswith(key + "=")]
    print(f"  {key:<16}= {val[0] if val else '?'}")
dist = md.distribution("torch")
print("có torch.distributed (USE_DISTRIBUTED):", torch.distributed.is_available())
print("console_scripts:", [e.name for e in dist.entry_points if e.group == "console_scripts"])
print("install_requires:", [r.split(";")[0].strip() for r in (dist.requires or []) if "extra ==" not in r])
dyn = subprocess.run(["readelf", "-d", torch._C.__file__], capture_output=True, text=True).stdout
print("rpath của _C:", [l.split("[")[1].rstrip("]") for l in dyn.splitlines() if "RPATH" in l or "RUNPATH" in l])
```

<!-- output -->
```output
  - GCC 13.3
  - C++ Version: 202002
  - CPU capability usage: AVX2
  BUILD_TYPE      = Release
  USE_CUDA        = 0
  USE_MKLDNN      = ON
  USE_OPENMP      = ON
  TORCH_VERSION   = 2.14.0
có torch.distributed (USE_DISTRIBUTED): True
console_scripts: ['torchrun', 'torchfrtrace']
install_requires: ['filelock', 'typing-extensions>=4.10.0', 'setuptools>=77.0.3', 'sympy>=1.13.3', 'networkx>=2.5.1', 'jinja2', 'fsspec>=0.8.5']
rpath của _C: ['$ORIGIN:$ORIGIN/lib']
```
<!-- /output -->

## Tài liệu tham khảo

- [`setup.py` của PyTorch](https://github.com/pytorch/pytorch/blob/main/setup.py)
- [setuptools: Building Extension Modules](https://setuptools.pypa.io/en/latest/userguide/ext_modules.html)
- [PyTorch CONTRIBUTING: build từ mã nguồn](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
