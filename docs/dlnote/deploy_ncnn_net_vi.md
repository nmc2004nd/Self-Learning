# Triển khai mô hình: phân tích mã nguồn ncnn, lớp Net

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/6-model_deploy/ncnn源码解析-Net类.md), [kho mã](https://github.com/harleyszhang/dl_note), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); phần dịch có chỉnh sửa). Số mục giữ như bản gốc. Chỗ bản gốc sai, thiếu hoặc đã lỗi thời được sửa ngay trong bài và ghi ở ô **Biên tập**. Các mục có chữ **Bổ sung** do người biên soạn viết thêm.

Bài này đọc hai phần trong mã nguồn ncnn: lớp `Net` (tệp `net.h`, `net.cpp`), nơi nạp mô hình và tạo `Extractor`, và lớp `ParamDict`, nơi phân tích từng dòng tham số của tệp `.param`. Trang [Code](dlnote/deploy_code_vi) đi kèm (mục 5.1) viết lại hàm `ParamDict::load_param` bằng Python và dùng nó để đọc các dòng của một tệp `.param` thật.

<div class="qa-theory">

**Lý thuyết:** [Ôn tập · 17.10.1 NCNN](qa/ch17_vi?id=_17101-ncnn)

</div>

## 1 Lớp mạng Net

`net.cpp/net.h` là nơi định nghĩa cấu trúc mô hình và các lớp suy luận, gồm chủ yếu:

- `Net`: lớp mạng, cung cấp giao diện nạp và phân tích tệp cấu trúc, tệp trọng số: `load_param` và `load_model`.
- `Extractor`: lớp thực thi suy luận, giao diện chính ra bên ngoài là hàm đưa dữ liệu vào `input` và hàm chạy suy luận, lấy kết quả `extract`.
- `NetPrivate`: các thành viên riêng của `Net` đều đặt trong lớp này, như `blobs`, `layers`, `input_blob_indexes`, `output_blob_indexes`, `custom_layer_registry` và `local_blob_allocator`…
- `ExtractorPrivate`: các thành viên riêng của `Extractor`.

ncnn gói các thành viên riêng của `ncnn::Net` vào một lớp `NetPrivate` riêng; nhiều lớp khác trong ncnn cũng làm vậy, như `Pipeline`, `ParamDict`, `Extractor`, `PoolAllocator`, `DataReaderFromMemory`…

> Cách làm này là thành ngữ **PIMPL** (pointer to implementation) của C++: tệp tiêu đề công khai chỉ chứa một con trỏ tới lớp cài đặt, nên thay đổi thành viên riêng không làm đổi bố cục bộ nhớ của lớp công khai. Nhờ vậy ncnn giữ được tương thích nhị phân (ABI) giữa các phiên bản, và tệp tiêu đề không lộ chi tiết cài đặt.

### 1.1 Phân tích lớp Net

Một phần định nghĩa của `Net` (đã lược bớt):

```cpp
class Net
{
public:
    // khởi tạo rỗng
    Net();
    // xoá và huỷ
    virtual ~Net();

public:
    // tuỳ chọn, có thể đổi trước khi nạp mô hình
    Option opt;
    int register_custom_layer(int index, layer_creator_func creator,
                              layer_destroyer_func destroyer = 0, void* userdata = 0);
    // các hàm nạp cấu trúc và trọng số thực sự
    int load_param_bin(const DataReader& dr);
    int load_model(const DataReader& dr);

    // nạp cấu trúc mạng từ tệp tham số
    int load_param(const char* protopath);
    // nạp trọng số từ tệp mô hình, trả về 0 nếu thành công
    int load_model(const char* modelpath);

private:
    NetPrivate* const d;  // con trỏ tới lớp chứa thành viên riêng
};
```

> **Biên tập:** Chú thích trong mã gốc ghi `load_param` nạp "từ tệp tham số nhị phân" (binary param file); thực ra `load_param(const char*)` đọc tệp `.param` **dạng văn bản** (cần macro `NCNN_STRING`), còn tệp tham số nhị phân `.param.bin` (sinh bằng công cụ `ncnn2mem`, không chứa tên tầng) được đọc bằng `load_param_bin`. Các hàm nhận `const char*` chỉ mở tệp rồi gọi bản nhận `DataReader`; ncnn cũng có bản nạp từ bộ nhớ (`load_param_mem`, `load_model(const unsigned char*)`) và từ Android asset.

Các biến thành viên chính của `NetPrivate`:

```cpp
// Blob ghi lại dữ liệu tensor của feature map
std::vector<Blob> blobs;
std::vector<Layer*> layers;

std::vector<int> input_blob_indexes;
std::vector<int> output_blob_indexes;
```

`Option opt` chứa các tuỳ chọn chạy: `num_threads` (số luồng OpenMP), `use_vulkan_compute` (chạy trên GPU qua Vulkan), `use_fp16_storage`, `use_fp16_arithmetic`, `use_bf16_storage` (lưu, tính bằng 16 bit trên CPU hỗ trợ), `use_int8_inference`, `use_winograd_convolution`, `use_sgemm_convolution`, `use_packing_layout`… Phải đặt chúng **trước** khi gọi `load_param`, vì ncnn chọn cài đặt của từng tầng ngay lúc nạp.

## 2 Lớp từ điển tham số ParamDict

Trong `Net::load_param()`, lớp `ParamDict` được dùng ở ba chỗ:

```cpp
Mat shape_hints = pd.get(30, Mat());
layer->featmask = pd.get(31, 0);
int lr = layer->load_param(pd);
```

Hai id 30, 31 là tham số chung cho mọi tầng: id 30 là gợi ý hình dạng của các blob ra (ghi sẵn bởi công cụ chuyển đổi, giúp cấp phát bộ nhớ trước), id 31 là mặt nạ đặc tính (`featmask`) để tắt riêng một số tối ưu (fp16, bf16, packing, Vulkan…) cho đúng tầng đó. Các id còn lại do từng loại tầng tự định nghĩa trong hàm `load_param` của nó.

`ParamDict` lưu thành viên riêng qua lớp `ParamDictPrivate`; định nghĩa như sau:

```cpp
class NCNN_EXPORT ParamDict
{
public:
    // rỗng
    ParamDict();

    virtual ~ParamDict();

    // sao chép
    ParamDict(const ParamDict&);

    // gán
    ParamDict& operator=(const ParamDict&);

    // lấy kiểu
    int type(int id) const;

    // lấy int
    int get(int id, int def) const;
    // lấy float
    float get(int id, float def) const;
    // lấy mảng
    Mat get(int id, const Mat& def) const;

    // đặt int
    void set(int id, int i);
    // đặt float
    void set(int id, float f);
    // đặt mảng
    void set(int id, const Mat& v);

protected:
    friend class Net;

    void clear();

    int load_param(const DataReader& dr);
    int load_param_bin(const DataReader& dr);

private:
    ParamDictPrivate* const d;
};
```

Phân tích hàm thành viên chính `load_param` của `ParamDict`:

```cpp
int ParamDict::load_param(const DataReader& dr)
{
    clear();

    //     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // phân tích từng cặp key=value
    int id = 0;
    while (dr.scan("%d=", &id) == 1)
    {
        // có phải kiểu mảng không
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }
        // id có vượt số tham số tối đa không
        if (id >= NCNN_MAX_PARAM_COUNT)
        {
            NCNN_LOGE("id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)", id, NCNN_MAX_PARAM_COUNT);
            return -1;
        }
        // nếu là mảng, phân tích như sau
        if (is_array)
        {
            int len = 0;
            int nscan = dr.scan("%d", &len);  // đọc độ dài mảng
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read array length failed");
                return -1;
            }

            d->params[id].v.create(len);  // tạo mảng
            // duyệt và phân tích từng phần tử
            for (int j = 0; j < len; j++)
            {
                // đọc một phần tử
                char vstr[16];
                nscan = dr.scan(",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict read array element failed");
                    return -1;
                }

                // là số thực nếu chuỗi có '.' hoặc 'e'
                // (dạng thập phân hoặc dạng khoa học)
                bool is_float = vstr_is_float(vstr);

                // chuyển sang kiểu tương ứng
                if (is_float)
                {
                    float* ptr = d->params[id].v;
                    ptr[j] = vstr_to_float(vstr);
                }
                else
                {
                    // gán vstr cho params[id].v[j]
                    int* ptr = d->params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                    if (nscan != 1)
                    {
                        NCNN_LOGE("ParamDict parse array element failed");
                        return -1;
                    }
                }
                // đặt kiểu tham số
                d->params[id].type = is_float ? 6 : 5;
            }
        }
        // nếu không phải mảng thì đọc một giá trị, các bước giống trong khối if
        else
        {
            char vstr[16];
            int nscan = dr.scan("%15s", vstr);
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read value failed");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
            {
                d->params[id].f = vstr_to_float(vstr);
            }
            else
            {
                nscan = sscanf(vstr, "%d", &d->params[id].i);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict parse value failed");
                    return -1;
                }
            }

            d->params[id].type = is_float ? 3 : 2;
        }
    }

    return 0;
}
#endif // NCNN_STRING
```

Tóm lại quy ước của một dòng tham số: mỗi cặp `id=giá trị`; id từ 0 tới 31 là một giá trị đơn (int nếu chuỗi không có `.` hay `e`, ngược lại là float); id âm $\le -23300$ là **mảng**, id thật là $-\text{id} - 23300$, và giá trị có dạng `độ dài,phần tử 1,phần tử 2,…`. Với dòng ví dụ trong chú thích, `-23303=5,0.1,0.2,0.4,0.8,1.0` là tham số id 3, mảng 5 số thực. Trang Code chạy bản Python của hàm này trên dòng đó và cho `{0: 100, 1: 1.25, 3: [0.1, 0.2, 0.4, 0.8, 1.0]}`; với dòng `Reduction` trong tệp ShuffleNet v2 do pnnx sinh, `-23303=2,1,2` là tham số 3 (các trục cần rút gọn) bằng `[1, 2]`, tức lấy trung bình theo chiều cao và chiều rộng.

`dr.scan` gọi tới `DataReaderFromMemory::scan`, định nghĩa như sau:

```cpp
#if NCNN_STRING  // chỉ biên dịch khi có macro NCNN_STRING
int DataReaderFromMemory::scan(const char* format, void* p) const
{
    // độ dài chuỗi định dạng
    size_t fmtlen = strlen(format);

    // thêm '%n' vào cuối chuỗi định dạng để lấy số ký tự đã đọc
    char* format_with_n = new char[fmtlen + 4];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;  // số ký tự đã đọc
    int nscan = sscanf((const char*)d->mem, format_with_n, p, &nconsumed);  // đọc dữ liệu
    d->mem += nconsumed;  // dời con trỏ tới phần chưa đọc

    delete[] format_with_n;  // giải phóng bộ nhớ cấp phát động

    return nconsumed > 0 ? nscan : 0;  // trả về số trường đọc được, hoặc 0
}
#endif // NCNN_STRING
```

Hàm `DataReaderFromMemory::scan` **đọc dữ liệu từ bộ nhớ và phân tích theo định dạng cho trước**. Nó lấy độ dài chuỗi định dạng, thêm `%n` vào cuối để biết đã đọc bao nhiêu ký tự, gọi `sscanf` đọc dữ liệu rồi dời con trỏ bộ nhớ tới phần chưa đọc, cuối cùng trả về số trường `sscanf` đọc được (hoặc 0 nếu không đọc được ký tự nào).

> **Biên tập:** Bản gốc viết hàm "trả về số ký tự đã đọc hoặc 0"; nó trả về `nscan`, tức **số trường** mà `sscanf` gán được (thường là 1), không phải số ký tự (số ký tự là `nconsumed`, chỉ dùng để dời con trỏ). Vì vậy vòng lặp `while (dr.scan("%d=", &id) == 1)` dừng khi không còn cặp `id=` nào trên dòng.

Lưu ý hàm này phụ thuộc macro `NCNN_STRING`, chỉ được biên dịch khi macro đó được định nghĩa. Khi tắt `NCNN_STRING` (để thư viện nhỏ hơn trên thiết bị nhúng), chỉ nạp được tệp tham số nhị phân.

Định nghĩa lớp `ParamDictPrivate`:

```cpp
#define NCNN_MAX_PARAM_COUNT 32
class ParamDictPrivate
{
public:
    struct
    {
        // 0 = null
        // 1 = int/float
        // 2 = int
        // 3 = float
        // 4 = array of int/float
        // 5 = array of int
        // 6 = array of float
        int type;
        union
        {
            int i;
            float f;
        };
        Mat v;
    } params[NCNN_MAX_PARAM_COUNT];
};
```

`NCNN_MAX_PARAM_COUNT` được định nghĩa bằng 32, nên `params` là mảng 32 cấu trúc: mỗi tầng có tối đa 32 tham số (id 0 tới 31). Cấu trúc này lưu giá trị tham số, `type` cho biết kiểu: số đơn được giữ trong `union` (int hoặc float dùng chung 4 byte), mảng được giữ trong `Mat v`.

> **Biên tập:** Bản gốc viết "số tham số trên mỗi dòng của tệp tham số không được vượt quá 32". Chính xác hơn, giới hạn là trên **giá trị id** (phải nhỏ hơn 32, như điều kiện `id >= NCNN_MAX_PARAM_COUNT` trong `load_param`), không phải số cặp; một dòng có thể ghi lại cùng một id nhiều lần (giá trị sau đè giá trị trước).

## 3 Bổ sung: định dạng tệp `.param` và `.bin`

Mục này do người biên soạn viết thêm, dựa trên tệp do pnnx sinh ở trang Code.

Tệp `.param` (dạng văn bản) có cấu trúc:

```text
7767517                       ← số "ma thuật" cho biết định dạng
108 124                       ← số tầng, số blob
Input        in0         0 1 in0
Convolution  convrelu_0  1 1 in0 1 0=24 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=648 9=1
...
InnerProduct linear_53   1 1 122 out0 0=1000 1=1 2=1024000
```

Mỗi dòng tầng gồm: **loại tầng**, **tên tầng**, **số blob vào**, **số blob ra**, tên các blob vào, tên các blob ra, rồi các cặp tham số. Với tầng `Convolution` ở trên: `0=24` số kênh ra, `1=3` kernel rộng 3 (`11=3` cao 3), `12` và `2` là dilation, `3=2`, `13=2` stride 2, `4=1`, `14=1` padding 1, `5=1` có bias, `6=648` số trọng số ($24 \times 3 \times 3 \times 3$), `9=1` hàm kích hoạt ReLU gộp sẵn trong tầng. Theo quy ước, id `1x` là bản theo chiều cao của id `x` (khi vắng thì bằng giá trị theo chiều rộng).

ShuffleNet v2 sau khi pnnx chuyển có 108 tầng: 37 `Convolution`, 19 `ConvolutionDepthWise`, 16 `ShuffleChannel`, 16 `Concat`, 13 `Slice`, 3 `Split` (ncnn tự chèn khi một blob được nhiều tầng dùng), cùng `Input`, `Pooling`, `Reduction`, `InnerProduct`.

Tệp `.bin` là trọng số của các tầng, ghi nối tiếp theo đúng thứ tự tầng trong `.param`. Trọng số mỗi tầng có thể bắt đầu bằng một thẻ 4 byte cho biết kiểu dữ liệu (fp32, fp16, hay int8 đã lượng tử hoá), nên cùng một `.param` có thể đi với `.bin` fp32 hoặc fp16. Tệp `.bin` không có tiêu đề, nên chỉ đọc đúng được khi đi kèm đúng tệp `.param`.

## Tài liệu tham khảo

1. [ncnn: net.h, net.cpp, paramdict.h, paramdict.cpp, datareader.cpp](https://github.com/Tencent/ncnn/tree/master/src)
2. [ncnn wiki: param and model file structure](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure)
3. [ncnn wiki: operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)
