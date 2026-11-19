# HuggingFace Transformers: nhập môn

> Ghi chú của người biên soạn: trang này dựa trên ghi chú *transformers库快速入门* trong *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/1-math_ml_basic/transformers库快速入门.md), [kho mã](https://github.com/harleyszhang/dl_note), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)). Bản gốc viết cho transformers 4.29 (2023); bài này **viết lại theo transformers 5** và mọi đoạn code được chạy thật (transformers 5.17, PyTorch 2.14, CPU). Bố cục và số mục giữ như bản gốc. Chỗ bản gốc sai hoặc đã lỗi thời được ghi ở ô **Biên tập**; mục 4 do người biên soạn viết thêm.

> **Biên tập:** Những thay đổi lớn từ transformers 4.29 đến 5.x ảnh hưởng tới bài này:
> - Thư viện chỉ còn hỗ trợ PyTorch; các lớp TensorFlow (`TFAutoModel`...) và Flax đã bị bỏ.
> - Tokenizer "chậm" (Python) và "nhanh" (Rust) được gộp làm một; `BertTokenizer` giờ chính là bản nhanh, tham số `use_fast` không còn cần.
> - Tham số `torch_dtype` đổi tên thành `dtype`; mặc định nạp trọng số theo kiểu lưu trong checkpoint (ví dụ bfloat16) thay vì luôn float32.
> - Nhiều tác vụ pipeline dành cho mô hình seq2seq cũ (`summarization`, `translation`, `question-answering`, `text2text-generation`, `vqa`) bị bỏ; các việc này giờ làm bằng mô hình ngôn ngữ lớn qua `text-generation` hoặc `image-text-to-text`.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống và dùng chung biến. Cần `pip install "transformers>=5" torch`. Lần chạy đầu sẽ tải bốn mô hình nhỏ từ Hub (tổng khoảng 1,9 GB): tokenizer của mBERT, `distilbert-base-uncased-finetuned-sst-2-english` (268 MB), `nlptown/bert-base-multilingual-uncased-sentiment` (669 MB) và `Qwen2.5-0.5B-Instruct` (988 MB). Mọi khối chạy được trên CPU.

<!-- run -->
```python
import warnings
warnings.filterwarnings("ignore")
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
transformers.logging.set_verbosity_error()      # tắt bớt thông báo khi nạp mô hình
torch.manual_seed(0)
print("transformers", transformers.__version__, "| torch", torch.__version__)
print("còn lớp TensorFlow không:", hasattr(transformers, "TFAutoModel"))
```

<!-- output -->
```output
transformers 5.17.0 | torch 2.14.0+cpu
còn lớp TensorFlow không: False
```
<!-- /output -->

## 1 Thuật ngữ

### 1.1 Token, tokenization và tokenizer

**Token** là đơn vị nhỏ nhất mà mô hình nhìn thấy. Tuỳ cách tách, token có thể là một từ, một ký tự hay một **mảnh từ** (subword); các mô hình hiện nay hầu hết dùng mảnh từ.

**Tokenization** là quá trình tách chuỗi đầu vào thành các token sao cho mỗi token mang một phần nghĩa tương đối trọn vẹn, để dùng cho bước sau (học embedding, làm đầu vào cho mô hình ngôn ngữ lớn).

**Tokenizer** là đối tượng thực hiện tokenization trong thư viện transformers. Mỗi tokenizer có một **bộ từ vựng** (vocabulary) riêng, và chỉ tách ra những token có trong bộ từ vựng đó.

Ví dụ:

- "VRAM" thường không có trong bộ từ vựng, nên bị tách thành "VR" và "##AM" (`##` của WordPiece nghĩa là "nối liền với token trước").
- Câu tiếng Trung 我是中国人 có thể được tách thành ['我', '是', '中国人'] với một tokenizer theo từ, nhưng mBERT tách theo từng chữ Hán.

<!-- run -->
```python
bert_tok = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

for text in ["A Titan RTX has 24GB of VRAM", "我是中国人", "Tôi là sinh viên học máy", "I am a machine learning student"]:
    print(f"{text!r}")
    print(f"  mBERT (WordPiece, {bert_tok.vocab_size:,} token): {bert_tok.tokenize(text)}")
    q = qwen_tok.tokenize(text)
    print(f"  Qwen2.5 (BPE theo byte, {qwen_tok.vocab_size:,} token): {len(q)} token")
```

<!-- output -->
```output
'A Titan RTX has 24GB of VRAM'
  mBERT (WordPiece, 119,547 token): ['A', 'Titan', 'RT', '##X', 'has', '24', '##GB', 'of', 'VR', '##AM']
  Qwen2.5 (BPE theo byte, 151,643 token): 12 token
'我是中国人'
  mBERT (WordPiece, 119,547 token): ['我', '是', '中', '国', '人']
  Qwen2.5 (BPE theo byte, 151,643 token): 2 token
'Tôi là sinh viên học máy'
  mBERT (WordPiece, 119,547 token): ['T', '##ôi', 'là', 'sinh', 'viên', 'học', 'máy']
  Qwen2.5 (BPE theo byte, 151,643 token): 6 token
'I am a machine learning student'
  mBERT (WordPiece, 119,547 token): ['I', 'am', 'a', 'machine', 'learning', 'student']
  Qwen2.5 (BPE theo byte, 151,643 token): 6 token
```
<!-- /output -->

> **Biên tập:** Bản gốc nói 我是中国人 được tách thành ['我', '是', '中国人']; đó là kết quả của bộ tách từ tiếng Trung (như jieba), không phải của các tokenizer trong transformers: mBERT tách từng chữ Hán như trên. Token của Qwen là các byte UTF-8 ghép lại (byte-level BPE) nên in ra không đọc được; ta chỉ in số token. Với câu ngắn ở trên, tiếng Việt và tiếng Anh tốn số token gần nhau trên Qwen; với các tokenizer có ít dữ liệu tiếng Việt khi huấn luyện, tiếng Việt thường tốn nhiều token hơn hẳn (mBERT tách "Tôi" thành hai mảnh), nghĩa là chậm hơn và tốn ngữ cảnh hơn.

### 1.2 Input IDs

Đầu vào bắt buộc duy nhất của mô hình ngôn ngữ là **input IDs**: chỉ số của từng token trong bộ từ vựng, tức một vector số nguyên. Hai bước:

- chuỗi đầu vào → các token (tokenization);
- các token → input IDs (mã hoá), mỗi số là chỉ số của token trong bộ từ vựng.

<!-- run -->
```python
sequence = "A Titan RTX has 24GB of VRAM"
tokenized_sequence = bert_tok.tokenize(sequence)     # chỉ tách token
inputs = bert_tok(sequence)                           # tách + đổi thành ID + thêm token đặc biệt
encoded_sequence = inputs["input_ids"]

print(tokenized_sequence)
print(encoded_sequence)
print("độ dài:", len(tokenized_sequence), len(encoded_sequence))
print("các khoá trả về:", list(inputs.keys()))
print("giải mã lại   :", bert_tok.decode(encoded_sequence))
```

<!-- output -->
```output
['A', 'Titan', 'RT', '##X', 'has', '24', '##GB', 'of', 'VR', '##AM']
[101, 138, 28318, 56898, 12674, 10393, 10233, 32469, 10108, 74727, 36535, 102]
độ dài: 10 12
các khoá trả về: ['input_ids', 'token_type_ids', 'attention_mask']
giải mã lại   : [CLS] A Titan RTX has 24GB of VRAM [SEP]
```
<!-- /output -->

Gọi `tokenizer(...)` trả về một **dict** chứa mọi tham số mô hình cần; chỉ số token nằm ở khoá `input_ids`. Tokenizer cũng **tự thêm token đặc biệt** nếu mô hình cần (với BERT là `[CLS]` ở đầu và `[SEP]` ở cuối), nên `encoded_sequence` dài hơn `tokenized_sequence` hai phần tử.

### 1.3 Attention mask

**Attention mask** là tham số tuỳ chọn, thường dùng khi đưa nhiều câu vào cùng một **batch**. Các câu dài ngắn khác nhau nên phải đệm (padding) cho bằng nhau; mask cho mô hình biết token nào là thật cần chú ý (giá trị `1`) và token nào là phần đệm cần bỏ qua (giá trị `0`).

<!-- run -->
```python
sentences = ["We are very happy to show you the 🤗 Transformers library.", "Deepspeed is faster"]
sent_tok = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
padded = sent_tok(sentences, padding=True, return_tensors="pt")
print(padded["input_ids"])
print(padded["attention_mask"])
print("token đệm:", sent_tok.pad_token, "=", sent_tok.pad_token_id)
```

<!-- output -->
```output
tensor([[  101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103,   100,
         58263, 13299,   119,   102],
        [  101, 15526, 65998, 54436, 10127, 51524,   102,     0,     0,     0,
             0,     0,     0,     0]])
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
token đệm: [PAD] = 0
```
<!-- /output -->

Câu thứ hai được đệm bằng token `[PAD]` (ID 0) ở cuối, và mask tương ứng bằng 0 ở các vị trí đó.

> **Biên tập:** Bản gốc có một ảnh minh hoạ attention mask trỏ sai đường dẫn (`../../images/...`), nên không hiện trong kho gốc; kết quả in ở trên thay cho ảnh đó. Mô hình sinh văn bản (decoder) thường **đệm bên trái** (`tokenizer.padding_side = "left"`) khi sinh theo batch, để token cuối của mọi câu thẳng hàng.

### 1.4 Ý nghĩa các token đặc biệt

Trong tệp cấu hình của checkpoint thường thấy các token đặc biệt sau:

1. `bos_token` (Beginning of Sequence): đánh dấu đầu chuỗi.
2. `eos_token` (End of Sequence): **đánh dấu cuối chuỗi**; khi sinh văn bản, mô hình dừng lúc sinh ra token này.
3. `pad_token` (Padding): token đệm để các chuỗi trong batch dài bằng nhau.
4. `unk_token`, `cls_token`, `sep_token`, `mask_token`: token cho từ lạ, đầu câu phân loại, phân cách câu và che từ (của BERT).

<!-- run -->
```python
for name, tok in [("mBERT", bert_tok), ("Qwen2.5-Instruct", qwen_tok)]:
    print(f"{name:<17}", {k: getattr(tok, k) for k in ["bos_token", "eos_token", "pad_token", "unk_token", "cls_token", "sep_token", "mask_token"] if getattr(tok, k, None)})
```

<!-- output -->
```output
mBERT             {'pad_token': '[PAD]', 'unk_token': '[UNK]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'mask_token': '[MASK]'}
Qwen2.5-Instruct  {'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>'}
```
<!-- /output -->

> **Biên tập:** Bản gốc liệt kê `eop_token` (End of Paragraph) như một token chuẩn. Token này chỉ có ở một số mô hình (như ChatGLM của Zhipu), không phải thuộc tính chung của tokenizer trong transformers. Ví dụ ở trên cũng cho thấy không phải mô hình nào cũng có đủ các token: Qwen2.5-Instruct không có `bos_token`, và dùng `<|im_end|>` (kết thúc một lượt hội thoại) làm `eos_token`.

### 1.5 Mô hình decoder

Mô hình decoder còn gọi là mô hình **tự hồi quy** (auto-regressive) hay mô hình ngôn ngữ nhân quả (causal language model): đọc văn bản theo thứ tự và phải dự đoán từ tiếp theo. Khi huấn luyện, nó dùng **mặt nạ nhân quả** để mỗi vị trí chỉ nhìn thấy các token đứng trước.

### 1.6 Kiến trúc và tham số

- **Kiến trúc** (architecture): bộ khung của mô hình, gồm loại và định nghĩa của từng lớp, cách các lớp nối với nhau.
- **Checkpoint**: bộ trọng số được nạp vào một kiến trúc cho trước.
- **Mô hình** (model): thuật ngữ chung, kém chính xác hơn "kiến trúc" hay "checkpoint", có thể chỉ cả hai.

Ví dụ: `BERT` là kiến trúc, còn `google-bert/bert-base-multilingual-cased` là một checkpoint của nó.

## 2 Chức năng của Transformers

Thư viện [Transformers](https://github.com/huggingface/transformers) cho phép tạo mô hình transformer và nạp, dùng các mô hình được chia sẻ; [Hub](https://huggingface.co/models) chứa hơn hai triệu mô hình tiền huấn luyện tải về dùng được, và người dùng cũng có thể tải mô hình của mình lên.

### Tổng quan API

`API` của thư viện gồm ba phần chính:

1. **Lớp chính** (main classes): cấu hình (configuration), mô hình (model), tokenizer và pipeline.
2. **Mô hình** (models): các lớp, hàm gắn với cài đặt của từng mô hình.
3. **Hàm trợ giúp nội bộ** (internal helpers): lớp và hàm dùng bên trong thư viện.

## 3 Bắt đầu nhanh

### 3.1 Các loại mô hình transformer

Kiến trúc Transformer gồm hai khối chính:

- **Encoder**: nhận đầu vào và xây dựng biểu diễn (đặc trưng) của nó; được tối ưu để **hiểu** đầu vào.
- **Decoder**: dùng biểu diễn của encoder cùng các đầu vào khác để sinh chuỗi đích; được tối ưu để **sinh** đầu ra.

Mỗi khối có thể dùng riêng, tuỳ nhiệm vụ:

- **Chỉ encoder** (encoder-only), còn gọi là mô hình tự mã hoá (auto-encoding), như họ BERT: hợp với nhiệm vụ cần hiểu đầu vào như phân loại câu, nhận dạng thực thể.
- **Chỉ decoder** (decoder-only), còn gọi là mô hình tự hồi quy, như họ GPT: hợp với nhiệm vụ sinh, như **sinh văn bản**.
- **Encoder-decoder**, còn gọi là mô hình chuỗi sang chuỗi (sequence-to-sequence), như BART, T5: hợp với nhiệm vụ sinh dựa trên đầu vào, như dịch hay tóm tắt.

| Loại | Ví dụ | Nhiệm vụ |
| --- | --- | --- |
| Encoder | BERT, RoBERTa, DeBERTa, DistilBERT, ModernBERT | Phân loại câu, nhận dạng thực thể, trích câu trả lời, embedding cho tìm kiếm |
| Decoder | GPT-2, Llama, Qwen, Mistral, Gemma | Sinh văn bản; hiện nay làm được hầu hết mọi nhiệm vụ qua chỉ dẫn (prompt) |
| Encoder-decoder | BART, T5, mBART, Whisper (âm thanh → văn bản) | Tóm tắt, dịch, nhận dạng giọng nói |

> **Biên tập:** Bảng của bản gốc (2023) liệt kê CTRL và Transformer-XL ở hàng decoder; hai mô hình này đã bị chuyển sang nhóm "deprecated" của thư viện. Bảng trên thêm các họ decoder-only đang phổ biến. Từ khoảng 2023, mô hình chỉ decoder cỡ lớn làm luôn cả dịch, tóm tắt, hỏi đáp, nên ranh giới "nhiệm vụ nào dùng loại nào" của bảng chỉ còn đúng với mô hình nhỏ chuyên dụng.

### 3.2 Pipeline

Hàm `pipeline()` nhận tham số `task` (loại nhiệm vụ) và chạy suy luận bằng một mô hình phù hợp, chỉ với một dòng code cho nhiều loại dữ liệu. Danh sách nhiệm vụ của phiên bản đang cài lấy được trực tiếp từ thư viện:

<!-- run -->
```python
from transformers.pipelines import SUPPORTED_TASKS, TASK_ALIASES
print(len(SUPPORTED_TASKS), "nhiệm vụ:", ", ".join(sorted(SUPPORTED_TASKS)))
print("tên gọi khác:", TASK_ALIASES)
```

<!-- output -->
```output
24 nhiệm vụ: any-to-any, audio-classification, automatic-speech-recognition, depth-estimation, document-question-answering, feature-extraction, fill-mask, image-classification, image-feature-extraction, image-segmentation, image-text-to-text, keypoint-matching, mask-generation, object-detection, table-question-answering, text-classification, text-generation, text-to-audio, token-classification, video-classification, zero-shot-audio-classification, zero-shot-classification, zero-shot-image-classification, zero-shot-object-detection
tên gọi khác: {'sentiment-analysis': 'text-classification', 'ner': 'token-classification', 'text-to-speech': 'text-to-audio'}
```
<!-- /output -->

Một số nhiệm vụ hay dùng:

| Nhiệm vụ | Mô tả | Loại dữ liệu | Pipeline |
| --- | --- | --- | --- |
| Phân loại văn bản | Gán nhãn cho một chuỗi văn bản (vd. cảm xúc) | Văn bản | `pipeline("text-classification")` (tên khác: `"sentiment-analysis"`) |
| Sinh văn bản | Sinh văn bản từ một prompt, kể cả hội thoại | Văn bản | `pipeline("text-generation")` |
| Nhận dạng thực thể | Gán nhãn cho từng token (người, tổ chức, địa điểm...) | Văn bản | `pipeline("token-classification")` (tên khác: `"ner"`) |
| Điền từ bị che | Dự đoán token bị che trong chuỗi | Văn bản | `pipeline("fill-mask")` |
| Phân loại không cần huấn luyện | Phân loại theo danh sách nhãn tuỳ ý | Văn bản | `pipeline("zero-shot-classification")` |
| Phân loại ảnh | Gán nhãn cho ảnh | Ảnh | `pipeline("image-classification")` |
| Phân đoạn ảnh | Gán nhãn cho từng điểm ảnh (ngữ nghĩa, thể hiện, toàn cảnh) | Ảnh | `pipeline("image-segmentation")` |
| Phát hiện vật thể | Dự đoán khung bao và lớp của vật thể trong ảnh | Ảnh | `pipeline("object-detection")` |
| Phân loại âm thanh | Gán nhãn cho đoạn âm thanh | Âm thanh | `pipeline("audio-classification")` |
| Nhận dạng giọng nói | Chuyển lời nói trong tệp âm thanh thành văn bản | Âm thanh | `pipeline("automatic-speech-recognition")` |
| Ảnh và văn bản sang văn bản | Trả lời câu hỏi về ảnh, mô tả ảnh | Đa phương thức | `pipeline("image-text-to-text")` |

> **Biên tập:** Bảng của bản gốc có `summarization`, `translation`, `question-answering` và `vqa`. Cả bốn không còn trong pipeline của transformers 5 (xem danh sách in ở trên); dịch, tóm tắt, hỏi đáp làm bằng `text-generation` với một mô hình hội thoại (mục 4), còn hỏi đáp về ảnh làm bằng `image-text-to-text`. Bản gốc cũng có một ảnh chụp trang Hub, trỏ sai đường dẫn nên không hiện; đã bỏ.

Ví dụ phân loại cảm xúc:

<!-- run -->
```python
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("I've been waiting for a HuggingFace course my whole life."))
print(classifier(["This library is great!", "The installation broke my environment."]))
```

<!-- output -->
```output
[{'label': 'POSITIVE', 'score': 0.9598048329353333}]
[{'label': 'POSITIVE', 'score': 0.9998798370361328}, {'label': 'POSITIVE', 'score': 0.94245845079422}]
```
<!-- /output -->

Câu thứ hai mang nghĩa tiêu cực nhưng mô hình vẫn cho POSITIVE với độ tin cậy cao. DistilBERT-SST-2 được huấn luyện trên câu bình luận phim, nên câu thuộc lĩnh vực khác (cài đặt phần mềm) dễ bị đoán sai; điểm "score" cao không có nghĩa là đúng. Trước khi dùng một mô hình trên Hub, nên thử nó trên vài mẫu của chính dữ liệu mình.

<!-- output -->
```output
[{'label': 'POSITIVE', 'score': 0.9598048329353333}]
[{'label': 'POSITIVE', 'score': 0.9998798370361328}, {'label': 'POSITIVE', 'score': 0.94245845079422}]
```
<!-- /output -->

> **Biên tập:** Bản gốc gọi `pipeline("sentiment-analysis")` không chỉ định mô hình. Cách này vẫn chạy (thư viện tự chọn đúng mô hình DistilBERT trên) nhưng in cảnh báo không nên dùng trong sản phẩm, vì mô hình mặc định có thể đổi giữa các phiên bản; nên luôn ghi rõ `model=` (và `revision=` nếu cần tái lập kết quả).

Với nhiệm vụ văn bản, ngoài mô hình mặc định, có thể truyền `model` và `tokenizer` để dùng mô hình tuỳ chọn trên Hub.

### 3.3 AutoClass

Đằng sau `pipeline()` là các lớp **AutoClass**: chỉ cần tên hoặc đường dẫn của checkpoint, chúng **tự tìm đúng kiến trúc** để nạp. Chọn AutoClass phù hợp với nhiệm vụ cùng lớp tiền xử lý đi kèm là tái hiện được kết quả của `pipeline()`.

#### 3.3.1 AutoTokenizer

Tokenizer lo tiền xử lý văn bản: biến chuỗi đầu vào (prompt) thành **mảng số** làm đầu vào cho mô hình. Quy tắc tokenization quyết định cách tách từ và tách tới mức nào. Lưu ý, tokenizer và mô hình phải nạp từ **cùng một** tên mô hình hay thư mục checkpoint.

Với mô hình ngôn ngữ lớn, thường dùng `AutoModelForCausalLM` và `AutoTokenizer` để nạp mô hình và tokenizer đi kèm:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto", device_map="auto")
```

> **Biên tập:** Đoạn code tương ứng của bản gốc import `AutoTokenizer` nhưng lại gọi `LlamaTokenizer` (chưa import, báo `NameError`), dùng `AutoModel` (chỉ có thân mô hình, không có đầu dự đoán token nên không sinh văn bản được), và `torch_dtype=torch.float16` (tên tham số cũ; float16 trên CPU chậm và có thể lỗi). `dtype="auto"` lấy đúng kiểu lưu trong checkpoint; `device_map="auto"` (cần thư viện `accelerate`) tự đặt mô hình lên GPU nếu có.

Thường dùng `AutoTokenizer` để nạp tokenizer:

<!-- run -->
```python
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)
```

<!-- output -->
```output
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
<!-- /output -->

Kết quả là dict gồm các khoá:

- [input_ids](https://huggingface.co/docs/transformers/glossary#input-ids): các token dưới dạng số.
- [attention_mask](https://huggingface.co/docs/transformers/glossary#attention-mask): token nào cần chú ý.
- `token_type_ids`: câu thứ nhất hay câu thứ hai (chỉ có ở mô hình như BERT, nhận cặp câu).

`tokenizer()` còn **nhận một danh sách câu, đệm và cắt để trả về batch có độ dài đều nhau**:

<!-- run -->
```python
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print({k: tuple(v.shape) for k, v in pt_batch.items()})
```

<!-- output -->
```output
{'input_ids': (2, 14), 'token_type_ids': (2, 14), 'attention_mask': (2, 14)}
```
<!-- /output -->

#### 3.3.2 AutoModel

Transformers nạp mô hình tiền huấn luyện theo cùng một cách đơn giản như `AutoTokenizer`: chỉ cần tên mô hình hoặc đường dẫn checkpoint là có đúng kiến trúc. Với phân loại văn bản, dùng `AutoModelForSequenceClassification` (thân mô hình + đầu phân loại):

<!-- run -->
```python
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_model.eval()
with torch.no_grad():
    pt_outputs = pt_model(**pt_batch)                 # ** mở dict thành tham số có tên
pt_predictions = torch.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)
print(pt_model.config.id2label)
for s, p in zip(["câu 1", "câu 2"], pt_predictions):
    print(s, "→", pt_model.config.id2label[int(p.argmax())], f"({float(p.max()):.2f})")
```

<!-- output -->
```output
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]])
{0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
câu 1 → 5 stars (0.77)
câu 2 → 5 stars (0.24)
```
<!-- /output -->

> **Biên tập:** Code của bản gốc truyền hai câu thành hai **tham số vị trí** `tokenizer("câu 1", "câu 2", ...)`, nên tokenizer hiểu đây là **một cặp câu** (một mẫu duy nhất, câu 2 nối sau `[SEP]`) chứ không phải batch hai câu, trong khi kết quả in trong bài lại có hai hàng xác suất. Phải truyền một **danh sách** như ở trên. Ghi chú của bản gốc "mô hình transformers mặc định cần nhiều câu, tokenizer thêm một chiều batch" đúng với `return_tensors="pt"`: kể cả một câu cũng trả về tensor có chiều batch bằng 1.

Đầu ra của mô hình là **logits** (điểm chưa chuẩn hoá); áp softmax lên logits mới được xác suất. Mô hình này dự đoán số sao (1 đến 5) của một bài đánh giá.

## 4 Bổ sung: sinh văn bản với mô hình hội thoại

Mục này do người biên soạn viết thêm, vì phần lớn việc dùng transformers hiện nay là chạy mô hình ngôn ngữ lớn. Mô hình hội thoại (instruct/chat) được huấn luyện với một **mẫu hội thoại** (chat template) riêng, đánh dấu vai trò và ranh giới từng lượt. Không được tự ghép chuỗi; dùng `apply_chat_template` để tokenizer chèn đúng các token đặc biệt:

<!-- run -->
```python
messages = [
    {"role": "system", "content": "Bạn là trợ lý dịch thuật. Chỉ trả lời bằng bản dịch."},
    {"role": "user", "content": "Dịch sang tiếng Anh: Mô hình nhỏ chạy được trên CPU."},
]
prompt_text = qwen_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(prompt_text)
```

<!-- output -->
```output
<|im_start|>system
Bạn là trợ lý dịch thuật. Chỉ trả lời bằng bản dịch.<|im_end|>
<|im_start|>user
Dịch sang tiếng Anh: Mô hình nhỏ chạy được trên CPU.<|im_end|>
<|im_start|>assistant
```
<!-- /output -->

`<|im_start|>` và `<|im_end|>` là token đặc biệt của Qwen đánh dấu đầu và cuối mỗi lượt; dòng cuối `<|im_start|>assistant` báo cho mô hình đến lượt nó trả lời. Sinh câu trả lời bằng `generate`:

<!-- run -->
```python
import time
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
print("số tham số:", f"{sum(p.numel() for p in llm.parameters()) / 1e6:.0f} triệu", "| kiểu:", llm.dtype)

def chat(messages, max_new_tokens=60):
    inputs = qwen_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    t = time.perf_counter()
    out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)   # giải mã tham lam: tái lập được
    new_tokens = out[0, inputs["input_ids"].shape[1]:]                            # bỏ phần prompt
    dt = time.perf_counter() - t
    return qwen_tok.decode(new_tokens, skip_special_tokens=True), len(new_tokens) / dt

answer, speed = chat(messages)
print(f"→ {answer}   ({speed:.0f} token/giây trên CPU)")
```

<!-- output -->
```output
số tham số: 494 triệu | kiểu: torch.bfloat16
→ Small model can run on the CPU.   (6 token/giây trên CPU)
```
<!-- /output -->

Mô hình 0,5 tỉ tham số dịch câu ngắn tốt, nhưng **hay bịa** khi được hỏi kiến thức. Hỏi chính khái niệm ở mục 1.3:

<!-- run -->
```python
answer, _ = chat([{"role": "user", "content": "Giải thích ngắn gọn attention mask trong mô hình Transformer là gì."}], max_new_tokens=80)
print(answer)
```

<!-- output -->
```output
Attention Mask trong mô hình Transformer là một thuật toán được sử dụng để cải thiện hiệu suất của các mô hình dự đoán văn bản. Đây là một bước quan trọng trong quá trình đào tạo và phân tích dữ liệu, giúp mô hình hiểu rõ hơn về ngữ cảnh và thông tin liên quan đến câu hỏi mà nó đang giải quyết.

Trong việc xử lý Attention Mask, mô hình Transformer
```
<!-- /output -->

Câu trả lời trên là ví dụ điển hình của "ảo giác" ở mô hình nhỏ: câu văn trôi chảy, tự tin, nhưng sai: attention mask không phải "thuật toán cải thiện hiệu suất", mà là vector 0/1 báo cho mô hình bỏ qua các token đệm (mục 1.3). Muốn dùng mô hình ngôn ngữ cho kiến thức chuyên môn cần mô hình lớn hơn, hoặc cung cấp tài liệu trong prompt (RAG).

Vài điều cần biết khi chạy mô hình lớn hơn:

- **Bộ nhớ**: mô hình $N$ tỉ tham số cần khoảng $2N$ GB ở bfloat16 chỉ để chứa trọng số (0,5 tỉ → khoảng 1 GB như trên; 7 tỉ → 14 GB). Lượng tử hóa 4 bit (xem [bài lượng tử hóa](dlnote/compress_quant_vi)) giảm xuống khoảng $N/2$ GB.
- **Tốc độ**: sinh văn bản bị giới hạn bởi băng thông bộ nhớ, mỗi token phải đọc lại toàn bộ trọng số; GPU nhanh hơn CPU hàng chục lần chủ yếu vì băng thông bộ nhớ cao hơn.
- **Phục vụ nhiều người dùng**: `generate` của transformers hợp cho thử nghiệm; khi cần thông lượng cao thì dùng máy chủ suy luận chuyên dụng như vLLM, SGLang hay llama.cpp.

## Tài liệu tham khảo

1. [Tài liệu HuggingFace Transformers](https://huggingface.co/docs/transformers/index) (bản gốc dẫn tài liệu v4.29 tiếng Trung)
2. [Khoá học LLM của HuggingFace](https://huggingface.co/learn/llm-course) (tên cũ: NLP Course)
3. [Transformers v5 release notes](https://github.com/huggingface/transformers/releases)
4. [NLP领域中的token和tokenization到底指的是什么](https://www.zhihu.com/question/64984731) (tiếng Trung)
5. [Chat templates](https://huggingface.co/docs/transformers/chat_templating)
