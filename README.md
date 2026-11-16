<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Giấy phép Creative Commons" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

# Hope for Future — Tự học AI bằng tiếng Việt

*Learn AI · Build Skills · Create a Better Tomorrow*

**Hope for Future** là một trang tự học trí tuệ nhân tạo bằng tiếng Việt, **phi thương mại**. Nội dung đi theo lộ trình **Toán nền tảng → Học máy → Học sâu → Học tăng cường → LLM**. Các khoá được xây từ những giáo trình mở có uy tín (dịch, trích nguyên văn hoặc tổng hợp lại, có ghi nguồn đầy đủ), rồi bổ sung thêm code chạy được, mô phỏng tương tác và ví dụ trên dữ liệu thật.

📖 **Đọc online:** https://nmc2004nd.github.io/easy-rl-vi/

Repo này ban đầu chỉ là bản dịch tiếng Việt của Sách Nấm EasyRL (vì vậy tên repo vẫn là `easy-rl-vi`). Bản dịch đó nay là một phần của khoá Học tăng cường.

## Các khoá học

| Khoá | Nội dung | Nguồn chính | Trạng thái |
| --- | --- | --- | --- |
| **Toán nền tảng** | Đại số tuyến tính, giải tích ma trận, xác suất, ước lượng tham số, tối ưu lồi, đối ngẫu | *Machine Learning cơ bản* (Vũ Hữu Tiệp), chương 1–4, 23–25 và phụ lục A | Đầy đủ |
| **Học máy** | 22 chương của sách: hồi quy, KNN, K-means, naive Bayes, gradient descent, mạng neuron, hệ thống gợi ý, SVD, PCA, LDA, SVM. Thêm 5 chủ đề bổ sung: đánh giá mô hình, cây quyết định, học tổ hợp, phân cụm nâng cao, tiền xử lý dữ liệu | *Machine Learning cơ bản* (Vũ Hữu Tiệp); các chủ đề bổ sung do người biên soạn viết | Đầy đủ |
| **Học tăng cường** | 16 chương EasyRL (từ MDP tới DQN, PPO, DDPG, học bắt chước), cùng bản tổng hợp 10 chương *Nền tảng toán học của RL* | Sách Nấm EasyRL (Datawhale); *Mathematical Foundations of RL* (Shiyu Zhao) | Đầy đủ |
| **Học sâu** | | | Sắp có |
| **LLM và ứng dụng** | | | Sắp có |

Ngoài ra còn có **Lộ trình học** cho toàn bộ site, **Bản đồ kiến thức** (sơ đồ tư duy Học máy và Học sâu, mỗi nút dẫn tới bài tương ứng) và **Quy ước ký hiệu** dùng chung.

## Cách học

Mỗi bài trong khoá Toán và Học máy có năm trang:

| Trang | Nội dung |
| --- | --- |
| **Bài đọc** | Lý thuyết và phân tích toán: đặt bài toán, suy ra lời giải, chỉ ra khi nào thuật toán thất bại |
| **Mô phỏng** | Trang tương tác chạy ngay trong trình duyệt: kéo thanh trượt, bấm từng bước, đổi dữ liệu |
| **Code từ đầu** | Cài đặt lại thuật toán chỉ với NumPy, rồi đối chiếu với thư viện |
| **Dùng thư viện** | Giải lại bằng scikit-learn hoặc PyTorch, so sánh các lựa chọn bằng kiểm định chéo |
| **Ví dụ áp dụng** | Một bài toán trọn vẹn trên dữ liệu thật (California, MNIST, Titanic, Ames, MovieLens, CIFAR-10, …) |

Khoá Học tăng cường có thêm ghi chú trọng tâm, bài tập, trang Code (chạy sẵn, kèm GIF) và các notebook Jupyter.

Mọi kết quả in ra trên các trang code đều được **chạy sẵn trên máy rồi nhúng vào trang**, nên đọc được ngay mà không cần cài gì. Mọi con số được dẫn trong bài viết đều lấy từ những lần chạy đó.

Gợi ý: học theo **Lộ trình học** trên trang chủ. Với mỗi bài, nếu muốn hiểu kỹ thì đi lần lượt Bài đọc → Mô phỏng → Code từ đầu → Dùng thư viện → Ví dụ áp dụng; nếu cần dùng ngay thì bắt đầu từ Mô phỏng và Dùng thư viện, rồi quay lại phần lý thuyết sau.

## Cấu trúc repo

| Thư mục | Nội dung |
| --- | --- |
| `docs/` | Trang web (docsify). `home_vi.md` là trang chủ, `_sidebar_vi.md` là mục lục của mọi khoá |
| `docs/math/`, `docs/ml/` | Khoá Toán nền tảng và khoá Học máy (năm trang cho mỗi bài) |
| `docs/chapter1/` … `docs/chapter16/` | 16 chương EasyRL đã dịch, kèm ghi chú, bài tập và trang Code |
| `docs/mathrl/` | Bản tổng hợp sách của Shiyu Zhao |
| `docs/viz/` | Các trang mô phỏng tương tác (HTML + JavaScript thuần) |
| `docs/code/py/` | Mã Python dùng chung cho các trang Code của khoá Học tăng cường |
| `docs/mindmap/` | Bản đồ kiến thức Học máy và Học sâu |
| `notebooks/` | 16 notebook Jupyter cài đặt thuật toán RL (đã dịch chú thích, markdown và log) |
| `papers/` | 20 bài đọc bài báo về DQN và Policy Gradient |
| `scripts/` | Script dựng nội dung: chuyển sách LaTeX sang Markdown (`mlcb/`), vẽ hình (`ml_extra/`), chạy code và nhúng kết quả (`build_code_pages.py`), dựng bản đồ kiến thức (`mindmap/`) |

## Đọc và chạy trên máy

Đọc offline (không cần mạng sau khi tải về):

```bash
git clone https://github.com/nmc2004nd/easy-rl-vi.git
cd easy-rl-vi/docs
python3 -m http.server 3000
# mở http://localhost:3000
```

Chạy lại code của một trang và cập nhật kết quả nhúng trong trang (cần Python 3.10+, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` ≥ 1.3; vài chương cần `torch`):

```bash
python3 -B scripts/build_code_pages.py docs/ml/prep_code_vi.md
```

Script chạy lần lượt các khối code có đánh dấu `<!-- run -->`, rồi ghi kết quả in ra và hình vẽ vào ngay dưới mỗi khối.

## Quy ước biên soạn

- **Ghi rõ đâu là của ai.** Nội dung dịch hoặc trích nguyên văn được giữ nguyên ý tác giả. Sửa lỗi của bản gốc được đánh dấu **"Ghi chú của người dịch"**; phần viết thêm được đánh dấu **"Ghi chú của người biên soạn"**.
- **Thuật ngữ** được dịch sang tiếng Việt và giữ tiếng Anh trong ngoặc ở lần đầu xuất hiện, ví dụ: tác tử (agent), chính sách (policy), hàm giá trị (value function), lợi tức (return). Tên chương của sách *Machine Learning cơ bản* giữ tiếng Anh như trong sách.
- **Thuật toán** được viết dạng pseudocode: từ khoá tiếng Anh (`for`, `while`, `if`), phép gán `←`, chú thích tiếng Việt.
- **Ký hiệu** của phần tổng hợp và các bài mới theo một chuẩn chung ([Quy ước ký hiệu](https://nmc2004nd.github.io/easy-rl-vi/#/notation_vi)). Các chương EasyRL giữ ký hiệu gốc, kèm bảng đối chiếu.
- **Tên người Trung Quốc** được ghi bằng phiên âm kèm chữ Hán, ví dụ: Wu Wenhao (吴文昊).
- Chữ trong một số hình vẽ sẵn của bản gốc vẫn là tiếng Trung vì không thể chỉnh sửa.

Nội dung có thể còn sai sót. Rất mong nhận được góp ý qua [Issues](https://github.com/nmc2004nd/easy-rl-vi/issues).

## Nguồn và ghi công

Người biên soạn không sở hữu và không nhận bất kỳ phần nội dung gốc nào là của mình. Nếu tác giả của một nguồn dưới đây không đồng ý với cách sử dụng, hoặc muốn nội dung được điều chỉnh, xin liên hệ qua [Issues](https://github.com/nmc2004nd/easy-rl-vi/issues); người biên soạn sẽ chỉnh sửa hoặc gỡ theo yêu cầu.

### Sách Nấm EasyRL (khoá Học tăng cường)

Bản dịch tiếng Việt **không chính thức** của [蘑菇书 EasyRL (Sách Nấm EasyRL)](https://github.com/datawhalechina/easy-rl), giáo trình nhập môn học tăng cường do cộng đồng [Datawhale](https://github.com/datawhalechina) biên soạn, theo giấy phép [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) của tác phẩm gốc: ghi công tác giả, không dùng cho mục đích thương mại, và phát hành lại dưới cùng giấy phép.

| Tác giả | Đóng góp |
|---|---|
| [Qi Wang (王琦)](https://github.com/qiwang067) | Thiết kế giáo trình (chương 1–12) |
| [Yiyuan Yang (杨毅远)](https://github.com/yyysjz1997) | Thiết kế bài tập và chương 13 |
| [John Jim (江季)](https://github.com/JohnJim0816) | Thực hành thuật toán |

Cùng các tác giả của từng bài đọc bài báo trong thư mục `papers/` (tên mỗi người được ghi ở cuối bài). Giáo trình gốc tham khảo [Deep Reinforcement Learning của Hung-yi Lee (李宏毅)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) (chương 4–11), [Intro to RL của Bolei Zhou (周博磊)](https://github.com/zhoubolei/introRL) (chương 1–2) và [khoá học của Li Kejiao (李科浇)](https://aistudio.baidu.com/aistudio/education/group/info/1335) (chương 3 và 12).

Repo gốc: https://github.com/datawhalechina/easy-rl · Bản online gốc: https://datawhalechina.github.io/easy-rl/ · Nếu thấy giáo trình hữu ích, hãy ⭐ repo gốc hoặc mua [sách giấy](https://item.jd.com/13075567.html).

### Machine Learning cơ bản (khoá Toán nền tảng và Học máy)

Văn bản của các chương được trích **nguyên văn** từ sách [*Machine Learning cơ bản*](https://github.com/tiepvupsu/ebookMLCB) của **Vũ Hữu Tiệp** (blog [machinelearningcoban.com](https://machinelearningcoban.com)), chuyển từ mã LaTeX của sách sang Markdown, giữ nguyên số chương, công thức, hình và bảng. Lỗi phát hiện trong sách được ghi ở phần ghi chú riêng, không sửa vào lời tác giả. Các trang Code, Thư viện, Ví dụ áp dụng, Mô phỏng và năm chủ đề bổ sung do người biên soạn viết thêm.

### Mathematical Foundations of Reinforcement Learning (khoá Học tăng cường)

Phần *Nền tảng toán học* là **bản tổng hợp tiếng Việt** (diễn giải lại, không dịch nguyên văn) của sách *Mathematical Foundations of Reinforcement Learning* của **Shiyu Zhao** (Springer và NXB Đại học Thanh Hoa, 2025). Tác giả công bố miễn phí bản PDF, slide và mã nguồn tại [GitHub MathFoundationRL](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning); hình được cắt từ bản PDF này và ghi nguồn dưới mỗi hình. Có bổ sung từ loạt ghi chú [Reinforcement Learning trên blog lyk-love](https://lyk-love.cn/tags/reinforcement-learning/) (CC BY-NC-SA 4.0), đánh dấu *(theo lyk-love)*.

### Mã nguồn và bản đồ kiến thức

- Mã của các trang Code khoá Học tăng cường chuyển thể từ [minrl](https://github.com/10-OASIS-01/minrl) (giấy phép MIT).
- Bản đồ kiến thức chuyển từ các sơ đồ [Machine Learning Mindmap](https://github.com/dformoso/machine-learning-mindmap) và [Deep Learning Mindmap](https://github.com/dformoso/deeplearning-mindmap) của Daniel Martinez; tên mục giữ tiếng Anh, phần giải thích được dịch.
- Dữ liệu cho các bài Ví dụ áp dụng lấy từ [scikit-learn](https://scikit-learn.org/stable/datasets.html), [OpenML](https://www.openml.org), [MovieLens](https://grouplens.org/datasets/movielens/) và [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

## Tài liệu tham khảo

Ngoài các nguồn chính ở trên, những tài liệu sau được dùng để viết các phần bổ sung và rất đáng đọc thêm:

**Sách**

- G. James, D. Witten, T. Hastie, R. Tibshirani, J. Taylor, [*An Introduction to Statistical Learning*](https://www.statlearning.com) (có bản Python, miễn phí).
- T. Hastie, R. Tibshirani, J. Friedman, [*The Elements of Statistical Learning*](https://web.stanford.edu/~hastie/ElemStatLearn/), 2009 (miễn phí).
- C. Bishop, [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), 2006.
- K. Murphy, [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html), 2022 (miễn phí).
- R. Sutton, A. Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2018 (miễn phí).
- I. Goodfellow, Y. Bengio, A. Courville, [*Deep Learning*](https://www.deeplearningbook.org), 2016 (miễn phí).

**Khoá học**

- Stanford [CS229: Machine Learning](https://cs229.stanford.edu).
- David Silver, [UCL Course on RL](https://www.davidsilver.uk/teaching/).
- OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com).

**Tài liệu thư viện**

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html), [NumPy](https://numpy.org/doc/stable/), [PyTorch](https://pytorch.org/docs/stable/index.html), [Gymnasium](https://gymnasium.farama.org).

Bài báo gốc của từng thuật toán được dẫn ngay trong các bài liên quan. Danh sách bài báo của các chủ đề bổ sung nằm ở cuối trang giới thiệu khoá Học máy.

## Giấy phép

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Giấy phép Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />Repo được phát hành theo giấy phép <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Ghi công – Phi thương mại – Chia sẻ tương tự 4.0 Quốc tế (CC BY-NC-SA 4.0)</a>, giống Sách Nấm EasyRL. Xem file [LICENSE](LICENSE). Phần nội dung lấy từ các nguồn khác vẫn thuộc về tác giả của chúng và tuân theo giấy phép gốc: *Machine Learning cơ bản* theo giấy phép do tác giả công bố trong [repo sách](https://github.com/tiepvupsu/ebookMLCB); mã chuyển thể từ minrl theo giấy phép MIT.
