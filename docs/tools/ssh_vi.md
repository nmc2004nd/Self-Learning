# SSH: đăng nhập máy chủ từ xa

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/1-math_ml_basic/ssh远程登录服务.md), [kho mã](https://github.com/harleyszhang/dl_note), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); phần dịch có chỉnh sửa). Số mục giữ như bản gốc. Chỗ bản gốc sai, thiếu hoặc đã lỗi thời được sửa ngay trong bài và ghi ở ô **Biên tập**. Các mục có chữ **Bổ sung** do người biên soạn viết thêm.

`SSH` (Secure Shell Protocol) là giao thức mạng có mã hoá, dùng để kết nối máy khách (client) với máy chủ (server) qua mạng. Trường hợp điển hình là từ máy tính cá nhân kết nối SSH vào máy chủ có GPU để làm việc. Windows, macOS và Linux đều có sẵn SSH client.

> **Biên tập:** Bản gốc viết SSH client trên Windows "dùng không tốt nên thường thay bằng `Xshell`". Từ Windows 10 (bản 1809) và Windows 11, OpenSSH client có sẵn: mở PowerShell và gõ `ssh` như trên Linux, nên không cần cài thêm. Xshell, PuTTY hay MobaXterm vẫn dùng được nếu thích giao diện đồ hoạ.

## 1 Chuẩn bị

### 1.1 SSH client

Một kết nối SSH cần hai thành phần: client và server. SSH client là phần mềm trên máy của bạn. Phía server có một tiến trình nền gọi là SSH daemon (`sshd`), liên tục lắng nghe một cổng TCP (mặc định 22) để chờ client kết nối. Khi client kết nối, `sshd` trả lời bằng phiên bản phần mềm và giao thức nó hỗ trợ, hai bên trao đổi thông tin định danh và thiết lập kênh mã hoá. Nếu thông tin xác thực đúng, SSH mở một phiên làm việc mới.

macOS, Windows 10/11 và hầu hết bản phân phối Linux có sẵn SSH client. Kiểm tra bằng `ssh -V`:

```bash
$ ssh -V
OpenSSH_9.6p1 Ubuntu-3ubuntu13.20, OpenSSL 3.0.13 30 Jan 2024
```

### 1.2 SSH server

Trên Linux có nhiều cách kiểm tra SSH server có chạy không, ví dụ xem có tiến trình `sshd`:

```bash
ps -e | grep sshd
```

Có dòng `sshd` trong kết quả là server đang chạy. Trên Ubuntu/Debian dùng systemd, cách rõ ràng hơn là hỏi trạng thái dịch vụ, và cài nếu chưa có:

```bash
systemctl is-active ssh                 # in "active" nếu đang chạy
sudo apt install openssh-server         # cài nếu chưa có
sudo systemctl enable --now ssh         # bật và cho chạy cùng hệ thống
```

> **Biên tập:** Bản gốc đề xuất kiểm tra server trên Ubuntu bằng `ssh localhost`. Lệnh đó kiểm tra cả client lẫn server cùng lúc và còn hỏi mật khẩu; `systemctl is-active ssh` trả lời thẳng. Tên dịch vụ là `ssh` trên Ubuntu/Debian và `sshd` trên CentOS/Fedora.

## 2 Đăng nhập bằng mật khẩu

Cách dùng điển hình, chỉ cần một lệnh:

```bash
# cổng mặc định là 22; nếu tên người dùng ở máy mình và máy chủ giống nhau thì bỏ được username@
ssh username@host_ip
# chỉ định cổng khác
ssh -p port username@host_ip
```

Lần đầu kết nối tới một máy chủ, SSH hỏi bạn có tin khoá của máy chủ không. Kiểm tra fingerprint (nếu quản trị viên cung cấp) rồi gõ `yes`, sau đó nhập mật khẩu:

```bash
The authenticity of host '203.0.113.7 (203.0.113.7)' can't be established.
ED25519 key fingerprint is SHA256:QW5nscbIadeqedp7ByOSUF+Z45rxWGYJvAs3TTmTb0M.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```

Khoá của máy chủ được ghi vào `~/.ssh/known_hosts`. Những lần sau, nếu khoá máy chủ đổi, SSH sẽ cảnh báo `REMOTE HOST IDENTIFICATION HAS CHANGED`: có thể máy chủ được cài lại, nhưng cũng có thể có người đang chặn giữa đường, nên cần hỏi lại quản trị viên trước khi xoá dòng cũ bằng `ssh-keygen -R host_ip`.

> **Biên tập:** Bản gốc dùng một địa chỉ IP thật trong ví dụ; đã thay bằng địa chỉ dành cho tài liệu `203.0.113.7` (RFC 5737).

## 3 Đăng nhập bằng khoá công khai

Cách trên phải **nhập mật khẩu** mỗi lần đăng nhập. SSH còn cho đăng nhập bằng khoá công khai, vừa bỏ được bước nhập mật khẩu vừa an toàn hơn (có thể tắt hẳn đăng nhập bằng mật khẩu trên máy chủ để chống dò mật khẩu).

> **Biên tập:** Bản gốc gọi cách ở mục 2 là đăng nhập "bằng mật khẩu (khoá riêng)". Mục 2 chỉ dùng mật khẩu; khoá riêng là thứ của mục 3.

Nguyên lý: trên máy của mình tạo một **cặp khoá**, khoá công khai (public key) và khoá riêng (private key), rồi chép **khoá công khai** lên máy chủ. Khi đăng nhập, máy chủ gửi một thử thách gắn với phiên kết nối; client dùng **khoá riêng** để ký lên dữ liệu đó và gửi chữ ký về. Máy chủ dùng khoá công khai đã lưu để kiểm tra chữ ký; đúng thì cho đăng nhập. Khoá riêng **không bao giờ rời khỏi máy của bạn**.

> **Biên tập:** Bản gốc có câu "chỉ cần gửi khoá riêng cho máy chủ, máy chủ kiểm tra khoá riêng có khớp với khoá công khai không". Câu này sai và nguy hiểm: ai có khoá riêng thì đăng nhập được thay bạn, nên khoá riêng không được gửi đi đâu cả. Bản gốc cũng mô tả cơ chế là "client mã hoá chuỗi ngẫu nhiên bằng khoá riêng, máy chủ giải mã bằng khoá công khai"; đó là cách nói gần đúng của giao thức SSH-1 cũ. SSH-2 dùng **chữ ký số** trên dữ liệu gắn với phiên (session identifier), như mô tả ở trên.

SSH hỗ trợ nhiều thuật toán khoá: RSA, ECDSA, Ed25519 (DSA đã bị loại khỏi OpenSSH). Nên dùng **Ed25519**: khoá ngắn, nhanh và là kiểu mặc định của `ssh-keygen` từ OpenSSH 9.5. Chỉ dùng RSA (ít nhất 3072 bit) khi phải kết nối tới hệ thống cũ không hỗ trợ Ed25519.

> **Biên tập:** Bản gốc chọn RSA "vì là thuật toán mặc định của giao thức SSH". Điều này đã cũ: OpenSSH 9.5 (2023) đổi kiểu mặc định của `ssh-keygen` sang Ed25519, và chữ ký `ssh-rsa` dùng SHA-1 đã bị tắt mặc định từ OpenSSH 8.8.

`ssh-keygen` là công cụ tạo cặp khoá. Hai tham số hay dùng:

- `-t`: kiểu khoá (`ed25519`, `rsa`...).
- `-b`: độ dài khoá (chỉ cần với RSA, ví dụ `-b 4096`).

### 3.1 Làm từng bước

Trên **máy của mình** chạy `ssh-keygen -t ed25519`. Lệnh hỏi nơi lưu khoá (Enter để dùng mặc định) và passphrase để mã hoá khoá riêng (nên đặt; nếu để trống thì ai lấy được tệp khoá riêng là dùng được ngay). Kết quả thật khi chạy:

```text
$ ssh-keygen -t ed25519 -C "ban@may-ca-nhan"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/ban/.ssh/id_ed25519):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/ban/.ssh/id_ed25519
Your public key has been saved in /home/ban/.ssh/id_ed25519.pub
The key fingerprint is:
SHA256:7XPjcYTM3bEQFTkCOAlMdduln82XqB/xLb/3rej3ZZI ban@may-ca-nhan
The key's randomart image is:
+--[ED25519 256]--+
|      ooo.oo...+o|
|       . +. o.+o |
|          .. +...|
|         . o o+o*|
|        S . +oo==|
|         .  ..o.o|
|          o.+.E.+|
|           +.=o=+|
|           .+o.+O|
+----[SHA256]-----+
```

Thư mục `~/.ssh/` có thêm hai tệp: `id_ed25519.pub` (khoá công khai, chép đi đâu cũng được) và `id_ed25519` (khoá riêng, chỉ chủ sở hữu đọc được, quyền `-rw-------`). `-C` chỉ là chú thích để nhận ra khoá.

> **Biên tập:** Bản gốc dùng ảnh chụp màn hình lệnh `ssh-keygen -t rsa -b 4096`; thay bằng kết quả chạy thật của lệnh Ed25519 ở trên (đường dẫn đổi thành `/home/ban`; khoá tạo ra chỉ để minh hoạ và đã xoá). Bản gốc cũng khuyên để trống passphrase; nếu để trống thì nên dùng `ssh-agent` hoặc ít nhất giữ quyền tệp chặt.

Sau đó thêm nội dung khoá công khai vào tệp `~/.ssh/authorized_keys` trên máy chủ. Nếu máy chủ chưa có thư mục `.ssh` thì tạo trước:

```bash
# 1. Trên máy của mình: xem khoá công khai và sao chép
cat ~/.ssh/id_ed25519.pub
# 2. Trên máy chủ: tạo thư mục và tệp nếu chưa có
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
# 3. Dán khoá công khai thành một dòng mới trong authorized_keys, lưu lại
```

Các bước trên viết ra để hiểu cơ chế. Trong thực tế, hai lệnh sau làm hết (chỉ phải nhập mật khẩu máy chủ một lần):

```bash
ssh-keygen -t ed25519                                    # tạo cặp khoá
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@host_ip    # chép khoá công khai lên máy chủ
```

Nếu làm xong mà vẫn bị hỏi mật khẩu, nguyên nhân hay gặp nhất là **quyền thư mục**: `sshd` từ chối dùng `authorized_keys` khi thư mục hay tệp ghi được bởi người khác. Trên máy chủ, sửa bằng:

```bash
chmod go-w ~                         # thư mục nhà: người khác không ghi được
chmod 700 ~/.ssh                     # chỉ chủ sở hữu vào được .ssh
chmod 600 ~/.ssh/authorized_keys     # chỉ chủ sở hữu đọc ghi được
```

Kết quả đúng trông như sau (`ls -al ~/.ssh/`, đã bỏ cột tên người dùng):

```text
drwx------  2  4096 Jun 17 11:33 .
drwxr-xr-x 24  4096 Dec 11 15:24 ..
-rw-------  1  1521 Dec 11 15:31 authorized_keys
-rw-------  1   444 Mar 20  2025 id_ed25519
-rw-r--r--  1   122 Mar 20  2025 id_ed25519.pub
-rw-------  1  1420 Apr 28  2025 known_hosts
```

Nếu vẫn lỗi, chạy `ssh -v username@host_ip` để xem client đã thử những khoá nào, và xem log trên máy chủ (`journalctl -u ssh`).

> **Biên tập:** Bản gốc ghi đường dẫn tuyệt đối của tác giả (`/home/honggao/.ssh/authorized_keys`) và đặt `chmod 755 ~`; ở đây dùng `~` và `go-w` cho đúng với mọi người dùng. Bản gốc cũng dặn "đừng chép tay tệp khoá công khai": chép tay vẫn được, miễn khoá nằm trọn trên **một dòng** (lỗi hay gặp là xuống dòng giữa khoá khi dán). Ảnh chụp `ls -al` của bản gốc được gõ lại thành khối văn bản.

### 3.2 Một lệnh duy nhất

Khi máy không có `ssh-copy-id` (ví dụ PowerShell trên Windows), có thể làm bằng một lệnh:

```bash
ssh username@host "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys" < ~/.ssh/id_ed25519.pub
```

Chỉ cần nội dung khoá công khai nằm trong `authorized_keys` của máy chủ là cài đặt xong; từ đó đăng nhập không cần mật khẩu.

Đẩy code lên `GitHub` qua SSH dùng đúng cơ chế này: dán khoá công khai vào **Settings › SSH and GPG keys** của tài khoản GitHub, rồi kiểm tra bằng `ssh -T git@github.com`.

> **Biên tập:** Bản gốc viết "thêm khoá công khai vào tệp `authorized_keys` của GitHub"; trên GitHub ta không sửa tệp nào, mà thêm khoá trong trang cài đặt tài khoản.

## 4 Kết nối từ xa bằng VS Code

VS Code kết nối được tới máy chủ qua tiện ích **Remote - SSH**: mở thư mục trên máy chủ, chỉnh sửa, chạy terminal và debug như trên máy mình.

1. Cài tiện ích [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).
2. Nhấn `Ctrl + Shift + P` (macOS: `Cmd + Shift + P`) mở bảng lệnh, gõ `Remote-SSH: Open SSH Configuration File...` rồi chọn tệp `~/.ssh/config`. Cũng có thể vào thanh bên **Remote Explorer**, mục **SSH**, bấm biểu tượng bánh răng.
3. Khai báo máy chủ trong `~/.ssh/config`. Ví dụ dưới có một máy trung gian (jump host) và máy đích có GPU chỉ vào được qua máy trung gian:

```bash
# Xem thêm: man ssh_config
Host JumpMachine
    HostName 203.0.113.10
    User username             # tên người dùng trên máy trung gian

Host T4
    HostName 10.0.0.25        # IP của máy đích (trong mạng nội bộ)
    User username             # tên người dùng trên máy đích
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump JumpMachine     # đi qua máy trung gian
```

4. Tạo khoá và thêm khoá công khai vào `authorized_keys` của **cả hai** máy như mục 3.
5. Lưu tệp, rồi trong **Remote Explorer › SSH** chọn `T4` để kết nối. Từ terminal cũng chỉ cần gõ `ssh T4`, mọi tham số lấy từ tệp config.

> **Biên tập:** Bản gốc dùng `ProxyCommand ssh -q -W %h:%p JumpMachine`. Từ OpenSSH 7.3 có `ProxyJump`, ngắn hơn và làm cùng việc (dùng được cả trên Windows). Ảnh chụp màn hình VS Code của bản gốc (có đường dẫn thư mục cá nhân của tác giả và giao diện tiếng Trung) được thay bằng mô tả các bước.

## 5 Bổ sung: vài thao tác hay dùng khi làm việc trên máy chủ GPU

Mục này do người biên soạn viết thêm.

- **Chép tệp**: `scp file.pt T4:~/ckpt/` hoặc tốt hơn `rsync -avP data/ T4:~/data/` (chép tiếp được khi đứt mạng, chỉ chép phần thay đổi).
- **Giữ tiến trình chạy khi mất kết nối**: chạy huấn luyện bên trong `tmux` (`tmux new -s train`, thoát tạm bằng `Ctrl+b d`, vào lại bằng `tmux attach -t train`). Nếu chạy trực tiếp, mất kết nối SSH là tiến trình bị dừng.
- **Chuyển tiếp cổng** để mở Jupyter hay TensorBoard chạy trên máy chủ: `ssh -L 8888:localhost:8888 T4`, rồi mở `http://localhost:8888` trên máy mình.
- **Giữ kết nối không bị ngắt khi để lâu**: thêm `ServerAliveInterval 60` vào `Host` trong `~/.ssh/config`.
- **Dùng khoá có passphrase mà không phải gõ lại mỗi lần**: `ssh-add ~/.ssh/id_ed25519` (nạp khoá vào `ssh-agent` một lần mỗi phiên).

## Tài liệu tham khảo

1. [Wikipedia - Secure Shell](https://en.wikipedia.org/wiki/Secure_Shell)
2. [How to Use ssh-keygen to Generate a New SSH Key?](https://www.ssh.com/academy/ssh/keygen)
3. [SSH原理与运用（一）：远程登录](https://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html) (Nguyễn Nhất Phong, tiếng Trung)
4. [OpenSSH 9.5 release notes](https://www.openssh.com/txt/release-9.5) (đổi kiểu khoá mặc định sang Ed25519)
5. [VS Code: Remote Development using SSH](https://code.visualstudio.com/docs/remote/ssh)
