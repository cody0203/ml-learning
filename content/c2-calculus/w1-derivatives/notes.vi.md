## Nền tảng đạo hàm

Nhóm này giới thiệu vì sao đạo hàm cần cho ML.
Nó nối loss, độ dốc trung bình, độ dốc tại điểm,
tiếp tuyến và ký hiệu đạo hàm thành một nền tảng.

### Động lực học máy

#### Trực giác
Huấn luyện là thử, đo sai, rồi sửa.
Mô hình dự đoán giá nhà hoặc nhãn câu.
Loss đo mức sai của dự đoán.
Đạo hàm cho biết loss đổi ra sao
khi một trọng số thay đổi.

#### Định nghĩa và công thức
Với một tham số, tín hiệu huấn luyện là
$\frac{d}{dw}J(w)$.
Với nhiều tham số, ta dùng gradient.
Dấu cho hướng; độ lớn cho độ nhạy.

#### Ví dụ giải chi tiết
Giả sử $J(w)=(w-3)^2$.
Khi đó $J'(w)=2(w-3)$.
Tại $w=1$, độ dốc là $-4$.
Với learning rate $0.1$,
$w_{\text{new}}=1-0.1(-4)=1.4$.
Bước cập nhật tiến về $3$.

#### NumPy / Python snippet
```python
w = 1.0
lr = 0.1
grad = 2*(w - 3)
w = w - lr*grad
```

> ⚠️ Sai lầm: dùng sai số dự đoán trực tiếp
> mà không lấy đạo hàm của loss.

#### Tóm tắt
- Đạo hàm nối loss với cập nhật tham số.
- Độ dốc dương gợi ý đi sang trái.
- Độ dốc âm gợi ý đi sang phải.

### Nhập môn đạo hàm

#### Trực giác
Tốc độ trung bình trong năm giây rất hữu ích.
Tốc độ đúng tại một thời điểm cần giới hạn.
Các slide thu nhỏ khoảng thời gian.
Độ dốc dây cung tiến tới độ dốc điểm.

#### Định nghĩa và công thức
Độ dốc trung bình từ $a$ đến $b$ là
$\frac{x(b)-x(a)}{b-a}$.
Đạo hàm là giá trị giới hạn.
Với vị trí $x(t)$, đạo hàm là vận tốc.

#### Ví dụ giải chi tiết
Từ bảng, $x(10)=122$ và $x(15)=202$.
Vận tốc trung bình trên $[10,15]$ là
$\frac{202-122}{15-10}=16$ m/s.
Dùng $x(12)=155$ và $x(13)=170$
cho ước lượng gần $12.5$ là $15$ m/s.

#### NumPy / Python snippet
```python
t = [12, 13]
x = [155, 170]
speed = (x[1]-x[0])/(t[1]-t[0])
```

> ⚠️ Sai lầm: kết luận xe có một vận tốc
> cho cả chuyến khi các khoảng tăng khác nhau.

#### Tóm tắt
- Đạo hàm tinh chỉnh tốc độ trung bình.
- Khoảng nhỏ cho thông tin cục bộ.
- Đơn vị ở đây là mét trên giây.

### Đạo hàm và tiếp tuyến

#### Trực giác
Tiếp tuyến chạm đường cong tại một điểm.
Độ dốc của nó là đạo hàm ở điểm đó.
Gần điểm chạm, đường cong và tiếp tuyến
cho xấp xỉ bậc nhất giống nhau.

#### Định nghĩa và công thức
Tại $x=a$, tiếp tuyến là
$y=f(a)+f'(a)(x-a)$.
Đây cũng là xấp xỉ tuyến tính cục bộ.

#### Ví dụ giải chi tiết
Cho $f(x)=x^2$ và $a=1$.
Ta có $f(1)=1$ và $f'(1)=2$.
Tiếp tuyến là $y=1+2(x-1)$.
Tại $x=1.1$, dự đoán là $1.2$.
Giá trị thật là $1.21$.

#### NumPy / Python snippet
```python
def tangent_at_one(x):
    return 1 + 2*(x-1)
```

> ⚠️ Sai lầm: lấy đường qua hai điểm xa
> rồi gọi nó là tiếp tuyến.

#### Tóm tắt
- Độ dốc tiếp tuyến bằng đạo hàm.
- Tiếp tuyến là mô tả cục bộ.
- Xấp xỉ tuyến tính chỉ tốt gần điểm gốc.

## Cực trị và độ dốc bằng không

Nhóm này giải thích ý nghĩa của độ dốc bằng không.
Tiếp tuyến ngang rất quan trọng,
nhưng vẫn phải phân loại bằng lân cận hoặc biên.

### Độ dốc, cực đại và cực tiểu

#### Trực giác
Ở đỉnh đồi hoặc đáy thung lũng trơn,
đường cong tạm thời không tăng hay giảm.
Vì vậy tiếp tuyến nằm ngang.
Độ dốc bằng không chỉ là ứng viên.

#### Định nghĩa và công thức
Với cực trị trong miền và trơn,
$f'(a)=0$.
Cần kiểm tra giá trị lân cận,
dấu của đạo hàm, hoặc đạo hàm bậc hai.

#### Ví dụ giải chi tiết
Với $f(x)=(x-4)^2+7$,
$f'(x)=2(x-4)$.
Giải $f'(x)=0$ được $x=4$.
Ta có $f(3)=8$, $f(4)=7$, $f(5)=8$.
Vậy $x=4$ là cực tiểu.

#### NumPy / Python snippet
```python
xs = [3, 4, 5]
vals = [(x-4)**2 + 7 for x in xs]
```

> ⚠️ Sai lầm: mọi điểm dốc không
> đều không phải cực tiểu; ví dụ $x^3$.

#### Tóm tắt
- Độ dốc bằng không tạo điểm tới hạn.
- Phải phân loại bằng ngữ cảnh.
- Biên của miền cần kiểm tra riêng.

### Ký hiệu đạo hàm

#### Trực giác
Mỗi ký hiệu nhấn mạnh một ý.
$f'(x)$ xem đạo hàm như hàm mới.
$\frac{dy}{dx}$ nhấn mạnh tỉ lệ đổi
của đầu ra theo đầu vào.

#### Định nghĩa và công thức
Nếu $y=f(x)$, thì
$f'(x)=\frac{d}{dx}f(x)=\frac{dy}{dx}$.
Tại một điểm có thể viết $f'(2)$
hoặc $\left.\frac{dy}{dx}\right|_{x=2}$.

#### Ví dụ giải chi tiết
Cho $y=3x^2-1$.
Ký hiệu Lagrange: $f'(x)=6x$.
Ký hiệu Leibniz: $\frac{dy}{dx}=6x$.
Tại $x=2$, cả hai bằng $12$.

#### NumPy / Python snippet
```python
def dy_dx(x):
    return 6*x
```

> ⚠️ Sai lầm: xem $dy$ và $dx$
> như hai số hữu hạn tùy ý.

#### Tóm tắt
- Dấu phẩy trên rất gọn.
- Leibniz thể hiện biến rõ hơn.
- Hai cách viết nói cùng một đạo hàm.

## Đạo hàm các hàm cơ bản

Nhóm này xây dựng bảng đạo hàm đầu tiên.
Nó bao gồm hằng số, đường thẳng, bậc hai,
quy tắc lũy thừa và hàm nghịch đảo $1/x$.

### Đạo hàm của hằng số và đường thẳng

#### Trực giác
Hàm hằng là đường phẳng ngang.
Đường thẳng tăng cùng một lượng ở mọi nơi.
Do đó đạo hàm của đường thẳng không đổi.

#### Định nghĩa và công thức
$\frac{d}{dx}c=0$.
Nếu $f(x)=ax+b$, thì $f'(x)=a$.
Hệ số chặn $b$ chỉ tịnh tiến đồ thị.

#### Ví dụ giải chi tiết
Với $f(x)=-3x+7$,
độ dốc luôn bằng $-3$.
Tại $x=0$, $x=5$, hay $x=-2$,
đạo hàm vẫn là $-3$.

#### NumPy / Python snippet
```python
import numpy as np
x = np.array([-2, 0, 5])
derivative = np.full_like(x, -3)
```

> ⚠️ Sai lầm: đạo hàm của $b$
> thành $b$ thay vì $0$.

#### Tóm tắt
- Hằng số có đạo hàm bằng không.
- Đạo hàm của $ax+b$ là $a$.
- Đường thẳng có tốc độ đổi không đổi.

### Đạo hàm của hàm bậc hai

#### Trực giác
Với $x^2$, đồ thị càng xa gốc
thì càng dốc.
Bên trái độ dốc âm,
ở đỉnh bằng không,
bên phải độ dốc dương.

#### Định nghĩa và công thức
$\frac{d}{dx}x^2=2x$.
Tổng quát hơn,
$\frac{d}{dx}(ax^2+bx+c)=2ax+b$.

#### Ví dụ giải chi tiết
Cho $f(x)=2x^2-4x+1$.
Khi đó $f'(x)=4x-4$.
Tại $x=3$, độ dốc bằng $8$.
Tại $x=1$, tiếp tuyến nằm ngang.

#### NumPy / Python snippet
```python
def quad_grad(x):
    return 4*x - 4
```

> ⚠️ Sai lầm: quên hệ số đứng trước $x^2$.

#### Tóm tắt
- Độ dốc của bậc hai là tuyến tính.
- Đỉnh nằm tại nơi đạo hàm bằng không.
- Hệ số làm thay đổi độ dốc.

### Đạo hàm đa thức bậc cao

#### Trực giác
Đa thức được ghép từ các lũy thừa.
Quy tắc lũy thừa xử lý từng hạng tử.
Vì vậy đường cong phức tạp vẫn dễ tính.

#### Định nghĩa và công thức
Với số nguyên $n$,
$\frac{d}{dx}x^n=nx^{n-1}$.
Lấy đạo hàm từng hạng tử rồi cộng lại.
Hằng số biến mất.

#### Ví dụ giải chi tiết
Với $p(x)=4x^5-2x^3+x-9$,
$p'(x)=20x^4-6x^2+1$.
Tại $x=1$, độ dốc là
$20-6+1=15$.

#### NumPy / Python snippet
```python
def pprime(x):
    return 20*x**4 - 6*x**2 + 1
```

> ⚠️ Sai lầm: giảm số mũ
> nhưng quên nhân với số mũ cũ.

#### Tóm tắt
- Mỗi đơn thức dùng quy tắc lũy thừa.
- Đạo hàm đa thức vẫn là đa thức.
- Hằng số có đạo hàm bằng không.

### Đạo hàm các lũy thừa khác

#### Trực giác
Số mũ âm tạo đường cong nghịch đảo.
Quy tắc lũy thừa vẫn dùng được,
nhưng miền xác định có thể loại $0$.

#### Định nghĩa và công thức
$x^{-1}=1/x$.
Theo quy tắc lũy thừa,
$\frac{d}{dx}x^{-1}=-x^{-2}$.
Vì vậy đạo hàm của $1/x$ là $-1/x^2$.

#### Ví dụ giải chi tiết
Tại $x=2$,
đạo hàm của $1/x$ là
$-1/2^2=-1/4$.
Đồ thị đang giảm,
nên dấu âm là hợp lý.

#### NumPy / Python snippet
```python
def reciprocal_grad(x):
    return -1/(x*x)
```

> ⚠️ Sai lầm: viết $1/x^2$
> và làm mất dấu âm.

#### Tóm tắt
- Số mũ âm vẫn theo quy tắc lũy thừa.
- Cần chú ý miền gần $0$.
- Với $x>0$, hàm nghịch đảo đang giảm.

## Hàm ngược, lượng giác, mũ và log

Nhóm này mở rộng ra ngoài đa thức.
Ta gặp độ dốc của hàm ngược,
sóng lượng giác, cơ số $e$, hàm mũ và log.

### Đạo hàm của hàm ngược

#### Trực giác
Hàm ngược đổi vai trò đầu vào và đầu ra.
Đồ thị phản chiếu qua đường $y=x$.
Phản chiếu làm độ dốc thành nghịch đảo.

#### Định nghĩa và công thức
Nếu $y=f(x)$ và $g=f^{-1}$,
thì $g'(y)=1/f'(x)$.
Điều này cần $f'(x)\ne0$ cục bộ.

#### Ví dụ giải chi tiết
Xét $f(x)=x^2$ trên $x>0$.
Hàm ngược là $g(y)=\sqrt y$.
Tại $x=2$, $y=4$ và $f'(2)=4$.
Do đó $g'(4)=1/4$.

#### NumPy / Python snippet
```python
y = 4.0
gprime = 1/(2*y**0.5)
```

> ⚠️ Sai lầm: dùng quy tắc hàm ngược
> khi hàm gốc không một-một.

#### Tóm tắt
- Độ dốc hàm ngược là nghịch đảo.
- Phải ghép đúng cặp điểm $(x,y)$.
- Tính khả nghịch cục bộ rất quan trọng.

### Đạo hàm lượng giác

#### Trực giác
Sóng sin dốc nhất tại điểm cắt trục.
Nó phẳng tại đỉnh và đáy.
Mẫu độ dốc đó chính là cos.

#### Định nghĩa và công thức
$\frac{d}{dx}\sin x=\cos x$.
$\frac{d}{dx}\cos x=-\sin x$.
Góc phải tính bằng radian.

#### Ví dụ giải chi tiết
Với $f(x)=3\sin x-2\cos x$,
$f'(x)=3\cos x+2\sin x$.
Tại $x=0$, ta được $f'(0)=3$.

#### NumPy / Python snippet
```python
import numpy as np
def trig_grad(x):
    return 3*np.cos(x) + 2*np.sin(x)
```

> ⚠️ Sai lầm: dùng độ thay vì radian
> trong công thức giải tích.

#### Tóm tắt
- Sin có đạo hàm là cos.
- Cos có đạo hàm là âm sin.
- Radian là đơn vị của giải tích.

### Ý nghĩa của hằng số e

#### Trực giác
Số $e$ xuất hiện khi lãi được ghép
ngày càng thường xuyên.
Nó là cơ số tự nhiên cho tăng trưởng liên tục.

#### Định nghĩa và công thức
$e=\lim_{n\to\infty}(1+1/n)^n$.
Các slide so sánh ghép lãi theo năm,
nửa năm, và nhiều lần hơn.

#### Ví dụ giải chi tiết
Với $n=10$,
$(1+1/10)^{10}\approx2.594$.
Với $n=1000$,
$(1+1/1000)^{1000}\approx2.717$.
Các giá trị tiến tới $2.71828\ldots$.

#### NumPy / Python snippet
```python
for n in [10, 1000]:
    print((1 + 1/n)**n)
```

> ⚠️ Sai lầm: nghĩ $e$ chỉ là lựa chọn
> tùy tiện để công thức đẹp hơn.

#### Tóm tắt
- $e$ là giới hạn tăng trưởng.
- Nó gắn với lãi kép liên tục.
- Nó làm đạo hàm mũ trở nên đơn giản.

### Đạo hàm của $e^x$

#### Trực giác
$e^x$ có tốc độ đổi bằng chính nó.
Chiều cao của đường cong bằng độ dốc.
Vì vậy nó mô hình hóa tăng trưởng tự nhiên.

#### Định nghĩa và công thức
$\frac{d}{dx}e^x=e^x$.
Với $a e^x$, đạo hàm là $a e^x$.

#### Ví dụ giải chi tiết
Cho $f(x)=5e^x$.
Khi đó $f'(x)=5e^x$.
Tại $x=0$, độ dốc là $5$.
Tại $x=2$, độ dốc là $5e^2$.

#### NumPy / Python snippet
```python
import numpy as np
def exp_grad(x):
    return 5*np.exp(x)
```

> ⚠️ Sai lầm: xem $e^x$
> giống lũy thừa $x^e$.

#### Tóm tắt
- Hàm mũ tự nhiên tự đạo hàm.
- Nhân hệ số sẽ nhân độ dốc.
- Tăng trưởng càng cao thì dốc càng lớn.

### Đạo hàm của $\log(x)$

#### Trực giác
Log tự nhiên là hàm ngược của mũ tự nhiên.
Khi $x$ lớn hơn, cùng một thay đổi tuyệt đối
có ý nghĩa tương đối nhỏ hơn.
Vì thế độ dốc $1/x$ giảm dần.

#### Định nghĩa và công thức
Với log tự nhiên,
$\frac{d}{dx}\log x=1/x$.
Miền xác định là $x>0$.
Công thức đến từ quy tắc hàm ngược.

#### Ví dụ giải chi tiết
Với $f(x)=\log x$,
$f'(4)=1/4$.
Với $g(x)=3\log x$,
$g'(4)=3/4$.

#### NumPy / Python snippet
```python
def log_grad(x):
    return 1/x
```

> ⚠️ Sai lầm: áp dụng $\log x$
> hoặc đạo hàm của nó tại $x\le0$.

#### Tóm tắt
- Log tự nhiên có đạo hàm $1/x$.
- Độ dốc giảm khi $x$ tăng.
- Miền xác định rất quan trọng.

## Sự tồn tại của đạo hàm

Nhóm này nói về lúc đạo hàm không tồn tại.
Các góc nhọn, độ dốc một phía,
và sự khác nhau giữa liên tục và trơn rất quan trọng.

### Sự tồn tại của đạo hàm

#### Trực giác
Đạo hàm tồn tại khi độ dốc trái và phải
cùng tiến tới một giá trị hữu hạn.
Góc nhọn, bước nhảy, hoặc tiếp tuyến đứng
có thể làm giới hạn thất bại.

#### Định nghĩa và công thức
Hàm khả vi tại $a$ nếu giới hạn định nghĩa
của $f'(a)$ tồn tại.
Với $|x|$ tại $0$,
độ dốc trái là $-1$ và phải là $1$.

#### Ví dụ giải chi tiết
Với $f(x)=|x|$,
$\frac{|0+h|-|0|}{h}=1$ khi $h>0$.
Khi $h<0$, thương số bằng $-1$.
Giới hạn hai phía không tồn tại.

#### NumPy / Python snippet
```python
def right_abs_slope(h):
    return abs(h)/h
```

> ⚠️ Sai lầm: lấy trung bình độ dốc trái
> và phải rồi gọi là đạo hàm.

#### Tóm tắt
- Khả vi cần giới hạn hai phía.
- Góc nhọn có thể liên tục nhưng không trơn.
- Luôn kiểm tra điểm đáng nghi.

## Các quy tắc đạo hàm

Nhóm này gom các quy tắc kết hợp đạo hàm.
Nhân hằng số, tổng, tích và hàm hợp
có những mẫu tính khác nhau.

### Tính chất: nhân với hằng số

#### Trực giác
Kéo đồ thị theo chiều dọc làm mọi độ tăng
bị nhân bởi cùng một hệ số.
Độ chạy ngang không đổi,
nên độ dốc cũng bị nhân như vậy.

#### Định nghĩa và công thức
Với hằng số $c$,
$(cf)'=c f'$.
Hệ số này không được phụ thuộc vào $x$.

#### Ví dụ giải chi tiết
Nếu $f(x)=x^2$ và $g(x)=7f(x)$,
thì $g(x)=7x^2$.
Do đó $g'(x)=14x$,
bằng $7$ lần đạo hàm $2x$.

#### NumPy / Python snippet
```python
def scaled_grad(x):
    return 7*(2*x)
```

> ⚠️ Sai lầm: dùng quy tắc này
> khi hệ số nhân là một hàm khác.

#### Tóm tắt
- Hằng số đi xuyên qua đạo hàm.
- Kéo dọc đồ thị kéo theo độ dốc.
- Hệ số biến thiên cần quy tắc tích.

### Tính chất: quy tắc tổng

#### Trực giác
Nếu hai hiệu ứng được cộng lại,
những thay đổi nhỏ của chúng cũng cộng lại.
Vì vậy đạo hàm phân phối qua phép cộng.

#### Định nghĩa và công thức
$(g+h)'=g'+h'$.
Ý tưởng này mở rộng cho nhiều hạng tử.

#### Ví dụ giải chi tiết
Cho $f(x)=x^2+\sin x$.
Khi đó $f'(x)=2x+\cos x$.
Tại $x=0$, độ dốc là $1$.
Phần bậc hai góp $0$;
phần sin góp $1$.

#### NumPy / Python snippet
```python
import numpy as np
def sum_grad(x):
    return 2*x + np.cos(x)
```

> ⚠️ Sai lầm: chỉ lấy đạo hàm
> của hạng tử đầu tiên.

#### Tóm tắt
- Lấy đạo hàm từng hạng tử.
- Sau đó cộng các mảnh đạo hàm.
- Đây là nền tảng cho đa thức.

### Tính chất: quy tắc tích

#### Trực giác
Khi hai đại lượng đang đổi được nhân với nhau,
cả hai đều làm tích thay đổi.
Một hạng giữ thừa số thứ hai cố định;
hạng kia giữ thừa số thứ nhất cố định.

#### Định nghĩa và công thức
$(gh)'=g'h+gh'$.
Không được thay bằng $g'h'$.

#### Ví dụ giải chi tiết
Với $f(x)=x^2\sin x$,
đặt $g=x^2$ và $h=\sin x$.
Ta có $g'=2x$ và $h'=\cos x$.
Vậy $f'(x)=2x\sin x+x^2\cos x$.

#### NumPy / Python snippet
```python
import numpy as np
def product_grad(x):
    return 2*x*np.sin(x) + x*x*np.cos(x)
```

> ⚠️ Sai lầm: nhân hai đạo hàm với nhau.

#### Tóm tắt
- Quy tắc tích có hai hạng.
- Mỗi hạng cho một thừa số thay đổi.
- Bỏ một hạng sẽ sai đáp án.

### Tính chất: quy tắc dây chuyền

#### Trực giác
Hàm hợp thay đổi theo nhiều lớp.
Lớp ngoài đổi theo giá trị bên trong.
Lớp bên trong đổi theo $x$.
Ta nhân hai tốc độ đó.

#### Định nghĩa và công thức
$\frac{d}{dx}g(h(x))=g'(h(x))h'(x)$.
Đây là quy tắc trung tâm của mạng nơ-ron.

#### Ví dụ giải chi tiết
Với $f(x)=(3x+1)^2$,
đạo hàm lớp ngoài là $2u$.
Đạo hàm lớp trong là $3$.
Vậy $f'(x)=2(3x+1)\cdot3$.
Tại $x=1$, độ dốc là $24$.

#### NumPy / Python snippet
```python
def chain_grad(x):
    return 6*(3*x + 1)
```

> ⚠️ Sai lầm: lấy đạo hàm lớp ngoài
> rồi quên nhân với đạo hàm lớp trong.

#### Tóm tắt
- Xác định hàm ngoài và hàm trong.
- Lấy đạo hàm của cả hai lớp.
- Nhân các tốc độ lại.

## Tối ưu hóa và loss trong ML

Nhóm này cho thấy vì sao các quy tắc trên hữu ích.
Đạo hàm dẫn hướng tìm kiếm,
squared loss dẫn tới trung bình,
và log-loss nối xác suất với huấn luyện.

### Nhập môn tối ưu hóa

#### Trực giác
Tối ưu hóa tìm đầu vào tốt nhất
theo một hàm mục tiêu.
Trong slide, mục tiêu có thể là nơi mát nhất
hoặc điểm nối dây rẻ nhất.

#### Định nghĩa và công thức
Với gradient descent một chiều,
$x_{\text{new}}=x-\alpha f'(x)$.
Learning rate $\alpha$ điều khiển độ dài bước.

#### Ví dụ giải chi tiết
Cho $f(x)=(x-6)^2$.
Tại $x=2$, $f'(x)=2(x-6)=-8$.
Với $\alpha=0.25$,
$x_{\text{new}}=2-0.25(-8)=4$.
Bước đi hướng về $6$.

#### NumPy / Python snippet
```python
x = 2.0
x = x - 0.25*2*(x - 6)
```

> ⚠️ Sai lầm: đi cùng chiều gradient
> trong bài toán cần giảm loss.

#### Tóm tắt
- Tối ưu hóa cần hàm mục tiêu.
- Đạo hàm cho hướng tìm kiếm.
- Kích thước bước ảnh hưởng kết quả.

### Tối ưu squared loss: một đường điện

#### Trực giác
Với một đường điện, vị trí tốt nhất
nằm đúng tại đường đó.
Bình phương khoảng cách bằng không ở đó
và dương ở mọi nơi khác.

#### Định nghĩa và công thức
Nếu đường nằm tại $a$,
$J(x)=(x-a)^2$.
Đạo hàm là $J'(x)=2(x-a)$.

#### Ví dụ giải chi tiết
Với $a=8$,
$J(x)=(x-8)^2$.
Giải $J'(x)=2(x-8)=0$
được $x=8$.
Chi phí nhỏ nhất là $0$.

#### NumPy / Python snippet
```python
a = 8
best_x = a
```

> ⚠️ Sai lầm: tối ưu khoảng cách có dấu
> thay vì khoảng cách bình phương.

#### Tóm tắt
- Một bình phương khoảng cách nhỏ nhất ở tâm.
- Đạo hàm kéo điểm về đường điện.
- Chi phí không thể âm.

### Tối ưu squared loss: hai đường điện

#### Trực giác
Với hai đường điện, điểm tốt nhất cân bằng
hai khoảng cách bình phương.
Kết quả là trung điểm của hai vị trí.

#### Định nghĩa và công thức
$J(x)=(x-a)^2+(x-b)^2$.
Khi đó $J'(x)=2(x-a)+2(x-b)$.
Giải ra $x=(a+b)/2$.

#### Ví dụ giải chi tiết
Cho $a=2$ và $b=10$.
$J'(x)=2(x-2)+2(x-10)=4x-24$.
Đặt $4x-24=0$.
Điểm tốt nhất là $x=6$.

#### NumPy / Python snippet
```python
a, b = 2, 10
x_star = (a + b)/2
```

> ⚠️ Sai lầm: chỉ chọn điểm gần
> một đường điện riêng lẻ.

#### Tóm tắt
- Hai squared loss cân bằng ở trung điểm.
- Phương trình đạo hàm là tuyến tính.
- Cả hai đường đều ảnh hưởng nghiệm.

### Tối ưu squared loss: ba đường điện

#### Trực giác
Với nhiều đường điện,
nghiệm squared loss là trung bình
các vị trí của chúng.
Đây là mầm mống của least squares.

#### Định nghĩa và công thức
$J(x)=\sum_i(x-a_i)^2$.
Do đó $J'(x)=2\sum_i(x-a_i)$.
Đặt bằng không cho
$x=\frac{1}{n}\sum_i a_i$.

#### Ví dụ giải chi tiết
Với các vị trí $1,4,10$,
trung bình là $(1+4+10)/3=5$.
Đạo hàm bằng
$2(x-1)+2(x-4)+2(x-10)=6x-30$.
Giải $6x-30=0$ được $x=5$.

#### NumPy / Python snippet
```python
import numpy as np
x_star = np.mean([1, 4, 10])
```

> ⚠️ Sai lầm: dùng median cho squared loss;
> median phù hợp hơn với absolute loss.

#### Tóm tắt
- Squared loss dẫn tới mean.
- Nhiều điểm thêm nhiều hạng đạo hàm.
- Nghiệm cân bằng tổng lực kéo.

### Tối ưu log-loss: phần 1

#### Trực giác
Với dữ liệu tung đồng xu,
likelihood nhân xác suất của các kết quả
đã thật sự quan sát.
Nhiều dự đoán tự tin mà sai làm tích rất nhỏ.

#### Định nghĩa và công thức
Với $H$ lần ngửa và $T$ lần sấp,
$L(p)=p^H(1-p)^T$.
Ta chọn $p$ làm dữ liệu có vẻ hợp lý nhất.

#### Ví dụ giải chi tiết
Với $7$ ngửa và $3$ sấp,
đồng xu $p=0.7$ cho $0.7^7 0.3^3$.
Đồng xu $p=0.5$ cho $0.5^{10}$.
Giá trị đầu lớn hơn,
nên phù hợp dữ liệu hơn.

#### NumPy / Python snippet
```python
p = 0.7
likelihood = p**7 * (1-p)**3
```

> ⚠️ Sai lầm: chỉ nhìn lần tung cuối
> thay vì toàn bộ chuỗi quan sát.

#### Tóm tắt
- Likelihood nhân xác suất quan sát.
- Xác suất phù hợp cho tích lớn hơn.
- Tích có thể trở nên rất nhỏ.

### Tối ưu log-loss: phần 2

#### Trực giác
Log biến tích thành tổng.
Điều đó làm đạo hàm dễ hơn.
Âm log-likelihood trở thành loss cần giảm.

#### Định nghĩa và công thức
Binary log-loss là
$-[y\log p+(1-y)\log(1-p)]$.
Với nhiều điểm, lấy trung bình các hạng.

#### Ví dụ giải chi tiết
Nếu $y=1$ và $p=0.9$,
loss là $-\log(0.9)\approx0.105$.
Nếu $y=1$ và $p=0.1$,
loss là $-\log(0.1)\approx2.303$.
Dự đoán sai mà tự tin bị phạt nặng.

#### NumPy / Python snippet
```python
import numpy as np
loss = -np.log(0.9)
```

> ⚠️ Sai lầm: giảm likelihood
> thay vì giảm negative log-likelihood.

#### Tóm tắt
- Log biến tích thành tổng.
- Âm log-likelihood là log-loss.
- Xác suất sai tự tin tạo loss lớn.

### Kết luận

#### Trực giác
Tuần 1 tạo một chuỗi ý tưởng:
độ dốc cục bộ thành đạo hàm,
quy tắc đạo hàm giúp tính nhanh,
và tối ưu hóa dùng chúng để huấn luyện ML.

#### Định nghĩa và công thức
Hãy nhớ ba tầng:
định nghĩa bằng giới hạn,
các quy tắc tính đạo hàm,
và cập nhật như
$x_{\text{new}}=x-\alpha f'(x)$.

#### Ví dụ giải chi tiết
Với $J(w)=(w-5)^2$,
đạo hàm là $2(w-5)$.
Bắt đầu ở $w=9$ với $\alpha=0.25$,
giá trị mới là
$9-0.25\cdot8=7$.
Tham số tiến về điểm cực tiểu.

#### NumPy / Python snippet
```python
w = 9
w -= 0.25*2*(w - 5)
```

> ⚠️ Sai lầm: học thuộc bảng đạo hàm
> mà không liên hệ với loss.

#### Tóm tắt
- Đạo hàm đo thay đổi cục bộ.
- Quy tắc làm việc tính toán thực tế.
- Tối ưu hóa dùng độ dốc để cải thiện mô hình.
