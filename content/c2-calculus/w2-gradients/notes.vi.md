# Tuần 2: Gradient và Gradient Descent

Slides tuần 2 bắt đầu từ mặt $f(x,y)=x^2+y^2$,
rồi dùng đạo hàm để tối ưu.  Điểm mới so với đạo hàm một biến
là mỗi tọa độ có một độ dốc riêng.  Các độ dốc đó tạo thành
gradient, và gradient hướng dẫn cách dịch chuyển tham số mô hình.

## Mặt phẳng tiếp tuyến

### Trực giác
Hàm hai biến là một mặt cao thấp.  Cố định $y=4$ tạo lát cắt
$f(x,4)$; cố định $x=2$ tạo lát cắt $f(2,y)$.  Ở điểm $(2,4)$,
mặt phẳng tiếp tuyến (tangent plane) là tấm phẳng khớp với cả
hai đường tiếp tuyến của hai lát cắt.

### Định nghĩa và công thức
Mô hình tuyến tính cục bộ tại $(a,b)$ là
$$L(x,y)=f(a,b)+f_x(a,b)(x-a)+f_y(a,b)(y-b).$$
Khi viết $z=L(x,y)$, ta được phương trình mặt phẳng tiếp tuyến.

### Ví dụ làm đầy đủ
Với hàm trong slide $f(x,y)=x^2+y^2$ tại $(2,4)$:
$f(2,4)=4+16=20$.  Đạo hàm lát cắt theo $x$ là $f_x=2x$,
nên $f_x(2,4)=4$.  Đạo hàm lát cắt theo $y$ là $f_y=2y$,
nên $f_y(2,4)=8$.  Vì vậy
$$z=20+4(x-2)+8(y-4).$$
Gần điểm đó, ví dụ $(2.1,3.9)$, mặt phẳng dự đoán
$20+0.4-0.8=19.6$.

> ⚠️ Mặt phẳng tiếp tuyến chỉ đúng cục bộ.  Nó không nói rằng
> cả paraboloid trở thành mặt phẳng ở xa $(2,4)$.

### NumPy snippet
```python
def f(x, y):
    return x*x + y*y
a, b = 2.0, 4.0
fx, fy = 2*a, 2*b
def plane(x, y):
    return f(a, b) + fx*(x-a) + fy*(y-b)
print(plane(2.1, 3.9))
```

### Tóm tắt
- Cắt mặt theo từng tọa độ để thấy độ dốc.
- Hai độ dốc tạo một xấp xỉ phẳng.
- Xấp xỉ bậc nhất chỉ đáng tin gần điểm tiếp xúc.

## Đạo hàm riêng

### Trực giác
Đạo hàm riêng (partial derivative) hỏi: "độ dốc là bao nhiêu
nếu chỉ cho một tọa độ thay đổi?"  Slides minh họa bằng cách
xem biến còn lại như hằng số rồi đạo hàm theo quy tắc quen thuộc.

### Định nghĩa và công thức
Ký hiệu $f_x$ hoặc $\partial f/\partial x$ nghĩa là giữ $y$
cố định.  Ký hiệu $f_y$ giữ $x$ cố định.  Trong biểu thức
$3x^2y^3$, phần không chứa biến đang đạo hàm đóng vai trò như
hệ số hằng.

### Ví dụ làm đầy đủ
Cho $f(x,y)=3x^2y^3$.  Khi tính $f_x$, giữ $y^3$:
$$f_x=3y^3\cdot2x=6xy^3.$$
Khi tính $f_y$, giữ $3x^2$:
$$f_y=3x^2\cdot3y^2=9x^2y^2.$$
Tại $(1,2)$, ta có $f_x=6\cdot1\cdot8=48$ và
$f_y=9\cdot1\cdot4=36$.

> ⚠️ "Xem là hằng số" không có nghĩa là thay bằng 0 hoặc 1.
> Hãy giữ ký hiệu đó trong suốt phép đạo hàm.

### NumPy snippet
```python
def f(x, y):
    return 3*x*x*y**3
h = 1e-5
x, y = 1.0, 2.0
dfdx = (f(x+h, y) - f(x-h, y)) / (2*h)
dfdy = (f(x, y+h) - f(x, y-h)) / (2*h)
print(round(dfdx), round(dfdy))
```

### Tóm tắt
- Đạo hàm riêng là độ dốc của một lát cắt.
- Biến không xét được giữ cố định, không bị xóa.
- Sai phân hữu hạn giúp kiểm tra kết quả tay.

## Gradient

### Trực giác
Gradient gom các đạo hàm riêng thành một vector.  Nó chỉ hướng
hàm tăng nhanh nhất trong không gian đầu vào.  Hướng ngược lại,
$-\nabla f$, là hướng đi xuống nhanh nhất.

### Định nghĩa và công thức
Với hai biến,
$$\nabla f(x,y)=\begin{bmatrix}f_x(x,y)\\ f_y(x,y)\end{bmatrix}.$$
Đạo hàm theo hướng đơn vị $u$ là $\nabla f\cdot u$.
Tích vô hướng này lớn nhất khi $u$ cùng hướng gradient.

### Ví dụ làm đầy đủ
Với $f(x,y)=x^2+y^2$, hai đạo hàm riêng là $2x$ và $2y$.
Tại $(2,3)$,
$$\nabla f(2,3)=\begin{bmatrix}4\\6\end{bmatrix}.$$
Đi một bước nhỏ theo $(4,6)$ làm hàm tăng nhanh nhất; đi theo
$(-4,-6)$ làm hàm giảm nhanh nhất.

> ⚠️ Gradient nằm trong mặt phẳng tọa độ $(x,y)$, không phải là
> chiều cao $z$ của mặt.

### NumPy snippet
```python
import numpy as np
def grad_xy(point):
    x, y = point
    return np.array([2*x, 2*y])
g = grad_xy(np.array([2.0, 3.0]))
print(g, -g)
```

### Tóm tắt
- Gradient là vector các độ dốc theo tọa độ.
- Nó chỉ hướng tăng nhanh nhất tại điểm hiện tại.
- Muốn giảm hàm thì dùng hướng đối của gradient.

## Gradient và cực trị

### Trực giác
Ở đỉnh hoặc đáy trơn bên trong miền, không còn hướng đi lên
bậc nhất nào.  Vì vậy slides đặt cả hai đạo hàm riêng bằng 0
khi tìm cực trị.

### Định nghĩa và công thức
Điểm dừng thỏa $\nabla f=0$.  Điều kiện này chỉ tạo ứng viên.
Muốn kết luận cực tiểu, cực đại hay yên ngựa, cần xét biên,
độ cong, hoặc so sánh trực tiếp giá trị hàm.

### Ví dụ làm đầy đủ
Với $f(x,y)=x^2+y^2$, giải $2x=0$ và $2y=0$.
Ứng viên duy nhất là $(0,0)$.  Vì bình phương luôn không âm,
$f(0,0)=0$ là cực tiểu toàn cục.  Trong ví dụ bản đồ nhiệt của
slides, ta cũng giải hai phương trình đạo hàm riêng, rồi loại
điểm nằm ngoài phòng và so sánh nhiệt độ còn lại.

> ⚠️ Gradient bằng 0 chưa đủ để gọi là minimum.  Hàm $x^2-y^2$
> có gradient bằng 0 tại gốc, nhưng đó là điểm yên ngựa.

### NumPy snippet
```python
candidates = [(0, 0), (4, 4), (6, 6)]
def simple_heat(x, y):
    return 85 - (x*x*(x-6)*y*y*(y-6))/90
for p in candidates:
    print(p, round(simple_heat(*p), 3))
```

### Tóm tắt
- Đặt mọi đạo hàm riêng bằng 0 để lấy ứng viên.
- Loại điểm không hợp lệ hoặc nằm ngoài miền.
- Phải phân loại trước khi tuyên bố tối ưu.

## Tối ưu giải tích

### Trực giác
Tối ưu giải tích nghĩa là giải chính xác hệ phương trình
gradient bằng 0.  Cách này đẹp với quadratic nhỏ, nhưng slides
dẫn tới gradient descent vì nhiều cost thực tế không giải tay
dễ như vậy.

### Định nghĩa và công thức
Với cost $E(m,b)$, phương pháp giải tích đặt
$$E_m(m,b)=0,\qquad E_b(m,b)=0.$$
Khi $E$ là quadratic, các phương trình này thường tuyến tính
theo tham số cần tìm.

### Ví dụ làm đầy đủ
Cost trong slides là
$$E=14m^2+3b^2+38+12mb-42m-20b.$$
Đạo hàm:
$$E_m=28m+12b-42,\qquad E_b=12m+6b-20.$$
Đặt cả hai bằng 0.  Nhân phương trình hai với 2 được
$24m+12b-40=0$.  Trừ khỏi phương trình một:
$4m-2=0$, nên $m=1/2$.  Thế vào $E_b=0$:
$6+6b-20=0$, suy ra $b=7/3$.

> ⚠️ Nếu chỉ giải $E_m=0$, ta có cả một đường ứng viên chứ
> không có cặp $(m,b)$ tối ưu duy nhất.

### NumPy snippet
```python
import numpy as np
A = np.array([[28, 12], [12, 6]], dtype=float)
r = np.array([42, 20], dtype=float)
print(np.linalg.solve(A, r))
```

### Tóm tắt
- Viết đầy đủ các phương trình bậc nhất.
- Giải hệ cùng lúc, không giải từng đạo hàm riêng lẻ.
- Dạng đóng hữu ích nhưng không phải lúc nào cũng có.

## Động lực hồi quy tuyến tính

### Trực giác
Hồi quy tuyến tính biến việc kẻ đường thẳng thành bài toán tối
ưu cost.  Dữ liệu trong slides là $(1,2)$, $(2,5)$, $(3,3)$.
Mô hình dự đoán $\hat y=mx+b$, nên biến cần tối ưu là $m,b$.

### Định nghĩa và công thức
Residual của điểm $i$ là $mx_i+b-y_i$.  Slides dùng tổng bình
phương residual:
$$E(m,b)=\sum_i (mx_i+b-y_i)^2.$$
Bình phương làm lỗi âm và lỗi dương đều bị phạt như nhau.

### Ví dụ làm đầy đủ
Với ba điểm trong slides, residual là
$m+b-2$, $2m+b-5$, và $3m+b-3$.
Khai triển rồi gom hạng tử:
$$E=14m^2+3b^2+12mb-42m-20b+38.$$
Nghiệm giải tích là $m=1/2$, $b=7/3$,
nên đường fit là $y=\tfrac12x+\tfrac73$.

> ⚠️ Tọa độ dữ liệu $x$ trên hình không cùng vai trò với tham số
> slope $m$.  Mặt cost nằm trên không gian $(m,b)$.

### NumPy snippet
```python
import numpy as np
x = np.array([1., 2., 3.])
y = np.array([2., 5., 3.])
m, b = 0.5, 7/3
err = m*x + b - y
print(np.sum(err**2))
```

### Tóm tắt
- Chọn tham số để giảm bình phương residual.
- Cost là một mặt trên không gian tham số.
- Lỗi lớn bị phạt mạnh vì có bình phương.

## Gradient descent một biến

### Trực giác
Gradient descent thay việc giải chính xác bằng chuyển động cục bộ
lặp lại.  Nếu đạo hàm âm, bước sang phải làm hàm giảm; nếu đạo
hàm dương, bước sang trái.

### Định nghĩa và công thức
Với hàm một biến,
$$x_{k+1}=x_k-\alpha f'(x_k).$$
Learning rate $\alpha$ quyết định lấy bao nhiêu phần của độ dốc
làm bước đi.

### Ví dụ làm đầy đủ
Cho $f(x)=(x-4)^2$, nên $f'(x)=2(x-4)$.
Bắt đầu $x_0=1$ với $\alpha=0.25$.
Ta có $x_1=1-0.25(-6)=2.5$.
Tiếp theo $x_2=2.5-0.25(-3)=3.25$.
Rồi $x_3=3.25-0.25(-1.5)=3.625$.
Dãy tiến dần về minimum $x=4$.

> ⚠️ Tối thiểu hóa dùng phép trừ đạo hàm.  Nếu cộng hạng tử đó,
> bạn đang làm gradient ascent.

### NumPy snippet
```python
x = 1.0
alpha = 0.25
for step in range(3):
    x = x - alpha * 2*(x-4)
    print(step + 1, x)
```

### Tóm tắt
- Lặp bước nhỏ thay vì giải phương trình chính xác.
- Dấu đạo hàm quyết định đi trái hay phải.
- Độ dài bước phụ thuộc vào slope và learning rate.

## Learning rate và cực tiểu cục bộ

### Trực giác
Learning rate là núm điều chỉnh biến slope thành chuyển động.
Quá nhỏ thì học rất chậm; quá lớn thì nhảy qua đáy và dao động.
Với hàm không lồi, descent còn có thể mắc ở cực tiểu cục bộ.

### Định nghĩa và công thức
Với $f(x)=x^2$, cập nhật gradient descent là
$x_{k+1}=x_k-2\alpha x_k=(1-2\alpha)x_k$.
Hệ số $1-2\alpha$ giải thích hội tụ chậm, dao động và phân kỳ.

### Ví dụ làm đầy đủ
Bắt đầu $x_0=1$, $\alpha=1.1$ cho $f(x)=x^2$.
Khi đó $x_1=(1-2.2)\cdot1=-1.2$.
Điểm kế tiếp $x_2=(-1.2)(-1.2)=1.44$.
Dấu luân phiên và độ lớn tăng, nên learning rate này quá lớn.

> ⚠️ Cost giảm vài bước đầu chưa chứng minh tối ưu toàn cục.
> Với hàm gợn sóng, nên thử nhiều điểm khởi tạo.

### NumPy snippet
```python
def run(alpha):
    x = 1.0
    history = []
    for _ in range(5):
        x = x - alpha * 2*x
        history.append(round(x, 3))
    return history
print(run(0.1))
print(run(1.1))
```

### Tóm tắt
- Rate nhỏ an toàn nhưng chậm.
- Rate lớn có thể dao động hoặc phân kỳ.
- Cực tiểu cục bộ làm điểm bắt đầu trở nên quan trọng.

## Gradient descent hai biến

### Trực giác
Với hai biến, vị trí hiện tại là một vector.  Ví dụ bản đồ nhiệt
trong slides bắt đầu ở một điểm trong phòng, tính gradient, rồi
đi một bước nhỏ theo hướng ngược để tìm điểm mát hơn.

### Định nghĩa và công thức
Với $p_k=[x_k,y_k]^T$,
$$p_{k+1}=p_k-\alpha\nabla f(p_k).$$
Cả hai tọa độ phải dùng cùng gradient tại điểm cũ.

### Ví dụ làm đầy đủ
Cho $f(x,y)=(x-1)^2+2(y+1)^2$.
Tại $(3,1)$, gradient là
$[2(3-1),4(1+1)]=[4,8]$.
Với $\alpha=0.1$,
$$[x_1,y_1]=[3,1]-0.1[4,8]=[2.6,0.2].$$
Điểm mới gần minimum $(1,-1)$ hơn.

> ⚠️ Đừng cập nhật $x$ trước rồi dùng $x$ mới để tính gradient
> của cùng vòng lặp cho $y$.

### NumPy snippet
```python
import numpy as np
p = np.array([3.0, 1.0])
g = np.array([2*(p[0]-1), 4*(p[1]+1)])
p_next = p - 0.1*g
print(p_next)
```

### Tóm tắt
- Điểm và gradient đều là vector.
- Hướng âm của gradient là hướng xuống nhanh nhất cục bộ.
- Mọi tọa độ trong một bước dùng cùng điểm xuất phát.

## Gradient descent cho least squares

### Trực giác
Least-squares gradient descent dùng cùng quy tắc vector cho tham
số mô hình.  Với đường thẳng, tham số là $m$ và $b$.  Mỗi bước
dùng toàn bộ residual để đẩy đường fit tốt hơn.

### Định nghĩa và công thức
Với
$$J(m,b)=\frac1{2n}\sum_i (mx_i+b-y_i)^2,$$
ta có
$$J_m=\frac1n\sum_i (mx_i+b-y_i)x_i,$$
$$J_b=\frac1n\sum_i (mx_i+b-y_i).$$
Sau đó cập nhật $m\leftarrow m-\alpha J_m$ và
$b\leftarrow b-\alpha J_b$.

### Ví dụ làm đầy đủ
Dữ liệu nhỏ $(0,1),(1,3),(2,5)$ nằm trên $y=2x+1$.
Tại $m=0,b=0$, dự đoán là $0,0,0$ và residual là
$-1,-3,-5$.  Do đó
$J_m=(-1\cdot0-3\cdot1-5\cdot2)/3=-13/3$ và
$J_b=(-1-3-5)/3=-3$.  Với $\alpha=0.1$,
$m$ thành $0.4333$ và $b$ thành $0.3$.

> ⚠️ Hệ số $1/(2n)$ làm đạo hàm gọn hơn.  Quên trung bình sẽ
> đổi scale của gradient và ảnh hưởng learning rate.

### NumPy snippet
```python
import numpy as np
x = np.array([0., 1., 2.])
y = np.array([1., 3., 5.])
m, b, alpha = 0.0, 0.0, 0.1
err = m*x + b - y
m -= alpha * np.mean(err*x)
b -= alpha * np.mean(err)
print(round(m, 4), round(b, 4))
```

### Tóm tắt
- Residual tạo gradient cho tham số.
- Lấy trung bình giúp scale ổn định khi đổi số điểm.
- Squared loss lồi cho đường thẳng có một vùng tối ưu.
