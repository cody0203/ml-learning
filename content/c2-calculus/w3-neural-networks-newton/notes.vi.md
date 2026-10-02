# Tuần 3: Tối ưu hóa mạng neural và phương pháp Newton

Hai PDF tạo thành một tuần dài.
Phần 1 xây công cụ tối ưu hóa.
Phần 2 đưa đạo hàm vào perceptron và mạng neural.

## Gradient descent và backpropagation

### Trực giác
Gradient descent hỏi một câu cục bộ:
"Dịch tham số theo hướng nào làm loss giảm nhanh?"
Gradient chỉ hướng tăng nhanh nhất.
Vì vậy khi tối thiểu hóa ta đi ngược gradient.
Backpropagation là quy tắc dây chuyền đi lùi
qua đồ thị tính toán của mạng.

### Định nghĩa và công thức
Với tham số $\theta$ và loss $L$:

$$\theta_{k+1}=\theta_k-\alpha\nabla_\theta L.$$

Learning rate $\alpha$ là độ dài bước.
Bước quá lớn có thể vượt qua đáy.
Bước quá nhỏ làm quá trình học rất chậm.

### Ví dụ làm đầy đủ
Cho $L(w)=\frac12(w-3)^2$.
Khi đó $L'(w)=w-3$.
Với $w_0=0$ và $\alpha=0.1$:

$$w_1=0-0.1(-3)=0.3.$$

Loss giảm từ $4.5$ xuống $3.645$.
Nếu loss nằm cuối một mạng,
backprop cũng trả về đạo hàm này.

> ⚠️ Lỗi thường gặp:
> cộng $\alpha\nabla L$ khi đang minimize.

### NumPy snippet
```python
w = w - alpha * grad_w
```

### Tóm tắt
- Gradient descent dùng đạo hàm bậc nhất.
- Backprop là chain rule dạng đi ngược.
- Dấu của update rất quan trọng.

## Phương pháp Newton

### Trực giác
Newton thay đường cong gần điểm hiện tại
bằng tiếp tuyến tại điểm đó.
Giao điểm của tiếp tuyến với trục hoành
trở thành ước lượng tiếp theo.

### Định nghĩa và công thức
Để tìm nghiệm:

$$x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.$$

Để tối thiểu $g(x)$,
áp dụng công thức cho $g'(x)=0$:

$$x_{k+1}=x_k-\frac{g'(x_k)}{g''(x_k)}.$$

### Ví dụ làm đầy đủ
Lấy $f(x)=x^2-2$ và $x_0=1$.
Ta có $f(1)=-1$ và $f'(1)=2$.

$$x_1=1-\frac{-1}{2}=1.5.$$

Bước sau là
$1.5-(0.25/3)=1.4167$,
đã rất gần $\sqrt2$.

> ⚠️ Lỗi thường gặp:
> dùng Newton khi $f'(x_k)$ gần bằng 0.

### NumPy snippet
```python
x = x - f(x) / fp(x)
```

### Tóm tắt
- Newton dựa trên hình học tiếp tuyến.
- Gần nghiệm tốt, nó hội tụ rất nhanh.
- Điểm bắt đầu xấu có thể làm phân kỳ.

## Ví dụ phương pháp Newton

### Trực giác
Slides tối thiểu
$g(x)=e^x-\log x$.
Miền hợp lệ là $x>0$.
Ta tìm nghiệm của $g'(x)$
thay vì nhìn trực tiếp vào đồ thị $g$.

### Định nghĩa và công thức
Với hàm này:

$$g'(x)=e^x-\frac1x,$$

và

$$g''(x)=e^x+\frac1{x^2}.$$

Do đó:

$$x_{k+1}=x_k-\frac{e^{x_k}-1/x_k}{e^{x_k}+1/x_k^2}.$$

### Ví dụ làm đầy đủ
Tại $x_0=0.05$:
$g'(x_0)\approx-18.949$.
$g''(x_0)\approx401.051$.
Suy ra

$$x_1\approx0.05-(-18.949/401.051)=0.097.$$

Các bước trong slides đi qua
$0.183,0.320,0.477$ rồi tới khoảng $0.567$.

> ⚠️ Lỗi thường gặp:
> chọn $x\le0$ khiến $\log x$ không xác định.

### NumPy snippet
```python
x = x - (np.exp(x) - 1/x) / (np.exp(x) + 1/x**2)
```

### Tóm tắt
- Newton optimization giải $g'(x)=0$.
- Curvature nằm trong $g''(x)$.
- Miền xác định phải được giữ trong từng bước.

## Đạo hàm bậc hai

### Trực giác
Đạo hàm bậc nhất giống vận tốc của hàm.
Đạo hàm bậc hai giống gia tốc của vận tốc đó.
Trên đồ thị, nó mô tả độ cong.

### Định nghĩa và công thức

$$f''(x)=\frac{d}{dx}f'(x)=\frac{d^2f}{dx^2}.$$

Nếu $f''(x)>0$, đồ thị cong lên.
Nếu $f''(x)<0$, đồ thị cong xuống.
Nếu $f''(x)=0$, kết luận chưa đủ.

### Ví dụ làm đầy đủ
Cho $f(x)=x^3-3x$.
Khi đó $f'(x)=3x^2-3$.
Tiếp tục lấy đạo hàm:
$f''(x)=6x$.
Tại $x=1$, $f''(1)=6>0$,
nên quanh điểm đó đồ thị cong lên.

> ⚠️ Lỗi thường gặp:
> nghĩ $f''(x)=0$ luôn loại trừ cực trị.

### NumPy snippet
```python
second = (f(x+h) - 2*f(x) + f(x-h)) / h**2
```

### Tóm tắt
- Đạo hàm bậc hai đo curvature.
- Dấu dương gợi ý dạng cái bát.
- Dấu âm gợi ý dạng cái mũ.

## Ma trận Hessian

### Trực giác
Với hai biến, curvature không còn là một số.
Bề mặt có thể cong theo trục $x$,
theo trục $y$,
và theo hướng trộn giữa hai biến.
Hessian gom các thông tin đó lại.

### Định nghĩa và công thức

$$
H_f(x,y)=
\begin{bmatrix}
f_{xx} & f_{xy}\\
f_{yx} & f_{yy}
\end{bmatrix}.
$$

Với hàm đủ trơn trong ví dụ khóa học,
hai đạo hàm trộn thường bằng nhau.

### Ví dụ làm đầy đủ
Với
$f(x,y)=2x^2+3y^2-xy$:

$$f_x=4x-y,\qquad f_y=6y-x.$$

Vì vậy

$$H=\begin{bmatrix}4&-1\\-1&6\end{bmatrix}.$$

Phần tử $-1$ ngoài đường chéo
cho biết hai biến có tương tác curvature.

> ⚠️ Lỗi thường gặp:
> đưa đạo hàm bậc nhất vào Hessian.

### NumPy snippet
```python
H = np.array([[4.0, -1.0], [-1.0, 6.0]])
```

### Tóm tắt
- Gradient là thông tin bậc nhất.
- Hessian là thông tin bậc hai.
- Mixed partials đo sự ghép giữa biến.

## Hessian và độ lõm/lồi

### Trực giác
Trong 1D, đạo hàm bậc hai chỉ có một dấu.
Trong 2D, Hessian có nhiều hướng curvature.
Eigenvalue cho biết dấu curvature theo các hướng chính.
Tất cả dương là local minimum.
Tất cả âm là local maximum.
Trái dấu là saddle point.

### Định nghĩa và công thức
Tại critical point:

$$
\lambda_i(H)>0\ \forall i
\Rightarrow \text{local minimum}.
$$

$$
\lambda_i(H)<0\ \forall i
\Rightarrow \text{local maximum}.
$$

Eigenvalue trái dấu nghĩa là saddle.

### Ví dụ làm đầy đủ
Với $H=\begin{bmatrix}4&-1\\-1&6\end{bmatrix}$:

$$\det(H-\lambda I)=\lambda^2-10\lambda+23.$$

Hai nghiệm xấp xỉ $3.59$ và $6.41$.
Cả hai dương,
nên critical point tương ứng là minimum.

> ⚠️ Lỗi thường gặp:
> chỉ nhìn determinant mà bỏ qua dấu curvature.

### NumPy snippet
```python
np.linalg.eigvals(H)
```

### Tóm tắt
- Eigenvalue dương biểu diễn curvature lồi.
- Eigenvalue âm biểu diễn curvature lõm.
- Trái dấu tạo điểm yên ngựa.

## Newton cho hai biến

### Trực giác
Trong 1D, Newton chia cho đạo hàm bậc hai.
Trong nhiều biến,
phép chia đó trở thành giải hệ tuyến tính
với ma trận Hessian.

### Định nghĩa và công thức

$$
\mathbf{x}_{k+1}
=\mathbf{x}_k-H(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k).
$$

Khi code, ta giải
$H\Delta=\nabla f$
rồi đặt $\mathbf{x}_{k+1}=\mathbf{x}_k-\Delta$.

### Ví dụ làm đầy đủ
Cho
$f=x^2+2y^2-2x-8y$.
Ta có
$\nabla f=(2x-2,4y-8)$
và
$H=\begin{bmatrix}2&0\\0&4\end{bmatrix}$.
Tại $(0,0)$, gradient là $(-2,-8)$.
Giải $H\Delta=(-2,-8)$ được $\Delta=(-1,-2)$.
Điểm mới là $(1,2)$.

> ⚠️ Lỗi thường gặp:
> nhân với $H$ thay vì giải hệ có $H$.

### NumPy snippet
```python
delta = np.linalg.solve(H, grad)
x = x - delta
```

### Tóm tắt
- Newton nhiều biến dùng Hessian.
- Thứ tự ma trận rất quan trọng.
- Giải hệ tốt hơn tính inverse tường minh.

## Hồi quy với perceptron

### Trực giác
Perceptron hồi quy dự đoán một số liên tục.
Trong ví dụ giá nhà,
feature có thể là diện tích và số phòng.
Weight cho biết mỗi feature kéo dự đoán bao nhiêu.

### Định nghĩa và công thức

$$\hat y=w_1x_1+w_2x_2+b.$$

Đây là một neuron tuyến tính
chưa dùng activation phi tuyến.

### Ví dụ làm đầy đủ
Dùng đơn vị đã scale:
$x=(2,4)$, $w=(10,5)$, $b=3$.
Khi đó

$$\hat y=10(2)+5(4)+3=43.$$

Nếu giá thật là $45$,
residual bằng $2$.

> ⚠️ Lỗi thường gặp:
> quên bias khi tính dự đoán theo batch.

### NumPy snippet
```python
yhat = X @ w + b
```

### Tóm tắt
- Output hồi quy là số liên tục.
- Neuron tính tổng có trọng số.
- Bias dịch mọi dự đoán lên hoặc xuống.

## Hàm mất mát hồi quy

### Trực giác
Loss biến sai số dự đoán thành một số.
Slides dùng bình phương sai số,
nên dự đoán cao và thấp đều bị phạt.
Hệ số $\frac12$ giúp đạo hàm gọn hơn.

### Định nghĩa và công thức

$$L(y,\hat y)=\frac12(y-\hat y)^2.$$

Đạo hàm theo $\hat y$ là $\hat y-y$.

### Ví dụ làm đầy đủ
Nếu $y=5$ và $\hat y=3$,
sai số là $2$.
Loss là

$$\frac12(5-3)^2=2.$$

Nếu $\hat y=1$,
loss tăng thành $8$.

> ⚠️ Lỗi thường gặp:
> lấy trung bình trước rồi mới bình phương.

### NumPy snippet
```python
loss = 0.5 * (y - yhat)**2
```

### Tóm tắt
- Bình phương xóa dấu của residual.
- Sai số lớn bị phạt mạnh hơn.
- Training tối thiểu hóa loss.

## Gradient descent cho hồi quy perceptron

### Trực giác
Dự đoán phụ thuộc vào $w_1,w_2,b$.
Loss phụ thuộc vào dự đoán.
Chain rule nối từng tham số với loss.

### Định nghĩa và công thức

$$
\frac{\partial L}{\partial w_i}
=(\hat y-y)x_i,
$$

và

$$\frac{\partial L}{\partial b}=\hat y-y.$$

### Ví dụ làm đầy đủ
Cho $x=(2,1)$, $y=5$, $\hat y=3$.
Khi đó $\hat y-y=-2$.
Gradient theo weight là $(-4,-2)$.
Gradient theo bias là $-2$.
Với $\alpha=0.1$,
weight tăng thêm $(0.4,0.2)$
và bias tăng thêm $0.2$.

> ⚠️ Lỗi thường gặp:
> dùng $(y-\hat y)x_i$ nhưng vẫn trừ như gradient chuẩn.

### NumPy snippet
```python
err = yhat - y
dw = err * x
db = err
```

### Tóm tắt
- Mỗi feature scale gradient weight riêng.
- Bias có input ngầm bằng 1.
- Quy ước dấu phải nhất quán.

## Phân loại với perceptron

### Trực giác
Phân loại cần nhãn hoặc xác suất.
Slides biến câu thành số lần xuất hiện
của Aack và Beep,
rồi đưa các số đó vào weighted sum.

### Định nghĩa và công thức

$$z=w\cdot x+b,\qquad \hat y=\sigma(z).$$

Score thô $z$ có thể là số thực bất kỳ.
Sigmoid đổi nó thành số trong $(0,1)$.

### Ví dụ làm đầy đủ
Cho $x=(1,3)$, $w=(4.5,1.5)$, $b=2$.
Ta được

$$z=4.5(1)+1.5(3)+2=11.$$

Sigmoid của 11 gần 1,
nên mô hình rất tự tin với class 1.

> ⚠️ Lỗi thường gặp:
> xem score thô là xác suất.

### NumPy snippet
```python
z = x @ w + b
p = 1 / (1 + np.exp(-z))
```

### Tóm tắt
- Linear score đứng trước activation.
- Sigmoid tạo xác suất nhị phân.
- Feature phải được mã hóa thành số.

## Hàm sigmoid

### Trực giác
Sigmoid là đường cong chữ S.
Score âm lớn cho xác suất gần 0.
Score dương lớn cho xác suất gần 1.
Gần 0, mô hình còn phân vân.

### Định nghĩa và công thức

$$\sigma(z)=\frac1{1+e^{-z}}.$$

Đạo hàm là

$$\sigma'(z)=\sigma(z)(1-\sigma(z)).$$

### Ví dụ làm đầy đủ
Tại $z=0$:

$$\sigma(0)=\frac12.$$

Đạo hàm là

$$0.5(1-0.5)=0.25.$$

Đây là slope lớn nhất của sigmoid.

> ⚠️ Lỗi thường gặp:
> quên đạo hàm dùng chính giá trị sigmoid.

### NumPy snippet
```python
s = 1 / (1 + np.exp(-z))
sp = s * (1 - s)
```

### Tóm tắt
- Sigmoid luôn nằm giữa 0 và 1.
- Đạo hàm lớn nhất tại gốc.
- Vùng bão hòa tạo gradient nhỏ.

## Gradient descent cho phân loại perceptron

### Trực giác
Khi ghép sigmoid với binary log-loss,
nhiều thừa số chain rule triệt tiêu.
Gradient cuối cùng rất gọn:
dự đoán trừ nhãn, nhân input.

### Định nghĩa và công thức

$$
\frac{\partial J}{\partial w_i}
=(\hat y-y)x_i,
$$

và

$$\frac{\partial J}{\partial b}=\hat y-y.$$

### Ví dụ làm đầy đủ
Nếu $\hat y=0.8$, $y=1$, $x=(3,1)$,
thì $\hat y-y=-0.2$.
Do đó
$dw=(-0.6,-0.2)$ và $db=-0.2$.
Với $\alpha=0.5$,
weight đổi thêm $(0.3,0.1)$.

> ⚠️ Lỗi thường gặp:
> nhân thêm đạo hàm sigmoid sau khi log-loss đã rút gọn.

### NumPy snippet
```python
err = yhat - y
dw = err * x
b = b - alpha * err
```

### Tóm tắt
- Log-loss kết hợp tốt với sigmoid.
- Error term là $\hat y-y$.
- Dự đoán sai nhưng tự tin bị phạt nặng.

## Tính đạo hàm cho phân loại

### Trực giác
Slides khai triển đạo hàm từng mắt xích.
Với một weight,
đường đi là loss tới prediction,
prediction tới score,
rồi score tới weight đó.

### Định nghĩa và công thức

$$
\frac{\partial L}{\partial w_i}
=\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
\frac{\partial z}{\partial w_i}.
$$

Dòng trên là tích các mắt xích.
Ngoài ra,
$\partial z/\partial w_i=x_i$.

### Ví dụ làm đầy đủ
Giả sử
$dL/d\hat y=2$,
$d\hat y/dz=0.25$,
và $dz/dw_1=3$.
Khi đó

$$dL/dw_1=2(0.25)(3)=1.5.$$

> ⚠️ Lỗi thường gặp:
> dùng $w_i$ thay vì $x_i$
> cho $\partial z/\partial w_i$.

### NumPy snippet
```python
grad_wi = dL_dyhat * dyhat_dz * x[i]
```

### Tóm tắt
- Backprop nhân đạo hàm cục bộ.
- Mỗi cạnh trong graph cho một thừa số.
- Giá trị input xuất hiện trong gradient weight.

## Phân loại với mạng neural

### Trực giác
Mạng 2-2-1 có hai input,
hai neuron hidden,
và một neuron output.
Activation hidden trở thành feature mới
cho neuron output.

### Định nghĩa và công thức

$$
z^{[1]}=W^{[1]}x+b^{[1]},\quad
a^{[1]}=\sigma(z^{[1]}).
$$

Sau đó

$$\hat y=\sigma(W^{[2]}a^{[1]}+b^{[2]}).$$

### Ví dụ làm đầy đủ
Cho $W^{[1]}$ là ma trận đơn vị,
$b^{[1]}=(1,-1)$,
và $x=(2,3)$.
Khi đó
$z^{[1]}=(3,2)$.
Sau sigmoid,
$a^{[1]}\approx(0.953,0.881)$.
Hai số này đi vào output layer.

> ⚠️ Lỗi thường gặp:
> lẫn quy ước vector cột và vector hàng.

### NumPy snippet
```python
a1 = sigmoid(W1 @ x + b1)
yhat = sigmoid(W2 @ a1 + b2)
```

### Tóm tắt
- Hidden unit tạo feature đã học.
- Output layer là một perceptron khác.
- Shape phải được kiểm tra ở từng layer.

## Tối thiểu log-loss trong mạng neural

### Trực giác
Output cuối là một xác suất.
Binary log-loss thưởng xác suất cao
cho nhãn đúng.
Nó phạt rất mạnh khi mô hình tự tin sai.

### Định nghĩa và công thức

$$
J(y,\hat y)=
-y\log(\hat y)-(1-y)\log(1-\hat y).
$$

Với output sigmoid,
tín hiệu lỗi cục bộ là $\hat y-y$.

### Ví dụ làm đầy đủ
Nếu $y=1$ và $\hat y=0.9$:

$$J=-\log(0.9)\approx0.105.$$

Nếu cùng ví dụ đó có $\hat y=0.1$:

$$J=-\log(0.1)\approx2.303.$$

Dự đoán thứ hai tệ hơn rất nhiều.

> ⚠️ Lỗi thường gặp:
> tính $\log(0)$ khi không clip xác suất.

### NumPy snippet
```python
p = np.clip(p, 1e-12, 1-1e-12)
loss = -(y*np.log(p) + (1-y)*np.log(1-p))
```

### Tóm tắt
- Log-loss phụ thuộc mạnh vào nhãn đúng.
- Clipping giúp code ổn định số.
- Backprop dùng loss này để chỉnh mọi weight.

## Tổng kết tối ưu hóa

### Trực giác
Tuần này nối giải tích với neural network.
Gradient descent tạo bước học.
Newton cho thấy curvature cải thiện bước học ra sao.
Backprop tính gradient cho rất nhiều tham số.

### Định nghĩa và công thức
Một vòng lặp training thực tế là:

$$
\text{forward}\rightarrow\text{loss}
\rightarrow\text{backprop}\rightarrow\text{update}.
$$

Bước Newton đầy đủ mạnh,
nhưng Hessian thường quá lớn trong deep learning.

### Ví dụ làm đầy đủ
Với một quadratic nhỏ,
Newton có thể tới optimum chỉ trong một bước.
Với mạng có hàng triệu weight,
lập Hessian gần như không thực tế.
First-order optimizer đổi curvature chính xác
lấy nhiều bước rẻ hơn.

> ⚠️ Lỗi thường gặp:
> tin rằng một optimizer tốt nhất cho mọi quy mô.

### NumPy snippet
```python
for batch in data:
    loss, grads = value_and_grad(params, batch)
    params = params - lr * grads
```

### Tóm tắt
- Gradient descent là công cụ chính.
- Newton giúp hiểu curvature.
- Backprop làm gradient trở nên khả thi.
