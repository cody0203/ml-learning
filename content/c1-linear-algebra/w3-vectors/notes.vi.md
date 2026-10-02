# Tuần 3: Vector, ma trận và biến đổi tuyến tính

Tuần này nối trực giác hình học với cách tính toán trong Machine Learning: vector biểu diễn dữ liệu và trọng số; chuẩn (norm) đo độ lớn; dot product biến nhiều đặc trưng thành một score; ma trận mô tả biến đổi tuyến tính; matrix multiplication là cách ghép nhiều biến đổi hoặc xử lý cả batch dữ liệu cùng lúc. Các slide cũng dùng ví dụ spam/perceptron để cho thấy những phép toán này không chỉ là ký hiệu đại số mà là “ngôn ngữ” của neural networks.

## Vector và tọa độ

### Trực giác
Vector (vector) có hai cách nhìn bổ sung cho nhau. Về hình học, vector là một mũi tên có hướng và độ dài. Về tính toán, vector là một danh sách số có thứ tự. Vector $(4,3)$ nghĩa là đi $4$ đơn vị theo trục $x$ và $3$ đơn vị theo trục $y$. Nếu thêm một chiều nữa, $(4,3,1)$ có thể là điểm/dịch chuyển trong không gian 3D hoặc là một vector đặc trưng có 3 feature.

Trong ML, mỗi email, ảnh, câu văn hoặc người dùng thường được biến thành vector. Ví dụ một email có thể được mô tả bởi số lần xuất hiện của các từ khóa: $x=(\text{Lottery},\text{Win})=(2,1)$. Khi ta học trọng số $w$, mô hình đang học cách “đọc” vector đó.

### Định nghĩa và ký hiệu
Vector cột thường được viết:
$$u=\begin{bmatrix}4\\3\end{bmatrix}.$$
Thành phần thứ nhất là $u_1=4$, thành phần thứ hai là $u_2=3$. Hai vector bằng nhau nếu từng thành phần tương ứng bằng nhau. Nếu vẽ hai mũi tên ở hai vị trí khác nhau nhưng cùng dịch chuyển $(4,3)$, trong linear algebra chúng vẫn biểu diễn cùng một vector tự do.

### Ví dụ làm từng bước
Tìm vector từ $A=(1,2)$ đến $B=(5,5)$.

1. Vector “từ $A$ đến $B$” là điểm cuối trừ điểm đầu: $B-A$.
2. Trừ từng tọa độ:
   $$B-A=(5-1,\;5-2).$$
3. Kết quả:
   $$B-A=(4,3).$$
4. Kiểm tra hình học: từ $(1,2)$ đi sang phải $4$ và đi lên $3$ sẽ tới $(5,5)$.

### NumPy nhỏ
```python
import numpy as np
A = np.array([1, 2])
B = np.array([5, 5])
u = B - A          # array([4, 3])
```

> ⚠️ Lỗi thường gặp: nhầm điểm $(4,3)$ với vector dịch chuyển $(4,3)$. Cùng một cặp số có thể biểu diễn vị trí hoặc độ dịch chuyển; hãy đọc ngữ cảnh. Một lỗi khác là coi vector như tập hợp không có thứ tự, trong khi $(4,3)\neq(3,4)$.

### Tóm tắt
- Vector là danh sách số có thứ tự hoặc mũi tên có hướng.
- Vector từ $A$ đến $B$ là $B-A$.
- Trong ML, vector thường biểu diễn feature, weight, embedding hoặc một mẫu dữ liệu.

## Chuẩn và khoảng cách

### Trực giác
Chuẩn (norm) đo “độ lớn” của vector. Nếu vector là một mũi tên, chuẩn là độ dài mũi tên theo một quy tắc đo cụ thể. Slides nhấn mạnh hai chuẩn: $L_1$ giống đi taxi trên lưới phố, còn $L_2$ giống bay thẳng như trực thăng.

Khoảng cách giữa hai vector không được tính bằng cách so từng vector riêng lẻ, mà bằng chuẩn của hiệu:
$$d(u,v)=\|u-v\|.$$
Nghĩa là: muốn biết $u$ và $v$ xa nhau bao nhiêu, trước hết tìm vector dịch chuyển từ $v$ đến $u$, rồi đo độ dài vector đó.

### Công thức
Với $u=(a,b)$:
$$\|u\|_1=|a|+|b|,$$
$$\|u\|_2=\sqrt{a^2+b^2}.$$
Với hai vector:
$$d_1(u,v)=\|u-v\|_1,\qquad d_2(u,v)=\|u-v\|_2.$$

### Ví dụ làm từng bước
Cho $u=(6,2)$ và $v=(1,5)$. Tính khoảng cách $L_1$ và $L_2$.

1. Tính hiệu:
   $$u-v=(6-1,\;2-5)=(5,-3).$$
2. Khoảng cách $L_1$:
   $$d_1(u,v)=|5|+|-3|=5+3=8.$$
3. Khoảng cách $L_2$:
   $$d_2(u,v)=\sqrt{5^2+(-3)^2}=\sqrt{25+9}=\sqrt{34}\approx5.83.$$
4. Diễn giải: nếu chỉ được đi ngang/dọc, cần $8$ bước; nếu đi đường thẳng, ngắn hơn.

### NumPy nhỏ
```python
import numpy as np
u = np.array([6, 2])
v = np.array([1, 5])
diff = u - v
l1 = np.sum(np.abs(diff))
l2 = np.sqrt(np.sum(diff * diff))
```

> ⚠️ Lỗi thường gặp: quên trị tuyệt đối trong $L_1$ nên nhận kết quả âm hoặc quá nhỏ. Với $L_2$, đừng tính $\sqrt{5}+\sqrt{9}$; phải cộng bình phương trước rồi mới lấy căn. Cũng đừng nhầm chuẩn của $u$ với khoảng cách giữa $u$ và $v$.

### Tóm tắt
- $\|u\|_1$ cộng trị tuyệt đối của các thành phần.
- $\|u\|_2$ là độ dài Euclidean theo Pythagoras.
- Khoảng cách giữa hai vector là chuẩn của hiệu vector.

## Hướng và nhân vô hướng

### Trực giác
Một vector có độ lớn và hướng. Vector $(4,3)$ không chỉ dài $5$ mà còn nghiêng lên so với trục $x$. Nhân vector với một scalar (vô hướng) thay đổi độ dài theo cùng một hướng nếu scalar dương, và đảo hướng nếu scalar âm. Đây là ý tưởng quan trọng vì trong ML, trọng số có thể phóng đại, làm yếu đi, hoặc đảo dấu ảnh hưởng của một feature.

### Công thức
Với $u=(a,b)$, nếu $a\neq0$:
$$\tan\theta=\frac{b}{a}.$$
Trong thực hành nên dùng `atan2(b, a)` để đúng góc phần tư. Nhân scalar:
$$\lambda u=\lambda(a,b)=(\lambda a,\lambda b).$$
Độ dài biến đổi theo:
$$\|\lambda u\|=|\lambda|\|u\|.$$

### Ví dụ làm từng bước
Tìm hướng của $u=(4,3)$ và tính $-2u$.

1. Tính tỉ số:
   $$\tan\theta=\frac{3}{4}.$$
2. Lấy arctangent:
   $$\theta=\arctan(3/4)\approx36.87^\circ.$$
3. Nhân scalar:
   $$-2u=-2(4,3)=(-8,-6).$$
4. Diễn giải: vector mới dài gấp đôi nhưng quay ngược hướng $180^\circ$.

### NumPy nhỏ
```python
import numpy as np
u = np.array([4, 3])
angle_deg = np.degrees(np.arctan2(u[1], u[0]))
scaled = -2 * u
```

> ⚠️ Lỗi thường gặp: dùng $\arctan(b/a)$ mà không xét dấu của $a,b$. Ví dụ $(-1,1)$ có $b/a=-1$, nhưng góc đúng là $135^\circ$, không phải $-45^\circ$. Một lỗi khác là nghĩ scalar âm chỉ đổi dấu từng số mà quên ý nghĩa hình học là đảo hướng.

### Tóm tắt
- Scalar dương giữ hướng, scalar âm đảo hướng.
- Độ dài nhân với trị tuyệt đối của scalar.
- Dùng `atan2` để lấy góc đúng phần tư.

## Cộng/trừ vector

### Trực giác
Cộng vector tương ứng với ghép các bước di chuyển: đi theo $u$, rồi đi tiếp theo $v$, kết quả là $u+v$. Trừ vector $u-v$ là vector từ đầu mũi tên $v$ đến đầu mũi tên $u$ khi đặt chung gốc. Đây là lý do phép trừ vector xuất hiện tự nhiên trong khoảng cách.

### Công thức
Với $u=(a,b)$ và $v=(c,d)$:
$$u+v=(a+c,b+d),$$
$$u-v=(a-c,b-d).$$
Phép cộng có tính giao hoán: $u+v=v+u$. Phép trừ thì không: $u-v=-(v-u)$.

### Ví dụ làm từng bước
Cho $u=(4,1)$ và $v=(1,3)$.

1. Cộng:
   $$u+v=(4+1,\;1+3)=(5,4).$$
2. Trừ:
   $$u-v=(4-1,\;1-3)=(3,-2).$$
3. Kiểm tra hình học: nếu đầu $u$ ở $(4,1)$ và đầu $v$ ở $(1,3)$, vector từ đầu $v$ đến đầu $u$ là $(4-1,1-3)$.

### NumPy nhỏ
```python
import numpy as np
u = np.array([4, 1])
v = np.array([1, 3])
sum_uv = u + v
diff_uv = u - v
```

> ⚠️ Lỗi thường gặp: do OCR trong slide có thể làm dấu trừ nhìn giống dấu chấm, nhưng hiệu vector không phải nhân component-wise. Hãy trừ từng thành phần theo đúng thứ tự. Cũng cần chú ý $u-v$ khác $v-u$.

### Tóm tắt
- Cộng/trừ vector thực hiện theo từng thành phần.
- $u+v$ là tổng dịch chuyển.
- $u-v$ là vector dùng để đo khoảng cách và sai khác.

## Tích vô hướng

### Trực giác
Tích vô hướng (dot product) biến hai vector cùng chiều thành một scalar. Nó là phép toán “nhân tương ứng rồi cộng lại”. Trong ví dụ giỏ hàng: quantities dot prices = tổng tiền. Trong ML: features dot weights = score.

Nếu $x$ là vector feature và $w$ là vector weight, dot product trả lời câu hỏi: “mẫu này khớp với trọng số của mô hình mạnh đến mức nào?”. Feature có weight lớn sẽ đóng góp nhiều hơn vào score.

### Công thức
Với $u=(u_1,\dots,u_n)$ và $v=(v_1,\dots,v_n)$:
$$u\cdot v=\sum_{i=1}^n u_i v_i.$$
Liên hệ với chuẩn:
$$\|u\|_2=\sqrt{u\cdot u}.$$

### Ví dụ làm từng bước
Tính $(2,4,1)\cdot(3,5,2)$.

1. Nhân từng cặp thành phần:
   $$2\cdot3=6,\qquad 4\cdot5=20,\qquad 1\cdot2=2.$$
2. Cộng lại:
   $$6+20+2=28.$$
3. Kết luận:
   $$(2,4,1)\cdot(3,5,2)=28.$$
4. Nếu đây là quantities và prices, tổng tiền là $28$.

### NumPy nhỏ
```python
import numpy as np
q = np.array([2, 4, 1])
p = np.array([3, 5, 2])
total = np.sum(q * p)   # or q @ p
```

> ⚠️ Lỗi thường gặp: trả về vector các tích $[6,20,2]$ mà quên cộng lại. Dot product trả về scalar. Ngoài ra, hai vector phải cùng số chiều; không thể dot vector dài 3 với vector dài 2 theo định nghĩa này.

### Tóm tắt
- Dot product = nhân từng thành phần rồi cộng.
- Kết quả là scalar.
- $\sqrt{u\cdot u}$ là chuẩn $L_2$.
- Linear model thường dùng score $w\cdot x$.

## Hình học của tích vô hướng

### Trực giác
Dot product còn có ý nghĩa hình học: nó đo mức “cùng hướng” của hai vector. Nếu hai vector cùng hướng nhiều, dot product dương lớn. Nếu vuông góc, dot product bằng 0. Nếu ngược hướng, dot product âm. Đây là nền tảng cho cosine similarity, projection, orthogonality và nhiều thuật toán ML.

### Công thức
Với hai vector khác zero:
$$u\cdot v=\|u\|\|v\|\cos\theta.$$
Từ đó:
$$\cos\theta=\frac{u\cdot v}{\|u\|\|v\|}.$$
Nếu $u\cdot v=0$, thì $\cos\theta=0$ và $\theta=90^\circ$.

### Ví dụ làm từng bước
Xét $u=(6,2)$ và $v=(-1,3)$.

1. Tính dot product:
   $$u\cdot v=6(-1)+2(3)=-6+6=0.$$
2. Hai vector đều khác zero.
3. Vì dot product bằng 0, góc giữa chúng là $90^\circ$.
4. Kết luận: $u$ và $v$ trực giao (orthogonal).

Ví dụ dấu dot product:
$$(6,2)\cdot(2,4)=12+8=20>0,$$
nên góc là góc nhọn. Còn
$$(6,2)\cdot(-4,1)=-24+2=-22<0,$$
nên góc là góc tù.

### NumPy nhỏ
```python
import numpy as np
u = np.array([6, 2])
v = np.array([-1, 3])
dot = u @ v
angle = np.degrees(np.arccos(dot / (np.linalg.norm(u) * np.linalg.norm(v))))
```

> ⚠️ Lỗi thường gặp: nói “dot product bằng 0” luôn có nghĩa là “hai vector có góc $90^\circ$” mà quên trường hợp vector zero. Vector zero không có hướng xác định, nên góc với nó không được định nghĩa tốt.

### Tóm tắt
- Dot dương: góc nhọn; dot bằng 0: trực giao; dot âm: góc tù.
- Công thức geometric dot product dùng cosine.
- Orthogonality là ý tưởng quan trọng trong projection và tối ưu hóa.

## Ma trận nhân vector

### Trực giác
Ma trận nhân vector là nhiều dot product chạy song song. Mỗi hàng của ma trận là một “bộ trọng số” tạo ra một thành phần output. Vì vậy một hệ phương trình tuyến tính có thể được viết gọn thành $Ax=b$.

Nếu $A$ là $m\times n$ và $x$ có $n$ thành phần, thì $Ax$ có $m$ thành phần. Output thứ $i$ được tạo bởi hàng $i$ của $A$ dot với $x$.

### Công thức
$$(Ax)_i=\sum_{j=1}^n A_{ij}x_j.$$
Với hệ:
$$a+b+c=10,\qquad a+2b+c=15,\qquad a+b+2c=12,$$
ta viết:
$$
\begin{bmatrix}1&1&1\\1&2&1\\1&1&2\end{bmatrix}
\begin{bmatrix}a\\b\\c\end{bmatrix}
=\begin{bmatrix}10\\15\\12\end{bmatrix}.
$$

### Ví dụ làm từng bước
Tính:
$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad x=\begin{bmatrix}2\\1\end{bmatrix}.$$

1. Hàng 1 dot với $x$:
   $$1\cdot2+2\cdot1=4.$$
2. Hàng 2 dot với $x$:
   $$3\cdot2+4\cdot1=10.$$
3. Xếp kết quả thành vector cột:
   $$Ax=\begin{bmatrix}4\\10\end{bmatrix}.$$

### NumPy nhỏ
```python
import numpy as np
A = np.array([[1, 2],
              [3, 4]])
x = np.array([2, 1])
y = A @ x
```

> ⚠️ Lỗi thường gặp: nhân sai shape. Ma trận $3\times2$ chỉ nhân được với vector dài 2, không phải vector dài 3. Một lỗi khác là lấy cột dot với vector thay vì hàng dot với vector.

### Tóm tắt
- $Ax$ là nhiều dot product giữa hàng của $A$ và vector $x$.
- Nếu $A$ là $m\times n$, $x$ phải dài $n$ và $Ax$ dài $m$.
- Hệ phương trình tuyến tính có thể viết thành một phương trình vector.

## Ma trận là biến đổi tuyến tính

### Trực giác
Ma trận không chỉ là bảng số; nó là một hàm biến vector input thành vector output. Với ma trận $2\times2$, ta có thể nhìn nó như phép biến đổi toàn bộ mặt phẳng: kéo giãn, nén, shear, xoay, phản chiếu hoặc kết hợp các thao tác đó.

Điểm quan trọng trong slides: chỉ cần biết ảnh của hai vector cơ sở $e_1=(1,0)$ và $e_2=(0,1)$ là biết toàn bộ biến đổi. Hai ảnh này chính là hai cột của ma trận.

### Công thức
Nếu:
$$A=\begin{bmatrix}3&1\\1&2\end{bmatrix},$$
thì:
$$Ae_1=\begin{bmatrix}3\\1\end{bmatrix},\qquad Ae_2=\begin{bmatrix}1\\2\end{bmatrix}.$$
Với vector bất kỳ $(a,b)$:
$$
A\begin{bmatrix}a\\b\end{bmatrix}
=aAe_1+bAe_2
=\begin{bmatrix}3a+b\\a+2b\end{bmatrix}.
$$

### Ví dụ làm từng bước
Biết $e_1\mapsto(3,-1)$ và $e_2\mapsto(2,3)$. Dựng ma trận.

1. Ảnh của $e_1$ là cột thứ nhất:
   $$\begin{bmatrix}3\\-1\end{bmatrix}.$$
2. Ảnh của $e_2$ là cột thứ hai:
   $$\begin{bmatrix}2\\3\end{bmatrix}.$$
3. Ghép hai cột:
   $$A=\begin{bmatrix}3&2\\-1&3\end{bmatrix}.$$
4. Kiểm tra: $A(1,1)=(3,-1)+(2,3)=(5,2)$.

### NumPy nhỏ
```python
import numpy as np
A = np.array([[3, 2],
              [-1, 3]])
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
transformed = points @ A.T   # points stored as rows
```

> ⚠️ Lỗi thường gặp: đặt ảnh của $e_1,e_2$ thành hàng thay vì cột. Nếu làm vậy, $Ae_1$ sẽ không còn là ảnh mong muốn. Một lỗi khác là quên rằng quy ước dữ liệu trong code có thể lưu điểm theo hàng, nên cần dùng $A^T$ khi nhân từ bên phải.

### Tóm tắt
- Cột của ma trận là ảnh của các vector cơ sở.
- Biến đổi tuyến tính bảo toàn cộng vector và nhân scalar.
- Hình vuông đơn vị biến thành hình bình hành xác định bởi hai cột của ma trận.

## Nhân ma trận

### Trực giác
Nhân ma trận là cách ghép các biến đổi tuyến tính. Nếu một điểm đi qua biến đổi $B$ trước, rồi qua biến đổi $A$, biến đổi tổng hợp là $AB$. Vì vậy thứ tự nhân quan trọng: “bên phải làm trước”.

Về mặt tính toán, mỗi phần tử của $AB$ là dot product giữa một hàng của $A$ và một cột của $B$. Điều này mở rộng trực tiếp từ matrix-vector product.

### Công thức
Nếu $A$ là $m\times k$ và $B$ là $k\times n$, thì $AB$ là $m\times n$:
$$(AB)_{ij}=\text{row}_i(A)\cdot\text{col}_j(B).$$

### Ví dụ làm từng bước
Tính:
$$
A=\begin{bmatrix}2&-1\\0&2\end{bmatrix},\qquad
B=\begin{bmatrix}3&1\\1&2\end{bmatrix}.
$$

1. Entry $(1,1)$:
   $$2\cdot3+(-1)\cdot1=5.$$
2. Entry $(1,2)$:
   $$2\cdot1+(-1)\cdot2=0.$$
3. Entry $(2,1)$:
   $$0\cdot3+2\cdot1=2.$$
4. Entry $(2,2)$:
   $$0\cdot1+2\cdot2=4.$$
5. Kết quả:
   $$AB=\begin{bmatrix}5&0\\2&4\end{bmatrix}.$$

### NumPy nhỏ
```python
import numpy as np
A = np.array([[2, -1],
              [0,  2]])
B = np.array([[3, 1],
              [1, 2]])
C = A @ B
```

> ⚠️ Lỗi thường gặp: giả sử $AB=BA$. Thường hai tích này khác nhau hoặc một trong hai thậm chí không hợp lệ vì shape không khớp. Hãy đọc thứ tự biến đổi: nếu $B$ áp dụng trước rồi $A$, viết $AB$.

### Tóm tắt
- Matrix multiplication là row-dot-column.
- Kích thước trong phải khớp.
- Thứ tự nhân biểu diễn thứ tự ghép biến đổi và thường không giao hoán.

## Ma trận đơn vị và nghịch đảo

### Trực giác
Ma trận đơn vị (identity matrix) là biến đổi “không làm gì”: mọi vector đi vào đều giữ nguyên. Ma trận nghịch đảo (inverse) làm ngược lại một biến đổi. Nếu $A$ kéo/shear mặt phẳng mà không làm mất chiều nào, $A^{-1}$ đưa kết quả trở về ban đầu.

Không phải ma trận nào cũng có inverse. Nếu một ma trận làm nhiều vector khác nhau rơi vào cùng một output, thông tin đã mất và không thể đảo ngược. Trong 2D, điều này tương ứng determinant bằng 0.

### Công thức
$$AI=IA=A,\qquad AA^{-1}=A^{-1}A=I.$$
Với:
$$A=\begin{bmatrix}a&b\\c&d\end{bmatrix},$$
nếu $ad-bc\neq0$ thì:
$$A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}.$$

### Ví dụ làm từng bước
Tìm inverse của:
$$A=\begin{bmatrix}5&2\\1&2\end{bmatrix}.$$

1. Tính determinant:
   $$\det(A)=5\cdot2-2\cdot1=8.$$
2. Vì $8\neq0$, inverse tồn tại.
3. Đổi vị trí $a,d$ và đổi dấu $b,c$:
   $$\begin{bmatrix}2&-2\\-1&5\end{bmatrix}.$$
4. Chia cho determinant:
$$
A^{-1}=\frac18\begin{bmatrix}2&-2\\-1&5\end{bmatrix}
   =\begin{bmatrix}1/4&-1/4\\-1/8&5/8\end{bmatrix}.
$$

Ví dụ singular:
$$\begin{bmatrix}1&1\\2&2\end{bmatrix}$$
có hàng 2 bằng 2 lần hàng 1, determinant $0$, nên không có inverse.

### NumPy nhỏ
```python
import numpy as np
A = np.array([[5, 2],
              [1, 2]], dtype=float)
det = np.linalg.det(A)
inv = np.linalg.inv(A)  # only safe when det is not near zero
```

> ⚠️ Lỗi thường gặp: áp dụng công thức inverse khi determinant bằng 0. Một lỗi khác là quên nhân toàn bộ ma trận bởi $\frac{1}{ad-bc}$, hoặc chỉ đổi vị trí $a,d$ mà quên đổi dấu $b,c$.

### Tóm tắt
- Identity giữ nguyên mọi vector.
- Inverse đảo ngược biến đổi tuyến tính.
- Ma trận vuông có inverse khi non-singular; với $2\times2$, determinant khác 0.

## Mạng neural và ma trận

### Trực giác
Neural networks dùng các phép toán tuần này ở quy mô lớn. Một perceptron tính score tuyến tính:
$$s=w\cdot x+b.$$
Sau đó so sánh score với threshold hoặc với 0 để quyết định class. Khi có nhiều ví dụ, ta xếp chúng thành ma trận $X$; khi có nhiều neuron, ta xếp trọng số thành ma trận $W$. Linear layer khi đó là:
$$Y=XW+b.$$

### Ví dụ từ slide spam
Feature là số lần xuất hiện “Lottery” và “Win”. Chọn:
$$w=\begin{bmatrix}1\\1\end{bmatrix},\qquad \text{threshold}=1.5.$$
Email “Win, win the lottery!” có $x=(1,2)$ nếu Lottery xuất hiện 1 lần và Win xuất hiện 2 lần.

1. Tính score:
   $$w\cdot x=1\cdot1+1\cdot2=3.$$
2. So sánh threshold:
   $$3>1.5.$$
3. Dự đoán: spam.

Thêm bias $b=-1.5$ biến quy tắc thành:
$$w\cdot x-1.5>0.$$

### Batch example
Với:
$$
X=\begin{bmatrix}1&1\\2&1\\0&0\\0&2\end{bmatrix},\qquad
w=\begin{bmatrix}1\\1\end{bmatrix},
$$
ta có:
$$Xw=\begin{bmatrix}2\\3\\0\\2\end{bmatrix}.$$
So sánh với $1.5$ cho kết quả: spam, spam, không spam, spam.

### NumPy nhỏ
```python
import numpy as np
X = np.array([[1, 1],
              [2, 1],
              [0, 0],
              [0, 2]])
w = np.array([1, 1])
scores = X @ w
pred = scores > 1.5
```

> ⚠️ Lỗi thường gặp: nhầm quy ước shape. Nếu mỗi hàng là một ví dụ, dùng $Xw$. Nếu mỗi cột là một ví dụ, công thức sẽ khác. Một lỗi khác là quên bias/threshold, làm decision boundary đi qua gốc thay vì được dịch chuyển.

### Tóm tắt
- Perceptron là dot product cộng bias rồi threshold.
- Bias có thể xem như weight của một feature luôn bằng 1.
- Matrix multiplication giúp tính score cho cả batch và cho nhiều neuron cùng lúc.
'''
root = Path(r'G:\ml-learning')
(root/'content'/'c1-linear-algebra'/'w3-vectors'/'notes.vi.md').write_text(notes_vi, encoding='utf-8')
print(len(notes_vi))
'@ | python -","description":"Prepare expanded Vietnamese notes","initial_wait":30} horrifying
