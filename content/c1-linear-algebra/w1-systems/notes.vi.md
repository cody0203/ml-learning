# Tuần 1: Hệ phương trình tuyến tính

## Động lực ML và phép toán ma trận

### Trực giác
Trong học máy, dữ liệu hiếm khi chỉ là một số đơn lẻ. Một ảnh có hàng nghìn pixel, một bảng dữ liệu có nhiều cột đặc trưng, và một mạng nơ-ron có nhiều lớp trọng số. Đại số tuyến tính cho ta ngôn ngữ để gom các số đó thành vector và ma trận, rồi biến đổi chúng bằng các phép nhân ma trận.

Tuần 1 bắt đầu từ một câu hỏi rất cơ bản: khi ta có nhiều thông tin tuyến tính về các biến, thông tin đó có đủ để xác định các biến không? Nếu đủ, ta có một nghiệm duy nhất. Nếu thông tin bị lặp, ta có nhiều nghiệm. Nếu thông tin mâu thuẫn, ta không có nghiệm. Đây cũng là trực giác đằng sau nhiều bài toán ML: dữ liệu và ràng buộc có đủ độc lập để xác định mô hình không?

### Định nghĩa và công thức
Một hệ phương trình tuyến tính có dạng tổng quát:

$$
Ax=b
$$

trong đó $A$ là ma trận hệ số, $x$ là vector ẩn, và $b$ là vector vế phải. Trong một lớp tuyến tính đơn giản của ML, ta cũng thường gặp dạng:

$$
y=Wx
$$

với $x$ là đầu vào, $W$ là ma trận trọng số, và $y$ là đầu ra trước khi áp dụng bias hoặc hàm kích hoạt.

### Ví dụ làm mẫu
Giả sử một mô hình toy tạo ra hai điểm số:

$$
\begin{cases}
s_1=a+b\\
s_2=a+2b
\end{cases}
$$

Với một đầu vào cụ thể, ta quan sát $s_1=10$ và $s_2=12$. Khi đó:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

Trừ phương trình thứ nhất khỏi phương trình thứ hai:

$$
(a+2b)-(a+b)=12-10
$$

$$
b=2
$$

Thay vào $a+b=10$:

$$
a+2=10 \Rightarrow a=8
$$

Vậy hai điểm số độc lập giúp khôi phục duy nhất $(a,b)=(8,2)$.

### NumPy liên quan
```python
import numpy as np

W = np.array([[1, 1],
              [1, 2]])
x = np.array([8, 2])
print(W @ x)  # [10 12]
```

Ở đây mỗi hàng của $W$ là một ràng buộc tuyến tính; phép `W @ x` tính đồng thời hai vế trái.

> ⚠️ Common mistakes: Đừng kết luận “có hai phương trình và hai ẩn nên chắc chắn có nghiệm duy nhất”. Hai phương trình có thể là cùng một thông tin lặp lại, hoặc có thể mâu thuẫn. Cũng đừng nghĩ ML chỉ là tuyến tính; mô hình hiện đại có phi tuyến, nhưng các khối tuyến tính vẫn là nền tảng tính toán.

### Tóm tắt
- Ma trận giúp biểu diễn nhiều số và nhiều phép tính cùng lúc.
- Hệ tuyến tính là mô hình nhỏ để hiểu thông tin đủ, thừa, hoặc mâu thuẫn.
- Trong ML, phép nhân ma trận xuất hiện trong xử lý ảnh, bảng dữ liệu, embedding, và neural networks.
- Trước khi giải, hãy hỏi: các ràng buộc có độc lập và nhất quán không?

## Hệ thông tin: đủ, thừa, mâu thuẫn

### Trực giác
Các slide dùng hệ câu về chó, mèo, chim để giúp ta hiểu “hệ” trước khi có số. Một hệ thông tin là tập các câu cùng mô tả một tình huống. Có hệ cho ta đúng một kết luận, có hệ lặp lại điều đã biết, và có hệ tự phủ định chính nó.

Đây là bản chất của singularity ở mức trực giác. Nếu thông tin đủ và không mâu thuẫn, ta có một trạng thái duy nhất. Nếu thông tin thừa, ta không phân biệt được nhiều khả năng. Nếu thông tin mâu thuẫn, không có khả năng nào thỏa tất cả.

### Định nghĩa và công thức
- **Complete / non-singular**: đủ thông tin và nhất quán để xác định một kết quả duy nhất.
- **Redundant / singular**: có thông tin thừa hoặc lặp, nên vẫn còn nhiều khả năng.
- **Contradictory / singular**: có thông tin mâu thuẫn, nên không có khả năng nào.

Trong hệ phương trình sau này, ba trường hợp này tương ứng với:

$$
\text{unique solution},\quad \text{infinitely many solutions},\quad \text{no solution}
$$

### Ví dụ làm mẫu
Xét hệ câu:

1. Trong chó, mèo, chim, đúng một con màu đỏ.
2. Trong chó và mèo, đúng một con màu cam.
3. Chó màu đen.

Giải từng bước:

1. Chó đã màu đen, nên chó không phải con màu cam trong cặp chó/mèo.
2. Vì đúng một trong chó/mèo màu cam, mèo phải màu cam.
3. Chó đen và mèo cam, nên trong ba con, con còn lại là chim.
4. Vì đúng một con màu đỏ, chim phải màu đỏ.

Hệ này là complete/non-singular đối với câu hỏi “chim màu gì?” vì nó xác định duy nhất: chim màu đỏ.

### NumPy liên quan
Hệ câu chưa cần NumPy, nhưng ta có thể mô phỏng ý tưởng “kiểm tra tất cả khả năng”:

```python
animals = ["dog", "cat", "bird"]
colors = {"dog": "black", "cat": "orange", "bird": "red"}
print(colors["bird"])  # red
```

Khi chuyển sang phương trình, “kiểm tra khả năng” sẽ được thay bằng giải hệ hoặc kiểm tra rank/determinant.

> ⚠️ Common mistakes: “Singular” không đồng nghĩa duy nhất với “vô nghiệm”. Redundant cũng singular, nhưng có nhiều nghiệm/khả năng. Contradictory mới là trường hợp không có nghiệm. Ngoài ra, một câu lặp lại không làm hệ mạnh hơn.

### Tóm tắt
- Complete: đủ và nhất quán, một kết luận.
- Redundant: có câu thừa, nhiều kết luận còn có thể.
- Contradictory: các câu không thể cùng đúng.
- Đây là phiên bản bằng ngôn ngữ tự nhiên của unique / infinite / none.

## Từ câu chữ đến hệ phương trình

### Trực giác
Khi câu chữ có số lượng và tổng tiền, ta có thể gán biến rồi viết phương trình. “Một táo và một chuối giá 10” trở thành $a+b=10$. Mỗi ngày mua hàng là một ràng buộc mới. Nếu các ngày cung cấp thông tin độc lập, ta tìm được giá từng món.

Điều quan trọng không chỉ là “giải ra số”, mà là nhận ra hệ thuộc loại nào. Một hệ có thể có nghiệm duy nhất, vô số nghiệm, hoặc không có nghiệm.

### Định nghĩa và công thức
Với hai biến $a,b$, một hệ tuyến tính có dạng:

$$
\begin{cases}
p_1a+q_1b=r_1\\
p_2a+q_2b=r_2
\end{cases}
$$

Một kỹ thuật cơ bản là **khử biến (elimination)**: cộng, trừ, hoặc nhân phương trình với hằng số khác 0 để loại một biến.

### Ví dụ làm mẫu
Ngày 1: mua một táo và một chuối, trả 10.

Ngày 2: mua một táo và hai chuối, trả 12.

Gọi $a$ là giá táo, $b$ là giá chuối:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

Bước 1: lấy phương trình 2 trừ phương trình 1:

$$
(a+2b)-(a+b)=12-10
$$

$$
b=2
$$

Bước 2: thay $b=2$ vào phương trình 1:

$$
a+2=10
$$

$$
a=8
$$

Bước 3: kiểm tra:

$$
8+2=10,\quad 8+2\cdot2=12
$$

Vậy táo giá 8, chuối giá 2.

### NumPy liên quan
```python
import numpy as np

A = np.array([[1, 1],
              [1, 2]], dtype=float)
b = np.array([10, 12], dtype=float)
x = np.linalg.solve(A, b)
print(x)  # [8. 2.]
```

Trong bài học, ta nên biết giải tay bằng khử biến; NumPy giúp kiểm tra nhanh và mở rộng lên hệ lớn.

> ⚠️ Common mistakes: Nếu phương trình thứ hai là $2a+2b=20$, nó chỉ là hai lần phương trình đầu $a+b=10$, không cho thông tin mới. Nếu là $2a+2b=24$, nó mâu thuẫn với phương trình đầu vì phương trình đầu nhân 2 phải cho 20 chứ không phải 24.

### Tóm tắt
- Câu chữ → chọn biến → viết phương trình.
- Khử biến là thao tác chính để giải hệ nhỏ.
- Nhân một phương trình với hằng số khác 0 không tạo thông tin mới.
- Luôn kiểm tra nghiệm bằng cách thay lại vào tất cả phương trình.

## Phương trình tuyến tính như đường thẳng

### Trực giác
Trong hai biến, một phương trình tuyến tính là một đường thẳng. Mỗi điểm trên đường là một nghiệm của phương trình đó. Một hệ hai phương trình là bài toán tìm điểm chung của hai đường.

Hình học giúp ta nhìn thấy ba trường hợp: hai đường cắt nhau tại một điểm, trùng nhau hoàn toàn, hoặc song song khác nhau. Đây chính là unique, infinite, none.

### Định nghĩa và công thức
Phương trình tuyến tính hai biến có dạng:

$$
pa+qb=r
$$

Nếu $q\ne0$, có thể viết:

$$
b=-\frac{p}{q}a+\frac{r}{q}
$$

Độ dốc là $-\frac{p}{q}$ và tung độ gốc là $\frac{r}{q}$.

### Ví dụ làm mẫu
Xét hệ:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

Đường thứ nhất:

$$
b=-a+10
$$

Một vài điểm trên đường: $(10,0)$, $(0,10)$, $(8,2)$.

Đường thứ hai:

$$
2b=-a+12 \Rightarrow b=-\frac12a+6
$$

Một vài điểm trên đường: $(12,0)$, $(0,6)$, $(8,2)$.

Hai đường có độ dốc khác nhau, nên cắt nhau. Điểm chung là $(8,2)$, vì:

$$
8+2=10,\quad 8+2\cdot2=12
$$

### NumPy liên quan
```python
import numpy as np

A = np.array([[1, 1],
              [1, 2]], dtype=float)
b = np.array([10, 12], dtype=float)
point = np.linalg.solve(A, b)
print(point)  # [8. 2.]
```

Nếu determinant của $A$ bằng 0, `np.linalg.solve` sẽ báo lỗi vì hai đường không có giao duy nhất.

> ⚠️ Common mistakes: Đừng chỉ nhìn hai phương trình “khác nhau” rồi kết luận chúng cắt nhau. $a+b=10$ và $2a+2b=24$ có cùng độ dốc nhưng tung độ gốc khác nhau, nên song song và vô nghiệm.

### Tóm tắt
- Một phương trình hai biến là một đường.
- Hai đường cắt nhau: một nghiệm, non-singular.
- Hai đường trùng: vô số nghiệm, singular redundant.
- Hai đường song song khác nhau: vô nghiệm, singular contradictory.

## Hệ dưới dạng ma trận và singularity

### Trực giác
Viết hệ dưới dạng ma trận giúp ta tách hai phần: cấu trúc hệ số và hằng số quan sát được. Ma trận hệ số $A$ nói các biến được kết hợp như thế nào. Vector $b$ nói các kết hợp đó phải bằng bao nhiêu.

Singularity là tính chất của $A$. Nếu $A$ non-singular, phép biến đổi từ $x$ sang $Ax$ không làm mất thông tin và có thể đảo ngược. Nếu $A$ singular, ít nhất một ràng buộc phụ thuộc vào ràng buộc khác, nên thông tin bị mất hoặc bị mâu thuẫn với $b$.

### Định nghĩa và công thức
Hệ:

$$
\begin{cases}
a+b=10\\
a+2b=12
\end{cases}
$$

có dạng:

$$
A=\begin{bmatrix}1&1\\1&2\end{bmatrix},\quad
x=\begin{bmatrix}a\\b\end{bmatrix},\quad
b=\begin{bmatrix}10\\12\end{bmatrix}
$$

và:

$$
Ax=b
$$

Với ma trận vuông:

- non-singular: có nghiệm duy nhất cho mọi $b$;
- singular: có thể vô số nghiệm hoặc vô nghiệm tùy $b$.

### Ví dụ làm mẫu
So sánh hai ma trận hệ số:

$$
A_1=\begin{bmatrix}1&1\\1&2\end{bmatrix}
$$

Hai hàng không phải bội của nhau, nên hệ có thể có nghiệm duy nhất.

$$
A_2=\begin{bmatrix}1&1\\2&2\end{bmatrix}
$$

Hàng 2 = 2 × hàng 1. Vì vậy phương trình thứ hai không thêm hướng thông tin mới. Nếu $b=\begin{bmatrix}10\\20\end{bmatrix}$, hệ có vô số nghiệm vì hai phương trình là cùng một đường. Nếu $b=\begin{bmatrix}10\\24\end{bmatrix}$, hệ vô nghiệm vì cùng một vế trái bị buộc bằng hai giá trị không tương thích.

### NumPy liên quan
```python
import numpy as np

A = np.array([[1, 1],
              [2, 2]], dtype=float)
print(np.linalg.matrix_rank(A))  # 1
print(np.linalg.det(A))          # 0.0
```

Rank nhỏ hơn số biến cho thấy thiếu ràng buộc độc lập.

> ⚠️ Common mistakes: Đừng đưa cột hằng số vào khi tính determinant của ma trận hệ số. Ma trận mở rộng $[A|b]$ hữu ích để xét nhất quán, nhưng singular/non-singular của hệ số là thuộc tính của riêng $A$.

### Tóm tắt
- $A$ chứa hệ số, $x$ chứa biến, $b$ chứa hằng số.
- Singularity phụ thuộc vào $A$.
- Với $A$ singular, cùng một $A$ có thể cho vô số nghiệm hoặc vô nghiệm tùy $b$.
- Rank và determinant là hai công cụ kiểm tra cấu trúc của $A$.

## Phụ thuộc và độc lập tuyến tính

### Trực giác
Một phương trình chỉ hữu ích nếu nó thêm một ràng buộc mới. Nếu một hàng của ma trận được tạo từ các hàng khác, nó không mang thông tin độc lập. Đó là phụ thuộc tuyến tính.

Trong 2×2, phụ thuộc thường dễ thấy: một hàng là bội của hàng kia. Trong 3×3, tinh tế hơn: một hàng có thể là tổng, hiệu, hoặc trung bình của hai hàng khác.

### Định nghĩa và công thức
Các hàng $r_1,\ldots,r_k$ phụ thuộc tuyến tính nếu tồn tại các hệ số không phải tất cả bằng 0 sao cho:

$$
c_1r_1+c_2r_2+\cdots+c_kr_k=0
$$

Nếu không có quan hệ như vậy, chúng độc lập tuyến tính.

Với ma trận vuông trong tuần này:

$$
\text{hàng độc lập} \Longleftrightarrow \text{non-singular} \Longleftrightarrow \det(A)\ne0
$$

### Ví dụ làm mẫu
Xét:

$$
A=\begin{bmatrix}
1&0&0\\
0&1&0\\
1&1&0
\end{bmatrix}
$$

Gọi các hàng là:

$$
r_1=(1,0,0),\quad r_2=(0,1,0),\quad r_3=(1,1,0)
$$

Ta thấy:

$$
r_1+r_2=(1,0,0)+(0,1,0)=(1,1,0)=r_3
$$

Vậy hàng 3 phụ thuộc vào hàng 1 và hàng 2. Ma trận không có ba ràng buộc độc lập, nên singular.

### NumPy liên quan
```python
import numpy as np

A = np.array([[1, 0, 0],
              [0, 1, 0],
              [1, 1, 0]], dtype=float)
print(np.linalg.matrix_rank(A))  # 2
```

Rank bằng 2 nghĩa là chỉ có hai hướng hàng độc lập.

> ⚠️ Common mistakes: Đừng chỉ kiểm tra xem có hàng nào là bội của hàng khác hay không. Trong 3×3, hàng phụ thuộc có thể là tổng của hai hàng khác, như ví dụ trên.

### Tóm tắt
- Phụ thuộc tuyến tính nghĩa là có ràng buộc thừa.
- Độc lập tuyến tính nghĩa là mỗi hàng thêm một thông tin mới.
- Rank đếm số hàng/cột độc lập.
- Với ma trận vuông, phụ thuộc tuyến tính liên hệ trực tiếp với determinant bằng 0.

## Định thức 2×2

### Trực giác
Định thức 2×2 là một con số tóm tắt liệu hai ràng buộc có độc lập hay không. Về hình học, determinant đo diện tích có hướng của hình bình hành tạo bởi hai vector hàng hoặc cột. Nếu diện tích bằng 0, hai vector nằm trên cùng một đường, nên thông tin bị “dẹt” và không thể đảo ngược.

### Định nghĩa và công thức
Với:

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
$$

định thức là:

$$
\det(A)=ad-bc
$$

Nếu $\det(A)=0$, ma trận singular. Nếu $\det(A)\ne0$, ma trận non-singular.

### Ví dụ làm mẫu
Ma trận 1:

$$
A=\begin{bmatrix}5&1\\-1&3\end{bmatrix}
$$

Tính:

$$
\det(A)=5\cdot3-1\cdot(-1)
$$

$$
=15+1=16
$$

Vì $16\ne0$, $A$ non-singular.

Ma trận 2:

$$
B=\begin{bmatrix}2&-1\\-6&3\end{bmatrix}
$$

Tính:

$$
\det(B)=2\cdot3-(-1)\cdot(-6)
$$

$$
=6-6=0
$$

Vì determinant bằng 0, $B$ singular.

### NumPy liên quan
```python
import numpy as np

A = np.array([[5, 1],
              [-1, 3]], dtype=float)
print(np.linalg.det(A))  # approximately 16
```

Khi học công thức, hãy tự tính $ad-bc$ trước rồi dùng NumPy kiểm tra.

> ⚠️ Common mistakes: Lỗi phổ biến nhất là dấu âm. Với $(-1)(-6)$, tích là $+6$, nên $6-6=0$, không phải $6-(-6)=12$. Cũng đừng dùng $ad+bc$.

### Tóm tắt
- 2×2 determinant = đường chéo chính trừ đường chéo phụ.
- Determinant bằng 0 ⇔ singular ⇔ không có nghiệm duy nhất.
- Determinant khác 0 ⇔ non-singular ⇔ có nghiệm duy nhất cho mọi $b$.
- Dấu âm trong $ad-bc$ rất quan trọng.

## Hệ phương trình 3×3

### Trực giác
Với ba biến, ta thường cần ba ràng buộc độc lập để xác định nghiệm duy nhất. Nhưng “ba phương trình” không tự động có nghĩa “đủ thông tin”. Nếu một phương trình là hệ quả của các phương trình khác, hệ vẫn còn tự do. Nếu các phương trình xung đột, hệ vô nghiệm.

Trong ví dụ trái cây của slide, ba ngày mua hàng cung cấp đủ thông tin độc lập để tìm giá táo, chuối, cherry.

### Định nghĩa và công thức
Hệ 3×3 có dạng:

$$
\begin{cases}
a_{11}x+a_{12}y+a_{13}z=b_1\\
a_{21}x+a_{22}y+a_{23}z=b_2\\
a_{31}x+a_{32}y+a_{33}z=b_3
\end{cases}
$$

Nếu ma trận hệ số 3×3 non-singular, hệ có nghiệm duy nhất. Nếu singular, ta phải xét thêm vế phải để biết vô số nghiệm hay vô nghiệm.

### Ví dụ làm mẫu
Hệ trái cây:

$$
\begin{cases}
a+b+c=10\\
a+2b+c=15\\
a+b+2c=12
\end{cases}
$$

Bước 1: lấy phương trình 2 trừ phương trình 1:

$$
(a+2b+c)-(a+b+c)=15-10
$$

$$
b=5
$$

Bước 2: lấy phương trình 3 trừ phương trình 1:

$$
(a+b+2c)-(a+b+c)=12-10
$$

$$
c=2
$$

Bước 3: thay vào phương trình 1:

$$
a+5+2=10
$$

$$
a=3
$$

Bước 4: kiểm tra:

$$
3+5+2=10,\quad 3+2\cdot5+2=15,\quad 3+5+2\cdot2=12
$$

Vậy $(a,b,c)=(3,5,2)$.

### NumPy liên quan
```python
import numpy as np

A = np.array([[1, 1, 1],
              [1, 2, 1],
              [1, 1, 2]], dtype=float)
b = np.array([10, 15, 12], dtype=float)
print(np.linalg.solve(A, b))  # [3. 5. 2.]
```

### Common mistakes
> ⚠️ Đừng kết luận hệ 3 phương trình 3 biến có nghiệm duy nhất chỉ vì số phương trình bằng số biến. Ví dụ $a+b+c=10$, $2a+2b+2c=20$, $3a+3b+3c=30$ chỉ lặp lại cùng một ràng buộc nên có vô số nghiệm.

### Tóm tắt
- Hệ 3×3 mở rộng trực tiếp ý tưởng 2×2.
- Khử biến vẫn là công cụ chính.
- Cần độc lập tuyến tính và nhất quán để có nghiệm duy nhất.
- Hệ singular có thể vô số nghiệm hoặc vô nghiệm.

## Mặt phẳng và hình học 3D

### Trực giác
Trong hai biến, một phương trình là một đường. Trong ba biến, một phương trình tuyến tính là một mặt phẳng. Nghiệm của hệ ba phương trình là phần giao chung của ba mặt phẳng.

Hình học 3D phong phú hơn 2D. Ba mặt phẳng có thể gặp nhau tại một điểm, cùng chứa một đường, trùng nhau thành một mặt phẳng, hoặc không có điểm chung. Vì vậy “không có nghiệm duy nhất” có nhiều hình dạng khác nhau.

### Định nghĩa và công thức
Phương trình mặt phẳng:

$$
pa+qb+rc=s
$$

Nếu một điểm $(a_0,b_0,c_0)$ thỏa:

$$
pa_0+qb_0+rc_0=s
$$

thì điểm đó nằm trên mặt phẳng.

### Ví dụ làm mẫu
Xét mặt phẳng:

$$
a+b+c=1
$$

Kiểm tra ba điểm:

$$
(1,0,0):\quad 1+0+0=1
$$

$$
(0,1,0):\quad 0+1+0=1
$$

$$
(0,0,1):\quad 0+0+1=1
$$

Vậy cả ba điểm đều nằm trên mặt phẳng. Chúng là các giao điểm với ba trục tọa độ.

Với mặt phẳng:

$$
3a-5b+2c=0
$$

điểm gốc $(0,0,0)$ nằm trên mặt phẳng vì:

$$
3\cdot0-5\cdot0+2\cdot0=0
$$

### NumPy liên quan
```python
import numpy as np

normal = np.array([1, 1, 1])
point = np.array([1, 0, 0])
rhs = 1
print(normal @ point == rhs)  # True
```

Vector hệ số $(p,q,r)$ là vector pháp tuyến của mặt phẳng $pa+qb+rc=s$.

> ⚠️ Common mistakes: Đừng hình dung ba mặt phẳng giống hệt hai đường trong 2D. Hai mặt phẳng có thể cắt theo một đường; ba mặt phẳng có thể không có giao chung dù từng cặp có thể cắt nhau.

### Tóm tắt
- 3 biến → mặt phẳng.
- Nghiệm hệ là giao chung của các mặt phẳng.
- Một điểm nằm trên mặt phẳng nếu thay tọa độ vào thỏa phương trình.
- Hình học giúp hiểu unique / infinite / none trước khi tính determinant.

## Định thức 3×3

### Trực giác
Định thức 3×3 đóng vai trò như định thức 2×2 nhưng trong không gian ba chiều. Nó kiểm tra xem ba ràng buộc hoặc ba vector có tạo thành một khối có thể tích khác 0 hay bị dẹt xuống một mặt phẳng/đường.

Nếu determinant khác 0, ba hướng độc lập và hệ vuông có nghiệm duy nhất. Nếu determinant bằng 0, có phụ thuộc tuyến tính và ma trận singular.

### Định nghĩa và công thức
Với:

$$
A=\begin{bmatrix}
a&b&c\\
d&e&f\\
g&h&i
\end{bmatrix}
$$

quy tắc Sarrus cho:

$$
\det(A)=aei+bfg+cdh-ceg-bdi-afh
$$

Ba tích “xuống phải” được cộng, ba tích “xuống trái” bị trừ.

### Ví dụ làm mẫu
Tính determinant:

$$
A=\begin{bmatrix}
1&1&1\\
1&2&1\\
1&1&2
\end{bmatrix}
$$

Ba tích cộng:

$$
1\cdot2\cdot2=4,\quad 1\cdot1\cdot1=1,\quad 1\cdot1\cdot1=1
$$

Ba tích trừ:

$$
1\cdot2\cdot1=2,\quad 1\cdot1\cdot1=1,\quad 1\cdot1\cdot2=2
$$

Vậy:

$$
\det(A)=4+1+1-2-1-2=1
$$

Vì determinant khác 0, ma trận non-singular.

### NumPy liên quan
```python
import numpy as np

A = np.array([[1, 1, 1],
              [1, 2, 1],
              [1, 1, 2]], dtype=float)
print(round(np.linalg.det(A)))  # 1
```

Với số thực, NumPy có thể trả về $0.9999999999$ thay vì đúng 1 do sai số dấu phẩy động, nên đôi khi cần `round` hoặc `np.isclose`.

> ⚠️ Common mistakes: Quy tắc Sarrus chỉ áp dụng trực tiếp cho 3×3. Ngoài ra, ba tích cuối phải bị trừ; nếu cộng cả sáu tích, kết quả sẽ sai.

### Tóm tắt
- Determinant 3×3 kiểm tra singularity trong hệ ba biến.
- Sarrus: ba tích cộng, ba tích trừ.
- Det bằng 0 ⇔ phụ thuộc tuyến tính ⇔ singular.
- Det khác 0 ⇔ non-singular ⇔ nghiệm duy nhất cho mọi $b$.
