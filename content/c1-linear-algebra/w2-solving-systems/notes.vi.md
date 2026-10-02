# Tuần 2: Giải hệ phương trình tuyến tính

## Động lực ML

### Trực giác
Tuần 2 cho thấy việc giải hệ không chỉ là thao tác giấy bút. Trong mạng nơ-ron, một lớp tính rất nhiều tổng có trọng số cùng lúc; trong nhận dạng âm thanh, ma trận có thể lưu đặc trưng theo thời gian; trong tạo nhạc, nén dữ liệu thường giữ lại vài hướng biến thiên quan trọng. Khử dòng là mô hình nhỏ của câu hỏi: hàng nào thật sự mang thông tin độc lập?

Hãy hình dung ma trận như một bàn trộn âm. Mỗi hàng nghe vector đầu vào theo một cách khác. Nếu hai hàng gần như nghe cùng một kiểu, mô hình có thông tin lặp. Nếu các hàng nghe theo nhiều hướng khác nhau, đầu ra giữ được nhiều thông tin hơn và hệ phương trình dễ xác định nghiệm hơn.

### Ý chính
Một hệ tuyến tính viết là $Ax=b$. Ma trận hệ số (coefficient matrix) $A$ chứa vế trái, vector ẩn (unknown vector) $x$ chứa các biến, còn $b$ là vector vế phải (right-hand side). Phép dòng thay hệ ban đầu bằng một hệ tương đương nhưng dễ đọc hơn.

Trong trực giác ML, hạng (rank) đo số hướng thông tin độc lập mà ma trận biểu diễn. Một bảng đặc trưng rank thấp có thể nén tốt, nhưng một ma trận hệ số vuông rank thấp không thể xác định mọi ẩn một cách duy nhất.

### Ví dụ làm mẫu
Giả sử hai cảm biến cho hệ

$$
\begin{cases}
2u+v=9\\
-u+3v=8
\end{cases}
$$

Cộng một nửa phương trình đầu vào phương trình hai: $-u+3v+(u+0.5v)=8+4.5$, nên $3.5v=12.5$ và $v=25/7$. Khi đó $2u+25/7=9$, do đó $u=19/7$. Hai hàng không phải bội của nhau, nên hai cảm biến cung cấp hai mẩu thông tin độc lập.

### Kiểm tra bằng NumPy
```python
import numpy as np
A = np.array([[2, 1], [-1, 3]], dtype=float)
b = np.array([9, 8], dtype=float)
print(np.linalg.solve(A, b))
```

> ⚠️ Sai lầm: ma trận có nhiều hàng chưa chắc có nhiều thông tin; nếu phần lớn hàng là tổ hợp tuyến tính của hàng trước thì rank vẫn thấp.

### Tóm tắt
- Hệ tuyến tính nối đại số với feature map và nén dữ liệu.
- Rank là số đo thông tin, không chỉ là kích thước ma trận.
- Khử dòng làm lộ pivot để nhận ra tính độc lập.
- Giải tay giúp hiểu nền tảng của các hàm số học tuyến tính.

## Hệ tuyến tính không suy biến

### Trực giác
Một hệ vuông không suy biến (non-singular) có đúng một nghiệm. Trong hai biến, đó là hai đường thẳng không song song cắt nhau tại một điểm. Về đại số, quá trình khử tạo được pivot cho mỗi biến, nên thế ngược xác định từng biến mà không cần chọn tham số tự do.

Các slide dùng ví dụ mua hàng như $a+b=10$ và $a+2b=12$. Phương trình thứ hai thay đổi hệ số của chuối nhưng giữ hệ số của táo, vì vậy trừ hai phương trình sẽ cô lập giá chuối.

### Ý chính
Với ma trận hệ số vuông $A$, non-singular nghĩa là full rank. Trong trường hợp $2\times2$, điều này tương đương với $\det(A)\ne0$. Khi khử, mỗi pivot phải khác 0; nếu phần tử hiện tại bằng 0 thì nên đổi hàng với một hàng phía dưới có phần tử khác 0.

### Ví dụ làm mẫu
Giải hệ

$$
\begin{cases}
5a+b=17\\
4a-3b=6
\end{cases}
$$

Chia phương trình đầu cho $5$ được $a+0.2b=3.4$. Chia phương trình hai cho $4$ được $a-0.75b=1.5$. Lấy phương trình đã chuẩn hóa thứ nhất trừ phương trình thứ hai: $0.95b=1.9$, nên $b=2$. Thay vào $a+0.2b=3.4$ được $a+0.4=3.4$, nên $a=3$.

### Kiểm tra bằng NumPy
```python
import numpy as np
A = np.array([[5, 1], [4, -3]], dtype=float)
b = np.array([17, 6], dtype=float)
print(np.linalg.det(A), np.linalg.solve(A, b))
```

> ⚠️ Sai lầm: đừng chia cho pivot trước khi chắc chắn nó khác 0; đổi hàng là phép hợp lệ, chia cho 0 thì không.

### Tóm tắt
- Mỗi biến có một pivot thì hệ có một nghiệm.
- Định thức khác 0 trong $2\times2$ xác nhận hai hàng độc lập.
- Khử tiến rồi thế ngược là quy trình tay quan trọng nhất.
- Luôn thay nghiệm vào hệ gốc để kiểm tra.

## Hệ tuyến tính suy biến

### Trực giác
Hệ suy biến (singular) không xác định duy nhất một điểm. Nó có thể dư thừa và cho vô số nghiệm, hoặc mâu thuẫn và vô nghiệm. Cùng một mẫu hệ số có thể rơi vào hai trường hợp khác nhau tùy vector vế phải.

Điều này quan trọng trong ML: đặc trưng lặp không tự động gây lỗi, nhưng ràng buộc mâu thuẫn thì không thể thỏa. Khử dòng tách hai tình huống bằng cách tạo hàng không với vế phải bằng 0 hoặc hàng không với vế phải khác 0.

### Ý chính
Nếu khử ra $0=0$, một phương trình không thêm thông tin. Nếu khử ra $0=c$ với $c\ne0$, hệ không nhất quán. Singular nghĩa là ma trận hệ số thiếu full rank; bản thân điều đó chưa nói $b$ có tương thích hay không.

### Ví dụ làm mẫu
So sánh hai hệ:

$$
\begin{cases}
a+b=10\\
2a+2b=20
\end{cases}
\qquad
\begin{cases}
a+b=10\\
2a+2b=24
\end{cases}
$$

Ở hệ đầu, lấy hàng 2 trừ $2$ lần hàng 1 được $0=0$. Mọi cặp có $a=10-b$ đều là nghiệm. Ở hệ sau, thao tác giống hệt cho $0=4$, điều không thể xảy ra. Hai ma trận hệ số đều singular, nhưng tập nghiệm khác nhau.

### Kiểm tra bằng NumPy
```python
import numpy as np
A = np.array([[1, 1], [2, 2]], dtype=float)
print(np.linalg.matrix_rank(A))
```

> ⚠️ Sai lầm: đừng nói “singular nghĩa là vô nghiệm”. Hệ dư thừa nhất quán cũng singular và có vô số nghiệm.

### Tóm tắt
- Dư thừa tạo biến tự do.
- Mâu thuẫn hiện ra dưới dạng $0=c$ với $c\ne0$.
- Ma trận hệ số chưa đủ để kết luận tính nhất quán.
- Cần khử ma trận mở rộng cùng với vế phải.

## Hệ có nhiều biến

### Trực giác
Với ba ẩn trở lên, khử biến vẫn là cùng một ý tưởng lặp lại. Dùng pivot thứ nhất để loại biến đầu khỏi các phương trình dưới. Sau đó dùng pivot thứ hai để loại biến thứ hai khỏi các hàng thấp hơn. Khi hệ đã tam giác, ta giải từ dưới lên.

Hệ tam giác giống một chuỗi khóa: hàng cuối mở được biến cuối, giá trị đó mở hàng ngay phía trên, rồi tiếp tục đi ngược lên.

### Ý chính
Với $n$ ẩn, nghiệm duy nhất cần pivot trong mọi cột biến. Nếu số pivot ít hơn và hệ nhất quán, ít nhất một biến là biến tự do. Nếu xuất hiện hàng bất khả thi, hệ vô nghiệm.

### Ví dụ làm mẫu
Xét hệ

$$
\begin{cases}
a+b+2c=12\\
3a-3b-c=3\\
2a-b+6c=24
\end{cases}
$$

Sau khi chuẩn hóa cột đầu như trong slide, ta được

$$
\begin{cases}
a+b+2c=12\\
-2b-\frac73c=-11\\
-\frac32b+c=0
\end{cases}
$$

Chuẩn hóa hai hàng cuối: $b+\frac76c=\frac{11}{2}$ và $b-\frac23c=0$. Lấy hiệu hai hàng được $\frac{11}{6}c=\frac{11}{2}$, nên $c=3$. Từ đó $b=2$, và hàng đầu cho $a=4$.

### Kiểm tra bằng NumPy
```python
import numpy as np
A = np.array([[1,1,2],[3,-3,-1],[2,-1,6]], dtype=float)
b = np.array([12,3,24], dtype=float)
print(np.linalg.solve(A, b))
```

> ⚠️ Sai lầm: sau khi có dạng tam giác, giải từ trên xuống thường vẫn còn nhiều ẩn; thế ngược phải bắt đầu từ hàng pivot thấp nhất.

### Tóm tắt
- Khử tiến tạo hình tam giác trên.
- Thế ngược đọc biến từ dưới lên.
- Cột thiếu pivot tạo biến tự do nếu hệ nhất quán.
- Một hàng mâu thuẫn là đủ để kết luận vô nghiệm.

## Khử dòng bằng ma trận

### Trực giác
Viết ma trận mở rộng $[A\mid b]$ giúp bỏ bớt ký hiệu biến lặp lại và tập trung vào các hàng. Mỗi hàng vẫn là một phương trình, bao gồm cả vế phải. Vì vậy khử dòng trong ma trận chính là biến đổi phương trình ở dạng bảng.

Các slide chuyển từ $5a+b=17$ sang hàng $[5\;1\mid17]$. Ký hiệu này đặc biệt hữu ích khi hệ có nhiều phương trình.

### Ý chính
Ba phép dòng hợp lệ là: đổi hai hàng; nhân một hàng với hằng số khác 0; cộng bội của một hàng vào hàng khác. Các phép này bảo toàn tập nghiệm của hệ mở rộng. Cột vế phải phải đi cùng mọi phép biến đổi trên hàng.

### Ví dụ làm mẫu
Bắt đầu với

$$
\left[\begin{array}{cc|c}
1&2&5\\
2&5&12
\end{array}\right].
$$

Dùng $R_2\leftarrow R_2-2R_1$:

$$
\left[\begin{array}{cc|c}
1&2&5\\
0&1&2
\end{array}\right].
$$

Hàng hai nói $y=2$. Hàng một nói $x+4=5$, nên $x=1$. Nếu chỉ khử hai cột hệ số mà giữ nguyên số $12$, ta đã tạo một bài toán khác.

### Kiểm tra bằng NumPy
```python
import numpy as np
M = np.array([[1,2,5],[2,5,12]], dtype=float)
M[1] = M[1] - 2*M[0]
print(M)
```

> ⚠️ Sai lầm: vạch dọc trong $[A\mid b]$ chỉ để nhìn; phép dòng đi qua vạch vì nó biến đổi cả phương trình.

### Tóm tắt
- Ma trận mở rộng mã hóa hệ một cách gọn gàng.
- Phép dòng phải áp dụng cho từng phần tử của cả hàng.
- Vế phải không phải là ghi chú bên lề.
- Dạng bậc thang làm cấu trúc nghiệm hiện rõ.

## Phép dòng và tính suy biến

### Trực giác
Phép dòng có thể làm ma trận trông khác đi nhưng không đổi trạng thái singular. Định thức có thể đổi dấu hoặc bị nhân với một hệ số, nhưng giá trị bằng 0 vẫn là 0 và giá trị khác 0 vẫn khác 0 khi phép dòng hợp lệ.

Đây là lý do khử biến an toàn: ta đơn giản hóa ma trận hệ số mà vẫn giữ câu hỏi cốt lõi “ma trận có đủ thông tin độc lập không?”.

### Ý chính
Đổi hai hàng làm determinant đổi dấu. Nhân một hàng với $k\ne0$ làm determinant bị nhân với $k$. Cộng bội của một hàng vào hàng khác không đổi determinant. Không phép nào trong ba phép hợp lệ biến determinant 0 thành khác 0 hoặc ngược lại.

### Ví dụ làm mẫu
Cho

$$
A=\begin{bmatrix}5&1\\4&3\end{bmatrix},
\quad \det(A)=5\cdot3-1\cdot4=11.
$$

Đổi hai hàng cho determinant $-11$. Nhân hàng đầu với $2$ cho determinant $22$. Thay hàng 2 bằng hàng 2 trừ hàng 1 được

$$
\begin{bmatrix}5&1\\-1&2\end{bmatrix},
$$

có determinant $10+1=11$. Trạng thái singular/non-singular giữ nguyên trong cả ba thao tác.

### Kiểm tra bằng NumPy
```python
import numpy as np
A = np.array([[5,1],[4,3]], dtype=float)
B = A.copy(); B[1] = B[1] - B[0]
print(np.linalg.det(A), np.linalg.det(B))
```

> ⚠️ Sai lầm: nhân một hàng với 0 không phải phép dòng hợp lệ vì nó xóa thông tin và có thể đổi tính suy biến.

### Tóm tắt
- Phép dòng hợp lệ bảo toàn trạng thái determinant bằng 0 hay khác 0.
- Row replacement đặc biệt hữu ích vì không đổi determinant.
- Nhân với số khác 0 là thao tác đảo được.
- Khử biến dựa vào các bước bảo toàn thông tin này.

## Hạng của ma trận

### Trực giác
Hạng (rank) đếm thông tin độc lập. Khi khử dòng, mỗi pivot đánh dấu một hướng hàng không giải thích được bằng các hàng trước. Hàng không sau khử nghĩa là hàng đó dư thừa.

Các slide nối rank với ví dụ hệ câu: hai câu giống hệt có vẻ là hai hàng, nhưng chỉ chứa một mẩu thông tin. Rank biến trực giác đó thành con số cho ma trận.

### Ý chính
Rank bằng số pivot trong row echelon form. Với ma trận vuông $n\times n$, rank $n$ nghĩa là non-singular; rank nhỏ hơn $n$ nghĩa là singular. Với hệ thuần nhất $Ax=0$, số chiều không gian nghiệm là $n-\operatorname{rank}(A)$.

### Ví dụ làm mẫu
Với

$$
A=\begin{bmatrix}1&1\\2&2\end{bmatrix},
$$

lấy hàng 2 trừ $2$ lần hàng 1:

$$
\begin{bmatrix}1&1\\0&0\end{bmatrix}.
$$

Chỉ có một pivot, nên rank bằng $1$. Hệ thuần nhất $a+b=0$ có một biến tự do; chẳng hạn đặt $b=t$ thì $a=-t$.

### Kiểm tra bằng NumPy
```python
import numpy as np
A = np.array([[1,1],[2,2]], dtype=float)
print(np.linalg.matrix_rank(A))
```

> ⚠️ Sai lầm: rank không phải số phần tử khác 0. Một hàng có nhiều số khác 0 vẫn có thể phụ thuộc tuyến tính vào hàng trước.

### Tóm tắt
- Hãy đếm pivot, không đếm entry khác 0.
- Hàng không sau khử không đóng góp vào rank.
- Ma trận vuông full rank là non-singular.
- Rank dự đoán số bậc tự do trong $Ax=0$.

## Hạng trong trường hợp tổng quát

### Trực giác
Ma trận chữ nhật cũng có rank. Rank tối đa bị giới hạn bởi kích thước nhỏ hơn: ma trận $m\times n$ có nhiều nhất $\min(m,n)$ pivot. Thêm hàng hoặc cột chỉ hữu ích khi chúng thêm hướng độc lập thật sự.

Trong bảng dữ liệu, điều này giải thích vì sao nhiều đặc trưng đo được vẫn có thể nằm trong không gian chiều thấp nếu các cột là tổ hợp của vài yếu tố ẩn.

### Ý chính
Với mọi ma trận, khử dòng làm lộ cột pivot và hàng không. Rank là duy nhất dù chuỗi phép dòng không duy nhất. Nếu ma trận có $n$ cột và rank $r$, hệ thuần nhất có $n-r$ biến tự do.

### Ví dụ làm mẫu
Khử

$$
B=\begin{bmatrix}
1&1&1\\
1&1&2\\
1&1&3
\end{bmatrix}.
$$

Lấy hàng 2 và hàng 3 trừ hàng 1:

$$
\begin{bmatrix}
1&1&1\\
0&0&1\\
0&0&2
\end{bmatrix}.
$$

Sau đó lấy hàng 3 trừ $2$ lần hàng 2 để được hàng không. Có hai pivot, ở cột 1 và cột 3, nên rank bằng $2$.

### Kiểm tra bằng NumPy
```python
import numpy as np
B = np.array([[1,1,1],[1,1,2],[1,1,3]], dtype=float)
print(np.linalg.matrix_rank(B))
```

> ⚠️ Sai lầm: cột bị bỏ qua không phải lỗi tính toán; nó cho biết cột đó không trở thành cột pivot trong quá trình khử.

### Tóm tắt
- Rank không vượt quá $\min(m,n)$.
- Số pivot không phụ thuộc con đường khử hợp lệ.
- Cột không pivot tương ứng với biến tự do tiềm năng.
- Rank chữ nhật là nền cho nén và trực giác least squares.

## Dạng bậc thang dòng

### Trực giác
Row echelon form (REF) giống cầu thang. Hàng không nằm dưới cùng, pivot của hàng dưới nằm bên phải pivot hàng trên, và các phần tử dưới pivot bằng 0. Ma trận chưa được giải sạch hoàn toàn, nhưng đã đủ có tổ chức để đọc rank và thế ngược.

Theo slide, pivot trong REF chỉ cần khác 0. Nó chưa cần bằng $1$ cho đến khi ta yêu cầu reduced row echelon form.

### Ý chính
Một ma trận ở REF khi thỏa ba điều kiện: mọi hàng khác không nằm trên hàng không; phần tử khác 0 đầu tiên của các hàng dịch sang phải khi đi xuống; các phần tử dưới mỗi pivot đều bằng 0. REF thường không duy nhất.

### Ví dụ làm mẫu
Bắt đầu với

$$
\begin{bmatrix}2&4&6\\1&3&5\\0&2&4\end{bmatrix}.
$$

Đổi hai hàng đầu để pivot đầu đơn giản hơn. Lấy hàng 2 mới trừ $2$ lần hàng 1:

$$
\begin{bmatrix}1&3&5\\0&-2&-4\\0&2&4\end{bmatrix}.
$$

Sau đó lấy hàng 3 cộng hàng 2:

$$
\begin{bmatrix}1&3&5\\0&-2&-4\\0&0&0\end{bmatrix}.
$$

Đây là REF với hai pivot.

### Kiểm tra bằng NumPy
```python
import numpy as np
M = np.array([[1,3,5],[0,-2,-4],[0,0,0]], dtype=float)
print(np.count_nonzero(np.any(M != 0, axis=1)))
```

> ⚠️ Sai lầm: đừng loại một ma trận khỏi REF chỉ vì pivot bằng $-2$; ở REF, chỉ cần pivot khác 0.

### Tóm tắt
- REF là dạng cầu thang để đọc pivot.
- Pivot dịch sang phải khi đi xuống.
- Các phần tử dưới mỗi pivot phải bằng 0.
- REF hỗ trợ đếm rank và thế ngược.

## Dạng bậc thang tổng quát

### Trực giác
Trong ma trận lớn hoặc chữ nhật, cầu thang có thể bỏ qua một vài cột. Cột bị bỏ qua chỉ có nghĩa là không hàng nào có phần tử khác 0 đầu tiên ở đó. Điều quan trọng là mẫu pivot, không phải mọi cột đều phải tham gia.

Chủ đề này mở rộng hình ảnh $2\times2$ sang các ma trận ML thực tế, nơi số hàng và số cột thường không bằng nhau.

### Ý chính
REF tổng quát dùng cho mọi kích thước. Hàng không vẫn ở dưới, leading entries dịch sang phải, và mỗi pivot xóa các phần tử phía dưới nó. Số pivot là rank. Các cột không pivot là cột biến tự do khi ma trận được dùng trong hệ phương trình.

### Ví dụ làm mẫu
Xét

$$
C=\begin{bmatrix}
0&2&4&1\\
0&0&3&6\\
0&0&0&0
\end{bmatrix}.
$$

Ma trận này đã ở REF. Pivot đầu ở cột 2, pivot thứ hai ở cột 3, và cột 1 không có pivot. Rank bằng $2$. Nếu đây là hệ thuần nhất bốn biến, số biến tự do là $4-2=2$.

### Kiểm tra bằng NumPy
```python
import numpy as np
C = np.array([[0,2,4,1],[0,0,3,6],[0,0,0,0]], dtype=float)
print(np.linalg.matrix_rank(C))
```

> ⚠️ Sai lầm: một cột đầu toàn số 0 không làm hỏng REF; nó chỉ đẩy pivot đầu tiên sang cột sau.

### Tóm tắt
- REF áp dụng cho ma trận không vuông.
- Cột pivot có thể cách nhau bởi cột không pivot.
- Rank vẫn là số pivot.
- Biến tự do nằm ở cột không pivot của hệ nhất quán.

## Dạng bậc thang rút gọn

### Trực giác
Reduced row echelon form (RREF) là dạng đã dọn sạch. Mỗi pivot bằng $1$, và cột pivot có số 0 cả phía dưới lẫn phía trên pivot. Với hệ vuông non-singular đã giải xong, phía hệ số trở thành ma trận đơn vị.

RREF hữu ích vì nó hiển thị nghiệm trực tiếp. Nó nghiêm ngặt hơn REF và là duy nhất cho một ma trận cho trước.

### Ý chính
RREF cần mọi điều kiện của REF, cộng thêm pivot bằng 1 và cột pivot sạch. Với ma trận mở rộng, một hàng RREF như $[0\;0\mid5]$ vẫn báo mâu thuẫn. Một hàng $[0\;0\mid0]$ không tạo pivot.

### Ví dụ làm mẫu
Bắt đầu từ ma trận mở rộng ở REF

$$
\left[\begin{array}{cc|c}
1&2&7\\
0&3&6
\end{array}\right].
$$

Nhân hàng 2 với $1/3$:

$$
\left[\begin{array}{cc|c}
1&2&7\\
0&1&2
\end{array}\right].
$$

Xóa phần tử phía trên pivot thứ hai bằng $R_1\leftarrow R_1-2R_2$:

$$
\left[\begin{array}{cc|c}
1&0&3\\
0&1&2
\end{array}\right].
$$

RREF cho ngay $x=3$, $y=2$.

### Kiểm tra bằng NumPy
```python
import numpy as np
M = np.array([[1,2,7],[0,3,6]], dtype=float)
M[1] = M[1] / 3
M[0] = M[0] - 2*M[1]
print(M)
```

> ⚠️ Sai lầm: dừng ở REF tam giác là đủ để thế ngược, nhưng chưa phải RREF nếu cột pivot chưa được xóa phía trên.

### Tóm tắt
- RREF thêm pivot bằng 1 và số 0 phía trên pivot.
- Dạng này cho thấy nghiệm với rất ít thế ngược.
- Khác REF, RREF là duy nhất cho một ma trận cố định.
- Hàng mâu thuẫn của ma trận mở rộng vẫn vô nghiệm ở mọi dạng.
