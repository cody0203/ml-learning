## Động lực ML: PCA

### Trực giác
PCA bắt đầu từ một vấn đề rất thực tế.
Dữ liệu có thể có nhiều cột.
Nhiều cột lại cùng thay đổi.
Khi đó đám mây điểm thường mỏng.
Nó gần một đường hay một mặt phẳng.
PCA tìm các hướng quan trọng đó.

Trong slide, ví dụ đi từ 2D xuống 1D.
Sau đó ví dụ đi từ 8D xuống 3D.
Ta không xóa cột một cách ngẫu nhiên.
Ta giữ hướng có độ trải lớn.

### Định nghĩa và công thức
Trước PCA, dữ liệu cần được center.
Ma trận covariance là ma trận đối xứng.
Eigenvectors của nó là các hướng chính.
Eigenvalues đo variance theo từng hướng.

Nếu eigenvalues là 9 và 1, PC1 giữ
$9/(9+1)=0.9$ variance.
Với dữ liệu đã center $X$, projection lên hướng đơn vị $u$
là $Xu$.

### Ví dụ làm đầy đủ
Giả sử covariance là $C=[[4,0],[0,1]]$.
Tọa độ thứ nhất biến thiên mạnh hơn.
Eigenvectors là $(1,0)$ và $(0,1)$.
Eigenvalues tương ứng là 4 và 1.
Nếu chỉ giữ một chiều, chọn trục x.
Tỉ lệ variance giữ lại là $4/5=0.8$.

### NumPy snippet
```python
import numpy as np
Xc = X - X.mean(axis=0)
C = np.cov(Xc, rowvar=False)
w, V = np.linalg.eigh(C)
pc1 = V[:, np.argmax(w)]
z = Xc @ pc1
```

> ⚠️ Lỗi thường gặp: chạy PCA khi chưa trừ mean.

### Tóm tắt
- PCA chọn hướng, không chỉ chọn tên cột.
- Eigenvalue lớn nghĩa là variance lớn.
- Projection biến nhiều feature thành ít tọa độ hơn.

## Suy biến và hạng của phép biến đổi tuyến tính

### Trực giác
Ma trận 2D biến hình vuông đơn vị.
Nếu hình vuông thành hình bình hành có diện tích,
hai hướng độc lập vẫn còn.
Nếu nó xẹp thành đường thẳng, một hướng đã mất.
Nếu nó xẹp về gốc, cả hai hướng đều mất.

Vì vậy rank có nghĩa hình học rõ ràng.
Rank 2 cho mặt phẳng.
Rank 1 cho một đường.
Rank 0 cho một điểm.

### Định nghĩa và công thức
Rank của phép biến đổi là dimension của ảnh.
Ma trận vuông non-singular khi rank đầy đủ.
Với ma trận 2×2, điều này tương đương determinant khác 0.

### Ví dụ làm đầy đủ
Cho $A=[[1,1],[2,2]]$.
Hai cột cùng hướng.
Ảnh của $(a,b)$ là $(a+b,2a+2b)$.
Mọi output đều là bội của $(1,2)$.
Vậy cả mặt phẳng bị ép lên một đường.
Determinant là $1\cdot2-1\cdot2=0$.
Do đó $A$ singular và rank 1.

### NumPy snippet
```python
A = np.array([[1, 1], [2, 2]])
np.linalg.matrix_rank(A)
```

> ⚠️ Lỗi thường gặp: singular không có nghĩa là không có output.
> Nó vẫn có output, nhưng thiếu hướng độc lập.

### Tóm tắt
- Rank đếm số hướng output độc lập.
- Singular làm mất ít nhất một hướng.
- Determinant khác 0 trong 2D nghĩa là rank 2.

## Định thức như diện tích có hướng

### Trực giác
Determinant đo cách diện tích thay đổi.
Dấu của nó còn ghi lại orientation.
Determinant âm nghĩa là thứ tự hai vector bị lật.

Slides dùng các cột $(3,1)$ và $(1,2)$.
Chúng tạo hình bình hành diện tích 5.
Ví dụ đảo chiều có determinant -5.
Diện tích hình học vẫn là 5.

### Định nghĩa và công thức
Với $A=[[a,b],[c,d]]$,
determinant là $ad-bc$.
Area scale là $|\det(A)|$.
Dấu cho biết orientation giữ hay đảo.

### Ví dụ làm đầy đủ
Với $A=[[1,3],[2,1]]$,
$\det(A)=1\cdot1-3\cdot2$.
Kết quả là $-5$.
Hình vuông đơn vị thành hình bình hành diện tích 5.
Dấu âm cho biết orientation bị đảo.

### NumPy snippet
```python
A = np.array([[1, 3], [2, 1]], dtype=float)
detA = np.linalg.det(A)
area = abs(detA)
```

> ⚠️ Lỗi thường gặp: báo diện tích âm.
> Determinant có dấu, còn diện tích thì không.

### Tóm tắt
- Determinant là hệ số diện tích có hướng.
- Determinant 0 nghĩa là hình bị xẹp.
- Trị tuyệt đối là diện tích hình học.

## Định thức của tích

### Trực giác
Nhân ma trận là ghép hai biến đổi.
Nếu $B$ nhân diện tích lên 3 lần,
rồi $A$ nhân tiếp lên 5 lần,
biến đổi tổng hợp nhân diện tích 15 lần.

Đó là lý do công thức trong slide xuất hiện.
Determinant của tích bằng tích determinant.

### Định nghĩa và công thức
$\det(AB)=\det(A)\det(B)$.
Thứ tự nhân ma trận vẫn quan trọng.
Nhưng hệ số diện tích là scalar nên nhân lại.

### Ví dụ làm đầy đủ
Nếu $\det(A)=5$ và $\det(B)=8$,
thì $\det(AB)=40$.
Nếu $B$ singular, $\det(B)=0$.
Khi đó $\det(AB)=\det(A)\cdot0=0$.
Vì vậy tích với một thừa số singular sẽ singular.

### NumPy snippet
```python
left = np.linalg.det(A @ B)
right = np.linalg.det(A) * np.linalg.det(B)
np.allclose(left, right)
```

> ⚠️ Lỗi thường gặp: áp dụng quy tắc này cho tổng.
> Nói chung det(A+B) không bằng det(A)+det(B).

### Tóm tắt
- Determinant của tích thì nhân.
- Một thừa số 0 làm tích singular.
- Quy tắc này phản ánh area scaling liên tiếp.

## Định thức của ma trận nghịch đảo

### Trực giác
Inverse hoàn tác một biến đổi.
Nếu ma trận nhân diện tích lên 5 lần,
inverse phải chia diện tích cho 5.
Vì vậy determinant inverse là reciprocal.

### Định nghĩa và công thức
Nếu $A$ khả nghịch thì $AA^{-1}=I$.
Vì $\det(I)=1$,
$\det(A)\det(A^{-1})=1$.
Do đó $\det(A^{-1})=1/\det(A)$.

### Ví dụ làm đầy đủ
Slides có ma trận determinant 5.
Inverse của nó có determinant 0.2.
Thật vậy $5\cdot0.2=1$.
Nếu determinant là 8, inverse determinant là 0.125.
Nếu determinant là 0, inverse không tồn tại.

### NumPy snippet
```python
A = np.array([[3, 1], [1, 2]], dtype=float)
np.linalg.det(np.linalg.inv(A))
```

> ⚠️ Lỗi thường gặp: cố invert ma trận singular.

### Tóm tắt
- Inverse dùng hệ số diện tích reciprocal.
- Identity có determinant 1.
- Ma trận singular không có inverse.

## Cơ sở

### Trực giác
Basis là một hệ tọa độ.
Trong 2D, hai vector không song song là đủ.
Chúng mô tả duy nhất mọi điểm trong mặt phẳng.
Nếu hai vector song song, chúng chỉ mô tả một đường.

### Định nghĩa và công thức
Basis phải span không gian.
Nó cũng phải độc lập tuyến tính.
Trong 2D, determinant khác 0 kiểm tra cả hai điều kiện.

### Ví dụ làm đầy đủ
Các vector $(3,1)$ và $(1,2)$ tạo basis của mặt phẳng.
Đặt chúng làm cột: $A=[[3,1],[1,2]]$.
Determinant là 5.
Vì khác 0, mọi vector 2D có tọa độ duy nhất trong basis này.

Các vector $(1,0)$ và $(2,0)$ thì không.
Chúng chỉ đi dọc trục x.

### NumPy snippet
```python
B = np.array([[3, 1], [1, 2]])
is_basis = abs(np.linalg.det(B)) > 1e-9
```

> ⚠️ Lỗi thường gặp: chỉ đếm số vector.
> Hai vector trong 2D vẫn có thể song song.

### Tóm tắt
- Basis nghĩa là span cộng independence.
- Số vector basis bằng dimension.
- Determinant khác 0 xác nhận basis 2D.

## Bao tuyến tính

### Trực giác
Span là tập mọi tổ hợp tuyến tính.
Một vector khác 0 span một đường qua gốc.
Hai vector không song song trong 2D span cả mặt phẳng.
Thêm vector cùng đường không tạo hướng mới.

### Định nghĩa và công thức
Span của $v_1$ và $v_2$ gồm mọi
$c_1v_1+c_2v_2$ với scalar thực.
Dimension của row span hay column span bằng rank.

### Ví dụ làm đầy đủ
Vector $(3,6)$ nằm trong span của $(1,2)$.
Nó bằng $3(1,2)$.
Vector $(2,3)$ thì không.
Không có scalar $c$ vừa thỏa $c=2$ vừa thỏa $2c=3$.

Với các hàng $(1,1)$ và $(2,2)$,
row span chỉ là một đường.
Rank là 1.

### NumPy snippet
```python
v = np.array([1, 2])
w = np.array([3, 6])
parallel = abs(v[0]*w[1] - v[1]*w[0]) < 1e-9
```

> ⚠️ Lỗi thường gặp: nghĩ span có thể bị tịnh tiến.
> Linear span luôn đi qua gốc.

### Tóm tắt
- Span chứa mọi tổ hợp tuyến tính.
- Vector phụ thuộc không mở rộng span.
- Rank là dimension của span.

## Cơ sở riêng

### Trực giác
Eigenbasis chọn các trục không bị trộn bởi ma trận.
Theo mỗi trục riêng, ma trận chỉ kéo dài hoặc co lại.
Điều này làm các phép lặp trở nên dễ hiểu.

### Định nghĩa và công thức
Nếu có đủ eigenvectors độc lập,
ta viết $A=PDP^{-1}$.
Các cột của $P$ là eigenvectors.
Các phần tử đường chéo của $D$ là eigenvalues.

### Ví dụ làm đầy đủ
Với $A=[[2,1],[0,3]]$,
$(1,0)$ là eigenvector cho eigenvalue 2.
Vector $(1,1)$ là eigenvector cho eigenvalue 3.
Hai vector này không song song nên tạo basis.
Trong basis đó, biến đổi chỉ nhân tọa độ với 2 và 3.

### NumPy snippet
```python
w, P = np.linalg.eig(A)
D = np.diag(w)
np.allclose(A, P @ D @ np.linalg.inv(P))
```

> ⚠️ Lỗi thường gặp: giả định mọi ma trận đều có eigenbasis thực.

### Tóm tắt
- Eigenbasis gồm eigenvectors độc lập.
- Dạng diagonal tách riêng các hướng.
- Nó hữu ích khi tính lũy thừa ma trận.

## Trị riêng và vector riêng

### Trực giác
Hầu hết vector bị đổi hướng bởi ma trận.
Eigenvectors là những vector đặc biệt.
Chúng vẫn nằm trên cùng đường thẳng.
Chỉ độ dài và dấu có thể thay đổi.

### Định nghĩa và công thức
Eigenvector khác 0 và thỏa $Av=\lambda v$.
Scalar $\lambda$ là eigenvalue.
Tìm eigenvalues từ $\det(A-\lambda I)=0$.
Sau đó giải $(A-\lambda I)v=0$.

### Ví dụ làm đầy đủ
Ma trận quiz là $[[9,4],[4,3]]$.
Trace bằng 12 và determinant bằng 11.
Characteristic polynomial là
$\lambda^2-12\lambda+11$.
Nó tách thành $(\lambda-11)(\lambda-1)$.
Eigenvalues là 11 và 1.
Eigenvectors tương ứng là bội của $(2,1)$ và $(-1,2)$.

### NumPy snippet
```python
A = np.array([[9, 4], [4, 3]], dtype=float)
w, V = np.linalg.eig(A)
```

> ⚠️ Lỗi thường gặp: chấp nhận vector 0.
> Nó thỏa phương trình với mọi lambda, nên bị loại.

### Tóm tắt
- Eigenvectors giữ hướng.
- Eigenvalues là hệ số co giãn.
- Characteristic equation tìm các ứng viên.

## Đa thức đặc trưng và PCA

### Trực giác
Characteristic polynomial nối ma trận với eigenvalues.
PCA dùng cầu nối đó trên covariance matrix.
Nghiệm lớn tương ứng hướng có variance lớn.

### Định nghĩa và công thức
Với ma trận 2×2, polynomial có thể viết là
$\lambda^2-\operatorname{tr}(A)\lambda+\det(A)$.
Nó tương đương $\det(A-\lambda I)$.
Với covariance, nên dùng eigensolver cho ma trận đối xứng.

### Ví dụ làm đầy đủ
Với $A=[[2,1],[0,3]]$,
trace bằng 5 và determinant bằng 6.
Polynomial là $\lambda^2-5\lambda+6$.
Nó tách thành $(\lambda-2)(\lambda-3)$.
Vậy nghiệm là 2 và 3.

Nếu covariance có roots 7, 2, 1,
hai PC đầu giữ $(7+2)/10=0.9$ total variance.

### NumPy snippet
```python
C = np.array([[4, 0], [0, 1]], dtype=float)
w, V = np.linalg.eigh(C)
order = np.argsort(w)[::-1]
```

> ⚠️ Lỗi thường gặp: sắp PCA theo độ dài vector.
> Phải sắp theo eigenvalue.

### Tóm tắt
- Nghiệm của polynomial là eigenvalues.
- Trace và determinant cho polynomial 2D nhanh.
- PCA xếp covariance eigenvectors theo eigenvalue.
