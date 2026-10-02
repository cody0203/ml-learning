# Tuần 2: Mô tả phân phối và phân phối nhiều biến

Tuần này học cách đọc một phân phối thay vì chỉ tính một xác suất riêng lẻ. Ta bắt đầu từ tâm, độ phân tán và hình dạng của một biến; sau đó chuyển sang bảng xác suất chung, phân phối biên, phân phối có điều kiện, hiệp phương sai, tương quan và Gaussian đa biến. Các ý này xuất hiện trực tiếp trong ML khi chuẩn hóa đặc trưng, so sánh mô hình, phát hiện ngoại lệ và mô tả dữ liệu nhiều chiều.

## Các thước đo xu hướng trung tâm

### Trực giác

Mean, median và mode trả lời ba câu hỏi khác nhau: điểm cân bằng ở đâu, điểm giữa sau khi sắp xếp là gì, và giá trị nào xuất hiện nhiều nhất. Trong câu chuyện lương sinh viên địa lý, một ngoại lệ rất lớn làm mean tăng mạnh, nhưng median vẫn mô tả người điển hình tốt hơn.

### Định nghĩa và công thức

Với dữ liệu $x_1,\ldots,x_n$,
$$
\bar x=\frac{1}{n}\sum_{i=1}^n x_i.
$$
Median là phân vị 50%. Mode là giá trị có tần suất lớn nhất. Với PMF rời rạc, mean cũng là kỳ vọng có trọng số theo xác suất.

### Ví dụ làm đầy đủ

Dữ liệu tuổi trong slide: $0,0,0,1,1,2,2,2,2,3$. Tổng là $13$, có $10$ quan sát, nên mean bằng $1.3$. Sau khi sắp xếp, hai vị trí giữa là $1$ và $2$, vậy median bằng $1.5$. Giá trị $2$ xuất hiện bốn lần, nên mode là $2$.

### NumPy snippet

```python
import numpy as np
x = np.array([0,0,0,1,1,2,2,2,2,3])
np.mean(x), np.median(x)
```

> ⚠️ Mean không phải lúc nào cũng là "người điển hình".
> Khi có ngoại lệ, hãy so sánh thêm median và box plot.

### Tóm tắt

- Mean dùng mọi giá trị và nhạy với ngoại lệ.
- Median dựa vào thứ tự nên bền hơn.
- Mode hữu ích khi phân phối có đỉnh rõ.

## Kỳ vọng

### Trực giác

Kỳ vọng (expected value) là trung bình dài hạn nếu lặp lại thí nghiệm rất nhiều lần. Nó không nhất thiết là giá trị có thể xảy ra. Trò tung đồng xu trả $10$ đô nếu heads và $0$ nếu tails có kỳ vọng $5$, dù không lần chơi nào trả đúng $5$.

### Định nghĩa và công thức

Với biến rời rạc,
$$
E[X]=\sum_x x\,p(x).
$$
Với biến liên tục, ta thay tổng bằng tích phân:
$$
E[X]=\int x f_X(x)\,dx.
$$
Phân phối đều trên $[a,b]$ có mean $(a+b)/2$.

### Ví dụ làm đầy đủ

Một đồng xu công bằng trả $10$ nếu heads và $0$ nếu tails. Khi đó
$$
E[X]=0.5\cdot 10+0.5\cdot 0=5.
$$
Nếu phí chơi là $6$, trung bình dài hạn lỗ $1$ mỗi lượt. Nếu phí là $4$, trung bình lãi $1$ mỗi lượt.

### NumPy snippet

```python
import numpy as np
xs = np.array([0, 10])
ps = np.array([0.5, 0.5])
np.sum(xs * ps)
```

> ⚠️ Không chọn giá trị có xác suất cao nhất rồi gọi đó là kỳ vọng.
> Kỳ vọng cần toàn bộ PMF.

### Tóm tắt

- Kỳ vọng là trung bình có trọng số.
- Kỳ vọng có thể nằm giữa các kết quả khả dĩ.
- Phí công bằng của trò chơi bằng kỳ vọng tiền thắng.

## Kỳ vọng của hàm

### Trực giác

Nhiều bài toán ML không chỉ cần $X$ mà cần hàm của $X$, ví dụ bình phương lỗi, log-loss hoặc payoff sau biến đổi. Quy tắc quan trọng là biến đổi từng giá trị trước, rồi mới lấy trung bình có trọng số.

### Định nghĩa và công thức

Với biến rời rạc,
$$
E[g(X)] = \sum_x g(x)p(x).
$$
Nói chung $E[g(X)]\ne g(E[X])$. Đẳng thức chỉ chắc chắn cho hàm tuyến tính $g(x)=ax+b$, khi đó $E[aX+b]=aE[X]+b$.

### Ví dụ làm đầy đủ

Với xúc xắc công bằng, $E[X]=3.5$. Nhưng
$$
E[X^2]=\frac{1^2+2^2+3^2+4^2+5^2+6^2}{6}
=\frac{91}{6}\approx 15.17.
$$
Trong khi $(E[X])^2=12.25$. Hai số khác nhau vì bình phương là hàm phi tuyến.

### NumPy snippet

```python
import numpy as np
xs = np.arange(1, 7)
np.mean(xs**2)
```

> ⚠️ Bình phương mean không thay thế được mean của bình phương.
> Đây là lỗi thường gặp khi tính variance.

### Tóm tắt

- Tính $g(x)$ cho từng kết quả trước.
- Hàm tuyến tính cho phép đưa $E$ vào trong.
- Hàm phi tuyến thường không cho phép làm vậy.

## Tổng các kỳ vọng

### Trực giác

Linearity of expectation nói rằng kỳ vọng của tổng bằng tổng các kỳ vọng. Điều đặc biệt là không cần độc lập. Đây là lý do bài ghép tên trong slide có thể giải bằng indicator thay vì liệt kê mọi hoán vị.

### Định nghĩa và công thức

Với mọi biến ngẫu nhiên có kỳ vọng hữu hạn,
$$
E[X_1+\cdots+X_n]=E[X_1]+\cdots+E[X_n].
$$
Độc lập không cần cho công thức này. Độc lập chỉ cần khi ta muốn nhân kỳ vọng hoặc cộng phương sai theo cách đơn giản.

### Ví dụ làm đầy đủ

Trò chơi gồm tung đồng xu: heads được $1$, tails được $0$, rồi tung xúc xắc và nhận số điểm hiện ra. Kỳ vọng phần đồng xu là $0.5$. Kỳ vọng xúc xắc là $3.5$. Tổng kỳ vọng là $4$ mà không cần liệt kê 12 kết quả.

### NumPy snippet

```python
E_coin = 0.5
E_die = (1 + 2 + 3 + 4 + 5 + 6) / 6
E_total = E_coin + E_die
```

> ⚠️ Đừng thêm giả thiết độc lập nếu bài chỉ hỏi kỳ vọng của tổng.
> Độc lập là điều kiện mạnh hơn mức cần thiết.

### Tóm tắt

- Kỳ vọng cộng được theo từng phần.
- Indicator giúp tính số lượng kỳ vọng.
- Không dùng quy tắc này cho variance nếu thiếu covariance.

## Phương sai

### Trực giác

Mean cho biết trung tâm, nhưng không cho biết rủi ro. Hai trò chơi cùng mean $0$ có thể rất khác nhau: thắng/thua $1$ đô ít rủi ro hơn thắng/thua $100$ đô. Phương sai đo độ xa bình phương so với mean.

### Định nghĩa và công thức

Nếu $\mu=E[X]$, thì
$$
\mathrm{Var}(X)=E[(X-\mu)^2].
$$
Công thức rút gọn:
$$
\mathrm{Var}(X)=E[X^2]-E[X]^2.
$$
Với biến đổi tuyến tính, $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$.

### Ví dụ làm đầy đủ

Với $X=\pm 1$ xác suất đều, mean bằng $0$ và variance bằng $1$. Với $Y=\pm 100$ xác suất đều, mean cũng bằng $0$, nhưng variance bằng $10000$. Vì vậy mean giống nhau không có nghĩa là độ rủi ro giống nhau.

### NumPy snippet

```python
import numpy as np
x = np.array([-1, 1])
np.mean((x - x.mean())**2)
```

> ⚠️ Không tính trung bình $X^2$ rồi dừng lại.
> Cần trừ $E[X]^2$ nếu dùng công thức rút gọn.

### Tóm tắt

- Variance dùng độ lệch bình phương quanh mean.
- Cộng hằng số không đổi variance.
- Nhân bởi $a$ làm variance nhân với $a^2$.

## Độ lệch chuẩn và chuẩn hóa

### Trực giác

Phương sai có đơn vị bình phương, nên đôi khi khó diễn giải. Độ lệch chuẩn lấy căn để quay lại đơn vị ban đầu. Chuẩn hóa z-score đưa phân phối về mean $0$ và độ lệch chuẩn $1$, giúp so sánh các thang đo khác nhau.

### Định nghĩa và công thức

Độ lệch chuẩn:
$$
\sigma=\sqrt{\mathrm{Var}(X)}.
$$
Z-score:
$$
Z=\frac{X-\mu}{\sigma}.
$$
Sau chuẩn hóa, $E[Z]=0$ và $\mathrm{std}(Z)=1$ nếu $\sigma>0$.

### Ví dụ làm đầy đủ

Nếu điểm quan sát là $74$, mean là $70$, và độ lệch chuẩn là $8$, thì
$$
z=\frac{74-70}{8}=0.5.
$$
Điểm này cao hơn mean nửa độ lệch chuẩn, không phải "cao hơn 0.5 điểm".

### NumPy snippet

```python
import numpy as np
x = np.array([66, 70, 74, 78])
z = (x - x.mean()) / x.std(ddof=0)
```

> ⚠️ Chuẩn hóa phải trừ mean trước khi chia.
> Chia trước rồi trừ sẽ cho thang đo sai.

### Tóm tắt

- Standard deviation có cùng đơn vị với dữ liệu.
- Z-score là số độ lệch chuẩn so với mean.
- Chuẩn hóa giúp so sánh đặc trưng ML khác đơn vị.

## Độ lệch và độ nhọn

### Trực giác

Skewness mô tả đuôi lệch về bên nào. Kurtosis mô tả mức độ nặng của đuôi sau khi đã chuẩn hóa. Slide lottery và insurance cho thấy hai phân phối có cùng mean và variance vẫn có thể khác đuôi.

### Định nghĩa và công thức

Skewness:
$$
E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right].
$$
Kurtosis:
$$
E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right].
$$
Moment bậc ba giữ dấu, nên phân biệt đuôi trái và phải. Moment bậc bốn nhấn mạnh giá trị cực xa.

### Ví dụ làm đầy đủ

Lottery: mất $1$ với xác suất $0.99$, thắng $99$ với xác suất $0.01$. Mean bằng $0$, nhưng một kết quả dương rất xa tạo skewness dương. Insurance đảo chiều: thường thắng nhỏ, hiếm khi thua lớn, nên skewness âm.

### NumPy snippet

```python
import numpy as np
z = (x - x.mean()) / x.std(ddof=0)
skew = np.mean(z**3)
kurt = np.mean(z**4)
```

> ⚠️ Không so sánh moment thô khi thang đo khác nhau.
> Hãy chuẩn hóa trước khi đọc skewness hoặc kurtosis.

### Tóm tắt

- Skewness dương nghĩa là đuôi phải mạnh hơn.
- Skewness âm nghĩa là đuôi trái mạnh hơn.
- Kurtosis lớn báo hiệu đuôi dày hoặc ngoại lệ xa.

## Phân vị, box plot, KDE, violin, QQ

### Trực giác

Phân vị chia dữ liệu theo thứ tự. Box plot dùng $Q_1$, median và $Q_3$ để mô tả phần giữa của dữ liệu. KDE và violin plot cho hình dạng mượt hơn histogram. QQ plot so sánh phân vị dữ liệu với phân vị Gaussian để kiểm tra giả định normality.

### Định nghĩa và công thức

$q_p$ là giá trị sao cho khoảng $p$ phần khối lượng nằm bên trái. Các phân vị quen thuộc: $Q_1=q_{0.25}$, $Q_2=q_{0.5}$, $Q_3=q_{0.75}$. IQR bằng $Q_3-Q_1$. Whisker thường dừng ở $Q_1-1.5IQR$ và $Q_3+1.5IQR$ hoặc ở min/max gần nhất.

### Ví dụ làm đầy đủ

Slide quảng cáo báo có dữ liệu đã sắp xếp gồm $8.7,14.2,18.3,18.4,23.2,25.9,29.7,35.2,51.2,54.7,65.9,75$. Median là $(25.9+29.7)/2=27.8$. $Q_1=(18.3+18.4)/2=18.35$, $Q_3=(51.2+54.7)/2=52.95$, nên $IQR=34.6$.

### NumPy snippet

```python
import numpy as np
np.quantile(x, [0.25, 0.5, 0.75])
```

> ⚠️ Quartile phụ thuộc quy ước nội suy.
> Khi so sánh kết quả, hãy nói rõ phương pháp.

### Tóm tắt

- Quantile dựa trên thứ tự dữ liệu.
- IQR mô tả nửa giữa của phân phối.
- QQ plot gần đường thẳng gợi ý dữ liệu gần Gaussian.

## Phân phối đồng thời rời rạc

### Trực giác

Khi có hai biến rời rạc, ta cần xác suất cho từng cặp giá trị. Bảng age-height trong slide là ví dụ: mỗi ô là xác suất một trẻ có tuổi và chiều cao tương ứng. Tổng mọi ô phải bằng $1$.

### Định nghĩa và công thức

Joint PMF:
$$
p_{XY}(x,y)=P(X=x,Y=y).
$$
Điều kiện hợp lệ:
$$
p_{XY}(x,y)\ge 0,\quad \sum_x\sum_y p_{XY}(x,y)=1.
$$
Nếu độc lập, $p_{XY}(x,y)=p_X(x)p_Y(y)$.

### Ví dụ làm đầy đủ

Trong bảng 10 trẻ, ô $X=9$, $Y=49$ có 3 trẻ. Vì tổng là 10, $p_{XY}(9,49)=3/10$. Với hai xúc xắc độc lập, mỗi cặp $(x,y)$ có xác suất $1/36$ vì có 36 kết quả đồng khả năng.

### NumPy snippet

```python
import numpy as np
P = np.array([[0.1, 0.2], [0.3, 0.4]])
P.sum()
```

> ⚠️ Không yêu cầu từng hàng cộng bằng $1$.
> Toàn bộ bảng joint mới phải cộng bằng $1$.

### Tóm tắt

- Joint PMF gán xác suất cho cặp.
- Tổng tất cả ô bằng $1$.
- Độc lập cho phép phân tích thành tích hai marginal.

## Phân phối đồng thời liên tục

### Trực giác

Với biến liên tục, xác suất tại một điểm riêng lẻ thường bằng $0$. Ta quan tâm xác suất rơi vào một vùng, ví dụ thời gian chờ từ 0 đến 5 phút và mức hài lòng từ 7 đến 10.

### Định nghĩa và công thức

Joint PDF $f_{XY}(x,y)$ phải không âm và có tổng thể tích bằng $1$:
$$
\iint f_{XY}(x,y)\,dx\,dy=1.
$$
Xác suất trên vùng $A$ là
$$
P((X,Y)\in A)=\iint_A f_{XY}(x,y)\,dx\,dy.
$$

### Ví dụ làm đầy đủ

Nếu mật độ đều trên hình chữ nhật $[0,2]\times[0,3]$, diện tích là $6$. Để tổng thể tích bằng $1$, mật độ phải là $1/6$. Xác suất rơi vào vùng có diện tích $1.5$ là $1.5\cdot(1/6)=0.25$.

### NumPy snippet

```python
import numpy as np
# xấp xỉ trên lưới
prob = np.sum(density[mask] * dx * dy)
```

> ⚠️ Chiều cao density không phải xác suất.
> Phải nhân với diện tích nhỏ hoặc tích phân trên vùng.

### Tóm tắt

- PDF liên tục được đọc qua diện tích/thể tích.
- Xác suất điểm đơn lẻ thường bằng $0$.
- Mật độ có thể lớn hơn $1$ nếu vùng đủ hẹp.

## Biên và có điều kiện

### Trực giác

Phân phối biên trả lời câu hỏi về một biến khi bỏ qua biến kia. Phân phối có điều kiện trả lời câu hỏi sau khi biết một biến đã nhận giá trị cụ thể. Trong bảng age-height, biết tuổi bằng 9 làm ta chỉ nhìn hàng tuổi 9 rồi chuẩn hóa hàng đó.

### Định nghĩa và công thức

Rời rạc:
$$
p_X(x)=\sum_y p_{XY}(x,y).
$$
Điều kiện:
$$
p_{Y|X=x}(y)=\frac{p_{XY}(x,y)}{p_X(x)}
$$
khi $p_X(x)>0$.

### Ví dụ làm đầy đủ

Từ slide, $P(X=9)=4/10$ và $P(X=9,Y=49)=3/10$. Do đó
$$
P(Y=49\mid X=9)=\frac{3/10}{4/10}=3/4.
$$
Ta không dùng toàn bộ bảng sau khi đã điều kiện hóa; chỉ hàng $X=9$ được chuẩn hóa lại.

### NumPy snippet

```python
import numpy as np
px = P.sum(axis=1)
cond_y_given_x0 = P[0] / px[0]
```

> ⚠️ Mẫu số của conditional là marginal của điều kiện.
> Không chia cho tổng toàn bảng nếu bảng đã là PMF.

### Tóm tắt

- Marginal cộng qua biến bị bỏ qua.
- Conditional chuẩn hóa một lát cắt.
- Conditional chỉ hợp lệ khi mẫu số dương.

## Hiệp phương sai, ma trận hiệp phương sai, tương quan

### Trực giác

Covariance đo hai biến cùng lệch khỏi mean theo chiều giống hay khác nhau. Nếu tuổi tăng và chiều cao tăng, covariance dương. Nếu tuổi tăng và số lần ngủ trưa giảm, covariance âm. Correlation chuẩn hóa covariance để so sánh sức mạnh quan hệ trên các đơn vị khác nhau.

### Định nghĩa và công thức

$$
\mathrm{Cov}(X,Y)=E[(X-\mu_X)(Y-\mu_Y)]
=E[XY]-E[X]E[Y].
$$
Correlation:
$$
\rho=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$
Ma trận covariance đặt variance trên đường chéo và covariance ở ngoài đường chéo.

### Ví dụ làm đầy đủ

Slide cho age-height có covariance $17>0$. Age-naps có covariance $-7.45<0$. Nếu $\mathrm{Var}(X)=9.17$ và $\mathrm{Var}(Y)=7.57$, correlation của age-naps xấp xỉ
$$
\rho=\frac{-7.45}{\sqrt{9.17}\sqrt{7.57}}\approx -0.894.
$$

### NumPy snippet

```python
import numpy as np
X = np.c_[x, y]
np.cov(X, rowvar=False, bias=True)
np.corrcoef(x, y)[0, 1]
```

> ⚠️ Covariance gần $0$ không chứng minh độc lập.
> Nó chỉ nói không có quan hệ tuyến tính rõ.

### Tóm tắt

- Dấu covariance cho chiều đồng biến/nghịch biến.
- Correlation nằm trong $[-1,1]$.
- Ma trận covariance là nền tảng của Gaussian đa biến.

## Gaussian đa biến

### Trực giác

Gaussian một biến có mean và variance. Gaussian nhiều biến cần mean vector và covariance matrix. Ma trận covariance quyết định độ rộng theo từng hướng và độ nghiêng của ellipsoid mật độ.

### Định nghĩa và công thức

Với $x\in\mathbb{R}^n$,
$$
f(x)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}
\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right).
$$
Nếu các biến độc lập, $\Sigma$ là đường chéo. Nếu có covariance khác $0$, contour bị nghiêng.

### Ví dụ làm đầy đủ

Giả sử height và weight độc lập, variance lần lượt là $4$ và $9$. Khi đó
$$
\Sigma=\begin{bmatrix}4&0\\0&9\end{bmatrix}.
$$
Nếu covariance bằng $3$, ma trận thành
$$
\begin{bmatrix}4&3\\3&9\end{bmatrix},
$$
và ellipse nghiêng theo quan hệ height-weight.

### NumPy snippet

```python
import numpy as np
d = x - mu
score = d @ np.linalg.inv(Sigma) @ d
```

> ⚠️ Không thay $\Sigma$ bằng một scalar khi có nhiều biến.
> Off-diagonal chứa thông tin phụ thuộc.

### Tóm tắt

- $\mu$ là vector tâm.
- $\Sigma$ chứa variance và covariance.
- Mahalanobis distance dùng $\Sigma^{-1}$.

## Tổng kết tuần 2

### Trực giác

Tuần 2 tạo một bộ công cụ mô tả dữ liệu: tâm, spread, tail, phân vị, hình vẽ, xác suất chung và quan hệ giữa biến. Trong ML, các công cụ này giúp nhìn dữ liệu trước khi chọn mô hình.

### Định nghĩa và công thức

Không có một công thức duy nhất cho mọi câu hỏi. Mean/median/mode nói về tâm. Variance/std nói về spread. Skewness/kurtosis nói về đuôi. Joint, marginal và conditional nói về nhiều biến. Covariance/correlation nói về quan hệ tuyến tính.

### Ví dụ làm đầy đủ

Với dữ liệu khách hàng, ta có thể báo cáo mean thời gian chờ, median nếu có ngoại lệ, IQR để thấy độ phân tán, KDE để xem hình dạng, rồi covariance giữa thời gian chờ và hài lòng để xem chờ lâu có đi cùng điểm thấp hay không.

### NumPy snippet

```python
import numpy as np
center = np.mean(x)
spread = np.std(x, ddof=0)
relation = np.corrcoef(x, y)[0, 1]
```

> ⚠️ Đừng chọn thước đo vì nó quen thuộc.
> Hãy chọn theo câu hỏi đang cần trả lời.

### Tóm tắt

- Một biến: tâm, spread, hình dạng.
- Hai biến: joint, marginal, conditional.
- Quan hệ: covariance, correlation, Gaussian đa biến.
