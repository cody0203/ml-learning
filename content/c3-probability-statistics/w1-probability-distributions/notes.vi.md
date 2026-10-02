## Xác suất là gì?

### Trực giác
Xác suất bắt đầu từ một thí nghiệm có thể lặp lại: chọn một trẻ, tung đồng xu, hoặc gieo xúc xắc. Slide xem biến cố là phần của không gian mẫu mà ta quan tâm. Nếu 3 trong 10 trẻ chơi bóng đá, biến cố có 3 trường hợp thuận lợi trên 10 trẻ có thể được chọn, nên xác suất là $0.3$. Trong ML, ta cũng bắt đầu bằng câu hỏi: đại lượng nào đang ngẫu nhiên?

### Định nghĩa và công thức
Không gian mẫu (sample space) $S$ chứa mọi kết quả có thể xảy ra. Biến cố (event) $A\subseteq S$ là tập kết quả thuận lợi. Nếu các kết quả đồng khả năng,
$$P(A)=\frac{|A|}{|S|}.$$
Giá trị xác suất nằm trong đoạn từ 0 đến 1.

### Ví dụ giải đầy đủ
Một xúc xắc công bằng có $S=\{1,2,3,4,5,6\}$. Biến cố ra số 6 có một kết quả nên $P(6)=1/6$. Với hai xúc xắc, cặp $(6,6)$ là một cặp có thứ tự trong 36 cặp, nên xác suất là $1/36$.

> ⚠️ Lỗi thường gặp: dùng mẫu số của một xúc xắc cho bài có hai xúc xắc.

### Đoạn NumPy / SciPy
```python
import numpy as np
rng = np.random.default_rng(1)
rolls = rng.integers(1, 7, size=20000)
print(np.mean(rolls == 6))
```

### Tóm tắt
- Xác định thí nghiệm trước khi tính.
- Đếm kết quả thuận lợi và tổng kết quả.
- Hai xúc xắc tạo các cặp có thứ tự.

## Quy tắc bù

### Trực giác
Đôi khi biến cố cần tìm khó liệt kê, nhưng biến cố đối lại rất đơn giản. Slide hỏi xác suất không được ba mặt ngửa. Thay vì liệt kê bảy chuỗi còn lại, ta tính xác suất $HHH$ rồi lấy 1 trừ đi. Biến cố và biến cố bù luôn phủ kín toàn bộ không gian mẫu.

### Định nghĩa và công thức
Biến cố bù (complement) của $A$ là $A^c$, gồm các kết quả mà $A$ không xảy ra. Do đúng một trong hai biến cố xảy ra,
$$P(A^c)=1-P(A).$$
Quy tắc này dùng được cho cả rời rạc và liên tục.

### Ví dụ giải đầy đủ
Ba lần tung đồng xu công bằng có $2^3=8$ chuỗi. Biến cố $HHH$ có xác suất $1/8$. Vậy xác suất không được ba mặt ngửa là
$$1-\frac18=\frac78=0.875.$$

> ⚠️ Lỗi thường gặp: hiểu "không phải $HHH$" thành "toàn sấp". Thực ra nó gồm mọi chuỗi trừ $HHH$.

### Đoạn NumPy / SciPy
```python
p_hhh = 0.5 ** 3
print(1 - p_hhh)
```

### Tóm tắt
- Bù là phần còn lại của không gian mẫu.
- Hữu ích cho bài "ít nhất một" hoặc "không phải tất cả".
- Hai xác suất bù nhau cộng bằng 1.

## Quy tắc cộng cho biến cố rời nhau và giao nhau

### Trực giác
"Hoặc" trong xác suất nghĩa là hợp. Nếu bóng đá và bóng rổ loại trừ nhau, ta cộng số trẻ ở hai nhóm. Nếu một trẻ có thể chơi cả hai môn, cộng thẳng sẽ đếm trùng vùng giao. Các slide dùng Venn diagram để so sánh hai tình huống này.

### Định nghĩa và công thức
Với hai biến cố bất kỳ,
$$P(A\cup B)=P(A)+P(B)-P(A\cap B).$$
Nếu $A$ và $B$ rời nhau, phần giao bằng 0 và công thức còn $P(A\cup B)=P(A)+P(B)$.

### Ví dụ giải đầy đủ
Trong 10 trẻ, 6 chơi bóng đá, 5 chơi bóng rổ, 3 chơi cả hai. Số trẻ chơi ít nhất một môn là $6+5-3=8$, nên xác suất là $0.8$. Nếu hai môn loại trừ nhau với xác suất 0.3 và 0.4, hợp bằng 0.7.

> ⚠️ Lỗi thường gặp: dùng công thức rời nhau dù hai vòng tròn Venn giao nhau.

### Đoạn NumPy / SciPy
```python
soccer = {1, 2, 3, 4, 5, 6}
basket = {4, 5, 6, 7, 8}
print(len(soccer | basket) / 10)
```

### Tóm tắt
- Hợp gồm kết quả thuộc ít nhất một biến cố.
- Trừ phần giao để không đếm hai lần.
- Rời nhau nghĩa là không có phần giao.

## Độc lập và quy tắc nhân

### Trực giác
Độc lập nghĩa là biết một biến cố xảy ra không làm đổi xác suất của biến cố kia. Trong ví dụ phòng học và bóng đá, nếu 40% trẻ chơi bóng đá và việc xếp phòng độc lập, thì từng phòng cũng nên có 40% trẻ chơi bóng đá.

### Định nghĩa và công thức
Hai biến cố độc lập khi $P(B\mid A)=P(B)$. Khi đó
$$P(A\cap B)=P(A)P(B).$$
Nếu không biết độc lập, không được tự động nhân.

### Ví dụ giải đầy đủ
Giả sử $P(S)=0.4$ cho bóng đá và $P(R_1)=0.3$ cho phòng 1. Nếu độc lập, xác suất vừa chơi bóng đá vừa ở phòng 1 là $0.4\cdot0.3=0.12$. Trong 100 trẻ, đó là 12 trẻ.

> ⚠️ Lỗi thường gặp: thấy chữ "và" rồi nhân ngay, dù đề chưa nói độc lập.

### Đoạn NumPy / SciPy
```python
p_soccer = 0.4
p_room1 = 0.3
print(p_soccer * p_room1)
```

### Tóm tắt
- Độc lập là xác suất có điều kiện không đổi.
- Quy tắc nhân chỉ là shortcut khi độc lập.
- Bài phụ thuộc phải dùng quy tắc nhân tổng quát.

## Xác suất có điều kiện

### Trực giác
Điều kiện làm nhỏ không gian mẫu. Slide hỏi xác suất tổng hai xúc xắc bằng 10, rồi hỏi lại khi biết xúc xắc thứ nhất là 6. Câu thứ hai chỉ còn sáu kết quả có thể, không còn 36.

### Định nghĩa và công thức
Với $P(A)>0$,
$$P(B\mid A)=\frac{P(A\cap B)}{P(A)}.$$
Dấu gạch dọc đọc là "biết rằng" và làm đổi mẫu số.

### Ví dụ giải đầy đủ
Gọi $B$ là biến cố tổng bằng 10, $A$ là biến cố xúc xắc thứ nhất bằng 6. Phần giao chỉ có $(6,4)$. Do đó
$$P(B\mid A)=\frac{1/36}{6/36}=\frac16.$$
Nếu xúc xắc thứ nhất bằng 1 thì xác suất sẽ bằng 0.

> ⚠️ Lỗi thường gặp: tráo $P(B\mid A)$ với $P(A\mid B)$.

### Đoạn NumPy / SciPy
```python
p_joint = 1 / 36
p_first6 = 6 / 36
print(p_joint / p_first6)
```

### Tóm tắt
- Điều kiện đổi tập tham chiếu.
- Tử số là phần giao.
- Mẫu số là xác suất của điều kiện.

## Định lý Bayes

### Trực giác
Bayes đảo chiều xác suất có điều kiện. Một xét nghiệm có thể đúng 99%, nhưng nếu bệnh cực hiếm thì phần lớn kết quả dương tính vẫn có thể là giả. Slide minh họa bằng một triệu người để nhấn mạnh vai trò của base rate.

### Định nghĩa và công thức
Với giả thuyết $A$ và bằng chứng $B$,
$$P(A\mid B)=\frac{P(A)P(B\mid A)}{P(A)P(B\mid A)+P(A^c)P(B\mid A^c)}.$$
Mẫu số là mọi con đường tạo ra bằng chứng.

### Ví dụ giải đầy đủ
Cho $P(\text{sick})=0.0001$, $P(+\mid\text{sick})=0.99$, và $P(+\mid\text{healthy})=0.01$. Khi đó
$$P(\text{sick}\mid +)=\frac{0.0001\cdot0.99}{0.0001\cdot0.99+0.9999\cdot0.01}\approx0.0098.$$

> ⚠️ Lỗi thường gặp: trả lời 99% vì nhầm độ nhạy với posterior.

### Đoạn NumPy / SciPy
```python
prior = 0.0001
post = prior * 0.99 / (prior * 0.99 + (1 - prior) * 0.01)
print(post)
```

### Tóm tắt
- Prior là xác suất trước bằng chứng.
- Likelihood mô tả bằng chứng trong từng lớp.
- Posterior là xác suất đã cập nhật.

## Naive Bayes

### Trực giác
Naive Bayes áp dụng Bayes cho nhiều đặc trưng bằng giả định đơn giản: các từ độc lập có điều kiện theo lớp. Ví dụ spam dùng hai từ "lottery" và "winning"; khi hai từ cùng xuất hiện, điểm spam tăng mạnh.

### Định nghĩa và công thức
Với lớp $C$ và các từ $w_i$,
$$P(C\mid w_1,\ldots,w_n)\propto P(C)\prod_i P(w_i\mid C).$$
Sau đó chuẩn hóa điểm của các lớp để tổng bằng 1.

### Ví dụ giải đầy đủ
Lấy $P(spam)=0.2$, $P(lottery\mid spam)=0.7$, $P(winning\mid spam)=0.75$, $P(lottery\mid ham)=0.125$, $P(winning\mid ham)=0.1$. Điểm spam là $0.105$, điểm ham là $0.01$, nên posterior là $0.105/0.115\approx0.913$.

> ⚠️ Lỗi thường gặp: tưởng giả định naive luôn đúng tuyệt đối. Nó chỉ là xấp xỉ hữu ích.

### Đoạn NumPy / SciPy
```python
s = 0.2 * 0.7 * 0.75
h = 0.8 * 0.125 * 0.1
print(s / (s + h))
```

### Tóm tắt
- Tính một điểm cho mỗi lớp.
- Nhân likelihood của từng từ.
- Chuẩn hóa để có xác suất.

## Xác suất trong học máy

### Trực giác
Mô hình ML thường trả về xác suất thay vì khẳng định tuyệt đối. Bộ nhận diện ảnh có thể trả $P(cat\mid pixels)=0.9$; mô hình y tế ước lượng $P(healthy\mid symptoms)$; phân tích cảm xúc ước lượng $P(happy\mid words)$.

### Định nghĩa và công thức
Bài toán phân loại thường học
$$P(y\mid x),$$
xác suất nhãn $y$ khi biết đặc trưng $x$. Mô hình sinh cũng dùng xác suất để tạo mẫu hợp lý.

### Ví dụ giải đầy đủ
Nếu bộ phân loại ảnh trả 0.9 cho cat và 0.1 cho not-cat, nhãn dự đoán là cat. Số 0.9 là độ tin cậy theo mô hình và dữ liệu huấn luyện, không phải chứng minh chắc chắn.

> ⚠️ Lỗi thường gặp: xem điểm xác suất như sự thật đã hiệu chỉnh hoàn hảo.

### Đoạn NumPy / SciPy
```python
probs = {"cat": 0.9, "not cat": 0.1}
print(max(probs, key=probs.get))
```

### Tóm tắt
- Phân loại dùng xác suất có điều kiện.
- Ngưỡng quyết định phụ thuộc chi phí sai.
- Sinh dữ liệu cũng dựa vào phân phối.

## Biến ngẫu nhiên

### Trực giác
Biến ngẫu nhiên biến kết quả phức tạp thành con số. Thay vì lưu toàn bộ chuỗi tung đồng xu, slide đặt $X$ là số mặt ngửa. Khi đó thí nghiệm 10 lần tung được tóm tắt bởi các giá trị từ 0 đến 10.

### Định nghĩa và công thức
Biến ngẫu nhiên là hàm từ kết quả sang số thực:
$$X:S\to\mathbb R.$$
Biến rời rạc nhận các giá trị đếm được; biến liên tục nhận giá trị trên khoảng.

### Ví dụ giải đầy đủ
Với một đồng xu, đặt $X=1$ nếu ngửa và $X=0$ nếu sấp. Khi đó $P(X=1)=0.5$ và $P(X=0)=0.5$. Với 10 lần tung, $X$ có thể là mọi số nguyên từ 0 đến 10.

> ⚠️ Lỗi thường gặp: nhầm biến ngẫu nhiên với kết quả thô. Nó là bản tóm tắt số của kết quả.

### Đoạn NumPy / SciPy
```python
rng = np.random.default_rng(2)
print(rng.binomial(n=10, p=0.5, size=6))
```

### Tóm tắt
- Biến ngẫu nhiên gán số cho kết quả.
- Nó giúp mô tả cả thí nghiệm bằng một hàm.
- Biến đếm thường là rời rạc.

## Phân phối rời rạc và PMF

### Trực giác
PMF đặt khối xác suất lên từng giá trị rời rạc. Slide xây PMF cho số mặt ngửa khi tung ba, bốn, năm đồng xu. Các cột cộng lại bằng 1 vì chắc chắn một trong các số đếm sẽ xảy ra.

### Định nghĩa và công thức
Với biến rời rạc,
$$p_X(x)=P(X=x),\qquad \sum_x p_X(x)=1.$$
Mỗi khối xác suất phải không âm.

### Ví dụ giải đầy đủ
Ba lần tung đồng xu công bằng cho xác suất $1/8,3/8,3/8,1/8$ tại $0,1,2,3$ mặt ngửa. Vì vậy $P(X\ge2)=3/8+1/8=1/2$.

> ⚠️ Lỗi thường gặp: đọc diện tích dưới cột PMF thay vì chiều cao cột.

### Đoạn NumPy / SciPy
```python
from math import comb
pmf = [comb(3, k) / 8 for k in range(4)]
print(pmf, sum(pmf))
```

### Tóm tắt
- Cột PMF chính là xác suất.
- Tổng các cột bằng 1.
- Biến đếm là ví dụ tự nhiên.

## Bernoulli và nhị thức

### Trực giác
Bernoulli mô tả một lần thử thành công/thất bại. Nhị thức đếm số lần thành công trong $n$ thử độc lập cùng xác suất $p$. Slide biến việc "gieo ra số 1" thành đồng xu lệch với $p=1/6$.

### Định nghĩa và công thức
Nếu $X\sim Bernoulli(p)$, thì $P(X=1)=p$ và $P(X=0)=1-p$. Nếu $X\sim Binomial(n,p)$,
$$P(X=k)=\binom nk p^k(1-p)^{n-k}.$$

### Ví dụ giải đầy đủ
Với năm lần tung đồng xu công bằng, xác suất đúng hai mặt ngửa là
$$\binom52(0.5)^2(0.5)^3=10/32=0.3125.$$
Với năm lần gieo xúc xắc, đúng ba số 1 dùng $p=1/6$.

> ⚠️ Lỗi thường gặp: quên hệ số nhị thức và chỉ tính một thứ tự.

### Đoạn NumPy / SciPy
```python
from math import comb
print(comb(5, 2) * 0.5**5)
```

### Tóm tắt
- Bernoulli là một thử.
- Nhị thức đếm số thành công.
- Hệ số nhị thức đếm các thứ tự.

## Biến liên tục và PDF

### Trực giác
Thời gian chờ có thể là 1.01, 2.43, hoặc vô số giá trị khác. Với biến liên tục, xác suất đúng một điểm bằng 0. Xác suất nằm trong khoảng được vẽ bằng diện tích dưới đường mật độ.

### Định nghĩa và công thức
PDF $f_X$ thỏa $f_X(x)\ge0$ và tổng diện tích bằng 1. Với một khoảng,
$$P(a<X<b)=\int_a^b f_X(x)\,dx.$$
Chiều cao $f_X(x)$ là mật độ, không phải xác suất tại điểm.

### Ví dụ giải đầy đủ
Nếu thời gian chờ đều từ 0 đến 5 phút, mật độ là $1/5$. Xác suất chờ từ 2 đến 3 phút bằng diện tích hình chữ nhật $1\cdot(1/5)=0.2$.

> ⚠️ Lỗi thường gặp: viết $P(X=2)=f_X(2)$. Với biến liên tục, $P(X=2)=0$.

### Đoạn NumPy / SciPy
```python
x = np.linspace(0, 5, 501)
print(np.trapz(np.full_like(x, 0.2), x))
```

### Tóm tắt
- Xác suất liên tục là diện tích.
- Điểm đơn lẻ có xác suất 0.
- PDF có thể cao hơn 1 nếu khoảng rất hẹp.

## Hàm phân phối tích lũy CDF

### Trực giác
CDF cộng dồn xác suất từ bên trái đến một giá trị đang xét. Slide vẽ đường bắt đầu ở 0, không bao giờ giảm, và cuối cùng lên 1. CDF dùng được cho cả biến rời rạc lẫn liên tục.

### Định nghĩa và công thức
Hàm phân phối tích lũy là
$$F_X(x)=P(X\le x).$$
Nó luôn nằm giữa 0 và 1, đồng thời không giảm.

### Ví dụ giải đầy đủ
Với $X\sim Uniform(0,4)$, ta có $F(2)=(2-0)/(4-0)=0.5$. Xác suất $P(1<X\le3)$ bằng $F(3)-F(1)=3/4-1/4=1/2$.

> ⚠️ Lỗi thường gặp: chấp nhận một đồ thị đi xuống là CDF.

### Đoạn NumPy / SciPy
```python
def uniform_cdf(x):
    return np.clip(x / 4, 0, 1)
print(uniform_cdf(2))
```

### Tóm tắt
- CDF là xác suất không vượt quá $x$.
- Hiệu hai CDF cho xác suất khoảng.
- CDF hợp lệ không giảm.

## Phân phối đều

### Trực giác
Phân phối đều nghĩa là các khoảng có cùng độ dài trong miền hỗ trợ có cùng xác suất. Ví dụ tổng đài kỹ thuật có thể trả lời bất kỳ lúc nào từ 0 đến 15 phút, không ưu tiên đoạn nào.

### Định nghĩa và công thức
Nếu $X\sim Uniform(a,b)$,
$$f_X(x)=\frac1{b-a}\quad a<x<b,$$
và CDF tăng tuyến tính từ 0 đến 1 trên khoảng đó.

### Ví dụ giải đầy đủ
Nếu $T\sim Uniform(0,15)$, mật độ là $1/15$. Xác suất chờ từ 6 đến 9 phút là $(9-6)/15=0.2$.

> ⚠️ Lỗi thường gặp: xem giá trị ở đầu mút như một xác suất riêng.

### Đoạn NumPy / SciPy
```python
a, b = 0, 15
print((9 - 6) / (b - a))
```

### Tóm tắt
- Mật độ hằng trên một khoảng hữu hạn.
- Xác suất bằng độ dài đoạn chia cho tổng độ dài.
- Ngoài khoảng, mật độ bằng 0.

## Phân phối chuẩn

### Trực giác
Phân phối chuẩn là đường chuông xuất hiện trong chiều cao, nhiễu, điểm số và tổng của nhiều ảnh hưởng độc lập nhỏ. Slide cho thấy $\mu$ dịch tâm chuông, còn $\sigma$ làm chuông rộng hoặc hẹp hơn.

### Định nghĩa và công thức
Nếu $X\sim\mathcal N(\mu,\sigma^2)$,
$$f_X(x)=\frac1{\sigma\sqrt{2\pi}}\exp\left[-\frac12\left(\frac{x-\mu}{\sigma}\right)^2\right].$$
Chuẩn hóa dùng $Z=(X-\mu)/\sigma$.

### Ví dụ giải đầy đủ
Với $X\sim\mathcal N(10,2^2)$, giá trị $x=14$ có $z=(14-10)/2=2$. Nó nằm cao hơn trung bình hai độ lệch chuẩn; diện tích chính xác cần bảng hoặc phần mềm.

> ⚠️ Lỗi thường gặp: đọc xác suất chuẩn trực tiếp từ chiều cao PDF.

### Đoạn NumPy / SciPy
```python
from math import erf, sqrt
z = (14 - 10) / 2
print(0.5 * (1 + erf(z / sqrt(2))))
```

### Tóm tắt
- $\mu$ điều khiển tâm.
- $\sigma$ điều khiển độ phân tán.
- Xác suất là diện tích dưới chuông.

## Phân phối Chi-squared

### Trực giác
Chi-squared xuất hiện khi cộng các bình phương của nhiễu chuẩn hóa. Slide dùng công suất nhiễu trong kênh truyền: nếu $Z$ là nhiễu chuẩn tắc, công suất $Z^2$ luôn không âm.

### Định nghĩa và công thức
Nếu $Z_1,\ldots,Z_k$ độc lập và chuẩn tắc, thì
$$W=\sum_{i=1}^k Z_i^2\sim \chi^2_k.$$
Tham số $k$ là bậc tự do.

### Ví dụ giải đầy đủ
Nếu hai lần truyền có nhiễu chuẩn hóa là $1$ và $-2$, công suất tích lũy là $1^2+(-2)^2=5$. Nếu các giá trị nhiễu là ngẫu nhiên chuẩn tắc, tổng đó theo phân phối chi-squared với 2 bậc tự do.

> ⚠️ Lỗi thường gặp: kỳ vọng giá trị chi-squared âm. Bình phương làm miền giá trị bắt đầu từ 0.

### Đoạn NumPy / SciPy
```python
rng = np.random.default_rng(4)
z = rng.normal(size=(10000, 2))
print(np.sum(z**2, axis=1).mean())
```

### Tóm tắt
- Chi-squared là tổng bình phương.
- Bậc tự do là số hạng chuẩn tắc.
- Nó mô hình hóa công suất nhiễu.

## Lấy mẫu từ phân phối

### Trực giác
Lấy mẫu biến xác suất thành kết quả mô phỏng. Slide chia đoạn $[0,1]$ theo xác suất tích lũy: xanh lá nhận $[0,0.3)$, xanh dương nhận $[0.3,0.8)$, cam nhận $[0.8,1]$.

### Định nghĩa và công thức
Với PMF rời rạc, xác suất tích lũy tạo các khoảng có độ dài bằng xác suất của từng loại. Lấy $u\sim Uniform(0,1)$ rồi chọn loại chứa $u$.

### Ví dụ giải đầy đủ
Với xác suất xanh lá 0.3, xanh dương 0.5, cam 0.2, số ngẫu nhiên $u=0.62$ nằm trong $[0.3,0.8)$ nên mẫu là xanh dương. Nếu $u=0.91$ thì chọn cam.

> ⚠️ Lỗi thường gặp: dùng các khoảng bằng nhau dù xác suất của các loại khác nhau.

### Đoạn NumPy / SciPy
```python
probs = np.array([0.3, 0.5, 0.2])
u = 0.62
print(np.searchsorted(np.cumsum(probs), u, side="right"))
```

### Tóm tắt
- Số ngẫu nhiên được ánh xạ vào khoảng tích lũy.
- CDF giúp lấy mẫu có hệ thống.
- Mô phỏng lớn nên gần PMF.
