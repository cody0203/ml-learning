# Tuần 4: Khoảng tin cậy và kiểm định

PDF đi từ ước lượng sang quyết định:
khoảng tin cậy khi biết $\sigma$,
sai số biên, cỡ mẫu, diễn giải,
Student's $t$, tỷ lệ, giả thuyết,
lỗi loại I/II, p-value, critical value,
power, t-test và A/B testing.

## Khoảng tin cậy khi biết σ

### Trực giác

Trung bình mẫu (sample mean) chỉ là ảnh chụp
có nhiễu.  Khi biết độ lệch chuẩn tổng thể
(population standard deviation) $\sigma$,
Central Limit Theorem cho biết mức nhiễu của
$\bar X$.  Vì vậy khoảng tin cậy là một
quy trình đã hiệu chỉnh, không phải lời hứa
tuyệt đối.

### Định nghĩa và công thức

Với confidence level $1-\alpha$:

$$
\bar x \pm z_{\alpha/2}
\frac{\sigma}{\sqrt n}.
$$

Giá trị $z_{\alpha/2}$ để lại diện tích
$\alpha/2$ ở mỗi đuôi normal.
Các mốc thường dùng:
90% dùng 1.645,
95% dùng 1.96,
99% dùng 2.576.

### Ví dụ giải đầy đủ

Slides dùng ví dụ chiều cao ở Statistopia.
Giả sử $n=49$, $\bar x=170$ cm,
và biết $\sigma=25$ cm.
Với confidence 95%, chọn $z=1.96$.

$$
SE=\frac{25}{\sqrt{49}}=\frac{25}{7}=3.571.
$$

$$
MOE=1.96(3.571)=7.00.
$$

Khoảng là

$$
170\pm 7=(163,177).
$$

Ta ước lượng mean height của người lớn
ở Statistopia nằm từ 163 đến 177 cm.

### Python snippet

```python
import math

xbar = 170
sigma = 25
n = 49
z = 1.96
me = z * sigma / math.sqrt(n)
print(xbar - me, xbar + me)
```

> ⚠️ Lỗi thường gặp:
> dùng $s$ và critical value $t$ khi đề bài
> đã nói rõ $\sigma$ được biết.

### Tóm tắt

- Biết $\sigma$ thì dùng z interval.
- Độ rộng phụ thuộc $z$, $\sigma$, và $n$.
- Estimate là $\bar x$; parameter là $\mu$.

## Sai số biên, cỡ mẫu và mức tin cậy

### Trực giác

Margin of error là nửa độ rộng khoảng.
Nó là cái giá của sự bất định.
Confidence cao hơn cần critical value lớn hơn.
Sample size lớn làm mean ổn định hơn,
nhưng tốc độ chỉ theo căn bậc hai.

### Định nghĩa và công thức

Với mean interval khi biết $\sigma$:

$$
MOE=z_{\alpha/2}\frac{\sigma}{\sqrt n}.
$$

Muốn margin không quá $m$:

$$
n\ge
\left(\frac{z_{\alpha/2}\sigma}{m}\right)^2.
$$

Kết quả cỡ mẫu luôn làm tròn lên.

### Ví dụ giải đầy đủ

Khoảng Statistopia ban đầu có margin 7 cm.
Bây giờ muốn margin tối đa 3 cm,
vẫn với $\sigma=25$ và confidence 95%.

$$
n\ge
\left(\frac{1.96\cdot25}{3}\right)^2
=266.78.
$$

Cỡ mẫu nguyên nhỏ nhất là 267.
Nếu lấy 266 thì chưa chắc đạt yêu cầu.

### Python snippet

```python
import math

sigma = 25
z = 1.96
target = 3
n = math.ceil((z * sigma / target) ** 2)
print(n)
```

> ⚠️ Lỗi thường gặp:
> tưởng tăng gấp đôi $n$ thì margin giảm nửa.
> Muốn margin giảm nửa cần gần bốn lần mẫu.

### Tóm tắt

- Confidence cao làm khoảng rộng hơn.
- $n$ lớn làm khoảng hẹp hơn.
- Planning sample size phải làm tròn lên.

## Diễn giải confidence interval

### Trực giác

Confidence nói về quy trình,
không phải xác suất của một $\mu$ cố định.
Sau khi quan sát dữ liệu,
hai endpoint của interval đã cố định.
True mean cũng cố định.
Khoảng đó hoặc chứa $\mu$, hoặc không.

### Định nghĩa và công thức

Một quy trình confidence 95% nghĩa là:
nếu lặp lại lấy mẫu rất nhiều lần,
khoảng 95% các khoảng sẽ chứa $\mu$.

Nó không nghĩa là:
95% cá thể nằm trong khoảng,
hoặc sau khi thấy dữ liệu thì
$P(\mu\in[L,U])=0.95$.

### Ví dụ giải đầy đủ

Hãy tưởng tượng tạo 100 khoảng cho cùng
một population mean, mỗi khoảng từ mẫu mới.
Nếu phương pháp có coverage 95%,
ta kỳ vọng khoảng 95 khoảng chứa $\mu$.
Con số thực có thể là 93 hoặc 98.
Sự dao động đó là ngẫu nhiên.

Với khoảng đã báo cáo, ví dụ $(163,177)$,
nên nói:
"Phương pháp này cover 95% trong dài hạn,
và khoảng quan sát là ước lượng của ta."

### Python snippet

```python
import numpy as np

rng = np.random.default_rng(7)
mu, sigma, n = 10, 2, 40
hits = 0
for _ in range(2000):
    sample = rng.normal(mu, sigma, n)
    xbar = sample.mean()
    me = 1.96 * sigma / np.sqrt(n)
    hits += (xbar - me <= mu <= xbar + me)
print(hits / 2000)
```

> ⚠️ Lỗi thường gặp:
> đọc CI của mean như prediction interval
> cho từng cá nhân.

### Tóm tắt

- Confidence là long-run coverage.
- Khoảng cố định không có xác suất di động.
- Mục tiêu là parameter, không phải cá thể.

## Khoảng tin cậy t và tỷ lệ

### Trực giác

Khi không biết $\sigma$,
ta thay bằng sample standard deviation $s$.
Việc ước lượng thêm này tạo bất định,
nên Student's $t$ có đuôi nặng hơn normal.
Với outcome nhị phân, parameter là proportion,
và variance đến từ Bernoulli trials.

### Định nghĩa và công thức

Mean khi chưa biết $\sigma$:

$$
\bar x\pm t_{\alpha/2,n-1}\frac{s}{\sqrt n}.
$$

Một proportion:

$$
\hat p\pm z_{\alpha/2}
\sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

### Ví dụ giải đầy đủ

Với mẫu height:
$n=10$, $\bar x=68.442$, $s=3.113$.
Degrees of freedom là 9.
Với CI 95%, $t^*=2.262$.

$$
SE=3.113/\sqrt{10}=0.984.
$$

$$
MOE=2.262(0.984)=2.227.
$$

Khoảng t là $(66.215,70.669)$.

Nếu 24 trong 30 người thích một design,
$\hat p=0.8$.
Margin 95% là
$1.96\sqrt{0.8(0.2)/30}=0.143$,
nên CI khoảng $(0.657,0.943)$.

### Python snippet

```python
import math
from scipy import stats

xbar, s, n = 68.442, 3.113, 10
tcrit = stats.t.ppf(0.975, df=n-1)
print(xbar - tcrit*s/math.sqrt(n))
print(xbar + tcrit*s/math.sqrt(n))
```

> ⚠️ Lỗi thường gặp:
> dùng cùng một công thức cho mean và
> proportion chỉ vì cả hai đều là interval.

### Tóm tắt

- Chưa biết $\sigma$ thì dùng $t$.
- One-sample t có $df=n-1$.
- Proportion dùng $\hat p(1-\hat p)$.

## Giả thuyết, lỗi loại I/II và mức ý nghĩa

### Trực giác

Hypothesis test bắt đầu từ một câu chuyện
mặc định.  Null hypothesis $H_0$ là baseline.
Alternative $H_1$ là claim cần bằng chứng.
Trong ví dụ spam, email thường là mặc định;
đưa vào spam cần dấu hiệu đủ mạnh.

### Định nghĩa và công thức

Type I error:
reject $H_0$ khi $H_0$ đúng.
Type II error:
không reject $H_0$ khi $H_0$ sai.

Significance level:

$$
\alpha=\max P(\text{reject }H_0
\mid H_0\text{ true}).
$$

### Ví dụ giải đầy đủ

Tung đồng xu 10 lần và thấy 8 heads.
Đặt $H_0:p=0.5$ và $H_1:p>0.5$.
Dưới null:

$$
P(X\ge 8)=
\frac{\binom{10}{8}+\binom{10}{9}
+\binom{10}{10}}{2^{10}}
=0.0547.
$$

Ở $\alpha=0.05$,
xác suất này chưa đủ nhỏ để reject.
Nếu thấy 80 heads trong 100 lần tung,
tail probability rất nhỏ,
nên reject sẽ hợp lý hơn nhiều.

### Python snippet

```python
from math import comb

p = sum(comb(10, k) for k in range(8, 11)) / 2**10
print(p)
```

> ⚠️ Lỗi thường gặp:
> nói "accept $H_0$" sau khi p-value lớn.
> Câu an toàn hơn là "fail to reject $H_0$".

### Tóm tắt

- $H_0$ là mô hình mặc định.
- $\alpha$ giới hạn false positive.
- Type II error là bỏ sót hiệu ứng thật.

## Kiểm định một/ hai phía, p-value và critical value

### Trực giác

Alternative quyết định dữ liệu nào là extreme.
Nếu $H_1:\mu>\mu_0$, bằng chứng nằm ở đuôi phải.
Nếu $H_1:\mu\ne\mu_0$, cả hai đuôi đều quan trọng.
p-value đo xác suất đi xa ít nhất như quan sát
theo hướng mà $H_1$ quy định.

### Định nghĩa và công thức

Mean test khi biết $\sigma$:

$$
z=\frac{\bar x-\mu_0}{\sigma/\sqrt n}.
$$

Right tail:
$p=P(Z\ge z_{obs})$.
Left tail:
$p=P(Z\le z_{obs})$.
Two tails:
$p=2P(Z\ge |z_{obs}|)$.

### Ví dụ giải đầy đủ

Với height example:
$\bar x=68.442$, $\mu_0=66.7$,
$\sigma=3$, và $n=10$.

$$
z=\frac{68.442-66.7}{3/\sqrt{10}}=1.837.
$$

Với $H_1:\mu>66.7$,
p-value khoảng 0.0407,
nên test 5% reject.
Với $H_1:\mu\ne66.7$,
p-value hai phía khoảng 0.0814,
nên cùng dữ liệu lại không reject.

### Python snippet

```python
import math
from scipy import stats

z = (68.442 - 66.7) / (3 / math.sqrt(10))
print(stats.norm.sf(z))
print(2 * stats.norm.sf(abs(z)))
```

> ⚠️ Lỗi thường gặp:
> chọn tail sau khi đã nhìn thấy mean tăng hay giảm.

### Tóm tắt

- Alternative cố định tail.
- p-value và critical value phải đồng nhất.
- Two-tailed test cần bằng chứng mạnh hơn.

## Power và diễn giải kết quả kiểm định

### Trực giác

Power hỏi test có đủ nhạy để phát hiện
hiệu ứng thật hay không.
Một p-value lớn chưa nói rằng hiệu ứng bằng 0.
Nếu study thiếu power,
kết quả không significant rất khó diễn giải.

### Định nghĩa và công thức

Với một giá trị thật thuộc alternative:

$$
\beta(\theta)=P(\text{fail to reject }H_0
\mid \theta).
$$

Power là:

$$
\text{power}(\theta)=1-\beta(\theta).
$$

### Ví dụ giải đầy đủ

Một right-tailed height test reject khi
$\bar X>68.26$.
Cho $\sigma=3$ và $n=10$.
Nếu true mean thật sự là 70,
standard error của $\bar X$ là $3/\sqrt{10}$.

$$
z=\frac{68.26-70}{3/\sqrt{10}}=-1.834.
$$

Vậy
$\beta=P(\bar X\le 68.26\mid\mu=70)=0.0333$.
Power bằng $1-0.0333=0.9667$.

### Python snippet

```python
import math
from scipy import stats

crit = 68.26
mu_true = 70
se = 3 / math.sqrt(10)
beta = stats.norm.cdf((crit - mu_true) / se)
print(beta, 1 - beta)
```

> ⚠️ Lỗi thường gặp:
> kết luận "không có effect" chỉ vì p-value lớn,
> mà không hỏi test có đủ power hay không.

### Tóm tắt

- Power là độ nhạy dưới alternative.
- Tăng $n$ thường tăng power.
- Giảm $\alpha$ thường làm power giảm.

## t-tests, kiểm định tỷ lệ và A/B testing

### Trực giác

Thiết kế dữ liệu khác nhau cần statistic khác nhau.
One-sample test so sánh mean với baseline.
Two-sample test so sánh hai nhóm độc lập.
Paired test so sánh thay đổi trong cùng đơn vị.
A/B testing dùng các ý này cho metric sản phẩm
như purchase amount hoặc conversion rate.

### Định nghĩa và công thức

One-sample t statistic:

$$
t=\frac{\bar x-\mu_0}{s/\sqrt n}.
$$

Welch two-sample statistic:

$$
t=\frac{\bar x-\bar y-\Delta_0}
{\sqrt{s_x^2/n_x+s_y^2/n_y}}.
$$

Two-proportion z statistic dưới
$H_0:p_A=p_B$:

$$
z=\frac{\hat p_A-\hat p_B}
{\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}}.
$$

### Ví dụ giải đầy đủ

Trong A/B conversion,
design A có $x_A=20$ conversions
trên $n_A=80$ users.
Design B có $x_B=8$ trên $n_B=20$.
Khi đó $\hat p_A=0.25$ và $\hat p_B=0.40$.
Dưới $H_0:p_A=p_B$,
pooled estimate là $28/100=0.28$.

$$
SE=\sqrt{0.28(0.72)(1/80+1/20)}=0.1123.
$$

$$
z=\frac{0.25-0.40}{0.1123}=-1.336.
$$

Với $H_1:p_A-p_B<0$,
p-value khoảng 0.091.
Ở mức 5%, chưa đủ bằng chứng rằng
B convert tốt hơn A.

### Python snippet

```python
import math
from scipy import stats

xA, nA, xB, nB = 20, 80, 8, 20
pA, pB = xA/nA, xB/nB
pool = (xA + xB) / (nA + nB)
se = math.sqrt(pool*(1-pool)*(1/nA + 1/nB))
z = (pA - pB) / se
print(z, stats.norm.cdf(z))
```

> ⚠️ Lỗi thường gặp:
> coi before/after trên cùng người là hai nhóm độc lập,
> làm mất lợi ích của pairing.

### Tóm tắt

- Chọn test theo thiết kế dữ liệu.
- Welch hữu ích khi variance không bằng nhau.
- A/B test cần cả ý nghĩa thống kê lẫn thực tế.
