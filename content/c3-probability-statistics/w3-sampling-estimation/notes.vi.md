# Tuần 3: Lấy mẫu và ước lượng điểm

Tuần này đi từ cách lấy mẫu đến ước lượng tham số: population/sample, thống kê mẫu, LLN, CLT, point estimation, MLE, Bayesian/MAP và regularization.

## Tổng thể, mẫu và lấy mẫu ngẫu nhiên

### Trực giác
Thống kê bắt đầu bằng việc tách **tổng thể (population)** ta muốn nghiên cứu khỏi **mẫu (sample)** ta có thể quan sát. Trong slides, Statistopia có 10,000 người; hỏi tất cả cho trung bình tổng thể, còn hỏi một nhóm con chỉ cho một ước lượng. Trong machine learning, mọi training set đều là một mẫu từ thế giới dữ liệu tương lai, nên chất lượng lấy mẫu ảnh hưởng trực tiếp đến khả năng khái quát.

### Định nghĩa và công thức
Tổng thể là toàn bộ nhóm cá thể/phần tử có hành vi chung mà ta quan tâm. Mẫu là tập con dùng để rút kết luận về tổng thể. Kích thước tổng thể thường ký hiệu $N$; kích thước mẫu là $n$. Mẫu ngẫu nhiên (random sample) cố tránh thiên lệch có hệ thống. Quan sát độc lập (independent) nghĩa là một giá trị được chọn không quyết định giá trị khác. Cùng phân phối (identically distributed) nghĩa là mỗi lần rút tuân theo cùng một phân phối.

Với dữ liệu i.i.d. $X_1,\dots,X_n$, mỗi $X_i$ độc lập và có cùng phân phối với $X$. Giả định này đứng sau luật số lớn, CLT, MLE và rất nhiều câu chuyện train/test trong ML.

### Ví dụ làm đầy đủ
Nếu bài toán hỏi “giá bơ bán ở Hoa Kỳ là bao nhiêu?”, tổng thể là tất cả quả bơ bán ở Hoa Kỳ. Nếu ta chỉ ghi giá bơ ở 4 cửa hàng đã chọn, các quả bơ ở 4 cửa hàng đó là mẫu. Một mẫu chỉ lấy từ cửa hàng cao cấp có thể ngẫu nhiên trong nhóm cửa hàng đó nhưng không đại diện cho tổng thể mục tiêu.

Một câu hỏi kiểm tra hữu ích là: “Nếu lặp lại quy trình lấy mẫu ngày mai, mỗi phần tử trong population mục tiêu có cơ hội hợp lý để xuất hiện không?” Nếu câu trả lời là không, sample vẫn có thể hữu ích, nhưng kết luận phải bị giới hạn. Ví dụ, model chỉ train bằng ảnh đường ban ngày không nên được đánh giá như thể nó đại diện đều cho đêm, mưa và tuyết.

Independence cũng là một giả định mô hình. Lấy mẫu không hoàn lại từ population rất nhỏ tạo phụ thuộc nhẹ, còn rút từ population rất lớn thường gần độc lập. Identically distributed nghĩa là cơ chế sinh dữ liệu ổn định: nếu nửa sample từ một quốc gia và nửa còn lại từ quốc gia có hành vi khác, dữ liệu gộp có thể vi phạm giả định này.

### NumPy snippet
```python
import numpy as np
rng = np.random.default_rng(7)
population = np.arange(151, 171, 2)
sample = rng.choice(population, size=4, replace=False)
print(sample, sample.mean(), population.mean())
```

> ⚠️ Lỗi thường gặp: xem training set là toàn bộ population; dùng convenience sample rồi gọi là random; hoặc quên rằng “independent” và “identically distributed” là hai điều kiện khác nhau.

Câu hỏi tự kiểm tra: population mục tiêu có đúng là nhóm mà model sẽ gặp khi triển khai không? Nếu không, estimate có thể rất chính xác cho sample nhưng sai cho quyết định ML thực tế.

### Tóm tắt
- Population = toàn bộ nhóm mục tiêu; sample = tập con quan sát được.
- Lấy mẫu tốt cần đại diện, độc lập và cùng phân phối.
- Dataset ML là mẫu, nên sampling bias có thể trở thành model bias.

## Trung bình mẫu, tỉ lệ mẫu và phương sai mẫu

### Trực giác
Vì thường không đo được mọi phần tử, ta tính các thống kê mẫu để ước lượng đại lượng tổng thể. Slides so sánh trung bình tổng thể $\mu$ với trung bình mẫu $\bar x$, tỉ lệ tổng thể $p$ với tỉ lệ mẫu $\hat p$, và phương sai tổng thể $\sigma^2$ với phương sai mẫu $s^2$.

### Định nghĩa và công thức
Trung bình mẫu:
$$\bar x=\frac1n\sum_{i=1}^n x_i.$$
Tỉ lệ mẫu cho một đặc tính nhị phân:
$$\hat p=\frac{x}{n},$$
trong đó $x$ là số phần tử trong mẫu có đặc tính đó. Phương sai mẫu dùng để ước lượng phương sai tổng thể:
$$s^2=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2.$$
Mẫu số $n-1$ là hiệu chỉnh Bessel: vì $\bar x$ đã được fit từ chính dữ liệu, các độ lệch mất một bậc tự do.

### Ví dụ làm đầy đủ
Với chiều cao Statistopia $151,153,\dots,169$, trung bình tổng thể là $160$. Mẫu $153,155,159,163,165,169$ có $\bar x=160.667$, còn mẫu $151,153,155,157,159,161$ có $\bar x=156$. Với dữ liệu $2,3,4,5,6$, $\bar x=4$, tổng bình phương độ lệch là $10$, nên phương sai mẫu là $10/(5-1)=2.5$.

Ba thống kê trả lời ba câu hỏi khác nhau. Mean ước lượng trung tâm của biến số, proportion ước lượng xác suất success của biến nhị phân, còn variance ước lượng độ phân tán. Cùng một dataset có thể có mean hợp lý nhưng variance gây hiểu nhầm nếu bỏ qua outlier hoặc subgroup.

Mẫu số $n-1$ không phải chi tiết trang trí. Sau khi tính $\bar x$, các deviation phải có tổng bằng 0, nên chỉ còn $n-1$ deviation tự do thay đổi. Chia cho $n$ thường underestimate population variance khi sample nhỏ; bảng trong slides minh họa điều này bằng cách lấy trung bình qua mọi sample size 2.

### NumPy snippet
```python
import numpy as np
x = np.array([2, 3, 4, 5, 6])
print(x.mean(), x.var(ddof=1), x.var(ddof=0))
```

> ⚠️ Lỗi thường gặp: chia phương sai mẫu cho $n$ khi đang ước lượng phương sai tổng thể; nhầm $\hat p$ với xác suất biết chính xác; hoặc so sánh các trung bình mẫu mà bỏ qua cỡ mẫu và cách lấy mẫu.

Câu hỏi tự kiểm tra: thống kê đang tính là parameter thật, estimator hay estimate cụ thể? Việc gọi đúng tên giúp tránh nhầm lẫn giữa công thức và giá trị quan sát được.

### Tóm tắt
- $\bar x$ ước lượng $\mu$; $\hat p$ ước lượng $p$.
- Phương sai mẫu để ước lượng chia cho $n-1$.
- Mỗi mẫu khác nhau tạo point estimate khác nhau.

## Luật số lớn

### Trực giác
Luật số lớn (law of large numbers) giải thích vì sao trung bình ổn định dần. Trong slides, gieo xúc xắc 4 mặt nhiều lần làm trung bình chạy tiến gần mean tổng thể $2.5$. Từng lần gieo vẫn nhiễu, nhưng trung bình của chúng đáng tin hơn.

### Định nghĩa và công thức
Với quan sát i.i.d. có mean hữu hạn $\mu=\mathbb E[X]$,
$$\bar X_n=\frac1n\sum_{i=1}^n X_i \to \mu$$
khi $n$ lớn. Slides nêu điều kiện thực hành: mẫu được rút ngẫu nhiên, cỡ mẫu đủ lớn, và quan sát độc lập.

### Ví dụ làm đầy đủ
Với xúc xắc 4 mặt công bằng có outcomes $1,2,3,4$, mean là $(1+2+3+4)/4=2.5$. Trung bình của 2 lần gieo có thể là $4$ hoặc $1$, nhưng sau hàng nghìn lần gieo, trung bình thường gần $2.5$.

LLN nói về việc lấy trung bình lặp lại, không phải sự chắc chắn tuyệt đối. Nếu true mean là $2.5$, sample average $2.7$ sau 100 lần gieo không mâu thuẫn; đó là dao động hữu hạn. Điều thay đổi khi $n$ tăng là các sai lệch lớn trở nên ít gặp hơn.

Trong ML, LLN giải thích vì sao metric trên training/validation có thể xấp xỉ metric population khi ví dụ đại diện và đủ nhiều. Nó cũng là cảnh báo: nhiều dữ liệu hơn nhưng từ sai distribution chỉ làm ổn định một câu trả lời sai.

### NumPy snippet
```python
import numpy as np
rng = np.random.default_rng(3)
rolls = rng.integers(1, 5, size=5000)
running = np.cumsum(rolls) / np.arange(1, len(rolls)+1)
print(running[-1])
```

> ⚠️ Lỗi thường gặp: nghĩ vài quan sát tiếp theo “phải bù” cho các quan sát trước (gambler's fallacy); đòi bằng đúng ở $n$ hữu hạn; hoặc áp dụng LLN cho mẫu phụ thuộc mạnh mà không kiểm tra giả định.

Câu hỏi tự kiểm tra: khi running mean chưa gần mean thật, đó có phải bằng chứng LLN sai không? Không; hãy nhìn xu hướng khi số quan sát tăng và nhớ rằng finite sample vẫn nhiễu.

### Tóm tắt
- LLN nói về hội tụ của trung bình mẫu.
- $n$ lớn hơn làm dao động ngẫu nhiên của trung bình nhỏ hơn.
- LLN không làm từng quan sát riêng lẻ bớt ngẫu nhiên.

## Định lý giới hạn trung tâm

### Trực giác
Định lý giới hạn trung tâm (central limit theorem, CLT) mô tả **hình dạng** của tổng hoặc trung bình. Slides có hai phiên bản: số lần ra heads khi tung nhiều đồng xu trở nên dạng chuông, và trung bình của các thời gian chờ Uniform$(0,15)$ gần normal dù từng thời gian chờ là uniform.

### Định nghĩa và công thức
Nếu $X_1,\dots,X_n$ i.i.d. với mean $\mu$ và variance $\sigma^2$, thì với $n$ lớn,
$$\bar X_n \approx \mathcal N\left(\mu,\frac{\sigma^2}{n}\right),$$
hoặc dạng chuẩn hóa:
$$\frac{\bar X_n-\mu}{\sigma/\sqrt n}\Rightarrow \mathcal N(0,1).$$
Với $X\sim\text{Binomial}(n,p)$, $\mu=np$ và $\sigma^2=np(1-p)$. Với $X\sim U(0,15)$, $\mu=7.5$ và $\sigma^2=18.75$, nên trung bình của $n$ lần chờ có variance $18.75/n$.

### Ví dụ làm đầy đủ
Với 10 lần tung đồng xu công bằng, số heads có mean $5$ và variance $2.5$, phân phối đã giống chuông hơn nhiều so với một lần tung. Với trung bình của 25 thời gian chờ uniform, $\mathbb E[\bar X]=7.5$ và $\operatorname{sd}(\bar X)=\sqrt{18.75/25}\approx0.866$.

CLT thêm xấp xỉ về hình dạng lên trên ý tưởng averaging. LLN nói mean tiến về $\mu$; CLT nói sai số đã scale $\bar X_n-\mu$ có hình dạng gần normal. Đây là lý do standard error và z-score xuất hiện khắp thống kê suy luận.

Với biến bị chặn như Uniform$(0,15)$, xấp xỉ thường tốt khá nhanh. Với biến lệch hoặc đuôi nặng, “đủ lớn” có thể lớn hơn nhiều. Luôn xác định bạn đang xấp xỉ count, sum hay average, vì mean và variance của chúng khác nhau bởi các hệ số $n$.

### NumPy snippet
```python
import numpy as np
rng = np.random.default_rng(4)
means = rng.uniform(0, 15, size=(4000, 25)).mean(axis=1)
print(means.mean(), means.std())
```

> ⚠️ Lỗi thường gặp: nói “dữ liệu trở thành normal” thay vì “sampling distribution của tổng/trung bình xấp xỉ normal”; quên standard error $\sigma/\sqrt n$; hoặc dùng CLT với $n$ rất nhỏ và đuôi cực nặng.

Câu hỏi tự kiểm tra: biểu đồ đang vẽ raw observations hay means của nhiều samples? CLT áp dụng cho phân phối của means/sums, nên nhãn trục và đơn vị variance rất quan trọng.

### Tóm tắt
- CLT nói về tổng/trung bình, không phải dữ liệu thô.
- Trung bình giữ mean $\mu$ và có variance $\sigma^2/n$.
- CLT biện minh cho xấp xỉ normal của nhiều estimator.

## Ước lượng điểm

### Trực giác
Ước lượng điểm (point estimate) nén dữ liệu mẫu thành một con số để xấp xỉ tham số tổng thể hoặc tham số mô hình chưa biết. Slides dùng $\bar x$ cho mean tổng thể, $s^2$ cho variance, $7/10$ cho xác suất heads của coin, và $(\beta_0,\beta_1)$ cho đường hồi quy.

### Định nghĩa và công thức
Tham số là đại lượng cố định nhưng chưa biết trong population/model, như $\mu$, $\sigma^2$, $p$, hoặc hệ số hồi quy. Statistic là đại lượng tính từ dữ liệu. Estimator là quy tắc; estimate là con số quan sát được. Ví dụ:
$$\hat\mu=\bar X,\qquad \hat p=\frac1n\sum_i X_i,\qquad s^2=\frac1{n-1}\sum_i(X_i-\bar X)^2.$$

### Ví dụ làm đầy đủ
Nếu 10 lần tung có 7 heads, point estimate cho $P(H)$ là $\hat p=0.7$. Nếu chiều cao là 10 giá trị trong slides, mean Gaussian ước lượng là trung bình của chúng, $68.442$.

Point estimate cố ý chưa đầy đủ: nó là best single guess theo một quy tắc đã chọn. Hai estimator có thể cùng target một parameter nhưng hành xử khác nhau. Một estimator có thể unbiased nhưng noisy; estimator khác hơi biased nhưng variance thấp hơn. Tradeoff này xuất hiện lại trong regularization.

Khi báo cáo point estimate, hãy kèm câu chuyện lấy mẫu. Con số $0.43$ từ survey không chỉ là thuộc tính của user; nó còn phụ thuộc thiết kế survey, non-response và sample size.

### NumPy snippet
```python
import numpy as np
x = np.array([66.75,70.24,67.19,67.09,63.65,64.64,69.81,69.79,73.52,71.74])
print(x.mean(), x.var(ddof=1), x.var(ddof=0))
```

> ⚠️ Lỗi thường gặp: xem một point estimate là tham số thật; không báo uncertainty; hoặc nhầm estimator (quy tắc ngẫu nhiên) với estimate (giá trị đã quan sát).

Câu hỏi tự kiểm tra: point estimate nào được chọn vì unbiased, vì MLE, hay vì dễ giải thích? Mỗi tiêu chí có thể dẫn tới estimator khác nhau trong cùng một bài toán.

### Tóm tắt
- Point estimate là tóm tắt một số cho tham số chưa biết.
- Nó thay đổi từ mẫu này sang mẫu khác.
- Estimator tốt được xét qua bias, variance, consistency và mục tiêu mô hình.

## MLE cho Bernoulli

### Trực giác
Maximum likelihood estimation hỏi: giá trị tham số nào làm dữ liệu quan sát được có khả năng xuất hiện cao nhất? Trong slides, popcorn trên sàn hợp lý hơn sau buổi xem phim so với ngủ trưa; coin có $p=0.7$ giải thích 8 heads 2 tails tốt hơn coin có $p=0.3$.

### Định nghĩa và công thức
Với dữ liệu Bernoulli i.i.d. $x_i\in\{0,1\}$,
$$L(p;x)=\prod_{i=1}^n p^{x_i}(1-p)^{1-x_i}=p^{k}(1-p)^{n-k},$$
trong đó $k=\sum_i x_i$. Log-likelihood:
$$\ell(p)=k\log p+(n-k)\log(1-p).$$
Cho đạo hàm bằng 0 thu được
$$\hat p_{\text{MLE}}=\frac{k}{n}=\bar x.$$

### Ví dụ làm đầy đủ
Với 8 heads và 2 tails, $L(p)=p^8(1-p)^2$. Các likelihood ứng viên: $0.7^8 0.3^2\approx0.00519$, $0.5^{10}\approx0.00098$, và $0.3^8 0.7^2\approx0.000032$. Cực đại liên tục là $\hat p=8/10=0.8$.

Likelihood curve không cần tích phân bằng 1 theo $p$. Nó là hàm chấm điểm các parameter ứng viên sau khi data đã cố định. Vì vậy nhân likelihood với một hằng số dương không đổi MLE.

Log-likelihood có cùng maximizer vì log là hàm tăng. Với Bernoulli, log biến $p^k(1-p)^{n-k}$ thành tổng hai hạng, làm đạo hàm rõ ràng. Ở biên, nếu tất cả quan sát là heads thì MLE là $p=1$; nếu tất cả là tails thì MLE là $p=0$.

### NumPy snippet
```python
import numpy as np
x = np.array([1,1,1,1,1,1,1,1,0,0])
p_hat = x.mean()
loglik = x.sum()*np.log(p_hat) + (len(x)-x.sum())*np.log(1-p_hat)
print(p_hat, loglik)
```

> ⚠️ Lỗi thường gặp: tối đa hóa $P(p\mid\text{data})$ nhưng gọi là MLE; nhầm likelihood theo $p$ với xác suất của tham số; hoặc nhân nhiều xác suất nhỏ mà không dùng log.

Câu hỏi tự kiểm tra: trong MLE, data đã cố định hay parameter đã cố định? Khi tối ưu likelihood, data cố định và ta quét qua các giá trị parameter ứng viên.

### Tóm tắt
- Likelihood là $P(\text{data}\mid\text{parameter})$ xem như hàm của parameter.
- Bernoulli MLE là sample proportion.
- Log-likelihood dễ tối ưu và an toàn số học hơn.

## MLE cho Gaussian và hồi quy tuyến tính

### Trực giác
Với dữ liệu Gaussian, MLE chọn đường cong đặt mật độ cao gần các điểm quan sát. Slides so sánh Gaussian khác mean và variance, rồi nối hồi quy tuyến tính với likelihood: nếu residual dọc là Gaussian, maximizing likelihood tương đương minimizing squared error.

### Định nghĩa và công thức
Với $x_i\sim\mathcal N(\mu,\sigma^2)$,
$$L(\mu,\sigma)=\prod_i \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right).$$
Khi $\sigma$ cố định, MLE cho mean là $\hat\mu=\bar x$. Nếu variance cũng được ước lượng bằng MLE,
$$\hat\sigma^2_{\text{MLE}}=\frac1n\sum_i(x_i-\bar x)^2.$$
Với hồi quy tuyến tính $y_i=mx_i+b+\epsilon_i$ và $\epsilon_i\sim\mathcal N(0,\sigma^2)$, MLE tối thiểu hóa $\sum_i (y_i-(mx_i+b))^2$.

### Ví dụ làm đầy đủ
Chiều cao trong slides có mean $68.442$. Theo quy ước Gaussian MLE, variance chia cho $10$, không phải $9$. Với residual hồi quy $d_1,\dots,d_5$, likelihood chứa $\exp[-\frac12\sum d_i^2]$, nên maximizing likelihood tương đương minimizing least squares.

Công thức variance Gaussian là nơi ngữ cảnh rất quan trọng. MLE chia cho $n$ vì nó tối đa hóa Gaussian likelihood. Sample variance không chệch chia cho $n-1$ vì nó nhắm tới population variance mà không underestimate có hệ thống. Cả hai công thức đều hữu ích; chúng trả lời hai câu hỏi khác nhau.

Với regression, câu chuyện xác suất là: đường thẳng cho expected $y$, còn Gaussian noise làm điểm lệch dọc quanh đường. Đường có squared residual nhỏ hơn gán mật độ lớn hơn cho các điểm quan sát, nên maximum likelihood chọn least-squares line.

### NumPy snippet
```python
import numpy as np
x = np.array([66.75,70.24,67.19,67.09,63.65,64.64,69.81,69.79,73.52,71.74])
mu = x.mean()
sigma2_mle = np.mean((x-mu)**2)
print(mu, sigma2_mle)
```

> ⚠️ Lỗi thường gặp: dùng $n-1$ trong Gaussian MLE variance mà không nhận ra objective đã khác; quên residual trong regression chuẩn là khoảng cách dọc; hoặc so likelihood thô bị underflow thay vì log-likelihood.

Câu hỏi tự kiểm tra: variance đang dùng để tối đa hóa likelihood hay để ước lượng không chệch population variance? Hai mục tiêu này giải thích sự khác nhau giữa chia $n$ và $n-1$.

### Tóm tắt
- Gaussian MLE mean là sample mean.
- Gaussian MLE variance chia cho $n$.
- Least squares là MLE dưới giả định nhiễu Gaussian.

## Bayes, MAP và regularization

### Trực giác
Bayesian statistics xem tham số chưa biết như đại lượng bất định với niềm tin tiên nghiệm (prior). Slides đối chiếu frequentist (xác suất là tần suất dài hạn, MLE là một điểm) với Bayesian (xác suất là mức độ tin tưởng, cập nhật bằng dữ liệu). MAP tối đa hóa posterior. Regularization xuất hiện khi prior phạt tham số mô hình quá phức tạp.

### Định nghĩa và công thức
Bayes cho tham số:
$$p(\theta\mid x)=\frac{p(x\mid\theta)p(\theta)}{p(x)}.$$
Posterior tỉ lệ với likelihood nhân prior:
$$p(\theta\mid x)\propto p(x\mid\theta)p(\theta).$$
MAP chọn
$$\hat\theta_{\text{MAP}}=\arg\max_\theta p(x\mid\theta)p(\theta).$$
Với tham số Bernoulli có prior Beta$(\alpha,\beta)$ và $k$ heads trong $n$ tosses, posterior là Beta$(\alpha+k,\beta+n-k)$ và MAP nội điểm là
$$\frac{\alpha+k-1}{\alpha+\beta+n-2}.$$
Nếu hệ số có Gaussian prior tâm 0, negative log posterior thêm một penalty $L_2$, chính là regularization.

### Ví dụ làm đầy đủ
Với prior Uniform$(0,1)$ cho coin bias, MAP bằng MLE cho 8 heads và 2 tails, tức $0.8$. Với prior Beta$(2,2)$, posterior là Beta$(10,4)$ và MAP là $(10-1)/(10+4-2)=0.75$, bị kéo nhẹ về coin công bằng. Trong polynomial regression, mô hình bậc cao có thể fit dữ liệu tốt nhưng bị prior phạt nếu hệ số lớn.

Bayesian updating tách evidence khỏi prior belief. Likelihood nói data thích điều gì; prior nói điều gì hợp lý trước khi thấy data. Khi dữ liệu ít, prior có thể ảnh hưởng mạnh. Khi dữ liệu nhiều, một prior hợp lý thường bị likelihood áp đảo.

Regularization là MAP dưới dạng optimization. Gaussian prior tâm 0 nói rằng coefficient rất lớn là ít khả dĩ trước khi thấy data. Lấy negative log biến prior đó thành penalty $L_2$, nên model chỉ được phức tạp hơn nếu cải thiện data fit đủ nhiều.

### NumPy snippet
```python
alpha, beta, k, n = 2, 2, 8, 10
map_est = (alpha+k-1)/(alpha+beta+n-2)
print(map_est)
```

> ⚠️ Lỗi thường gặp: bỏ normalizing constant khi cần xác suất thật chứ không chỉ argmax; nghĩ MAP luôn bằng MLE; hoặc xem regularization là mẹo tùy tiện thay vì prior ưu tiên tham số đơn giản.

Câu hỏi tự kiểm tra: prior đang thêm thông tin thật hay chỉ là cách phạt complexity? Với MAP, cả hai đều đi vào cùng công thức posterior và ảnh hưởng estimate.

### Tóm tắt
- Posterior $\propto$ likelihood $\times$ prior.
- MAP là point estimate từ posterior.
- Prior không thông tin có thể làm MAP trùng MLE; prior có thông tin kéo estimate.
- $L_2$ regularization tương ứng Gaussian prior trên coefficients.
