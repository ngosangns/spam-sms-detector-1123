#import "template.typ": *

#h1("Tổng quan về công nghệ học máy")

#h2("Giới thiệu")

Học máy (Machine Learning) là một lĩnh vực con của trí tuệ nhân tạo (AI) tập trung vào việc phát triển các thuật toán cho phép máy tính học từ dữ liệu mà không cần được lập trình cụ thể cho từng nhiệm vụ. Học máy cho phép hệ thống tự động cải thiện hiệu suất khi có thêm dữ liệu, từ đó giúp đưa ra dự đoán hoặc quyết định một cách chính xác.

Vòng đời học máy là một quy trình tuần hoàn nhằm xây dựng một dự án học máy hiệu quả. Mục đích chính của vòng đời này là tìm ra giải pháp cho vấn đề hoặc dự án:

1. Thu thập Dữ liệu: Thu thập và xác định các vấn đề liên quan đến dữ liệu.
2. Chuẩn bị Dữ liệu: Sắp xếp dữ liệu để sử dụng trong huấn luyện.
3. Dọn dẹp Dữ liệu: Làm sạch và chuyển đổi dữ liệu thành định dạng có thể sử dụng.
4. Phân tích Dữ liệu: Sử dụng các kỹ thuật phân tích để xây dựng mô hình.
5. Huấn luyện Mô hình: Cải thiện hiệu suất của mô hình.
6. Kiểm tra Mô hình: Đánh giá độ chính xác của mô hình bằng cách sử dụng tập dữ liệu kiểm tra.
7. Triển khai: Đưa mô hình vào hệ thống thực tế.

#h2("Phân loại")

Học máy được chia thành các loại chính sau:

#h3("Học có giám sát (Supervised Learning)")

Học giám sát là một phương pháp học máy trong đó mô hình được huấn luyện dựa trên một tập dữ liệu đã được gán nhãn. Mỗi mẫu trong tập dữ liệu bao gồm đầu vào và đầu ra tương ứng, cho phép mô hình học cách dự đoán đầu ra từ các đầu vào mới.

Quá trình học giám sát thường được chia thành các bước chính:

- Chuẩn bị dữ liệu: Tập hợp và làm sạch dữ liệu, bao gồm gán nhãn cho các mẫu dữ liệu.
- Phân chia dữ liệu: Tách dữ liệu thành tập huấn luyện và tập kiểm tra để đánh giá hiệu suất mô hình.
- Lựa chọn mô hình: Chọn một thuật toán học máy phù hợp, như cây quyết định, hồi quy logistic, hay mạng nơ-ron.
- Huấn luyện mô hình: Sử dụng tập huấn luyện để điều chỉnh trọng số của mô hình nhằm tối ưu hóa khả năng dự đoán.
- Đánh giá mô hình: Sử dụng tập kiểm tra để đo lường độ chính xác và khả năng tổng quát của mô hình.

Các thuật toán học giám sát phổ biến:

- Cây quyết định: Thuật toán đơn giản nhưng hiệu quả trong việc phân loại và hồi quy.
- Hồi quy logistic: Thích hợp cho bài toán phân loại nhị phân.
- Mạng nơ-ron: Mô hình mạnh mẽ cho cả phân loại và hồi quy, đặc biệt là trong các bài toán phức tạp như nhận diện hình ảnh và ngôn ngữ.

Ưu điểm:

- Độ chính xác cao khi có dữ liệu chất lượng.
- Dễ dàng giải thích và áp dụng cho nhiều lĩnh vực khác nhau.

Nhược điểm:

- Cần một lượng lớn dữ liệu đã được gán nhãn, tốn thời gian và công sức.
- Khả năng tổng quát có thể bị hạn chế nếu mô hình không được huấn luyện đầy đủ.

Học giám sát là phương pháp chủ yếu trong việc phát hiện tin nhắn lừa đảo. Trong phương pháp này, các mô hình được huấn luyện trên tập dữ liệu có gán nhãn, cho phép chúng học từ các ví dụ thực tế. Thuật toán có thể sử dụng:

- Naive Bayes: Một thuật toán đơn giản nhưng hiệu quả, Naive Bayes dựa trên lý thuyết xác suất để phân loại tin nhắn. Nó thích hợp cho các bài toán phân loại văn bản nhờ vào khả năng xử lý các đặc trưng từ văn bản một cách nhanh chóng.
- Support Vector Machines (SVM): SVM tìm kiếm siêu phẳng tốt nhất để phân loại dữ liệu, có hiệu quả cao trong không gian đặc trưng cao và giúp tách biệt các tin nhắn lừa đảo khỏi các tin nhắn hợp lệ.
- Decision Trees: Là một phương pháp trực quan, Decision Trees xây dựng một cây phân quyết để phân loại các tin nhắn dựa trên các đặc trưng của chúng. Mặc dù dễ hiểu, nó cũng có thể gặp phải vấn đề quá khớp.
- Random Forest: Là phiên bản nâng cao của Decision Trees, Random Forest sử dụng nhiều cây quyết định để cải thiện độ chính xác và giảm thiểu hiện tượng quá khớp.

#h3("Học không giám sát (Unsupervised Learning)")

Học không giám sát là một phương pháp học máy trong đó mô hình được huấn luyện trên một tập dữ liệu không có nhãn. Mục tiêu của phương pháp này là tìm ra cấu trúc hoặc mẫu trong dữ liệu mà không cần bất kỳ hướng dẫn nào từ nhãn.

Quá trình học không giám sát có thể được chia thành các bước chính:

- Chuẩn bị dữ liệu: Tập hợp và làm sạch dữ liệu, đảm bảo tính đầy đủ và nhất quán.
- Phân tích dữ liệu: Sử dụng các kỹ thuật như phân tích thành phần chính (PCA) để giảm số chiều dữ liệu và tìm ra các đặc trưng quan trọng.
- Chọn thuật toán: Lựa chọn một thuật toán phù hợp như clustering hoặc các kỹ thuật học sâu.
- Huấn luyện mô hình: Áp dụng thuật toán để nhóm hoặc phân loại dữ liệu mà không cần nhãn.
- Đánh giá kết quả: Sử dụng các chỉ số như silhouette score hoặc Davies-Bouldin index để đánh giá chất lượng của các nhóm được tạo ra.

Các thuật toán học không giám sát phổ biến:

- K-means: Một thuật toán phân cụm đơn giản nhưng hiệu quả, sử dụng khoảng cách Euclid để xác định độ tương đồng giữa các điểm dữ liệu.
- DBSCAN: Được sử dụng để phát hiện các cụm có hình dạng phức tạp và có khả năng phát hiện nhiễu (noise).
- Phân cụm theo chiều sâu: Cho phép tạo ra các cấu trúc phân cụm ở nhiều cấp độ khác nhau.

Ưu điểm:

- Không cần dữ liệu đã gán nhãn, tiết kiệm thời gian và công sức trong việc chuẩn bị dữ liệu.
- Có khả năng phát hiện các mẫu và cấu trúc ẩn mà không bị giới hạn bởi nhãn.

Nhược điểm:

- Khó khăn trong việc đánh giá chất lượng của các kết quả, vì không có nhãn để đối chiếu.
- Kết quả có thể nhạy cảm với lựa chọn tham số và thuật toán, đôi khi cần thử nghiệm nhiều phương pháp để đạt được kết quả tốt nhất.

#h3("Học bán giám sát (Semi-Supervised Learning)")

Kết hợp giữa học có giám sát và học không giám sát, sử dụng cả dữ liệu có nhãn và không có nhãn để cải thiện hiệu suất mô hình.

#h3("Học tăng cường (Reinforcement Learning)")

- Một tác nhân học cách hành động trong một môi trường thông qua việc nhận phần thưởng hoặc phạt từ các hành động của nó.
- Ứng dụng: Chơi game, robot tự động.
- Thuật toán phổ biến: Q-learning, Deep Q-Networks (DQN), Policy Gradients.

#h4("Học sâu (Deep Learning)")

Học sâu là một nhánh của học máy, trong đó sử dụng các mạng nơ-ron nhân tạo để mô hình hóa và giải quyết các bài toán phức tạp. Khác với các phương pháp học máy truyền thống, học sâu có khả năng tự động trích xuất các đặc trưng từ dữ liệu mà không cần sự can thiệp của con người, từ đó cải thiện độ chính xác trong các nhiệm vụ phân loại, nhận dạng và dự đoán.

Mạng nơ-ron là thành phần cơ bản trong học sâu. Cấu trúc mạng nơ-ron bao gồm:

- Đầu vào (Input Layer): Nhận dữ liệu đầu vào từ bộ dữ liệu.
- Các lớp ẩn (Hidden Layers): Thực hiện các phép toán để trích xuất và chuyển đổi đặc trưng từ dữ liệu. Số lượng lớp và số lượng nơ-ron trong mỗi lớp có thể thay đổi tùy thuộc vào bài toán.
- Đầu ra (Output Layer): Cung cấp kết quả cuối cùng, như phân loại tin nhắn là lừa đảo hoặc không lừa đảo.

Học sâu bao gồm nhiều loại mạng nơ-ron, mỗi loại phù hợp với những bài toán khác nhau:

- Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN): Thường được sử dụng trong nhận diện hình ảnh và phân loại, CNN có khả năng tự động phát hiện và học các đặc trưng từ dữ liệu đầu vào thông qua các phép toán chập.
- Mạng nơ-ron hồi tiếp (Recurrent Neural Networks - RNN): Phù hợp cho các dữ liệu có tính tuần tự, như chuỗi thời gian và văn bản. RNN có khả năng ghi nhớ thông tin từ các bước trước đó, giúp xử lý các bài toán như dịch máy và phân tích ngữ nghĩa.
- Mạng nơ-ron dài ngắn hạn (Long Short-Term Memory - LSTM): Là một dạng cải tiến của RNN, LSTM giúp giải quyết vấn đề mất mát thông tin trong các chuỗi dài. Nó cho phép mạng ghi nhớ thông tin trong khoảng thời gian dài hơn, rất hữu ích trong các bài toán xử lý ngôn ngữ tự nhiên.

Thuật toán học sâu phổ biến

- Xử lý ngôn ngữ tự nhiên (NLP): Các mô hình như BERT (Bidirectional Encoder Representations from Transformers) đã cách mạng hóa cách thức xử lý ngôn ngữ tự nhiên, cho phép phát hiện các mối quan hệ phức tạp trong dữ liệu văn bản.

Ưu điểm:

- Học sâu có khả năng xử lý và học từ dữ liệu lớn, giúp phát hiện mẫu lừa đảo một cách hiệu quả.
- Tự động trích xuất đặc trưng, giảm thiểu sự cần thiết phải can thiệp thủ công từ con người.

Nhược điểm:

- Cần một lượng lớn dữ liệu để huấn luyện và có thể tốn kém về tài nguyên tính toán.
- Mô hình phức tạp có thể khó giải thích, dẫn đến việc khó xác định lý do vì sao một tin nhắn cụ thể được phân loại là lừa đảo.

Gần đây, các mô hình học sâu đã trở thành công cụ mạnh mẽ trong phát hiện tin nhắn lừa đảo nhờ khả năng học từ dữ liệu lớn và tự động trích xuất các đặc trưng phức tạp.

- Recurrent Neural Networks (RNN): RNN được thiết kế để xử lý dữ liệu chuỗi, giúp nhận diện các mẫu trong văn bản theo ngữ cảnh. Tuy nhiên, RNN có thể gặp khó khăn với các chuỗi dài do vấn đề gradient biến mất.
- Long Short-Term Memory (LSTM): Là một biến thể của RNN, LSTM khắc phục vấn đề gradient biến mất, cho phép mô hình ghi nhớ thông tin lâu hơn và xử lý hiệu quả các chuỗi dài trong văn bản.
- Convolutional Neural Networks (CNN): Mặc dù thường được sử dụng trong xử lý hình ảnh, CNN cũng cho thấy hiệu quả trong việc phân tích văn bản nhờ khả năng phát hiện các đặc trưng cục bộ.
- BERT (Bidirectional Encoder Representations from Transformers): Là một trong những mô hình tiên tiến nhất trong NLP, BERT sử dụng cơ chế attention để học ngữ cảnh từ cả hai chiều. Mô hình này đã đạt được độ chính xác cao trong nhiều bài toán phân loại văn bản, bao gồm cả phát hiện tin nhắn lừa đảo.

#h2("Nguyên lý thuật toán học máy")

#h3("Mô hình Naive Bayes")

Mô hình phân lớp Naive Bayes là một phương pháp phân loại xác suất dựa trên Định lý Bayes và giả định tính độc lập của các thuộc tính. Đây là một phương pháp đơn giản nhưng mạnh mẽ, thường được sử dụng trong các bài toán phân loại văn bản như phân loại email spam, phân tích cảm xúc và phát hiện tin nhắn lừa đảo.

Định lý Bayes cung cấp một cách tính toán xác suất của một sự kiện dựa trên kiến thức có sẵn về các sự kiện liên quan, được biểu diễn bằng công thức, trong đó:

- P(y|X) gọi là posterior probability: xác suất của mục tiêu y với điều kiện có đặc trưng X
- P(X|y) gọi là likelihood: xác suất của đặc trưng X khi đã biết mục tiêu y
- P(y) gọi là prior probability của mục tiêu y
- P(X) gọi là prior probability của đặc trưng X

Trong mô hình Naive Bayes, có hai giả thiết được đặt ra:

- Các đặc trưng đưa vào mô hình là độc lập với nhau. Tức là sự thay đổi giá trị của một đặc trưng không ảnh hưởng đến các đặc trưng còn lại.
- Các đặc trưng đưa vào mô hình có ảnh hưởng ngang nhau đối với đầu ra mục tiêu.

Cả hai giả thiết gần như không tồn tại trong thực tế trên, mô hình này mới được gọi là naive (ngây thơ). Tuy nhiên, chính sự đơn giản của nó với việc dự đoán rất nhanh kết quả đầu ra khiến nó được sử dụng rất nhiều trong thực tế trên những bộ dữ liệu lớn, đem lại kết quả khả quan.

Một số kiểu mô hình Naive Bayes:

- Multinomial Naive Bayes: Mô hình này chủ yếu được sử dụng trong phân loại văn bản. Đặc trưng đầu vào ở đây chính là tần suất xuất hiện của từ trong văn bản đó.
- Bernoulli Naive Bayes: Mô hình này được sử dụng khi các đặc trưng đầu vào chỉ nhận giá trị nhị phân 0 hoặc 1 (phân bố Bernoulli).
- Gaussian Naive Bayes: Khi các đặc trưng nhận giá trị liên tục, ta giả sử các đặc trưng đó có phân phối Gaussian. Khi đó, likelihood sẽ có dạng:

Mô hình Naive Bayes là mô hình phân lớp đơn giản dễ cài đặt, có tốc độ xử lý nhanh. Tuy nhiên có nhược điểm lớn là yêu cầu các đặc trưng đầu vào phải độc lập, mà điều này khó xảy ra trong thực tế làm giảm chất lượng của mô hình.

#h3("Mô hình Random Forest")

Thuật toán Random Forest là một trong những phương pháp học máy mạnh mẽ và phổ biến, đặc biệt trong các bài toán phân loại và hồi quy. Được xây dựng dựa trên ý tưởng của thuật toán "tập hợp" (ensemble learning), Random Forest phát huy hiệu quả bằng cách kết hợp nhiều cây quyết định (Decision Trees) nhằm cải thiện độ chính xác và giảm thiểu rủi ro của việc quá khớp (overfitting).

Thuật toán hoạt động tuần tự theo bốn bước:

Bước 1: Chọn các mẫu ngẫu nhiên từ tập dữ liệu đã cho.
Bước 2: Thiết lập cây quyết định cho từng mẫu và nhận kết quả dự đoán từ mỗi quyết định cây.
Bước 3: Hãy bỏ phiếu cho mỗi kết quả dự đoán.
Bước 4: Chọn kết quả được dự đoán nhiều nhất là dự đoán cuối cùng.

#img("image copy.png", cap: "Mô hình Random Forest", width: 80%)

Random Forests được xem như một phương pháp mạnh mẽ và có độ chính xác cao nhờ sự tham gia của nhiều cây quyết định trong quá trình phân tích.

#h3("Mô hình Cây Quyết Định")

Thuật toán Decision Tree (Cây Quyết Định) là một trong những phương pháp học máy đơn giản nhưng rất hiệu quả, thường được sử dụng trong các bài toán phân loại và hồi quy. Cấu trúc của mô hình này giống như một cây phân cấp, nơi mỗi nút biểu thị một đặc tính của dữ liệu, mỗi nhánh biểu thị kết quả của một phép thử, và mỗi lá biểu thị một kết quả hoặc quyết định cuối cùng.

Quy trình hoạt động của thuật toán Decision Tree có thể được chia thành các bước chính như sau:

Bước 1: Chọn Thuộc Tính Tốt Nhất: Ở mỗi nút, mô hình chọn đặc tính tốt nhất để chia dữ liệu. Tiêu chí chọn thường dựa trên mức độ giảm độ bất định (impurity) sau khi chia dữ liệu. Hai tiêu chí phổ biến là:

- Information Gain (tăng thông tin), sử dụng trong cây phân loại dựa trên lý thuyết thông tin (entropy).
- Gini Index, sử dụng trong thuật toán CART (Classification and Regression Tree).

Bước 2: Chia Dữ Liệu: Mỗi nút chia dữ liệu thành hai hoặc nhiều nhánh con dựa trên đặc tính được chọn.

Bước 3: Lặp Lại Quy Trình: Quá trình lặp lại này tiếp tục cho đến khi một điều kiện dừng được thỏa mãn, chẳng hạn như:

- Tất cả các điểm dữ liệu trong một nút thuộc cùng một lớp.
- Không còn thuộc tính nào để chia hoặc độ sâu tối đa của cây đã đạt.

Bước 4: Tạo Quyết Định: Các lá của cây là những dự đoán cuối cùng, giúp phân loại dữ liệu đầu vào mới.

#h3("Mô hình Support Vector Machine")

Support Vector Machine (SVM) là một thuật toán học máy được sử dụng chủ yếu cho các bài toán phân loại và đôi khi cho các bài toán hồi quy.

#img("image copy 2.png", cap: "Mô hình Support Vector Machine trong không gian hai chiều và ba chiều")

SVM giải bài toán tối ưu hóa để tìm siêu phẳng có dạng:

$w dot x - b = 0$

trong đó:

- w là vector trọng số (định hướng của siêu phẳng).
- b là bias (độ lệch của siêu phẳng). Bài toán tối ưu hóa được xây dựng sao cho:

Dữ liệu thuộc lớp +1 thỏa mãn: $w dot x_i - b >= 1$

Dữ liệu thuộc lớp -1 thỏa mãn: $w dot x_i - b <= -1$

Mục tiêu là tối thiểu hóa $\| w \|^2$, tức là làm cho siêu phẳng càng "mỏng" và margin càng lớn càng tốt.

Mục tiêu chính của SVM là tìm một siêu phẳng (hyperplane) tối ưu nhất để phân chia các lớp dữ liệu khác nhau trong không gian nhiều chiều. Các siêu phẳng này giúp tối đa hóa khoảng cách biên (maximal margin) giữa các lớp dữ liệu, tạo nên một mô hình mạnh mẽ và hiệu quả cho việc phân loại.

#img("image copy 3.png", cap: "Siêu phẳng tối ưu có lề cực đại", width: 80%)

#img("image copy 4.png", cap: "Siêu mặt phẳng cực đại trong không gian 3D")

Một điểm trong không gian véc tơ có thể được coi là một véc tơ từ gốc tọa độ tới điểm đó. Các điểm dữ liệu nằm trên hoặc gần nhất với siêu phẳng được gọi là véc tơ hỗ trợ, chúng ảnh hưởng đến vị trí và hướng của siêu phẳng. Các véc tơ này được sử dụng để tối ưu hóa lề và nếu xóa các điểm này, vị trí của siêu phẳng sẽ thay đổi. Một điểm lưu ý nữa đó là các véc tơ hỗ trợ phải cách đều siêu phẳng.

#img("image copy 5.png", cap: "Siêu mặt phẳng cực đại trong không gian 2D")

Khi dữ liệu không thể phân tách bằng một siêu phẳng duy nhất (ví dụ, trong trường hợp không tuyến tính), SVM sử dụng hai kỹ thuật:

Một là lề mềm (Soft margin): Thuật toán này cho phép SVM mắc một số lỗi nhất định và giữ cho lề càng rộng càng tốt để các điểm khác vẫn có thể được phân loại chính xác. Có hai kiểu phân loại sai có thể xảy ra:

- Dữ liệu nằm ở đúng bên nhưng phạm vào lề
- Dữ liệu nằm ở sai bên

Mức độ chấp nhận lỗi càng lớn có nghĩa là SVM càng bị phạt nặng khi thực hiện phân loại sai. Do đó, lề càng hẹp và càng ít vectơ hỗ trợ được sử dụng. Khi lập trình với sklearn, mức độ chấp nhận lỗi được coi như một tham số phạt (C).

#img("image copy 6.png", cap: "Hình thể hiện SVM với các giá trị C khác nhau")

Hai gọi là kernel trick. Kỹ thuật này cho phép chuyển đổi không gian dữ liệu về một không gian cao hơn, nơi mà dữ liệu có thể được phân tách một cách hiệu quả. Có nhiều loại hàm kernel, như linear, polynomial và radial basis function (RBF), mỗi loại có những ưu điểm riêng tùy thuộc vào tính chất của dữ liệu.

#h3("Mô hình Logistic Regression")

Logistic Regression là một mô hình phân loại phổ biến trong học máy, thường được sử dụng để dự đoán khả năng một đối tượng thuộc vào một trong hai nhóm hoặc lớp. Mặc dù có tên gọi “hồi quy”, Logistic Regression chủ yếu được sử dụng cho các bài toán phân loại.

Logistic Regression sử dụng một hàm sigmoid (hay logistic) để biến đổi đầu ra tuyến tính của mô hình thành một xác suất:

$p(y=1|x) = sigma(w^T dot x) = frac(1, 1 + exp(-w^T x))$

trong đó:

- w là vector trọng số.
- x là vector đặc trưng của mẫu.
- $sigma(z)$ là hàm sigmoid, chuyển giá trị z thành một xác suất từ 0 đến 1.

#h3("Hàm mất mát")

Hàm mất mát trong Logistic Regression là hàm log-loss, được định nghĩa như sau:

$L(w) = -frac(1, N) sum_(i=1)^N [y_i log p(y_i|x_i)+(1-y_i) log (1-p(y_i|x_i))]$

trong đó:

- N là số lượng mẫu trong tập huấn luyện.
- $y_i$ là nhãn thực tế của mẫu thứ i (có giá trị 0 hoặc 1).
- $p(y_i|x_i)$ là xác suất dự đoán mà mô hình ước lượng.

Để tối ưu hóa hàm mất mát L(w), ta sử dụng thuật toán Gradient Descent. Gradient Descent cập nhật trọng số w theo hướng ngược lại của gradient của hàm mất mát:

$w arrow.l w - eta nabla L(w)w arrow.l w - eta nabla L(w)$

trong đó:

- $eta$ là tốc độ học (learning rate).
- $nabla L(w)$ là gradient của hàm mất mát $L(w)$.

Gradient $eta nabla L(w)$ được tính theo công thức:

$eta L(w) = -frac(1, N) sum_(i=1)^N (y_i - sigma(w^T x_i)) x_i$

Khi áp dụng công thức này, ta có thể cập nhật w như sau:

$w arrow.l w + eta frac(1, N) sum_(i=1)^N (y_i - sigma(w^T x_i)) x_i$

Với $y_i - sigma(w^T x_i)$ là sự chênh lệch giữa nhãn thực tế $y_i$ và xác suất dự đoán $sigma(w^T x_i​)$. Giá trị này cho biết hướng và mức độ điều chỉnh cần thiết cho trọng số $w$.

Bằng cách điều chỉnh $w$ theo gradient, mô hình dần dần học cách tối thiểu hóa hàm mất mát, dẫn đến việc cải thiện dự đoán.

#h3("Mạng nơ-ron hồi quy")

Mạng nơ-ron hồi quy (Recurrent Neural Network, RNN) là một loại mạng nơ-ron chuyên biệt, được thiết kế để xử lý các dữ liệu dạng chuỗi, như chuỗi thời gian, ngôn ngữ tự nhiên, hoặc bất kỳ dạng dữ liệu nào có thứ tự liên tiếp. Điểm khác biệt chính của RNN so với các mạng nơ-ron truyền thống là khả năng ghi nhớ thông tin từ các bước thời gian trước đó, giúp mô hình hóa các mối quan hệ và phụ thuộc theo thời gian.

RNN có một chuỗi tính toán lặp lại qua từng bước thời gian, khiến mỗi đầu vào và đầu ra đều phụ thuộc vào ngữ cảnh của các bước trước. Các phép tính lặp này cho phép RNN ghi nhớ các thông tin ngắn hạn trong chuỗi. Tuy nhiên, RNN gặp vấn đề với "quên dần" khi phải ghi nhớ các chuỗi dài hơn do vấn đề lan truyền ngược qua thời gian, dẫn đến hiện tượng mất mát hoặc quá lớn của gradient.

RNN xử lý mỗi đầu vào theo trình tự, và trạng thái ẩn $h_t$ ở mỗi bước thời gian $t$ phụ thuộc vào đầu vào $x_t$ và trạng thái ẩn của bước trước đó $h_(t-1)$. Công thức tính trạng thái ẩn được biểu diễn như sau:

$h_t = f(W_h dot h_(t-1) + W_x dot x_t + b)$

Trong đó:

- $h_t$: Trạng thái ẩn tại thời điểm t.
- $x_t$: Đầu vào hiện tại tại bước t.
- $W_h$​: Ma trận trọng số áp dụng lên trạng thái ẩn trước đó.
- $W_x$: Ma trận trọng số của đầu vào hiện tại.
- $b$: Hệ số điều chỉnh (bias).
- $f$: Hàm kích hoạt (thường là hàm tanh hoặc ReLU) để tạo ra đầu ra phi tuyến.

RNN sử dụng thuật toán lan truyền ngược để cập nhật trọng số, nhưng trong quá trình này, các gradient có thể trở nên rất nhỏ (gradient tiêu hao) hoặc rất lớn (gradient bùng nổ), dẫn đến sai số lớn trong quá trình học. Điều này gây khó khăn cho RNN trong việc ghi nhớ các thông tin quan trọng từ các bước xa trong chuỗi.

Để khắc phục vấn đề này, các mô hình RNN nâng cao như LSTM (Long Short-Term Memory) và GRU (Gated Recurrent Unit) được phát triển, cho phép mô hình ghi nhớ các thông tin dài hạn tốt hơn bằng cách sử dụng các cổng điều khiển trạng thái.

#h3("Mô hình Long Short-Term Memory")

Long Short-Term Memory (LSTM) là một dạng mạng nơ-ron hồi tiếp (Recurrent Neural Network - RNN) được thiết kế để xử lý và dự đoán dữ liệu chuỗi thời gian bằng cách giải quyết những hạn chế của RNN truyền thống, đặc biệt là trong việc lưu trữ và truy xuất thông tin dài hạn. Mô hình LSTM được phát triển nhằm giảm thiểu vấn đề về suy giảm đạo hàm (vanishing gradient problem) trong quá trình huấn luyện RNN với chuỗi dữ liệu dài.

Thuật toán LSTM hoạt động bằng cách sử dụng một cấu trúc ô nhớ (memory cell), nơi các trạng thái nhớ được kiểm soát qua các cổng: cổng quên, cổng nhập, và cổng đầu ra.

1. Các Cổng Chính trong LSTM

Cổng Quên (Forget Gate): Quyết định giữ lại bao nhiêu thông tin từ trạng thái trước đó. Công thức của cổng quên là:

- $f_t = sigma (W_f dot [h_(t-1), x_t] + b_f)$
  - $W_f$: Trọng số cho cổng quên.
  - $h_(t-1)$: Đầu ra từ trạng thái trước.
  - $x_t$: Dữ liệu hiện tại.
  - $sigma$: Hàm sigmoid, giới hạn đầu ra trong khoảng $(0, 1)$, xác định tỷ lệ thông tin được giữ lại.

Cổng Nhập (Input Gate): Quyết định thông tin mới sẽ được thêm vào trạng thái ô nhớ. Cổng nhập có hai bước:

- Xác định giá trị cần cập nhật: $i_t = sigma( W_i dot [h_(t-1), x_t] + b_i)$
  - Tạo một vector ứng viên $tilde(C)_t$ để thêm vào ô nhớ: $tilde(C)_t = tanh(W_C dot [h_(t-1), x_t] + b_C)$
  - Trạng thái ô nhớ mới: $C_t = f_t ast C_(t-1) + i_t ast tilde(C)_t$

Cổng Đầu Ra (Output Gate): Quyết định phần nào của trạng thái ô nhớ sẽ được đưa ra. Công thức tính toán là:

- $o_t = sigma(W_o dot [h_(t-1), x_t] + b_o)$

- Sau đó, đầu ra cuối cùng: $h_t = o_t ast tanh( C_t)$

Mỗi bước trong chuỗi dữ liệu, LSTM sẽ thực hiện các bước sau:

Bước 1: Xác định thông tin nào cần bỏ qua từ trạng thái nhớ trước qua cổng quên.
Bước 2: Thêm thông tin mới vào ô nhớ thông qua cổng nhập.
Bước 3: Tính toán đầu ra dựa trên cổng đầu ra và trạng thái mới của ô nhớ.

#h3("Mô hình Bidirectional Encoder Representation from Transformer")

BERT là viết tắt của cụm từ Bidirectional Encoder Representation from Transformer có nghĩa là mô hình biểu diễn từ theo 2 chiều ứng dụng kỹ thuật Transformer. BERT được thiết kế để huấn luyện trước các biểu diễn từ (pre-train word embedding). Điểm đặc biệt ở BERT đó là nó có thể điều hòa cân bằng bối cảnh theo cả 2 chiều trái và phải.

Cơ chế attention của Transformer sẽ truyền toàn bộ các từ trong câu văn đồng thời vào mô hình một lúc mà không cần quan tâm đến chiều của câu. Do đó Transformer được xem như là huấn luyện hai chiều (bidirectional) mặc dù trên thực tế chính xác hơn chúng ta có thể nói rằng đó là huấn luyện không chiều (non-directional). Đặc điểm này cho phép mô hình học được bối cảnh của từ dựa trên toàn bộ các từ xung quanh nó bao gồm cả từ bên trái và từ bên phải.
