#import "template.typ": *

#h1("Lời nói đầu", numbering: false)

Trong môi trường kỹ thuật số ngày càng phát triển, việc đảm bảo an toàn thông tin cá nhân và xác thực danh tính người dùng đã trở thành một yếu tố thiết yếu trong việc bảo vệ không gian mạng. Các cuộc tấn công lừa đảo qua tin nhắn, đặc biệt là thông qua SMS, đang có xu hướng gia tăng cả về số lượng lẫn mức độ phức tạp, đặt ra những thách thức đáng kể cho các chuyên gia bảo mật và các tổ chức trên toàn thế giới. Với mục tiêu đánh cắp thông tin cá nhân nhạy cảm hoặc chiếm đoạt tài sản, các cuộc tấn công này thường dựa vào sự bất cẩn của người dùng và khả năng giả mạo các tổ chức đáng tin cậy.

Trong bối cảnh này, việc nghiên cứu và phát triển các giải pháp phòng ngừa trở nên vô cùng quan trọng. Đồ án này tập trung vào việc phân tích chuyên sâu các phương pháp phát hiện lừa đảo qua tin nhắn, sử dụng các thuật toán học máy tiên tiến nhằm nhận diện các dấu hiệu đặc trưng của tin nhắn lừa đảo. Bên cạnh việc tìm hiểu các yếu tố kỹ thuật như cơ chế hoạt động của các mô hình học máy, em cũng nghiên cứu các tiêu chuẩn bảo mật liên quan và các thách thức trong việc triển khai trên quy mô lớn.

Thông qua việc thu thập và phân tích dữ liệu thực tế, em đã xây dựng và đánh giá hiệu quả của các mô hình nhận diện tin nhắn lừa đảo, từ đó đề xuất các phương pháp cải tiến để giảm thiểu tỷ lệ cảnh báo sai và tăng cường độ chính xác của hệ thống. Em mong rằng những kết quả nghiên cứu này sẽ không chỉ đóng góp vào việc nâng cao nhận thức về an ninh thông tin mà còn cung cấp cơ sở khoa học cho việc phát triển các giải pháp bảo mật hiệu quả hơn trong tương lai. Đồ án này không chỉ là cơ hội để em khám phá sâu hơn về các kỹ thuật bảo mật thông tin mà còn giúp em rèn luyện khả năng nghiên cứu, phân tích, và trình bày một cách chuyên nghiệp và có hệ thống.

#h1("Tổng quan")

#h2("Đặt vấn đề")

Ngày nay, cùng với sự phát triển không ngừng của công nghệ số đã làm thay đổi cách thức con người giao tiếp và thực hiện các giao dịch hàng ngày. Trong đó, các thiết bị thông minh như máy vi tính, điện thoại thông minh (smart phone),... đóng vai trò quan trọng, trở thành công cụ chính để trao đổi thông tin, thanh toán trực tuyến và lưu trữ các dữ liệu cá nhân quan trọng. Tuy nhiên, cùng với sự nhanh chóng và tiện lợi đó là nguy cơ mất an toàn thông tin, đặc biệt là qua các cuộc tấn công lừa đảo dưới dạng tin nhắn

#h2("Lừa đảo thông qua tin nhắn (Smishing)")

Tin nhắn văn bản hoặc SMS (Tin nhắn văn bản ngắn) là một phương thức nhắn tin ngắn gọn giữa các điện thoại di động, thường giới hạn trong khoảng 160 ký tự cho mỗi tin nhắn. SMS phishing còn được gọi là Smishing, một từ ghép giữa SMS và Phishing. David Rayhawk của McAfee đã sử dụng từ "SMISHING" lần đầu tiên vào ngày 25 tháng 8 năm 2006. [4] liên quan các hình thức lừa đảo qua tin nhắn. Trong smishing, kẻ tấn công gửi tin nhắn SMS có chứa liên kết hoặc yêu cầu cung cấp thông tin cá nhân, thường núp dưới danh nghĩa của ngân hàng, công ty dịch vụ tài chính, hoặc các tổ chức uy tín khác. Hình thức tấn công này ngày càng phổ biến, bởi người dùng thường có xu hướng tin tưởng vào những tin nhắn nhận được qua ứng dụng nhắn tin trên điện thoại hơn là qua email.

Ngoài ra, với sự phát triển của các nền tảng mạng xã hội, phishing hiện nay còn mở rộng sang các ứng dụng nhắn tin như Messenger, WhatsApp, và Telegram. Về mặt kỹ thuật không được xem là smishing, nhưng chúng có mối liên hệ chặt chẽ với nhau. Các tài khoản giả mạo của công ty hoặc bạn bè thường được sử dụng để tạo lòng tin, từ đó khiến nạn nhân dễ dàng cung cấp thông tin.

Một tin nhắn Smishing có thể bao gồm các thành phần sau đây:

#img("image.png", cap: "Các thảnh phần của một tin nhắn Smishing")

Liên kết URL nhúng: Trong kỹ thuật này, kẻ tấn công gửi tin nhắn văn bản chứa liên kết URL đến nạn nhân. Khi người dùng đọc tin nhắn và nhấp vào URL, các hoạt động sau sẽ được kích hoạt:
Mã độc hại nhúng trong URL sẽ được cài đặt vào điện thoại của nạn nhân.

Một số URL chuyển hướng người dùng đến một trang web phishing độc hại có giao diện giống với một trang web hợp pháp và yêu cầu người dùng điền thông tin đăng nhập hoặc thông tin thẻ tín dụng vào biểu mẫu.
Một tệp \*.apk được tải xuống điện thoại của nạn nhân, và sau đó gây ra các hoạt động độc hại.

Số điện thoại và/hoặc địa chỉ email: Kẻ tấn công gửi tin nhắn SMS đến nạn nhân, tuyên bố có phiếu giảm giá, quà tặng miễn phí hoặc ưu đãi, và trong tin nhắn đó có đề cập đến số điện thoại hoặc địa chỉ email của kẻ tấn công. Nạn nhân liên hệ với kẻ tấn công qua số điện thoại hoặc email, sau đó kẻ tấn công yêu cầu thông tin đăng nhập của họ.

Tin nhắn tự trả lời: Kẻ tấn công gửi tin nhắn tự trả lời đến nạn nhân yêu cầu họ đăng ký hoặc hủy đăng ký một dịch vụ. Những liên kết này sẽ chuyển hướng người dùng đến các trang web độc hại.

#h2("Tình trạng bảo mật thông tin trên các thiết bị thông minh")

Mặc dù nhiều người dùng không nhận thức được mối liên hệ giữa các cuộc tấn công phishing và tin nhắn cá nhân, nhưng thực tế cho thấy kẻ tấn công có thể dễ dàng truy tìm số điện thoại của nạn nhân hơn so với địa chỉ email. Điều này là bởi số điện thoại có một tập hợp giới hạn các sự kết hợp - ví dụ, tại Mỹ, số điện thoại chỉ gồm 10 chữ số. Ngược lại, địa chỉ email không bị giới hạn về độ dài và có thể bao gồm chữ cái, số, cùng với các ký hiệu như !, \#, %. Vì vậy, việc tạo ra một chuỗi 10 chữ số ngẫu nhiên để thực hiện tấn công trở nên dễ dàng hơn nhiều so với việc tìm kiếm một địa chỉ email hợp lệ.

Bên cạnh đó, kẻ tấn công có thể dễ dàng gửi tin nhắn đến bất kỳ sự kết hợp nào của các chữ số có độ dài tương ứng với số điện thoại mà không gặp phải rủi ro đáng kể. Theo báo cáo của Gartner, 98% tin nhắn SMS được người dùng đọc và 45% trong số đó nhận được phản hồi, điều này cho thấy SMS là một kênh tấn công lý tưởng đối với kẻ lừa đảo. Trái ngược với đó, chỉ có 6% email nhận được phản hồi từ người nhận.

Trên thực tế 46 triệu người dùng di động của Malaysia đã bị rò rỉ dữ liệu nhạy cảm và quan trọng như địa chỉ, số thẻ định danh, thông tin thẻ, sim điện thoại và thông tin cá nhân khác vào năm 2017; dữ liệu liên quan tới 7,6 triệu chủ tài khoản hiện tại và 65,4 triệu chủ tài khoản trước đây của Công ty viễn thông AT&T - nhà cung cấp dịch vụ không dây bán lẻ lớn thứ 3 của Mỹ đã bị rò rỉ từ trước năm 2019 và mới bị phát hiện vào đầu năm 2024. Theo báo cáo "State of the Phish" năm 2024 của Proofpoint, 75% các tổ chức đã gặp phải các cuộc tấn công smishing trong năm 2023.

Theo thống kê của Techreport, năm 2020, có tới 3 triệu tin nhắn lừa đảo liên quan đến lĩnh vực tài chính - ngân hàng. Thống kê mới nhất của Robokiller - ứng dụng chặn cuộc gọi và tin nhắn rác hàng đầu cho thấy, người Mỹ đã nhận được 225 tỉ tin nhắn rác vào năm 2023 (tăng 157% so với năm 2022) và 19,2 tỉ tin nhắn rác vào tháng 3/2024. Truecaller và Ủy ban Thương mại Liên bang cho biết, 68,4 triệu người Mỹ trở thành nạn nhân của các vụ lừa đảo qua điện thoại với tổng thiệt hại hơn 326 triệu USD. Ở Ailen, chỉ một vụ lừa đảo qua tin nhắn mà thiệt hại lên đến 800 nghìn USD. Theo báo cáo của Dimensional Enterprise Mobile Security, tấn công Smishing đứng thứ hai trong tất cả các cuộc tấn công vào thiết bị di động.

Tại Việt Nam, vào năm 2021, khách hàng của nhiều ngân hàng thương mại như Ngân hàng Thương mại cổ phần (NHTMCP) Ngoại thương Việt Nam (Vietcombank), NHTMCP Á Châu (ACB), NHTMCP Sài Gòn Thương Tín (Sacombank), NHTMCP Tiên Phong (TPBank) đã nhận được tin nhắn chứa đường dẫn mạo danh ngân hàng (Đại Việt, 2021; N.M, 2021). Những khách hàng này do không biết hoặc chủ quan nên đã nhấp vào và bị chiếm quyền truy cập tài khoản ngân hàng dẫn đến mất số tiền lớn. Năm 2022, Công an tỉnh Tuyên Quang cũng đã phải phát đi cảnh báo thủ đoạn lừa đảo qua SMS. Theo đó, các đối tượng giả lập trạm phát sóng di động của các nhà mạng trong nước rồi phát tán tin nhắn với nội dung thông báo khách hàng đã đăng kí các dịch vụ phát sinh chi phí (từ 3 - 6 triệu đồng) qua tài khoản ngân hàng (Hoàn, 2022).

Mặc dù Smishing là một loại lừa đảo, nhưng nó khác với lừa đảo ở nhiều khía cạnh như lượng thông tin có sẵn trong SMS, chiến lược tấn công, v.v. Do đó, việc phát triển của một ứng dụng có khả năng phát hiện và ngăn chặn tin nhắn lừa đảo trở thành một nhu cầu cấp thiết.

#h2("Phương pháp học máy")

Có hai phương pháp phòng vệ chính được sử dụng để phát hiện tin nhắn SMS giả mạo:

Phương pháp đầu tiên là kỹ thuật dựa trên danh sách đen, ngăn chặn các tin nhắn đến từ các nguồn giả mạo. Tuy nhiên, kỹ thuật này không bao quát được tất cả các nguồn giả mạo, vì tội phạm có thể mua bất kỳ số điện thoại di động nào để gửi tin nhắn giả mạo.

Phương pháp thứ hai là sử dụng các thuật toán học máy, trong đó các đặc trưng khác nhau được trích xuất từ tin nhắn SMS để đưa ra quyết định. Ưu điểm của phương pháp này là có thể phát hiện các tin nhắn giả mạo đến từ bất kỳ nguồn nào. Các phương pháp khai thác dữ liệu giúp trích xuất đặc trưng và tìm mối quan hệ giữa chúng, từ đó nhận diện kiến thức tiềm ẩn từ các tập dữ liệu dưới dạng các quy tắc và đưa ra quyết định dựa trên các quy tắc đó.

Có hai cách tiếp cận phổ biến trong việc xây dựng các thuật toán để phát hiện tin nhắn giả mạo là phương pháp phát hiện dựa trên luật (Rule-based method) và phương pháp lọc dựa trên nội dung (Xia và Chen, 2021).

#h3("Phương pháp dựa trên luật")

Phương pháp phát hiện dựa trên luật được những tập đoàn công nghệ lớn như Google, Symantec, McAfee ứng dụng để loại bỏ những tin nhắn, thư điện tử rác (M. Hameed và Hussein Ali, 2021) là một trong những cách tiếp cận truyền thống trong việc phát hiện và lọc tin nhắn rác.

Phương pháp này dựa trên các quy tắc được xác định sẵn để phân loại tin nhắn là rác hoặc không rác. Các quy tắc này được xây dựng dựa trên các mẫu và đặc điểm phổ biến trong các tin nhắn rác đã biết như từ ngữ đặc trưng, cấu trúc câu, hoặc các thông tin cụ thể về người gửi. Các quy tắc sẽ giúp hệ thống phân biệt giữa tin nhắn thông thường và tin nhắn có nội dung quảng cáo hoặc lừa đảo.

Ưu điểm:

- Dễ dàng triển khai: Phương pháp này không yêu cầu tập dữ liệu lớn hoặc các thuật toán phức tạp, do đó dễ triển khai và sử dụng trong các hệ thống đơn giản.
- Có tính tức thời: Hệ thống có thể xác định tin nhắn rác ngay khi nhận được mà không cần phải huấn luyện mô hình phức tạp.
- Khả năng kiểm soát cao: Các quy tắc có thể được thiết lập và điều chỉnh để phù hợp với nhu cầu của tổ chức hoặc cá nhân.

Nhược điểm:

- Tăng kích thước tập luật: Để duy trì hiệu quả, các quy tắc cần được cập nhật liên tục nhằm bắt kịp các kỹ thuật mới của tin nhắn rác. Điều này có thể dẫn đến tập hợp quy tắc trở nên phức tạp và khó quản lý.
- Thiếu tính linh hoạt: Phương pháp này dựa trên quy tắc cố định, nên khi có các mẫu tin rác mới, phương pháp này sẽ không thể phát hiện cho đến khi quy tắc được cập nhật.
- Hiệu quả thấp khi đối mặt với nội dung mới: Tin nhắn độc hại thường xuyên thay đổi mẫu, từ ngữ và cấu trúc để tránh bị phát hiện. Điều này làm cho phương pháp dựa trên luật không còn hiệu quả nếu không có cập nhật thường xuyên.

#h3("Phương pháp lọc dựa trên nội dung")

Phương pháp lọc tin nhắn độc hại dựa trên nội dung sử dụng nền tảng học máy được quan tâm nghiên cứu nhiều hơn trong những năm trở lại đây (Hsu, 2020; Xia và Chen, 2021) Khác với phương pháp phát hiện dựa trên luật, phương pháp này tập trung vào việc phân tích nội dung thực tế của tin nhắn.

Các kỹ thuật và thuật toán phổ biến có thể kể đến như:
- Naive Bayes: Là một trong những thuật toán đơn giản và hiệu quả trong việc phân loại văn bản. Naive Bayes giả định rằng các đặc trưng là độc lập với nhau, điều này giúp giảm độ phức tạp tính toán.
- Support Vector Machines (SVM): Là một thuật toán mạnh mẽ cho các bài toán phân loại, SVM tìm kiếm siêu phẳng (hyperplane) tối ưu để phân loại dữ liệu.
- Deep Learning: Các mô hình mạng nơ-ron, đặc biệt là RNN (Recurrent Neural Network) và LSTM (Long Short-Term Memory), đang được sử dụng ngày càng nhiều trong việc phân loại văn bản, nhờ vào khả năng nắm bắt mối quan hệ trong chuỗi văn bản.

Ưu điểm: Phương pháp lọc dựa trên nội dung mang lại độ chính xác cao, có khả năng phát hiện các tin nhắn rác mới mà không cần cập nhật quy tắc, đồng thời mô hình học máy có thể cải thiện qua thời gian khi được cung cấp thêm dữ liệu mới.

Nhược điểm: Để đạt được độ chính xác cao, phương pháp này yêu cầu một lượng lớn dữ liệu đã gán nhãn để huấn luyện mô hình, và một số thuật toán, đặc biệt là Deep Learning, có thể cần tài nguyên tính toán lớn.

Ngày nay với nhiều nghiên cứu khảo sát có thể thấy, hầu hết những giải pháp hiện có được đề xuất cho bài toán phát hiện tin nhắn rác đạt kết quả chính xác rất cao, thậm chí tiệm cận tới 100% có thể kể đến như:

- Nghiên cứu xác định phát hiện tin nhắn lừa đảo của Sandhya Mishra and Devpriya Soni bằng cách sử dụng mô hình bảo mật thông qua phân tích nội dung tin nhắn và phân tích hành vi URL đã đạt được độ hiệu quả tổng thể lên đến 96.29%.
- Thực hiện thử nghiệm với hai phương pháp phân lớp SVM và Naive Bayes trên kho SMS của Almeida và cộng sự (2011), Sonowal, Kuppusamy và cộng sự (2018) đã phát triển hai mô hình phân loại tin nhắn rác có độ chính xác 94,2%. Cùng sử dụng bộ dữ liệu tương tự, Jain và Gupta (2019) cũng đạt được độ chính xác 94,2% khi áp dụng cây quyết định và SVM vào bài toán phát hiện tin nhắn rác.
- Sjarif và cộng sự (2019) cùng Mishra và Soni (2020), sử dụng kỹ thuật phân lớp Naive Bayes trên cùng bộ dữ liệu của Almeida và cộng sự (2011), đã đạt được các mô hình phát hiện tin nhắn rác với độ chính xác lần lượt là 97,5% và 96,29%.
- Xia và Chen đã liên tục cải thiện mô hình Markov ẩn trên kho SMS của Almeida và cộng sự (2011) để phát triển công cụ phân loại tin nhắn rác, đạt độ chính xác 95,9% trong nghiên cứu (Xia và Chen, 2020) và 96,9% trong nghiên cứu (Xia và Chen, 2021).
- Roy và cộng sự (2020) đã phát triển một loạt mô hình phát hiện tin nhắn rác dựa trên các kỹ thuật học sâu như LSTM, CNN và mô hình kết hợp LSTM-CNN, sử dụng bộ dữ liệu SMS của Almeida và cộng sự (2011). Mô hình kết hợp LSTM-CNN đạt độ chính xác cao nhất là 99,44%.
- Cùng sử dụng phương pháp kết hợp LSTM và CNN, Ghourabi và cộng sự (2020) đã đề xuất một công cụ lọc tin nhắn rác dựa trên cùng bộ dữ liệu, đạt độ chính xác 98,37%. Kết quả này tương đương với các mô hình phân loại tin nhắn rác sử dụng thuật toán KNN của Sousa và cộng sự (2021), mô hình tuần tự trong học sâu của Liu và cộng sự (2021), CNN của Giri và cộng sự (2023), và mô hình kết hợp giữa CNN và Bi-LSTM của Mambina và cộng sự (2024).
- Một số nghiên cứu đã phát triển mô hình phát hiện tin nhắn rác gần như chính xác tuyệt đối khi sử dụng kho SMS của Almeida và cộng sự (2011), như công trình của Ghourabi và Alohaly (2023) kết hợp các kỹ thuật SVM, KNN, LightGBM và CNN; nghiên cứu của Maqsood và cộng sự (2023) áp dụng kỹ thuật SVM; hay công trình của Hussein và cộng sự (2023) kết hợp LSTM và CNN.

Ngoài các mô hình phát hiện tin nhắn rác tiếng Anh, đã có một số nghiên cứu phát hiện tin nhắn rác tiếng Indonesia. Hikmaturokhman và cộng sự (2022) sử dụng các kỹ thuật mạng nơ-ron dày đặc, LSTM và Bi-LSTM với độ chính xác lần lượt là 95,63%, 94,76% và 94,75%. Tuấn và cộng sự (2022) đã phát triển mô hình phát hiện tin nhắn rác tiếng Việt dựa trên sự kết hợp DNN và PhoBERT với độ chính xác 99,53%. Mambina và cộng sự (2024) xây dựng công cụ phát hiện tin nhắn rác tiếng Tanzania bằng cách kết hợp CNN và LSTM, đạt độ chính xác 99,98%. Cuối cùng, Ayaz và cộng sự (2024) sử dụng kho tin nhắn ngôn ngữ Latinh riêng tư để huấn luyện mô hình SVM và Naive Bayes, đạt được độ chính xác lần lượt là 97,33% và 99,42%.

Tuy nhiên, những số liệu thống kê lại phản ánh thực tế vấn nạn tin nhắn rác và mức độ thiệt hại của người dùng thiết bị di động không có dấu hiệu thuyên giảm trong những năm gần đây Những nghiên cứu điển hình được phân tích ở trên sử dụng hầu hết các bộ dữ liệu tin nhắn bị hạn chế về mặt số lượng tin nhắn và dữ liệu tin nhắn không được cập nhật thường xuyên (Salman và cộng sự, 2024), trong khi đó tin tặc lại sử dụng nhiều thủ đoạn khác nhau nhằm thay đổi liên tục mẫu tin nhắn rác để tránh bị phát hiện.

#h2("Mục tiêu đề tài")

Mục tiêu chính của đề tài này là xây dựng một ứng dụng phát hiện tin nhắn lừa đảo bằng cách áp dụng các thuật toán học máy dựa trên bộ dữ liệu tin nhắn phức hợp được cập nhật và tích hợp vào các nền tảng SMS hoặc công cụ chatbot tiện dụng. Cụ thể, đề tài sẽ tập trung vào những mục tiêu chính như:

- Phát hiện hiệu quả các tin nhắn SMS spam và lừa đảo: Phân tích và ứng dụng các mô hình học máy như Naive Bayes, Random Forest, LSTM, và Transformer để phát hiện và phân loại tin nhắn lừa đảo dựa trên nội dung và hành vi URL. Mục đích giúp tăng khả năng phát hiện các tin nhắn chứa đường link giả mạo hoặc yêu cầu cung cấp thông tin cá nhân.
- Tối ưu hóa độ chính xác và giảm tỷ lệ báo động sai: Mục tiêu tiếp theo là cải thiện độ chính xác của hệ thống phát hiện lừa đảo, giảm thiểu tỷ lệ báo động sai (false-positive) thông qua việc sử dụng các mô hình học sâu (Deep Learning) và tích hợp các thuật toán kiểm tra hành vi URL. Điều này đảm bảo rằng chỉ những tin nhắn thực sự nguy hiểm mới bị chặn.
- Tích hợp hệ thống phát hiện lừa đảo với các nền tảng SMS hiện có: Phát triển một hệ thống có khả năng tích hợp trực tiếp vào các nền tảng SMS hiện tại của các tổ chức tài chính hoặc nhà cung cấp dịch vụ viễn thông. Điều này sẽ giúp phát hiện và ngăn chặn các tin nhắn lừa đảo trước khi chúng đến tay người dùng, đồng thời bảo vệ dữ liệu cá nhân và tài chính của người dùng.
- Nâng cao nhận thức và bảo mật cho người dùng di động: Hướng đến việc tạo ra một hệ thống không chỉ phát hiện mà còn cung cấp các cảnh báo và giáo dục người dùng về cách nhận diện tin nhắn lừa đảo. Mục tiêu này giúp người dùng nắm vững kiến thức cơ bản về bảo mật và nâng cao khả năng tự bảo vệ trước các cuộc tấn công.

#h2("Các đóng góp chính của đề tài")

#h3("Mô tả kịch bản tấn công Smishing")

Phân loại các kiểu tấn công Smishing: Đề tài phân tích chi tiết các kịch bản lừa đảo thường gặp, bao gồm những tin nhắn giả mạo ngân hàng, cơ quan chính phủ, dịch vụ tài chính, và các tin nhắn yêu cầu người dùng cung cấp thông tin cá nhân hoặc truy cập vào các liên kết giả mạo. Mỗi kịch bản được mô tả cụ thể, nhằm hiểu rõ cơ chế tấn công và hành vi của kẻ xấu.

Nhận diện các yếu tố nhận dạng Smishing: Đề tài nghiên cứu và mô tả các yếu tố đặc trưng của tin nhắn lừa đảo, bao gồm nội dung chứa URL độc hại, từ ngữ khẩn cấp như "khóa tài khoản", "trúng thưởng", "liên hệ ngay", nhằm đánh lừa và kích thích người dùng tương tác.

#h3("Thu thập và xử lý dữ liệu")

Dữ liệu là yếu tố trung tâm quyết định chất lượng của hệ thống học máy. Phần này tập trung vào việc thu thập dữ liệu tin nhắn lừa đảo và hợp lệ, cũng như quy trình tiền xử lý để đảm bảo rằng mô hình học máy hoạt động hiệu quả nhất.

#h4("Các nguồn dữ liệu tin nhắn lừa đảo và hợp lệ")

Để phát hiện chính xác tin nhắn lừa đảo, cần phải có một tập dữ liệu đa dạng và phong phú bao gồm cả tin nhắn hợp lệ và tin nhắn lừa đảo.

Dữ liệu từ các nguồn công khai và học thuật:

- Kaggle datasets: Đây là một nguồn quan trọng cung cấp các tập dữ liệu về tin nhắn SMS bao gồm cả lừa đảo và hợp lệ. Kaggle có nhiều tập dữ liệu liên quan đến tin nhắn lừa đảo (spam) được thu thập từ nhiều quốc gia khác nhau, bao gồm cả ngôn ngữ Anh, Việt, và nhiều ngôn ngữ khác.
- SpamAssassin Public Corpus: Là một trong những tập dữ liệu nổi tiếng về spam và lừa đảo, chứa các tin nhắn từ nhiều nguồn email khác nhau. Tuy dữ liệu ban đầu là email, nhưng nhiều mô hình xử lý văn bản có thể tái sử dụng các phương pháp này cho tin nhắn SMS.
- Nghiên cứu học thuật và báo cáo từ các tổ chức an ninh: Các nghiên cứu từ các tổ chức lớn như Symantec, McAfee, và nhiều trường đại học đã cung cấp các tập dữ liệu về Smishing (SMS Phishing), trong đó chứa đựng các mẫu tin nhắn lừa đảo.

Dữ liệu tự tạo hoặc thu thập thủ công:

- Một số tổ chức có thể tự xây dựng tập dữ liệu bằng cách giả lập các cuộc tấn công lừa đảo qua tin nhắn, nhằm hiểu rõ cách thức tấn công và tạo ra các mẫu lừa đảo chuẩn mực.
- Thu thập từ các diễn đàn, trang web và mạng xã hội: Nhiều kẻ tấn công sử dụng các diễn đàn, website hoặc mạng xã hội để chia sẻ các mẫu tin nhắn lừa đảo. Việc theo dõi và thu thập các mẫu này từ những nguồn trên cũng là một cách để xây dựng tập dữ liệu.

#h4("Quy trình tiền xử lý dữ liệu")

Tiền xử lý dữ liệu là bước rất quan trọng trong quá trình phát triển mô hình học máy, đặc biệt là với dữ liệu văn bản như tin nhắn SMS. Việc tiền xử lý sẽ giúp làm sạch dữ liệu, loại bỏ nhiễu, và trích xuất các đặc trưng quan trọng phục vụ cho việc học máy. Một số các bước trong quy trình tiền xử lý bao gồm:

Làm sạch văn bản (Text Cleaning):

- Loại bỏ ký tự đặc biệt: Tin nhắn SMS thường chứa các ký tự đặc biệt, chẳng hạn như biểu tượng, số điện thoại, hoặc các ký tự đặc biệt mà không cần thiết cho mô hình học máy. Loại bỏ các ký tự này giúp giảm nhiễu và tập trung vào nội dung chính của tin nhắn.
- Xử lý viết tắt và từ viết sai chính tả: Tin nhắn SMS thường chứa các từ viết tắt hoặc viết sai chính tả. Các kỹ thuật như lemmatization hoặc stemming sẽ giúp chuyển đổi các từ này về dạng chuẩn để giúp mô hình hiểu rõ hơn nội dung.

Loại bỏ từ dừng (Stopwords Removal): Các từ dừng là những từ phổ biến trong văn bản nhưng không mang nhiều ý nghĩa cho việc phân loại, chẳng hạn như: "và", "là", "nhưng", "hoặc" (trong tiếng Việt), hay "is", "and", "but", "or" (trong tiếng Anh). Việc loại bỏ từ dừng giúp giảm số lượng từ cần xử lý mà vẫn giữ lại ý nghĩa chính của tin nhắn.

Chuyển đổi văn bản thành số:

- Bag of Words (BoW): Mô hình BoW tạo ra ma trận từ điển trong đó mỗi từ trong tập dữ liệu được biểu diễn bằng một cột, và mỗi tin nhắn được biểu diễn bằng số lần xuất hiện của các từ này. Tuy nhiên, BoW có hạn chế là không xem xét được thứ tự của các từ.
- TF-IDF (Term Frequency-Inverse Document Frequency): Đây là một trong những phương pháp phổ biến để trích xuất đặc trưng từ văn bản. TF-IDF không chỉ tính toán tần suất của từ trong tin nhắn mà còn điều chỉnh mức độ quan trọng của từ đó dựa trên tần suất xuất hiện của nó trong toàn bộ tập dữ liệu. TF-IDF giúp tăng trọng số cho các từ có ý nghĩa trong việc phân loại tin nhắn lừa đảo.

Chuẩn hóa và xử lý biến số: Sau khi chuyển văn bản thành số liệu, việc chuẩn hóa dữ liệu là cần thiết để đảm bảo rằng tất cả các đặc trưng (features) có cùng tỷ lệ và trọng số. Điều này rất quan trọng đối với các mô hình học máy như SVM và Naive Bayes để đảm bảo rằng các đặc trưng không bị mất cân bằng ảnh hưởng đến kết quả.

Xử lý dữ liệu bị thiếu (Missing Data): Dữ liệu bị thiếu hoặc không đầy đủ có thể dẫn đến sai lệch trong việc huấn luyện mô hình. Quá trình này bao gồm việc kiểm tra và xử lý các trường hợp dữ liệu trống hoặc không hợp lệ.

#h3("Phân tích và đề xuất mô hình")

#h4("Phân tích đặc trưng của tin nhắn")

Việc phân tích đặc trưng của tin nhắn là cần thiết để nhận biết các dấu hiệu giúp phát hiện tin nhắn lừa đảo:

- Phân tích từ khóa: Các tin nhắn lừa đảo thường chứa các từ khóa hoặc cụm từ gây hoang mang hoặc thúc giục người dùng hành động ngay lập tức, chẳng hạn như "trúng thưởng", "khóa tài khoản", "liên hệ ngay". Những từ khóa này là tín hiệu rõ ràng của hành vi lừa đảo.
- URL trong tin nhắn: Các tin nhắn lừa đảo thường chứa các liên kết (URL) giả mạo, dẫn đến các trang web giả để lừa người dùng cung cấp thông tin cá nhân. Việc phân tích cấu trúc và độ tin cậy của các URL trong tin nhắn giúp phát hiện các tin nhắn lừa đảo hiệu quả hơn.
- Phân tích ngữ nghĩa: Việc hiểu rõ ngữ cảnh và nội dung của tin nhắn cũng là một bước quan trọng. Ví dụ, các tin nhắn yêu cầu cung cấp mã OTP, mật khẩu, hoặc số thẻ tín dụng thường là dấu hiệu rõ ràng của lừa đảo.
- Độ dài và cấu trúc của tin nhắn: Các tin nhắn lừa đảo thường có cấu trúc khác biệt so với tin nhắn hợp lệ. Chúng thường ngắn gọn, sử dụng từ ngữ khẩn cấp hoặc đôi khi dài dòng nhưng chứa nội dung không rõ ràng.

#h4("Ứng dụng thực tiễn")

- Phát triển ứng dụng phát hiện lừa đảo: Đề tài phát triển một ứng dụng thực tiễn, dễ sử dụng để người dùng có thể nhập tin nhắn và kiểm tra xem tin nhắn đó có phải lừa đảo hay không. Giao diện người dùng được thiết kế đơn giản, thân thiện, và có khả năng xử lý nhanh chóng các tin nhắn đầu vào.

- Cảnh báo bảo mật và tích hợp: Ứng dụng tích hợp các tính năng cảnh báo bảo mật cho người dùng, giúp họ phòng ngừa các mối đe dọa tiềm tàng từ các tin nhắn lừa đảo.

- Lựa chọn mô hình tối ưu: Đề tài đánh giá và so sánh giữa các mô hình học máy truyền thống và học sâu, nhằm tìm ra mô hình tối ưu nhất dựa trên độ chính xác, tốc độ và khả năng mở rộng. Sau khi đánh giá, mô hình có độ chính xác cao nhất và phù hợp với yêu cầu của đề tài sẽ được sử dụng để triển khai ứng dụng thực tiễn.
