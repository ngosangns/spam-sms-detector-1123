#let cover(border: true) = {
  rect(
    width: 100%,
    height: 100%,
    stroke: if (border) {
      2.5pt
    } else {
      0pt
    },
    [
      #set align(center)
      \
      BAN CƠ YẾU CHÍNH PHỦ \
      *PHÂN HIỆU HỌC VIỆN KTMM TẠI TP. HỒ CHÍ MINH*
      #line(start: (0%, 20pt), length: 3.5cm)
      #image("./images/logo.png", width: 3.5cm) \
      ĐỒ ÁN TỐT NGHIỆP \ \
      #text(weight: "bold", size: 16pt, "PHÁT TRIỂN ỨNG DỤNG PHÁT HIỆN TIN NHẮN LỪA ĐẢO BẰNG PHƯƠNG PHÁP HỌC MÁY") \ \ \
      #set align(left)
      #pad(
        left: 8cm,
        [
          #"\t\t\tNgành: An toàn thông tin" \
          #"Mã số: 7.48.02.02" \
        ],
      ) \

      #pad(
        left: 2cm,
        [
          _Sinh viên thực hiện_: \
          #pad(left: 1cm, top: 0.5cm, [*Đàm Chí Nguyên*])
          #pad(left: 1cm, [Lớp: AT15H])
          _Giảng viên hướng dẫn_: \
          #pad(left: 1cm, top: 0.5cm, [*Nguyễn Xuân Sâm*])
          #pad(left: 1cm, [Giảng viên trường khoa CNTT, HCMUTE])
        ],
      )

      #set align((center + bottom))

      #pad(bottom: .25cm, "Thành phố Hồ Chí Minh, 2023")
    ],
  )
}