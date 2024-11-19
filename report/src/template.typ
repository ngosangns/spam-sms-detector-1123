#let textSize = 14pt
#let tabSize = 1cm
#let prefixh1 = "CHƯƠNG"

#let h1(children, numbering: true, pageBreak: true) = {
  if (pageBreak) [#pagebreak()]
  set align(center)
  heading(
    upper(text(size: textSize, weight: "bold", children)),
    level: 1,
    numbering: if (numbering) {
      (..nums) => text(
        size: textSize,
        weight: "bold",
        prefixh1 + " " + nums.pos().map(str).join(".") + ".",
      )
    } else {
      none
    },
  )
}

#let h2(children) = {
  heading(
    level: 2,
    numbering: (..nums) => text(size: textSize, weight: "bold", nums.pos().map(str).join(".") + "."),
    text(size: textSize, weight: "bold", children),
  )
}

#let h3(children) = {
  heading(
    level: 3,
    numbering: (..nums) => text(
      size: textSize,
      style: "italic",
      weight: "regular",
      h(1cm) + nums.pos().map(str).join(".") + ".",
    ),
    text(size: textSize, style: "italic", weight: "regular", children),
  )
}

#let h4(children) = {
  heading(
    level: 4,
    numbering: (..nums) => text(
      size: textSize,
      style: "italic",
      weight: "regular",
      h(1cm) + nums.pos().map(str).join(".") + ".",
    ),
    text(size: textSize, style: "italic", weight: "regular", children),
  )
}

#let img(src, cap: "", width: 100%) = {
  src = "images/" + src
  figure(
    [
      #block(inset: (top: 0pt))
      #image(src, width: width)
      #block(inset: (top: -(1 * textSize)))
    ],
    caption: cap,
    supplement: "Hình",
  )
}

#let tabl(..fields, cap: false) = {
  if type(cap) == "string" {
    block(inset: (top: -(2 * textSize)))
    figure(placement: none, caption: cap, supplement: "Bảng", table())
    block(inset: (top: -(2 * textSize)))
  }
  table(inset: 10pt, align: left, ..fields)
}

#let template(body) = {
  set page("a4", margin: (top: 2cm, right: 2cm, bottom: 2cm, left: 3cm))
  set text(size: textSize, font: ("Times New Roman"), fallback: false, region: "VN")
  set block(inset: 0pt, outset: 0pt, above: 0pt, below: 1.5 * textSize)
  set par(
        justify: true,
        leading: textSize, // 1.5 lines
        first-line-indent: tabSize,
    )
  set list(
    indent: tabSize,
    body-indent: .75em,
    marker: ([•], [◦], [--]),
    tight: true,
  )
  set enum(
    indent: tabSize,
    body-indent: .75em,
    tight: true,
  )

  show heading: it => {
    block(inset: (top: 0pt))
    it
    block(inset: (top: -(2.5 * textSize)))
    par("")
  }
  show list: it => {
    block(inset: (top: -(0.25 * textSize)))
    set par(hanging-indent: -(2 * tabSize - 1.5em))
    it
    block(inset: (top: -(3 * textSize)))
    par("")
  }
  show enum: it => {
    block(inset: (top: -(0.25 * textSize)))
    set par(hanging-indent: -(2 * tabSize - 0.75em))
    it
    block(inset: (top: -(3 * textSize)))
    par("")
  }
  show figure: it => {
    it
    block(inset: (top: -(3 * textSize)))
    par("")
  }
  show table: it => {
    it
    block(inset: (top: -(3 * textSize)))
    par("")
  }
  show raw.where(block: true): it => {
    block(inset: (top: 0pt))
    set par(justify: false, leading: 8pt)
    set text(size: 8pt)
    grid(
      columns: (100%, 100%),
      column-gutter: -100%,
      block(radius: 0em, fill: luma(246), stroke: 1pt, width: 100%, inset: 1em, it),
    )
    block(inset: (top: -(3 * textSize)))
    par("")
  }

  show bibliography: it => {
    block(inset: (top: (textSize)))
    set text(lang: "vi", region: "VN")
    it
  }

  body
}