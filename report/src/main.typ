#import "template.typ": *
#import "state.typ": bib_state, bib
#import "cover.typ": *

#bib_state.update(none)

#show: template

#cover()
#cover(border: false)

// --- set page number
#set page(numbering: "1")
#counter(page).update(1)

#include "pre.typ"
#include "chapter1.typ"
#include "chapter2.typ"
#include "chapter3.typ"

#h1("Tài liệu tham khảo", numbering: false)

#bib