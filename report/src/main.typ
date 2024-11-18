#import "template.typ": *
#import "state.typ": bib_state, bib
#bib_state.update(none)

#show: template

// --- set page number
#set page(numbering: "1")
#counter(page).update(1)

#include "chapter3.typ"

#h1("Tài liệu tham khảo", numbering: false)

#bib