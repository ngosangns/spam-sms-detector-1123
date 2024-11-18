# Report

## Installation

- Typst: https://github.com/typst/typst

## Edit

Refer to using VSCode with the Tinymist Typst and LaTeX Workshop extensions.

**Note**: If you are using the LaTeX Workshop extension, run the command `LaTeX Workshop: Refresh all LaTeX viewers` and set LaTeX Workshop as the default opener for `.pdf` files.

To export `.pdf` while watching for file changes:

```sh
RUST_BACKTRACE=full typst watch --font-path fonts ./src/main.typ main.pdf
```
