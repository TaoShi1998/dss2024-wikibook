# dss2024-wikibook

## Installation of mdBook
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  ## install Rust on Unix-like OS
cargo install mdbook  ## install mdBook
cargo install mdbook-katex  ## install mdBook-KaTeX
```

## Read the Wikibook on Localhost
```
git clone https://github.com/TaoShi1998/dss2024-wikibook.git
cd dss2024-wikibook-main
mdbook serve --open
```

## Publish the Wikibook
```
mdbook build
```
This will generate a directory named book which contains the HTML content of your book. You can then place this directory on any web server to host it.
