# huggingface-dataset-fs

[![Go Reference](https://pkg.go.dev/badge/github.com/dolmen-go/huggingface-dataset-fs.svg)](https://pkg.go.dev/github.com/dolmen-go/huggingface-dataset-fs)
[![License](https://img.shields.io/github/license/dolmen-go/huggingface-dataset-fs)](LICENSE)

`huggingface-dataset-fs` provides a virtual read-only filesystem ([`io/fs.FS`](https://pkg.go.dev/io/fs#FS)) for Go programs, accessing [Hugging Face](https://huggingface.co) datasets directly from their [Apache Parquet](https://parquet.apache.org/) exports.

This library allows you to browse and read dataset files as if they were on your local disk, without downloading the entire dataset upfront. It uses the Hugging Face [Datasets Server API](https://huggingface.co/docs/dataset-viewer/parquet) to discover Parquet files and streams them on demand.

## Features

- **Standard [`io/fs`](https://pkg.go.dev/io/fs) Interface**: Use familiar Go filesystem tools like `fs.WalkDir`, `fs.ReadFile`, or `fs.ReadDir`.
- **On-demand Streaming**: Only downloads the files you actually open.
- **Support for Private Datasets**: Easily authenticate using Hugging Face tokens.
- **Structured Hierarchy**: Maps dataset configs and splits to a directory structure: `config/split/file.parquet`.

## Installation

```bash
go get github.com/dolmen-go/huggingface-dataset-fs@main
```

## Usage

### Simple Example (Public Dataset)

```go
package main

import (
	"fmt"
	"io/fs"
	"net/http"

	"github.com/dolmen-go/huggingface-dataset-fs/hfdatasetfs"
)

func main() {
	// Initialize the filesystem for a specific dataset
	fsys := hfdatasetfs.New(http.DefaultClient, "stanfordnlp/imdb", nil)

	// Walk the dataset and list files
	fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		fmt.Println(path)
		return nil
	})
}
```

### Private Datasets & Authentication

To access private datasets, use the `hfclient` package to handle authentication via your `HF_TOKEN`.

```go
package main

import (
	"context"
	"io/fs"
	"log"

	"github.com/dolmen-go/huggingface-dataset-fs/hfclient"
	"github.com/dolmen-go/huggingface-dataset-fs/hfdatasetfs"
)

func main() {
	// Create a client that uses HF_TOKEN from environment
	hf := hfclient.NewClient(nil)
	httpClient, _ := hf.HTTPClient(context.Background())

	fsys := hfdatasetfs.New(httpClient, "your-org/private-dataset", nil)

	f, err := fsys.Open("default/train/0000.parquet")
	if err != nil {
		log.Fatal(err)
	}
	// ... use data
}
```

## Filesystem Structure

The filesystem reproduces Hugging Face dataset API structure:
- `.` (root)
  - `config_name/`
    - `split_name/`
      - `0000.parquet`
      - `0001.parquet`

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
