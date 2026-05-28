package hfdatasetfs_test

import (
	"fmt"
	"io/fs"
	"net/http"

	"github.com/dolmen-go/huggingface-dataset-fs/hfdatasetfs"
)

func Example_anonymous() {
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

	// Output:
	// .
	// plain_text
	// plain_text/test
	// plain_text/test/0000.parquet
	// plain_text/train
	// plain_text/train/0000.parquet
	// plain_text/unsupervised
	// plain_text/unsupervised/0000.parquet
}
