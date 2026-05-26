/*
Copyright 2026 Olivier Mengué

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package hfdatasetfs

import (
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/dolmen-go/huggingface-dataset-fs/hfclient"
)

func TestWalkMock(t *testing.T) {

	mux := http.NewServeMux()
	mux.HandleFunc("/parquet", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("dataset") != "org/dataset" {
			http.Error(w, "dataset not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"parquet_files":[
		{"config":"default","split":"train","url":"https://huggingface.co/datasets/org/dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet","filename":"0000.parquet","size":12}
]}`)
	})
	mux.HandleFunc("/datasets/org/dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", func(w http.ResponseWriter, r *http.Request) {
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(func() { ts.Close() })

	backupDefaultClient := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = backupDefaultClient })
	http.DefaultClient = ts.Client()

	client, err := hfclient.NewClient(hfclient.Options{
		Token: "test-token",
		Logger: slog.New(slog.NewTextHandler(t.Output(), &slog.HandlerOptions{Level: slog.LevelDebug})).
			WithGroup("hfclient").
			With(slog.String("test", t.Name())),
	}).HTTPClient(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	testWalk(t, client, "org/dataset", ts.URL)
}

func TestWalkReal(t *testing.T) {
	if os.Getenv("HF_TOKEN") == "" {
		t.Skip("HF_TOKEN environment variable not set")
	}

	client, err := hfclient.NewClient(hfclient.Options{
		Logger: slog.New(slog.NewTextHandler(t.Output(), &slog.HandlerOptions{Level: slog.LevelDebug})).
			WithGroup("hfclient").
			With(slog.String("test", t.Name())),
	}).HTTPClient(t.Context())
	if err != nil {
		t.Fatal(err)
	}

	testWalk(t, client, "macrolens/MacroLens", "")
}

func testWalk(t *testing.T, client *http.Client, dataset string, baseURL string) {
	fsys := New(client, dataset, &Options{BaseURL: baseURL})
	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Found %d entries.", len(entries))
	for _, entry := range entries {
		t.Log(fs.FormatDirEntry(entry))
	}
	t.Log("Walking the filesystem:")
	fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			t.Logf("Error walking to %s: %v", path, err)
			return nil
		}
		t.Logf("%s: %s", path, fs.FormatDirEntry(d))
		return nil
	})
}
