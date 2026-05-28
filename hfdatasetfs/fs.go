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

// Package hfdatasetfs implements a read-only [fs.FS] for remote Hugging Face datasets in Parquet format.
package hfdatasetfs

import (
	"cmp"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"slices"
	"strings"
	"time"
)

type datasetFS struct {
	client  *http.Client
	baseURL string
	dataset string
	err     error // error encountered during initialization, which will be returned by all methods

	files []fs.DirEntry // backed by *fileinfo
	dirs  []dir
}

type Options struct {
	BaseURL string
}

// New creates a new datasetFS for the given dataset, using the provided HTTP client and options.
//
// Credentials for accessing private datasets should be handled by the HTTP client, for example using the [hfclient.Client] and its [hfclient.Client.HTTPClient] method.
func New(client *http.Client, dataset string, opts *Options) fs.FS {
	fsys := &datasetFS{
		client:  client,
		baseURL: opts.BaseURL,
		dataset: dataset,
	}
	if fsys.baseURL == "" {
		fsys.baseURL = "https://datasets-server.huggingface.co"
	}

	resp, err := client.Get(fsys.baseURL + "/parquet?dataset=" + dataset)
	if err != nil {
		fsys.err = &fs.PathError{Op: "init", Path: ".", Err: err}
		return fsys
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fsys.err = &fs.PathError{Op: "init", Path: ".", Err: errors.New("unexpected status code: " + resp.Status)}
		return fsys
	}

	dec := json.NewDecoder(resp.Body)
	var data struct {
		ParquetFiles []*parquetFile `json:"parquet_files"`
	}
	err = dec.Decode(&data)
	if err != nil {
		fsys.err = &fs.PathError{Op: "init", Path: ".", Err: err}
		return fsys
	}
	if dec.More() {
		fsys.err = &fs.PathError{Op: "init", Path: ".", Err: errors.New("unexpected data after JSON")}
		return fsys
	}
	fsys.buildIndex(data.ParquetFiles)
	return fsys
}

// parquetFile represents a parquetFile in the dataset, with its metadata and URL for downloading.
//
// Reference: https://huggingface.co/docs/dataset-viewer/parquet
type parquetFile struct {
	Config   string `json:"config"`
	Split    string `json:"split"`
	URL      string `json:"url"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

// fileinfo implements [fs.FileInfo] and [fs.DirEntry] for representing a parquet file
// in the filesystem index.
type fileinfo struct {
	path string
	name string
	size int64
	url  string
}

var (
	_ fs.FileInfo = (*fileinfo)(nil)
	_ fs.DirEntry = (*fileinfo)(nil)
)

// Info implements [fs.DirEntry].
func (f *fileinfo) Info() (fs.FileInfo, error) {
	return f, nil
}

// Name implements [fs.FileInfo] and [fs.DirEntry].
func (f *fileinfo) Name() string {
	return f.name
}

// Size implements [fs.FileInfo].
func (f *fileinfo) Size() int64 {
	return f.size
}

// Mode implements [fs.FileInfo].
func (f *fileinfo) Mode() fs.FileMode {
	return 0444
}

// Type implements [fs.DirEntry] and [fs.FileInfo].
func (f *fileinfo) Type() fs.FileMode {
	return 0
}

// ModTime implements [fs.FileInfo].
func (f *fileinfo) ModTime() time.Time {
	return time.Time{}
}

// IsDir implements [fs.DirEntry] and [fs.FileInfo].
func (f *fileinfo) IsDir() bool {
	return false
}

// Sys implements [fs.FileInfo].
// We use it to expose the URL of the parquet file.
func (f *fileinfo) Sys() any {
	return f.url
}

type dir struct {
	path    string
	entries []fs.DirEntry
}

func (fsys *datasetFS) buildIndex(parquetFiles []*parquetFile) {
	slices.SortFunc(parquetFiles, func(p, q *parquetFile) int {
		return cmp.Compare(p.URL, q.URL)
	})

	files := make([]fs.DirEntry, len(parquetFiles))
	var dirs []dir
	lastDir := ""
	startDir := 0
	for i, pf := range parquetFiles {
		splitDir := pf.Config + "/" + pf.Split
		// FIXME reject '/' in config or split names
		files[i] = &fileinfo{
			path: splitDir + "/" + pf.Filename,
			name: pf.Filename,
			size: pf.Size,
			url:  pf.URL,
		}
		if splitDir != lastDir {
			if i > 0 {
				dirs[len(dirs)-1].entries = files[startDir:i:i]
			}
			dirs = append(dirs, dir{path: splitDir})
			lastDir = splitDir
			startDir = i
		}
	}
	if len(dirs) > 0 {
		dirs[len(dirs)-1].entries = files[startDir:len(files):len(files)]
	}
	fsys.files = files
	fsys.dirs = dirs
}

// Open implements [fs.FS]. It supports three levels of paths:
// - "config": lists the splits available for the config
// - "config/split": lists the parquet files available for the config and split
// - "config/split/filename": opens the parquet file for reading
func (fsys *datasetFS) Open(name string) (fs.File, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrInvalid}
	}
	if fsys.err != nil {
		return nil, fsys.err
	}

	nbSlash := strings.Count(name, "/")
	switch nbSlash {
	case 0:
		if name == "." {
			return &root{dirRead{fsys: fsys, name: "."}}, nil
		}
		return fsys.openConfig(name)
	case 1:
		return fsys.openSplit(name)
	case 2:
		return fsys.openFile(name)
	default:
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}
}

// ReadDir implements [fs.ReadDirFS] for the datasetFS, which allows listing directories in the filesystem.
func (fsys *datasetFS) ReadDir(name string) ([]fs.DirEntry, error) {
	// Our implementation of ReadDir is just a wrapper around Open + ReadDir, which is simpler to implement and test.
	// Compared to [fs.ReadDir] it just avoids sorting entries as they are already sorted by the buildIndex method.

	f, err := fsys.Open(name)
	if err != nil {
		if pe, ok := err.(*fs.PathError); ok {
			pe.Op = "readdir"
		}
		return nil, err
	}
	defer f.Close()
	dir, ok := f.(fs.ReadDirFile)
	if !ok {
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: errors.New("not a directory")}
	}
	return dir.ReadDir(-1)
}

// dirRead is the common reprresentation of the root directory, a config directory, or a split directory
// being opened or listed.
type dirRead struct {
	fsys    *datasetFS
	name    string
	entries []fs.DirEntry
}

type dirInfo string

var (
	_ fs.FileInfo    = dirInfo("")
	_ fs.DirEntry    = dirInfo("")
	_ fs.File        = (*dirRead)(nil)
	_ fs.ReadDirFile = (*dirRead)(nil)
)

// Name implements [fs.DirEntry] and [fs.FileInfo].
func (d dirInfo) Name() string {
	return string(d)
}

// IsDir implements [fs.DirEntry] and [fs.FileInfo].
func (d dirInfo) IsDir() bool {
	return true
}

// Mode implements [fs.FileInfo].
func (d dirInfo) Mode() fs.FileMode {
	return fs.ModeDir | 0555
}

// Type implements [fs.DirEntry] and [fs.FileInfo].
func (d dirInfo) Type() fs.FileMode {
	return fs.ModeDir
}

// Info implements [fs.DirEntry].
func (d dirInfo) Info() (fs.FileInfo, error) {
	return d, nil
}

// Size implements [fs.FileInfo].
func (d dirInfo) Size() int64 {
	return 0
}

// ModTime implements [fs.FileInfo].
func (d dirInfo) ModTime() time.Time {
	return time.Time{}
}

// Sys implements [fs.FileInfo].
func (d dirInfo) Sys() any {
	return nil
}

// Stat implements [fs.File].
func (d *dirRead) Stat() (fs.FileInfo, error) {
	if d.fsys == nil {
		return nil, fs.ErrClosed
	}
	return dirInfo(d.name), nil
}

// Read implements [fs.File] and always returns an error as directories cannot be read.
func (*dirRead) Read([]byte) (n int, err error) {
	return 0, fs.ErrInvalid
}

// ReadDir implements [fs.ReadDirFile] and returns the entries of the directory.
func (d *dirRead) ReadDir(n int) ([]fs.DirEntry, error) {
	if d.fsys == nil {
		return nil, fs.ErrClosed
	}
	if n <= 0 {
		entries := d.entries
		d.entries = []fs.DirEntry{} // avoid returning the same entries again
		return entries, nil
	}
	if len(d.entries) == 0 {
		return nil, io.EOF
	}
	if n > len(d.entries) {
		n = len(d.entries)
	}
	entries := d.entries[:n]
	d.entries = d.entries[n:]
	if len(d.entries) == 0 {
		return entries, io.EOF
	}
	return entries, nil
}

// Close implements [fs.File].
func (d *dirRead) Close() error {
	if d.fsys == nil {
		return fs.ErrClosed
	}
	d.fsys = nil
	d.entries = nil
	return nil
}

// root represents the root directory of the dataset, which lists the config directories.
type root struct {
	dirRead
}

var (
	_ fs.File        = (*root)(nil)
	_ fs.ReadDirFile = (*root)(nil)
)

// ReadDir implements [fs.ReadDirFile] for the root directory, which lists the config directories.
func (r *root) ReadDir(n int) ([]fs.DirEntry, error) {
	if r.dirRead.fsys == nil {
		return nil, fs.ErrClosed
	}
	if r.dirRead.entries == nil {
		var configs []fs.DirEntry
		lastConfig := ""
		for i := range r.fsys.dirs {
			p := r.fsys.dirs[i]
			configName, _, _ := strings.Cut(p.path, "/")
			if configName != lastConfig {
				configs = append(configs, dirInfo(configName))
				lastConfig = configName
			}
		}
		r.dirRead.entries = configs
	}
	return r.dirRead.ReadDir(n)
}

// configEntry represents a config directory, which lists the split directories.
type configEntry struct {
	dirRead
}

func (dc *configEntry) ReadDir(n int) ([]fs.DirEntry, error) {
	if dc.dirRead.fsys == nil {
		return nil, fs.ErrClosed
	}
	if dc.dirRead.entries == nil {
		var entries []fs.DirEntry
		dirSlash := dc.name + "/"
		lenDirSlash := len(dirSlash)
		allSplits := dc.dirRead.fsys.dirs
		for i := range allSplits {
			if strings.HasPrefix(allSplits[i].path, dirSlash) {
				entries = append(entries, dirInfo(allSplits[i].path[lenDirSlash:]))
			}
		}
		dc.dirRead.entries = entries
	}
	return dc.dirRead.ReadDir(n)
}

func (fsys *datasetFS) openConfig(name string) (*configEntry, error) {
	configSlash := name + "/"
	_, found := slices.BinarySearchFunc(fsys.dirs, configSlash, func(d dir, configSlash string) int {
		diff := cmp.Compare(d.path, configSlash)
		if diff < 0 {
			return -1
		}
		if strings.HasPrefix(d.path, configSlash) {
			return 0
		}
		return 1
	})
	if !found {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}

	return &configEntry{
		dirRead: dirRead{
			fsys:    fsys,
			name:    name,
			entries: nil,
		},
	}, nil
}

func (fsys *datasetFS) openSplit(name string) (*dirRead, error) {
	i, found := slices.BinarySearchFunc(fsys.dirs, name, func(d dir, name string) int {
		return cmp.Compare(d.path, name)
	})
	if !found {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}

	_, name, _ = strings.Cut(name, "/")

	return &dirRead{
		fsys:    fsys,
		name:    name,
		entries: fsys.dirs[i].entries,
	}, nil
}

func (fsys *datasetFS) openFile(name string) (fs.File, error) {
	i, found := slices.BinarySearchFunc(fsys.files, name, func(de fs.DirEntry, name string) int {
		return cmp.Compare(de.(*fileinfo).path, name)
	})
	if !found {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}

	return &file{
		fsys: fsys,
		info: fsys.files[i].(*fileinfo),
	}, nil
}

type file struct {
	fsys *datasetFS
	info *fileinfo
	r    io.ReadCloser
}

func (f *file) Close() error {
	if f.fsys == nil {
		return fs.ErrClosed
	}
	if f.r != nil {
		r := f.r
		f.r = nil
		f.fsys = nil
		return r.Close()
	}
	return nil
}

func (f *file) Stat() (fs.FileInfo, error) {
	if f.fsys == nil {
		return nil, fs.ErrClosed
	}
	return f.info, nil
}

func (f *file) Read(p []byte) (n int, err error) {
	if f.fsys == nil {
		return 0, fs.ErrClosed
	}
	if f.r == nil {
		resp, err := f.fsys.client.Get(f.info.url)
		if err != nil {
			return 0, err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return 0, errors.New("unexpected status code: " + resp.Status)
		}
		f.r = resp.Body
	}
	return f.r.Read(p)
}
