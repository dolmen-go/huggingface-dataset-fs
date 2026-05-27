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
	err     error
	files   []fs.DirEntry
	dirs    []dir
}

type Options struct {
	BaseURL string
}

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

type fileinfo struct {
	name     string
	splitDir string
	size     int64
	url      string
}

func (f *fileinfo) Info() (fs.FileInfo, error) {
	return f, nil
}

func (f *fileinfo) Name() string {
	return f.name
}

func (f *fileinfo) Size() int64 {
	return f.size
}

func (f *fileinfo) Mode() fs.FileMode {
	return 0444
}

func (f *fileinfo) Type() fs.FileMode {
	return 0
}

func (f *fileinfo) ModTime() time.Time {
	return time.Time{}
}

func (f *fileinfo) IsDir() bool {
	return false
}

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
		dirPath := pf.Config + "/" + pf.Split
		// FIXME reject '/' in config or split names
		files[i] = &fileinfo{
			name:     pf.Filename,
			splitDir: dirPath,
			size:     pf.Size,
			url:      pf.URL,
		}
		if dirPath != lastDir {
			if i > 0 {
				dirs[len(dirs)-1].entries = files[startDir:i:i]
			}
			dirs = append(dirs, dir{path: dirPath})
			lastDir = dirPath
			startDir = i
		}
	}
	if len(dirs) > 0 {
		dirs[len(dirs)-1].entries = files[startDir:len(files):len(files)]
	}
	fsys.files = files
	fsys.dirs = dirs
}

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
			return &root{dirEntry{fsys: fsys, name: "."}}, nil
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

func (fsys *datasetFS) ReadDir(name string) ([]fs.DirEntry, error) {
	// Implementation is similar to Open, but we just avoid sort of entries
	// as they are already sorted in buildIndex.

	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: fs.ErrInvalid}
	}
	if fsys.err != nil {
		return nil, fsys.err
	}

	nbSlash := strings.Count(name, "/")
	var dir fs.ReadDirFile
	var err error
	switch nbSlash {
	case 0:
		if name == "." {
			dir = (&root{dirEntry{fsys: fsys, name: "."}})
		} else {
			dir, err = fsys.openConfig(name)
		}
	case 1:
		dir, err = fsys.openSplit(name)
	case 2:
		f, err := fsys.openFile(name)
		if err != nil {
			return nil, err
		}
		f.Close()
		// a file is never a directory
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: fs.ErrInvalid}
	default:
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: fs.ErrNotExist}
	}
	if err != nil {
		return nil, err
	}
	entries, err := dir.ReadDir(-1)
	dir.Close()
	return entries, err
}

// dirEntry is the common reprresentation of the root directory, a config directory, or a split directory
// being opened.
type dirEntry struct {
	fsys    *datasetFS
	name    string
	entries []fs.DirEntry
}

var (
	_ fs.FileInfo    = (*dirEntry)(nil)
	_ fs.DirEntry    = (*dirEntry)(nil)
	_ fs.File        = (*dirEntry)(nil)
	_ fs.ReadDirFile = (*dirEntry)(nil)
)

func (d *dirEntry) Name() string {
	return d.name
}

func (d *dirEntry) IsDir() bool {
	return true
}

func (d *dirEntry) Mode() fs.FileMode {
	return fs.ModeDir | 0555
}

func (d *dirEntry) Type() fs.FileMode {
	return fs.ModeDir
}

func (d *dirEntry) Info() (fs.FileInfo, error) {
	return d, nil
}

func (d *dirEntry) Size() int64 {
	return 0
}

func (d *dirEntry) ModTime() time.Time {
	return time.Time{}
}

func (d *dirEntry) Sys() any {
	return nil
}

func (d *dirEntry) Stat() (fs.FileInfo, error) {
	if d.fsys == nil {
		return nil, fs.ErrClosed
	}
	return d, nil
}

func (*dirEntry) Read([]byte) (n int, err error) {
	return 0, fs.ErrInvalid
}

func (d *dirEntry) ReadDir(n int) ([]fs.DirEntry, error) {
	if d.fsys == nil {
		return nil, fs.ErrClosed
	}
	if n <= 0 {
		// ReadDir(-1) must not advance the position, so we return all entries without modifying d.entries.
		return d.entries, nil
	}
	if len(d.entries) == 0 {
		return nil, io.EOF
	}
	if n > len(d.entries) {
		n = len(d.entries)
	}
	entries := d.entries[:n]
	d.entries = d.entries[n:]
	return entries, nil
}

func (d *dirEntry) Close() error {
	if d.fsys == nil {
		return fs.ErrClosed
	}
	d.fsys = nil
	d.entries = nil
	return nil
}

type root struct {
	dirEntry
}

func (r *root) ReadDir(n int) ([]fs.DirEntry, error) {
	if r.dirEntry.fsys == nil {
		return nil, fs.ErrClosed
	}
	if r.dirEntry.entries == nil {
		var configs []fs.DirEntry
		lastConfig := ""
		for i := range r.fsys.dirs {
			p := r.fsys.dirs[i]
			configName, _, _ := strings.Cut(p.path, "/")
			if configName != lastConfig {
				configs = append(configs,
					&configEntry{
						dirEntry: dirEntry{
							fsys:    r.fsys,
							name:    configName,
							entries: nil,
						},
					})
				lastConfig = configName
			}
		}
		r.dirEntry.entries = configs
	}
	return r.dirEntry.ReadDir(n)
}

type configEntry struct {
	dirEntry
}

func (dc *configEntry) ReadDir(n int) ([]fs.DirEntry, error) {
	if dc.dirEntry.fsys == nil {
		return nil, fs.ErrClosed
	}
	if dc.dirEntry.entries == nil {
		var entries []fs.DirEntry
		dirSlash := dc.name + "/"
		allSplits := dc.dirEntry.fsys.dirs
		for i := range allSplits {
			if strings.HasPrefix(allSplits[i].path, dirSlash) {
				entries = append(entries,
					&dirEntry{
						fsys:    dc.fsys,
						name:    strings.TrimPrefix(allSplits[i].path, dirSlash),
						entries: allSplits[i].entries,
					},
				)
			}
		}
		dc.dirEntry.entries = entries
	}
	return dc.dirEntry.ReadDir(n)
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
		dirEntry: dirEntry{
			fsys:    fsys,
			name:    name,
			entries: nil,
		},
	}, nil
}

func (fsys *datasetFS) openSplit(name string) (*dirEntry, error) {
	i, found := slices.BinarySearchFunc(fsys.dirs, name, func(d dir, name string) int {
		return cmp.Compare(d.path, name)
	})
	if !found {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}

	_, name, _ = strings.Cut(name, "/")

	return &dirEntry{
		fsys:    fsys,
		name:    name,
		entries: fsys.dirs[i].entries,
	}, nil
}

func (fsys *datasetFS) openFile(name string) (fs.File, error) {
	p := strings.LastIndex(name, "/")
	splitDir := name[:p]
	filename := name[p+1:]
	i, found := slices.BinarySearchFunc(fsys.files, splitDir, func(de fs.DirEntry, splitDir string) int {
		diff := cmp.Compare(de.(*fileinfo).splitDir, splitDir)
		if diff != 0 {
			return diff
		}
		return cmp.Compare(de.(*fileinfo).name, filename)
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
