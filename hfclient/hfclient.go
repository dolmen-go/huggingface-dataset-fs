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

package hfclient

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

type Client struct {
	credentials string
	logger      *slog.Logger
}

func NewClient(opts Options) *Client {
	if opts.Token == "" {
		opts.Token = os.Getenv("HF_TOKEN")
	}

	c := &Client{
		credentials: opts.Token,
		logger:      opts.Logger,
	}
	if c.logger == nil {
		c.logger = slog.Default()
	}
	return c
}

type Options struct {
	Token  string
	Logger *slog.Logger
}

func (c *Client) HTTPClient(_ context.Context) (*http.Client, error) {
	auth := ""
	if c.credentials != "" {
		auth = "Bearer " + c.credentials
	}

	client := *http.DefaultClient
	client.Transport = &transport{
		baseTransport:     http.DefaultTransport,
		credentialsBearer: auth,
		logger:            c.logger.WithGroup("http").With(slog.String("component", "transport")),
	}
	return &client, nil
}

type baseTransport = http.RoundTripper

type transport struct {
	baseTransport
	credentialsBearer string

	logger *slog.Logger
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req
	t.logger.DebugContext(req.Context(), "RoundTrip", slog.String("URL", req.URL.String()))
	const tld = "huggingface.co"
	if t.credentialsBearer != "" && (req.Host == tld || strings.HasSuffix(req.Host, "."+tld)) && req.URL.Scheme == "https" {
		clone = req.Clone(req.Context())
		clone.URL.User = nil
		clone.Header.Set("Authorization", t.credentialsBearer)
	}
	r, err := t.baseTransport.RoundTrip(clone)
	if err != nil {
		return nil, err
	}
	if r.Request != req {
		// Do not leak the credentials to the caller.
		r.Request.Header.Del("Authorization")
	}
	return r, nil
}
