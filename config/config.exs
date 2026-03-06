import Config

case System.get_env("NX_BACKEND", "emlx") do
  "exla" ->
    config :nx, default_backend: EXLA.Backend
    config :nx, :default_defn_options, compiler: EXLA

  "emlx_cpu" ->
    config :nx, default_backend: {EMLX.Backend, device: :cpu}
    config :nx, :default_defn_options, compiler: EMLX

  _emlx ->
    config :nx, default_backend: {EMLX.Backend, device: :gpu}
    config :nx, :default_defn_options, compiler: EMLX
end

config :ex_nano_gpt, ExNanoGPTWeb.Endpoint,
  url: [host: "localhost"],
  http: [port: 4000],
  adapter: Bandit.PhoenixAdapter,
  secret_key_base: String.duplicate("nanochat_elixir_secret_", 4),
  live_view: [signing_salt: "nanochat_lv"],
  render_errors: [formats: [html: ExNanoGPTWeb.Layouts]],
  server: false
