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
