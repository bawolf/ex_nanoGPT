defmodule ExNanoGPTWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :ex_nano_gpt

  @session_options [
    store: :cookie,
    key: "_ex_nano_gpt_key",
    signing_salt: "nanochat_elixir",
    same_site: "Lax"
  ]

  socket "/live", Phoenix.LiveView.Socket,
    websocket: [connect_info: [session: @session_options]]

  plug Plug.Static,
    at: "/",
    from: :ex_nano_gpt,
    gzip: false,
    only: ~w(assets)

  plug Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Jason

  plug Plug.Session, @session_options
  plug ExNanoGPTWeb.Router
end
